from __future__ import annotations

import numpy as np
import tensorflow as tf

from configs.base import BaseConfig
from configs.schwarzschild import SchwarzschildConfig
from geometry.base import compute_ricci_tensor
from geometry.ricci_opt import (
    compute_ricci_and_kretschmann_embed_opt,
    speciality_index_from_invariants,
)
from geometry.schwarzschild import (Analytic_Kretschmann,
                                    PenroseRadiusWeighting,
                                    compute_kretschmann_scalar_embed,
                                    compute_ricci_and_kretschmann_embed,
                                    compute_ricci_tensor_embed,
                                    embed_S2_coords,
                                    embedding_jacobian_stereo,
                                    riemannian_inverse_metric_embed)
from helper_functions.helper_functions import RadiusWeighting, cholesky_from_vec
from losses.base import WeightBase
from network.schedulers import FloatScheduler
from sampling.penrose import disc_to_penrose_tf

tf.keras.backend.set_floatx("float64")


def euclidean_inverse_metric(metric_mat: tf.Tensor) -> tf.Tensor:
    """Return an SPD ("Euclidean") inverse of a symmetric (possibly indefinite,
    Lorentzian) metric, obtained by flipping the sign of negative eigenvalues:

        g = V diag(lambda) V^T   ->   g_E^{-1} = V diag(1/|lambda|) V^T.

    This guarantees the contracted loss norm error_ij g_E^{ik} g_E^{jl} error_kl
    = || g_E^{-1/2} error g_E^{-1/2} ||_F^2 is non-negative (a genuine norm),
    which the indefinite Lorentzian inverse does not.  For a diagonal Lorentzian
    metric (e.g. the analytic Schwarzschild diag(-F, F, s2, s2)) this spectral
    construction coincides exactly with the eta -> I reconstruction.

    Use this only where the eta -> I pulled-back inverse
    ``riemannian_inverse_metric_embed`` (which requires the submodel + intrinsic
    coords) is unavailable, e.g. the supervised analytic-target inverse.  In the
    embedding losses prefer ``riemannian_inverse_metric_embed`` so the Euclidean
    inverse matches the Einstein-loss convention exactly.
    """
    eigvals, eigvecs = tf.linalg.eigh(metric_mat)
    inv_abs_eigvals = 1.0 / tf.abs(eigvals)
    return tf.einsum("sij,sj,skj->sik", eigvecs, inv_abs_eigvals, eigvecs)


# Registry of Ricci / Kretschmann kernels.  Both entries share the same
# signature:
#   (x_vars, submodel, lorentzian, need_ricci, need_kretschmann,
#    need_speciality_index)
#     -> (metric (batch, 4, 4),
#         ricci  (batch, 4, 4) or None,
#         kretschmann (batch,) or None,
#         weyl_i (batch,) or None,
#         weyl_j (batch,) or None)
# so ``TotalSchwarzschildLoss`` can swap them out transparently.
_RICCI_KERNELS = {
    "standard": compute_ricci_and_kretschmann_embed,
    "optimised": compute_ricci_and_kretschmann_embed_opt,
}


def relative_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Per-component relative MSE: mean( (pred-true)^2 / (true^2 + 1) ).

    Using eps=1 means:
      - Large-magnitude components (|true| >> 1) contribute like a fractional error.
      - Near-zero components (off-diagonal entries that should be 0) contribute an
        absolute squared error with unit scale.
    This makes the loss scale-invariant across the Schwarzschild metric's large
    dynamic range and prevents high-F singularity-region samples from dominating.
    """
    eps = tf.constant(1.0, dtype=tf.float64)
    return tf.reduce_mean(tf.square(y_pred - y_true) / (tf.square(y_true) + eps))


def make_supervised_metric_loss(
    use_area_measure_weight: bool = False,
    use_metric_contraction: bool = False,
):
    """Factory returning a supervised loss that mirrors the unsupervised Einstein
    loss options for area-measure weighting and metric contraction.

    The analytic (target) metric encoded in ``y_true`` (shape ``(batch, 16)``)
    is used for both the inverse-metric contraction and the area-measure weight,
    exactly as the predicted metric is used in the unsupervised case.

    Args:
        use_area_measure_weight: multiply per-sample norm by sqrt(|det(g_analytic)|).
        use_metric_contraction:  contract the error tensor with the inverse analytic
            metric, i.e. norm_s = |error_ij g^{ik} g^{jl} error_kl|, instead of
            the default component-wise relative MSE.
    """

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        g_analytic = tf.reshape(y_true, [-1, 4, 4])  # (batch, 4, 4)
        g_pred = tf.reshape(y_pred, [-1, 4, 4])  # (batch, 4, 4)
        error = g_pred - g_analytic  # (batch, 4, 4)

        if use_metric_contraction:
            # Euclidean (SPD) inverse of the analytic target so the contracted
            # norm is genuinely non-negative. No submodel is available in this
            # Keras-fit closure, so use the spectral construction; for the
            # diagonal analytic Schwarzschild metric it equals the eta -> I
            # inverse used by the unsupervised Einstein loss.
            g_inv = euclidean_inverse_metric(g_analytic)
            norm = tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error)
            norm = tf.abs(norm)  # SPD inverse => already >= 0; kept as a guard
        else:
            eps = tf.constant(1.0, dtype=tf.float64)
            norm = tf.reduce_sum(
                tf.square(error) / (tf.square(g_analytic) + eps), axis=[1, 2]
            )  # (batch,)

        if use_area_measure_weight:
            area_weight = tf.sqrt(tf.abs(tf.linalg.det(g_analytic)))
            norm = norm * area_weight

        return tf.reduce_mean(norm)

    return loss_fn


def _use_volume_scaling(config: SchwarzschildConfig) -> bool:
    """Return the Schwarzschild volume-scaling flag with legacy-name support."""
    model_specific = config.model_specific
    use_volume_scaling = getattr(model_specific, "use_volume_scaling", None)
    if use_volume_scaling is not None:
        return bool(use_volume_scaling)
    return bool(getattr(model_specific, "use_area_measure_weight", False))


_LOSS_COMPONENTS = frozenset(
    {
        "einstein",
        "kretschmann",
        "r2_det",
        "killing_symmetry",
        "k_repeller",
        "speciality_index_rprofile",
    }
)


def _component_flag(
    config: SchwarzschildConfig,
    mapping_name: str,
    component: str,
    default: bool,
) -> bool:
    mapping = getattr(config.model_specific, mapping_name, None)
    if mapping is None:
        return default

    unknown = set(mapping) - _LOSS_COMPONENTS
    if unknown:
        raise ValueError(
            f"Unknown Schwarzschild loss component(s) in {mapping_name}: "
            f"{sorted(unknown)}. Expected components: {sorted(_LOSS_COMPONENTS)}."
        )
    return bool(mapping.get(component, default))


def _use_component_volume_scaling(
    config: SchwarzschildConfig, component: str
) -> bool:
    return _component_flag(
        config,
        "volume_scaling_loss_components",
        component,
        _use_volume_scaling(config),
    )


def _use_component_metric_contraction(
    config: SchwarzschildConfig, component: str
) -> bool:
    return _component_flag(
        config,
        "metric_contraction_loss_components",
        component,
        bool(getattr(config.model_specific, "use_metric_contraction", False)),
    )


def _apply_volume_scaling(
    norm: tf.Tensor,
    metric_pred_mat: tf.Tensor | None,
    enabled: bool,
) -> tf.Tensor:
    if not enabled:
        return norm
    if metric_pred_mat is None:
        return norm
    volume_weight = tf.sqrt(tf.abs(tf.linalg.det(metric_pred_mat)))
    return norm * volume_weight


def _metric_contracted_rank2_norm(
    tensor: tf.Tensor,
    inv_metric: tf.Tensor,
) -> tf.Tensor:
    """Contract a symmetric rank-2 tensor with a *provided* inverse metric:
    norm_s = | tensor_ij inv^{ik} inv^{jl} tensor_kl |.

    The inverse metric is passed in (rather than computed here) so the caller
    can supply a Euclidean (SPD) inverse: with an SPD inverse the contraction
    equals || inv^{1/2} tensor inv^{1/2} ||_F^2 >= 0 (a genuine norm), whereas
    the indefinite Lorentzian inverse would make the quadratic form sign-
    indefinite (only forced non-negative by the abs below).
    """
    norm = tf.einsum("sij,sik,sjl,skl->s", tensor, inv_metric, inv_metric, tensor)
    return tf.abs(norm)


class WeightSchwarzschild(WeightBase):
    def __init__(self, config: BaseConfig, weight: bool = True) -> None:
        super().__init__(config, weight)

    def _weight_patch(self, x_vars: tf.Tensor, norm: tf.Tensor):
        # Do nothing
        return norm


class EinsteinLossEmbed:
    """
    Einstein equation loss for the S^2-embedding architecture.

    Calls compute_ricci_tensor_embed which differentiates through the
    stereographic embedding map, so all chain-rule corrections are included.
    """

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.config = config
        self.weighter = weighter
        self.einstein_constant = config.geometry.einstein_constant
        self.lorentzian = getattr(config.model_specific, "lorentzian", False)
        self.use_volume_scaling = _use_component_volume_scaling(config, "einstein")
        self.use_metric_contraction = _use_component_metric_contraction(
            config, "einstein"
        )
        # (3) Einstein-residual form. "curvature_normalized" divides the
        # SPD-contracted residual by (|K| + eps) instead of weighting by the
        # volume element, yielding a homothety-invariant (intensive) loss.
        self.loss_mode = getattr(
            config.model_specific, "einstein_loss_mode", "volume_integral"
        )
        self.curvature_norm_epsilon = float(
            getattr(config.model_specific, "curvature_norm_epsilon", 1e-3)
        )
        # (3b) capped curvature normalization. Cap the reference curvature in the
        # denominator at K_cap = kappa * K_hor, the horizon Kretschmann scale,
        # K_hor = 48 m^2 / (2m)^6 = 0.75 / m^4. Only used for the
        # "curvature_normalized_capped" mode; precomputed here as a constant.
        self.curvature_norm_cap_kappa = float(
            getattr(config.model_specific, "curvature_norm_cap_kappa", 2.0)
        )
        _m = float(getattr(config.model_specific, "m", 1.0))
        self._k_hor = 48.0 * _m * _m / ((2.0 * _m) ** 6)  # = 0.75 / m^4
        self._k_cap = self.curvature_norm_cap_kappa * self._k_hor

    def compute(self, x_vars, metric_pred, model):
        """
        Args:
            x_vars:      (batch, 5) = [T, X, q1, q2, patch_idx_float]
            metric_pred: (batch, 4, 4) metric matrix
            model:       SchwarzschildGlobalModel (exposes .submodel)
        """
        ricci_tensor = compute_ricci_tensor_embed(
            x_vars, model.submodel, self.lorentzian
        )
        kretschmann_scalar = None
        if self.loss_mode in ("curvature_normalized", "curvature_normalized_capped"):
            kretschmann_scalar = compute_kretschmann_scalar_embed(
                x_vars, model.submodel, self.lorentzian
            )
        return self.compute_from_precomputed(
            x_vars, metric_pred, ricci_tensor, model.submodel, kretschmann_scalar
        )

    def compute_from_precomputed(
        self, x_vars, metric_pred_mat, ricci_tensor, submodel, kretschmann_scalar=None,
        weyl_i=None,
    ):
        """Compute Einstein loss from already-computed metric and Ricci tensor.

        weyl_i (the complex Weyl invariant I, optional) is used only by the
        "weyl_normalized" mode; the other modes ignore it (kept None by default), so
        their behaviour is unchanged.
        """
        error = self.einstein_constant * metric_pred_mat - ricci_tensor

        if self.loss_mode == "weyl_normalized":
            # Homothety-invariant, Ricci-DECOUPLED form: ||C g - Ric||^2_M / (|I| + eps),
            # |I| = |weyl_i| the self-dual Weyl invariant. Like curvature_normalized
            # (numerator and |I| both scale as lambda^-2 under g -> lambda g, so the
            # ratio is homothety-invariant -> inflation cannot game it), but normalising
            # by the Weyl part ALONE: the denominator is independent of Ricci, so it
            # cannot be lowered by trading Ricci into K (K = C^2 + 2|Ric|^2 - R^2/3).
            g_inv = riemannian_inverse_metric_embed(x_vars, submodel)
            num = tf.abs(tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error))
            denom = tf.abs(weyl_i) + tf.cast(self.curvature_norm_epsilon, num.dtype)
            norm = num / denom
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        if self.loss_mode == "curvature_normalized":
            # Intensive form: ||C g - Ric||^2_contracted / (|K| + eps). Both the
            # SPD-contracted residual and |K| scale as s^{-2} under g -> s g, so
            # the ratio is homothety-invariant -> cannot be reduced by shrinking
            # the metric. No volume factor, no extra metric-contraction toggle.
            g_inv = riemannian_inverse_metric_embed(x_vars, submodel)
            num = tf.abs(tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error))
            denom = tf.abs(kretschmann_scalar) + tf.cast(
                self.curvature_norm_epsilon, num.dtype
            )
            norm = num / denom
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        if self.loss_mode == "curvature_normalized_capped":
            # Same intensive SPD-contracted residual, but the reference curvature
            # is CAPPED at K_cap = kappa * K_hor (horizon Kretschmann). For
            # |K| < K_cap (exterior/bulk) this is exactly curvature_normalized;
            # for |K| > K_cap (interior/near-singularity, where 1/|K| suppresses
            # the loss ~12x) the denominator saturates at the fixed horizon scale,
            # restoring uniform Ricci-flatness pressure in the strong field.
            # Mass-scaled BY DESIGN (the cap is the physical curvature scale); the
            # det barrier prevents the metric collapse a homothety knob could give.
            g_inv = riemannian_inverse_metric_embed(x_vars, submodel)
            num = tf.abs(tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error))
            k_cap = tf.cast(self._k_cap, num.dtype)
            denom = tf.minimum(tf.abs(kretschmann_scalar), k_cap) + tf.cast(
                self.curvature_norm_epsilon, num.dtype
            )
            norm = num / denom
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        if self.loss_mode == "contracted_plain":
            # Uniform-weighted SPD-contracted tensorial norm (no volume, no 1/K).
            g_inv = riemannian_inverse_metric_embed(x_vars, submodel)
            norm = tf.abs(tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error))
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        if self.loss_mode == "contracted_volume":
            # SPD-contracted tensorial residual weighted by the proper volume
            # element sqrt(|det g|) = sqrt(-det g) for the Lorentzian metric
            # (NO 1/K normalization): ||C g - Ric||^2_M * sqrt|det g|. The
            # covariant-integral sibling of "contracted_plain" (the volume factor
            # is intrinsic to this mode and applied here unconditionally, so it
            # does NOT depend on the use_volume_scaling flag). Extensive (the det
            # barrier guards against the shrink-the-metric collapse a volume
            # weight would otherwise reward).
            g_inv = riemannian_inverse_metric_embed(x_vars, submodel)
            norm = tf.abs(tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error))
            norm = _apply_volume_scaling(norm, metric_pred_mat, True)
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        if self.use_metric_contraction:
            # Contract D_mn D_ab g^ma g^nb -> scalar; take abs for Lorentzian
            g_inv = riemannian_inverse_metric_embed(x_vars, submodel)
            norm = tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error)
            norm = tf.abs(norm)
        else:
            norm = tf.norm(error, axis=(1, 2))

        norm = _apply_volume_scaling(norm, metric_pred_mat, self.use_volume_scaling)
        norm = self.weighter._weight_patch(x_vars, norm)
        return tf.reduce_mean(norm)


class KretschmannLossEmbed:
    """
    Kretschmann scalar loss for the S^2-embedding architecture.

    Compares the predicted Kretschmann scalar K against the analytic value
    in log-space for scale-invariant training across the Penrose diagram.
    """

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.config = config
        self.weighter = weighter
        self.m = config.model_specific.m
        self.lorentzian = getattr(config.model_specific, "lorentzian", False)
        self.use_volume_scaling = _use_component_volume_scaling(
            config, "kretschmann"
        )
        # (4) Kretschmann "scale" form.
        self.loss_mode = getattr(
            config.model_specific, "kretschmann_loss_mode", "log_profile"
        )
        self.sqrt_epsilon = float(
            getattr(config.model_specific, "kretschmann_sqrt_epsilon", 1e-12)
        )
        # "weyl_invariant" mode: prescribe |weyl_i| (= c*K) to its analytic value.
        # c relates this code's Weyl invariant I = Ctilde^2/32 to K (=1/16: |I|=K/16).
        self.weyl_invariant_norm = float(
            getattr(config.model_specific, "weyl_invariant_norm", 0.0625)
        )
        # "weyl_invariant_alt" non-Schwarzschild prescriber target:
        #   sqrt(|weyl_i|) * r^(weyl_alt_r_power) -> weyl_alt_target.
        self.weyl_alt_r_power = float(
            getattr(config.model_specific, "weyl_alt_r_power", 3.0)
        )
        self.weyl_alt_target = float(
            getattr(config.model_specific, "weyl_alt_target", 1.7320508075688772)
        )

    def compute(self, x_vars, metric_pred, model):
        """
        Args:
            x_vars:      (batch, 5) = [T, X, q1, q2, patch_idx_float]
            metric_pred: (batch, 4, 4) metric matrix; used for area-measure
                         weighting when use_area_measure_weight=True.
            model:       SchwarzschildGlobalModel (exposes .submodel)
        """
        kretschmann_scalar = compute_kretschmann_scalar_embed(
            x_vars, model.submodel, self.lorentzian
        )
        return self.compute_from_precomputed(
            x_vars, kretschmann_scalar, metric_pred_mat=metric_pred
        )

    def compute_from_precomputed(
        self, x_vars, kretschmann_scalar, metric_pred_mat=None, weyl_i=None
    ):
        """Compute Kretschmann loss from an already-computed scalar.

        weyl_i (the complex Weyl invariant I, optional) is used only by the
        "weyl_invariant" mode; existing modes ignore it (kept None by default), so
        their behaviour is unchanged.
        """
        if self.loss_mode in ("weyl_invariant", "weyl_invariant_volume"):
            # Ricci-DECOUPLED prescriber: same sqrt_const form on the Weyl invariant
            # |I| instead of the Kretschmann K. K = C^2 + 2|Ric|^2 - R^2/3 couples to
            # Ricci; |I| does not, so this term no longer rewards R!=0. Target
            #   sqrt(|I|) * r^3 -> sqrt(c*48) * m   (|I| = c*K = 3 m^2/r^6 for c=1/16).
            r = PenroseRadiusWeighting(x_vars[:, :4], m=self.m)
            wi_abs = tf.abs(weyl_i)  # |I|; weyl_i is complex (Im ~ 0 for Schwarzschild)
            sqrt_W = tf.sqrt(wi_abs + tf.cast(self.sqrt_epsilon, wi_abs.dtype))
            pred = sqrt_W * tf.pow(r, 3)
            target = tf.cast(
                tf.sqrt(self.weyl_invariant_norm * 48.0) * self.m, pred.dtype
            )
            norm = tf.square(pred - target)
            if self.loss_mode == "weyl_invariant_volume":
                # Multiply the (unweighted) Weyl-invariant residual by the proper
                # volume element sqrt(|det g|) = sqrt(-det g) (covariant integral
                # measure). "weyl_invariant" is left exactly unweighted, so this
                # branch is a strict, mode-gated extension. Applied here
                # unconditionally (intrinsic to the mode), independent of the
                # use_volume_scaling flag.
                norm = _apply_volume_scaling(norm, metric_pred_mat, True)
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        if self.loss_mode == "weyl_invariant_alt":
            # Ricci-DECOUPLED prescriber with a NON-Schwarzschild target profile:
            #   sqrt(|weyl_i|+eps) * r^p -> T,   prescribing |I| = T^2 / r^(2p),
            # with p = weyl_alt_r_power, T = weyl_alt_target. p != 3 prescribes a curvature
            # falloff different from Schwarzschild's r^-6 (p=3). Like "weyl_invariant" but
            # parametrised; still on |I| (not K), so it does not reward R != 0.
            r = PenroseRadiusWeighting(x_vars[:, :4], m=self.m)
            wi_abs = tf.abs(weyl_i)
            sqrt_W = tf.sqrt(wi_abs + tf.cast(self.sqrt_epsilon, wi_abs.dtype))
            pred = sqrt_W * tf.pow(r, tf.cast(self.weyl_alt_r_power, r.dtype))
            target = tf.cast(self.weyl_alt_target, pred.dtype)
            norm = tf.square(pred - target)
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        if self.loss_mode == "sqrt_const":
            # Degree-2 (Weyl-scalar-like) quantity matched to a CONSTANT target:
            #   sqrt(|K|) * r^3  ->  sqrt(48) * m   (since K = 48 m^2 / r^6).
            # Removes the huge dynamic range of K and its quartic-g^{-1}
            # conditioning. r(T,X) is the analytic areal radius. No volume factor.
            r = PenroseRadiusWeighting(x_vars[:, :4], m=self.m)
            sqrt_K = tf.sqrt(tf.abs(kretschmann_scalar) + tf.cast(self.sqrt_epsilon, kretschmann_scalar.dtype))
            pred = sqrt_K * tf.pow(r, 3)
            target = tf.cast(tf.sqrt(48.0) * self.m, pred.dtype)
            norm = tf.square(pred - target)
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        # Analytic Kretschmann only depends on R^2 (Penrose) coords
        kretschmann_analytic = Analytic_Kretschmann(x_vars[:, :4], m=self.m)

        # Log-space loss for scale invariance across many orders of magnitude
        log_pred = tf.math.log1p(tf.abs(kretschmann_scalar))
        log_analytic = tf.math.log1p(kretschmann_analytic)
        norm = tf.square(log_pred - log_analytic)

        norm = _apply_volume_scaling(norm, metric_pred_mat, self.use_volume_scaling)
        norm = self.weighter._weight_patch(x_vars, norm)
        return tf.reduce_mean(norm)


class R2DetFinitenessLoss:
    """
    Finiteness barrier for the \u211d\u00b2 block of the metric.

    Computes mean(1/|det(g_R2)|) where g_R2 = g[:, :2, :2] is the upper-left
    2\u00d72 Lorentzian block (T, X components).  As |det(g_R2)| \u2192 0 the loss \u2192 \u221e,
    creating a repulsive barrier that prevents the degenerate area-weight
    attractor in which both Lorentzian Cholesky diagonals collapse toward \u03b5.
    """

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.weighter = weighter
        self.use_volume_scaling = _use_component_volume_scaling(config, "r2_det")
        # (5) Barrier scope. "both_blocks" also guards the S^2 block (whose scale
        # is otherwise unconstrained and observed to collapse) and is NEVER
        # volume-scaled (a 1/|det| barrier must not be suppressed by sqrt|det g|
        # exactly where it should diverge).
        self.barrier_mode = getattr(
            config.model_specific, "det_barrier_mode", "r2_only"
        )

    def compute(self, x_vars, metric_pred_mat):
        eps = tf.cast(1e-30, metric_pred_mat.dtype)
        if self.barrier_mode == "both_blocks":
            det_R2 = tf.abs(tf.linalg.det(metric_pred_mat[:, :2, :2]))
            det_S2 = tf.abs(tf.linalg.det(metric_pred_mat[:, 2:, 2:]))
            norm = 1.0 / (det_R2 + eps) + 1.0 / (det_S2 + eps)
            # Deliberately NOT volume-scaled.
            norm = self.weighter._weight_patch(x_vars, norm)
            return tf.reduce_mean(norm)

        g_R2 = metric_pred_mat[:, :2, :2]  # (batch, 2, 2)
        det_R2 = tf.abs(tf.linalg.det(g_R2))  # (batch,)
        norm = 1.0 / (det_R2 + eps)  # (batch,) barrier
        norm = _apply_volume_scaling(norm, metric_pred_mat, self.use_volume_scaling)
        norm = self.weighter._weight_patch(x_vars, norm)
        return tf.reduce_mean(norm)


# Schwarzschild/type-D speciality index in the S = 27 J^2 / I^3 convention.
_SPECIALITY_INDEX_SCHWARZSCHILD = 1.0

# Schwarzschild/type-D value for rho = J_cube / K^(3/2), where
# J_cube = R_ab^cd R_cd^ef R_ef^ab.  The sign of Weyl J depends on convention,
# so evaluation reports both signed rho and |rho|.
_RHO_TYPE_D_ABS = 1.0 / np.sqrt(12.0)


def _speciality_index_summary(
    weyl_i: tf.Tensor,
    weyl_j: tf.Tensor,
    weyl_floor: tf.Tensor | float = 1e-6,
    eps_weyl: tf.Tensor | float = 1e-12,
) -> dict[str, float | int]:
    real_dtype = tf.math.real(weyl_i).dtype
    weyl_floor = tf.cast(weyl_floor, real_dtype)
    eps_weyl = tf.cast(eps_weyl, real_dtype)
    valid = tf.abs(weyl_i) > weyl_floor
    n_valid = tf.reduce_sum(tf.cast(valid, tf.int32))
    n_total = tf.size(weyl_i)
    n_valid_int = int(tf.get_static_value(n_valid))
    n_total_int = int(tf.get_static_value(n_total))

    empty = {
        "speciality_index_real_mean": float("nan"),
        "speciality_index_real_median": float("nan"),
        "speciality_index_real_trimmed_mean": float("nan"),
        "speciality_index_real_std": float("nan"),
        "speciality_index_real_trimmed_std": float("nan"),
        "speciality_index_real_trimmed_outlier_count": 0,
        "speciality_index_imag_mean": float("nan"),
        "speciality_index_imag_median": float("nan"),
        "speciality_index_imag_trimmed_mean": float("nan"),
        "speciality_index_imag_std": float("nan"),
        "speciality_index_imag_trimmed_std": float("nan"),
        "speciality_index_imag_trimmed_outlier_count": 0,
        "speciality_index_n_valid": n_valid_int,
        "speciality_index_n_total": n_total_int,
    }
    if n_valid_int == 0:
        return empty

    s_vals = speciality_index_from_invariants(
        tf.boolean_mask(weyl_i, valid),
        tf.boolean_mask(weyl_j, valid),
        eps_weyl,
    )
    s_real = tf.sort(tf.math.real(s_vals))
    s_imag = tf.sort(tf.math.imag(s_vals))
    n = tf.shape(s_real)[0]
    trim = tf.cast(tf.math.floor(0.05 * tf.cast(n, tf.float64)), tf.int32)

    def median(sorted_vals: tf.Tensor) -> tf.Tensor:
        n_vals = tf.shape(sorted_vals)[0]
        mid = n_vals // 2
        return tf.cond(
            tf.equal(n_vals % 2, 0),
            lambda: 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid]),
            lambda: sorted_vals[mid],
        )

    def trimmed_mean(sorted_vals: tf.Tensor) -> tf.Tensor:
        trimmed = tf.cond(
            tf.greater(2 * trim, n - 1),
            lambda: sorted_vals,
            lambda: sorted_vals[trim : n - trim],
        )
        return tf.reduce_mean(trimmed)

    def trimmed_std(sorted_vals: tf.Tensor) -> tf.Tensor:
        trimmed = tf.cond(
            tf.greater(2 * trim, n - 1),
            lambda: sorted_vals,
            lambda: sorted_vals[trim : n - trim],
        )
        return tf.math.reduce_std(trimmed)

    trimmed_outlier_count = 2 * trim
    return {
        "speciality_index_real_mean": float(tf.get_static_value(tf.reduce_mean(s_real))),
        "speciality_index_real_median": float(tf.get_static_value(median(s_real))),
        "speciality_index_real_trimmed_mean": float(
            tf.get_static_value(trimmed_mean(s_real))
        ),
        "speciality_index_real_std": float(tf.get_static_value(tf.math.reduce_std(s_real))),
        "speciality_index_real_trimmed_std": float(
            tf.get_static_value(trimmed_std(s_real))
        ),
        "speciality_index_real_trimmed_outlier_count": int(
            tf.get_static_value(trimmed_outlier_count)
        ),
        "speciality_index_imag_mean": float(tf.get_static_value(tf.reduce_mean(s_imag))),
        "speciality_index_imag_median": float(tf.get_static_value(median(s_imag))),
        "speciality_index_imag_trimmed_mean": float(
            tf.get_static_value(trimmed_mean(s_imag))
        ),
        "speciality_index_imag_std": float(tf.get_static_value(tf.math.reduce_std(s_imag))),
        "speciality_index_imag_trimmed_std": float(
            tf.get_static_value(trimmed_std(s_imag))
        ),
        "speciality_index_imag_trimmed_outlier_count": int(
            tf.get_static_value(trimmed_outlier_count)
        ),
        "speciality_index_n_valid": n_valid_int,
        "speciality_index_n_total": n_total_int,
    }


def _rho_constant_summary(
    weyl_j: tf.Tensor,
    kretschmann_scalar: tf.Tensor,
    kretschmann_floor: tf.Tensor | float = 1e-12,
) -> dict[str, float | int | str]:
    """Summarise rho = J_cube / K^(3/2) on evaluation samples.

    The current Weyl invariant J is the normalized complex cubic Weyl
    invariant used in the speciality index.  With the historical project
    convention, the real cubic invariant is reconstructed as
    J_cube = -96 Re(J).  This sign is convention-sensitive; |rho| is the stable
    comparison against the Schwarzschild/type-D value 1/sqrt(12).
    """
    real_dtype = tf.math.real(weyl_j).dtype
    kretschmann_floor = tf.cast(kretschmann_floor, real_dtype)
    k_abs = tf.abs(tf.cast(kretschmann_scalar, real_dtype))
    j_real = tf.math.real(weyl_j)

    valid = (
        tf.math.is_finite(k_abs)
        & tf.math.is_finite(j_real)
        & tf.math.is_finite(tf.math.imag(weyl_j))
        & (k_abs > kretschmann_floor)
    )
    n_valid = tf.reduce_sum(tf.cast(valid, tf.int32))
    n_total = tf.size(kretschmann_scalar)
    n_valid_int = int(tf.get_static_value(n_valid))
    n_total_int = int(tf.get_static_value(n_total))

    empty = {
        "rho_constant_target_abs": float(_RHO_TYPE_D_ABS),
        "rho_constant_signed_convention": "-96*Re(WeylJ)/abs(K)^(3/2)",
        "rho_constant_n_valid": n_valid_int,
        "rho_constant_n_total": n_total_int,
        "rho_constant_signed_mean": float("nan"),
        "rho_constant_signed_median": float("nan"),
        "rho_constant_signed_trimmed_mean": float("nan"),
        "rho_constant_signed_std": float("nan"),
        "rho_constant_signed_trimmed_std": float("nan"),
        "rho_constant_signed_cov": float("nan"),
        "rho_constant_abs_mean": float("nan"),
        "rho_constant_abs_median": float("nan"),
        "rho_constant_abs_trimmed_mean": float("nan"),
        "rho_constant_abs_std": float("nan"),
        "rho_constant_abs_trimmed_std": float("nan"),
        "rho_constant_abs_cov": float("nan"),
        "rho_constant_abs_target_error_mean": float("nan"),
        "rho_constant_abs_target_error_median": float("nan"),
        "rho_constant_trimmed_outlier_count": 0,
    }
    if n_valid_int == 0:
        return empty

    k_valid = tf.boolean_mask(k_abs, valid)
    j_valid = tf.boolean_mask(j_real, valid)
    denom = tf.pow(k_valid, tf.cast(1.5, real_dtype))
    rho_signed = -tf.cast(96.0, real_dtype) * j_valid / denom
    rho_abs = tf.abs(rho_signed)
    rho_signed_sorted = tf.sort(rho_signed)
    rho_abs_sorted = tf.sort(rho_abs)
    abs_error_sorted = tf.sort(tf.abs(rho_abs - tf.cast(_RHO_TYPE_D_ABS, real_dtype)))
    n = tf.shape(rho_signed_sorted)[0]
    trim = tf.cast(tf.math.floor(0.05 * tf.cast(n, tf.float64)), tf.int32)

    def median(sorted_vals: tf.Tensor) -> tf.Tensor:
        n_vals = tf.shape(sorted_vals)[0]
        mid = n_vals // 2
        return tf.cond(
            tf.equal(n_vals % 2, 0),
            lambda: 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid]),
            lambda: sorted_vals[mid],
        )

    def trimmed(sorted_vals: tf.Tensor) -> tf.Tensor:
        return tf.cond(
            tf.greater(2 * trim, n - 1),
            lambda: sorted_vals,
            lambda: sorted_vals[trim : n - trim],
        )

    def trimmed_mean(sorted_vals: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(trimmed(sorted_vals))

    def trimmed_std(sorted_vals: tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_std(trimmed(sorted_vals))

    def cov(vals: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(vals)
        return tf.math.reduce_std(vals) / (tf.abs(mean) + tf.cast(1e-30, real_dtype))

    trimmed_outlier_count = 2 * trim
    return {
        "rho_constant_target_abs": float(_RHO_TYPE_D_ABS),
        "rho_constant_signed_convention": "-96*Re(WeylJ)/abs(K)^(3/2)",
        "rho_constant_n_valid": n_valid_int,
        "rho_constant_n_total": n_total_int,
        "rho_constant_signed_mean": float(tf.get_static_value(tf.reduce_mean(rho_signed))),
        "rho_constant_signed_median": float(tf.get_static_value(median(rho_signed_sorted))),
        "rho_constant_signed_trimmed_mean": float(
            tf.get_static_value(trimmed_mean(rho_signed_sorted))
        ),
        "rho_constant_signed_std": float(tf.get_static_value(tf.math.reduce_std(rho_signed))),
        "rho_constant_signed_trimmed_std": float(
            tf.get_static_value(trimmed_std(rho_signed_sorted))
        ),
        "rho_constant_signed_cov": float(tf.get_static_value(cov(rho_signed))),
        "rho_constant_abs_mean": float(tf.get_static_value(tf.reduce_mean(rho_abs))),
        "rho_constant_abs_median": float(tf.get_static_value(median(rho_abs_sorted))),
        "rho_constant_abs_trimmed_mean": float(
            tf.get_static_value(trimmed_mean(rho_abs_sorted))
        ),
        "rho_constant_abs_std": float(tf.get_static_value(tf.math.reduce_std(rho_abs))),
        "rho_constant_abs_trimmed_std": float(
            tf.get_static_value(trimmed_std(rho_abs_sorted))
        ),
        "rho_constant_abs_cov": float(tf.get_static_value(cov(rho_abs))),
        "rho_constant_abs_target_error_mean": float(
            tf.get_static_value(tf.reduce_mean(abs_error_sorted))
        ),
        "rho_constant_abs_target_error_median": float(
            tf.get_static_value(median(abs_error_sorted))
        ),
        "rho_constant_trimmed_outlier_count": int(
            tf.get_static_value(trimmed_outlier_count)
        ),
    }


class KillingSymmetryLossEmbed:
    """SO(3) spherical-symmetry loss from finite-difference Lie derivatives.

    The three rotational Killing fields are written in stereographic S2 chart
    coordinates.  For each generator xi, the loss approximates

        L_xi g = (phi_eps^* g - g) / eps

    and penalises the squared Frobenius norm relative to the metric size.
    """

    _EPS_FLOW: float = 1e-4
    _EPS_NORM: float = 1e-12

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.config = config
        self.weighter = weighter
        self.lorentzian = getattr(config.model_specific, "lorentzian", False)
        self.use_volume_scaling = _use_component_volume_scaling(
            config, "killing_symmetry"
        )
        self.use_metric_contraction = _use_component_metric_contraction(
            config, "killing_symmetry"
        )

    def _metric_from_submodel(self, x_vars: tf.Tensor, submodel) -> tf.Tensor:
        q_4d = x_vars[:, :4]
        patch_idx = tf.cast(x_vars[:, 4], tf.int32)
        x_5d = embed_S2_coords(q_4d, patch_idx)
        G_5d_vec = submodel(x_5d)
        G_5d = cholesky_from_vec(G_5d_vec, lorentzian=self.lorentzian)
        J_emb = embedding_jacobian_stereo(q_4d, patch_idx)
        return tf.einsum("sAB,sAm,sBn->smn", G_5d, J_emb, J_emb)

    @staticmethod
    def _killing_fields(q1: tf.Tensor, q2: tf.Tensor):
        zero = tf.zeros_like(q1)
        one = tf.ones_like(q1)
        half = tf.cast(0.5, q1.dtype)
        return (
            ((-q2, q1), (zero, -one, one, zero)),
            (
                (-q1 * q2, -half * (one - tf.square(q1) + tf.square(q2))),
                (-q2, -q1, q1, -q2),
            ),
            (
                (half * (one + tf.square(q1) - tf.square(q2)), q1 * q2),
                (q1, -q2, q2, q1),
            ),
        )

    def compute_from_precomputed(
        self, x_vars: tf.Tensor, metric_pred_mat: tf.Tensor, submodel
    ) -> tf.Tensor:
        dtype = x_vars.dtype
        eps_flow = tf.cast(self._EPS_FLOW, dtype)
        eps_norm = tf.cast(self._EPS_NORM, metric_pred_mat.dtype)
        q1 = x_vars[:, 2]
        q2 = x_vars[:, 3]
        batch = tf.shape(x_vars)[0]
        eye = tf.eye(4, batch_shape=[batch], dtype=metric_pred_mat.dtype)
        if self.use_metric_contraction:
            # Euclidean (SPD) inverse metric, identical to the Einstein-loss
            # convention (riemannian_inverse_metric_embed = inv(J^T (L L^T) J)),
            # so the metric-contracted Killing residual is a genuine non-negative
            # norm rather than a sign-indefinite Lorentzian contraction.
            g_inv_euclid = riemannian_inverse_metric_embed(x_vars, submodel)
            metric_norm = _metric_contracted_rank2_norm(metric_pred_mat, g_inv_euclid)
        else:
            metric_norm = tf.reduce_sum(tf.square(metric_pred_mat), axis=(1, 2))

        losses = []
        for (xi2, xi3), (d1xi2, d2xi2, d1xi3, d2xi3) in self._killing_fields(q1, q2):
            shifted = tf.concat(
                [
                    x_vars[:, :2],
                    (q1 + eps_flow * xi2)[:, tf.newaxis],
                    (q2 + eps_flow * xi3)[:, tf.newaxis],
                    x_vars[:, 4:5],
                ],
                axis=1,
            )
            metric_shifted = self._metric_from_submodel(shifted, submodel)

            flow_jac = eye
            row2 = tf.stack(
                [
                    tf.zeros_like(q1),
                    tf.zeros_like(q1),
                    1.0 + eps_flow * d1xi2,
                    eps_flow * d2xi2,
                ],
                axis=1,
            )
            row3 = tf.stack(
                [
                    tf.zeros_like(q1),
                    tf.zeros_like(q1),
                    eps_flow * d1xi3,
                    1.0 + eps_flow * d2xi3,
                ],
                axis=1,
            )
            flow_jac = tf.concat(
                [
                    flow_jac[:, :2, :],
                    row2[:, tf.newaxis, :],
                    row3[:, tf.newaxis, :],
                ],
                axis=1,
            )

            pulled_back = tf.einsum(
                "sab,sam,sbn->smn", metric_shifted, flow_jac, flow_jac
            )
            lie_metric = (pulled_back - metric_pred_mat) / eps_flow
            if self.use_metric_contraction:
                # Reuse the Euclidean (SPD) inverse computed above.
                lie_norm = _metric_contracted_rank2_norm(lie_metric, g_inv_euclid)
            else:
                lie_norm = tf.reduce_sum(tf.square(lie_metric), axis=(1, 2))
            rel_norm = lie_norm / (metric_norm + eps_norm)
            rel_norm = _apply_volume_scaling(
                rel_norm, metric_pred_mat, self.use_volume_scaling
            )
            rel_norm = self.weighter._weight_patch(x_vars, rel_norm)
            losses.append(tf.reduce_mean(rel_norm))

        return tf.add_n(losses) / tf.cast(len(losses), metric_pred_mat.dtype)


class KRepellerLoss:
    """
    Minkowski-repeller loss: mean(epsilon / (|K| + epsilon)).

    Penalises near-zero Kretschmann scalar, preventing the network from
    collapsing to the flat (Minkowski) solution.  The loss is large when K ~ 0
    and drops to ~0 for K >> epsilon, so it acts as a bounded soft barrier
    without over-constraining the magnitude of K once it is non-trivial.
    """

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.weighter = weighter
        self.epsilon = getattr(config.model_specific, "k_repeller_epsilon", 1e-4)
        self.use_volume_scaling = _use_component_volume_scaling(config, "k_repeller")

    def compute_from_precomputed(
        self,
        x_vars: tf.Tensor,
        kretschmann_scalar: tf.Tensor,
        metric_pred_mat: tf.Tensor | None = None,
    ) -> tf.Tensor:
        eps = tf.cast(self.epsilon, kretschmann_scalar.dtype)
        norm = eps / (tf.abs(kretschmann_scalar) + eps)  # (batch,)
        norm = _apply_volume_scaling(norm, metric_pred_mat, self.use_volume_scaling)
        norm = self.weighter._weight_patch(x_vars, norm)
        return tf.reduce_mean(norm)


class HorizonAnchorLoss:
    """Horizon curvature anchor (Campaign 7).

    Pins the curvature near the horizon r=2m to a fixed value, which (i) fixes the overall
    scale (so the Kretschmann is physical, no metric inflation), and (ii) enforces a regular,
    FINITE-curvature horizon (vs the divergence of a naked-singularity solution). Because it
    anchors at ONE radius only (the horizon band), the bulk profile stays free to be
    algebraically general (type I) -- unlike the everywhere-prescriber weyl_invariant(_alt),
    which over-constrains and is dragged toward Schwarzschild.

    Built on the SAME degree-2 (Weyl-scalar-like) quantity as the weyl_invariant prescriber,
    sqrt(|inv|+eps)*r^3, which for Schwarzschild is the r-INDEPENDENT constant sqrt(c*48)*m
    (= sqrt(3) m for inv=|I|, c=1/16; = sqrt(48) m for inv=|K|), so it is O(1)-scaled and a
    FINITE target encodes a regular horizon (a naked singularity has inv -> inf, i.e. this
    quantity -> inf). This is precisely the weyl_invariant prescriber RESTRICTED (Gaussian-
    weighted) to the horizon band -- it fixes the scale and horizon regularity while leaving the
    bulk profile free (the everywhere-prescriber over-constrains and is dragged to Schwarzschild).

        L = sum_s w_s (sqrt(|inv_s|+eps)*r_s^3 - target)^2 / sum_s w_s,
        w_s = exp(-((r_s - 2m)/(band*2m))^2),
    inv = |weyl_i| (Ricci-DECOUPLED Weyl invariant I; default, does not fight vacuum) or |K|.
    """

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.weighter = weighter
        ms = config.model_specific
        self.m = float(getattr(ms, "m", 1.0))
        self.target = float(getattr(ms, "horizon_anchor_target", 1.7320508075688772))
        self.band = float(getattr(ms, "horizon_anchor_band", 0.1))
        self.invariant = getattr(ms, "horizon_anchor_invariant", "weyl_i")
        self.sqrt_epsilon = float(getattr(ms, "kretschmann_sqrt_epsilon", 1e-12))

    def compute_from_precomputed(
        self,
        x_vars: tf.Tensor,
        weyl_i: tf.Tensor | None = None,
        kretschmann_scalar: tf.Tensor | None = None,
    ) -> tf.Tensor:
        if self.invariant == "kretschmann":
            inv = tf.abs(kretschmann_scalar)
        else:
            inv = tf.abs(weyl_i)  # |I|; weyl_i complex -> abs is real
        real_dtype = inv.dtype
        r = tf.cast(PenroseRadiusWeighting(x_vars[:, :4], m=self.m), real_dtype)
        rh = tf.cast(2.0 * self.m, real_dtype)
        sigma = tf.cast(self.band, real_dtype) * rh
        w = tf.exp(-tf.square((r - rh) / sigma))  # (batch,) horizon-band weight
        pred = tf.sqrt(inv + tf.cast(self.sqrt_epsilon, real_dtype)) * tf.pow(
            r, tf.cast(3.0, real_dtype)
        )
        target = tf.cast(self.target, real_dtype)
        sq = tf.square(pred - target)
        sq = self.weighter._weight_patch(x_vars, sq)
        wsum = tf.reduce_sum(w) + tf.cast(1e-30, real_dtype)
        return tf.reduce_sum(w * sq) / wsum


class TrappedSurfaceLoss:
    """Trapped-surface (Misner-Sharp) prescriber (Campaign 7b, LC-0017).

    Supplies the black-hole CAUSAL structure that the c7 type-I near-vacuum lacks. The
    Misner-Sharp invariant of the round-S^2 foliation,

        chi = g^{mn} d_m R d_n R,   R^2 = sqrt(det h_ang) * (1+|q|^2)^2 / 4  (= r^2 for round S^2),

    is the squared norm of the areal-radius gradient; for Schwarzschild chi = 1 - 2m/r, so
    chi < 0 <=> the 2-sphere is TRAPPED, chi = 0 <=> marginally trapped (the horizon), chi > 0
    <=> untrapped. A genuine black hole needs a trapped INTERIOR (chi<0 for r<2m) bounded by a
    MARGINAL horizon (chi->0 at r=2m). The c7 solution has chi>0 everywhere (no horizon), so we
    add a term that fixes only the SIGN on each side of r=2m plus chi->0 in the horizon band --
    NOT the radial profile -- leaving the type-I distortion free:

        L = mean_s [ relu(-chi_s * sign(r_s - 2m) + margin)^2  +  hor_weight * w_hor(r_s) * chi_s^2 ],
        w_hor = exp(-((r_s - 2m)/(band*2m))^2).

    The sign hinge drives chi<=-margin inside and chi>=+margin outside; the band term sharpens the
    marginal surface. chi is computed in a SINGLE GradientTape pass over the coordinates (one tape,
    cheaper than the Ricci double-tape). Default multiplier 0.0 reproduces existing behaviour exactly.
    """

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.weighter = weighter
        ms = config.model_specific
        self.m = float(getattr(ms, "m", 1.0))
        self.band = float(getattr(ms, "trapped_surface_band", 0.1))
        self.hor_weight = float(getattr(ms, "trapped_surface_horizon_weight", 1.0))
        self.margin = float(getattr(ms, "trapped_surface_margin", 0.0))
        self.margin_mode = getattr(ms, "trapped_surface_margin_mode", "constant")
        self.margin_slope = float(getattr(ms, "trapped_surface_margin_slope", 1.0))
        self.margin_floor = float(getattr(ms, "trapped_surface_margin_floor", 0.0))
        self.lorentzian = bool(getattr(ms, "lorentzian", True))

    def compute(self, x_vars: tf.Tensor, submodel) -> tf.Tensor:
        q4 = x_vars[:, :4]
        pidx = tf.cast(x_vars[:, 4], tf.int32)
        with tf.GradientTape() as tp:
            tp.watch(q4)
            G = cholesky_from_vec(submodel(embed_S2_coords(q4, pidx)), lorentzian=self.lorentzian)
            Jc = embedding_jacobian_stereo(q4, pidx)
            g = tf.einsum("sAB,sAm,sBn->smn", G, Jc, Jc)
            h = g[:, 2:4, 2:4]
            deth = h[:, 0, 0] * h[:, 1, 1] - h[:, 0, 1] * h[:, 1, 0]
            qsq = q4[:, 2] ** 2 + q4[:, 3] ** 2
            R2 = tf.sqrt(tf.abs(deth)) * tf.square(1.0 + qsq) / 4.0   # areal radius squared
            R2c = tf.reshape(R2, (-1, 1))
        dR2 = tp.batch_jacobian(R2c, q4)[:, 0, :]                     # (batch,4) = d R^2 / d x
        real_dtype = R2.dtype
        g_inv = tf.linalg.inv(g)
        chiS = tf.einsum("sm,smn,sn->s", dR2, g_inv, dR2)            # g^{mn} d_m R^2 d_n R^2
        chi = chiS / (4.0 * R2 + tf.cast(1e-30, real_dtype))         # = g^{mn} d_m R d_n R
        r = tf.cast(PenroseRadiusWeighting(x_vars[:, :4], m=self.m), real_dtype)
        rh = tf.cast(2.0 * self.m, real_dtype)
        if self.margin_mode == "profile":
            # LC-0019: single r-dependent ONE-SIDED bound, no band/suppression. The margin is the
            # Schwarzschild-linearized trapped depth inside, ->0 at the horizon, 0 outside:
            #   margin(r) = slope * max(0, 2m - r) / (2m),
            #   L = relu( margin(r) - sign(r-2m) * chi )^2.
            # => chi <= -margin(r) inside (TRAPPED, growing inward), chi = 0 at r=2m (marginal),
            #    chi >= 0 outside; profile left free to be MORE trapped above the bound (type-I free).
            # margin(r) = (floor + slope*max(0,2m-r)/(2m)) inside, 0 outside; the floor keeps a live
            # push just inside r=2m (where slope*depth ~ 0 is too weak to move the near-horizon bulk).
            depth = tf.nn.relu(rh - r) / rh                                # max(0,2m-r)/(2m)
            inside = tf.cast(r < rh, real_dtype)
            margin_r = tf.cast(self.margin_floor, real_dtype) * inside \
                + tf.cast(self.margin_slope, real_dtype) * depth
            per_point = tf.square(tf.nn.relu(margin_r - tf.sign(r - rh) * chi))
        else:
            # "constant" (LC-0017 original; reproduces c7b exactly):
            margin = tf.cast(self.margin, real_dtype)
            sigma = tf.cast(self.band, real_dtype) * rh
            w_hor = tf.exp(-tf.square((r - rh) / sigma))           # 1 at horizon -> 0 away
            # marginal-horizon term: chi -> 0 in the Gaussian band around r=2m
            hor_term = tf.cast(self.hor_weight, real_dtype) * w_hor * tf.square(chi)
            # sign hinge (acts AWAY from the band, so it does not fight chi->0 at the horizon):
            # want sign(chi)=sign(r-2m), i.e. chi<=-margin inside, chi>=+margin outside. A nonzero
            # margin gives a live gradient even at chi=0, so the interior is driven genuinely TRAPPED.
            sign_term = (1.0 - w_hor) * tf.square(tf.nn.relu(-chi * tf.sign(r - rh) + margin))
            per_point = sign_term + hor_term
        per_point = self.weighter._weight_patch(x_vars, per_point)
        return tf.reduce_mean(per_point)


class SpecialityIndexRProfileLoss:
    """
    Speciality-index profile loss for type-I Zipoy-Voorhees/gamma runs.

    Modes are supported:
      - "value": mean(epsilon / (|S - 1| + epsilon))
      - "profile": fit a smooth non-constant S profile inspired by static
        axisymmetric type-I Zipoy-Voorhees/gamma metrics away from the
        Schwarzschild type-D limit S = 1.
      - "gradient": mean(epsilon / (clip(||dS||^2, cap) + epsilon))
      - "discriminant": mean(epsilon / (Delta_norm + epsilon))
      - "hybrid": clipped-gradient repeller + discriminant

    The discriminant mode uses the algebraically-special relation
    Delta = I^3 - 27 J^2, normalised by |I|^3.

    Points with near-zero I are masked to avoid numerical blow-up.
    """

    _WEYL_FLOOR: float = 1e-6  # mask points with |I| below this threshold
    _EPS_WEYL: float = 1e-12  # additive guard in I^3 denominator
    _GRAD_POWER_CAP: float = 1.0
    _PROFILE_AMPLITUDE: float = 0.25

    def __init__(
        self, config: SchwarzschildConfig, weighter: WeightSchwarzschild
    ) -> None:
        self.weighter = weighter
        self.mode = getattr(
            config.model_specific, "speciality_index_rprofile_mode", "profile"
        )
        if self.mode == "variance":
            self.mode = "gradient"
        self.epsilon = getattr(
            config.model_specific, "speciality_index_rprofile_epsilon", 1e-2
        )
        self.profile_centre = getattr(
            config.model_specific, "speciality_index_rprofile_centre", 2.0
        )
        self.use_metric_contraction = _use_component_metric_contraction(
            config, "speciality_index_rprofile"
        )
        self.use_volume_scaling = _use_component_volume_scaling(
            config, "speciality_index_rprofile"
        )

        if self.mode not in ("value", "profile", "gradient", "discriminant", "hybrid"):
            raise ValueError(
                "speciality_index_rprofile_mode must be one of "
                "['value', 'profile', 'gradient', 'discriminant', 'hybrid']; "
                f"got '{self.mode}'."
            )

    @property
    def needs_speciality_index_gradient(self) -> bool:
        return False

    def _weighted_valid_mean(
        self,
        x_vars: tf.Tensor,
        values: tf.Tensor,
        valid: tf.Tensor,
        metric_pred_mat: tf.Tensor | None = None,
    ) -> tf.Tensor:
        values = tf.where(valid, values, tf.zeros_like(values))
        values = _apply_volume_scaling(
            values, metric_pred_mat, self.use_volume_scaling
        )
        values = self.weighter._weight_patch(x_vars, values)
        n_valid = tf.reduce_sum(tf.cast(valid, values.dtype))
        return tf.reduce_sum(values) / n_valid

    def _speciality_index_gradient_repeller(
        self,
        x_vars: tf.Tensor,
        speciality_index_grad: tf.Tensor,
        metric_pred_mat: tf.Tensor | None,
        valid: tf.Tensor,
        eps: tf.Tensor,
    ) -> tf.Tensor:
        if speciality_index_grad is None:
            raise ValueError(
                "speciality_index_grad must be supplied for speciality_index_rprofile_mode "
                "'gradient' or 'hybrid'."
            )

        if self.use_metric_contraction:
            if metric_pred_mat is None:
                raise ValueError(
                    "metric_pred_mat must be supplied when use_metric_contraction=True."
                )
            # Euclidean (SPD) inverse so the contracted gradient power is
            # non-negative. NOTE: this branch is currently unreachable (the
            # 'gradient'/'hybrid' modes route to _profile_loss) and has no
            # submodel in scope, so the spectral SPD inverse is used here rather
            # than riemannian_inverse_metric_embed; revisit if this mode is revived.
            inv_metric = euclidean_inverse_metric(metric_pred_mat)
            grad_power = tf.abs(
                tf.einsum(
                    "sa,sab,sb->s",
                    speciality_index_grad,
                    inv_metric,
                    speciality_index_grad,
                )
            )
        else:
            grad_power = tf.reduce_sum(tf.square(speciality_index_grad), axis=1)

        grad_power = tf.minimum(
            grad_power, tf.cast(self._GRAD_POWER_CAP, grad_power.dtype)
        )
        norm = eps / (grad_power + eps)
        return self._weighted_valid_mean(x_vars, norm, valid, metric_pred_mat)

    def _profile_target(self, x_vars: tf.Tensor, valid: tf.Tensor) -> tf.Tensor:
        """Smooth bounded type-I ZV/gamma-inspired speciality-index target."""
        dtype = x_vars.dtype
        X = x_vars[:, 1]
        q1 = x_vars[:, 2]
        q2 = x_vars[:, 3]
        patch_idx = tf.cast(x_vars[:, 4], tf.int32)

        pi = tf.cast(np.pi, dtype)
        radial = X / (0.5 * pi)
        radial_modulation = 1.0 + 0.25 * radial

        r_sq = tf.square(q1) + tf.square(q2)
        z_north = (1.0 - r_sq) / (1.0 + r_sq)
        z_south = (r_sq - 1.0) / (1.0 + r_sq)
        z_axis = tf.where(tf.equal(patch_idx, 0), z_north, z_south)
        p2_axis = 0.5 * (3.0 * tf.square(z_axis) - 1.0)

        raw_profile = radial_modulation * p2_axis
        profile_valid = tf.boolean_mask(raw_profile, valid)
        mean = tf.reduce_mean(profile_valid)
        std = tf.math.reduce_std(profile_valid)
        profile = (raw_profile - mean) / (std + tf.cast(1e-12, dtype))

        amplitude = tf.cast(self._PROFILE_AMPLITUDE, dtype)
        centre = tf.cast(self.profile_centre, dtype)
        return centre + amplitude * profile

    def _profile_loss(
        self,
        x_vars: tf.Tensor,
        speciality_index: tf.Tensor,
        valid: tf.Tensor,
        metric_pred_mat: tf.Tensor | None = None,
    ) -> tf.Tensor:
        target = self._profile_target(x_vars, valid)
        target = tf.cast(target, speciality_index.dtype)
        amplitude = tf.cast(self._PROFILE_AMPLITUDE, tf.math.real(speciality_index).dtype)
        norm = tf.square(tf.abs(speciality_index - target) / amplitude)
        return self._weighted_valid_mean(x_vars, norm, valid, metric_pred_mat)

    def _discriminant_repeller(
        self,
        x_vars: tf.Tensor,
        weyl_i: tf.Tensor,
        weyl_j: tf.Tensor,
        valid: tf.Tensor,
        eps: tf.Tensor,
        metric_pred_mat: tf.Tensor | None = None,
    ) -> tf.Tensor:
        weyl_i_abs = tf.abs(weyl_i)
        delta = tf.abs(weyl_i ** 3 - 27.0 * weyl_j ** 2)
        delta_norm = delta / (
            weyl_i_abs ** 3 + tf.cast(self._EPS_WEYL, weyl_i_abs.dtype)
        )
        norm = eps / (delta_norm + eps)
        return self._weighted_valid_mean(x_vars, norm, valid, metric_pred_mat)

    def compute_from_precomputed(
        self,
        x_vars: tf.Tensor,
        weyl_i: tf.Tensor,
        weyl_j: tf.Tensor,
        metric_pred_mat: tf.Tensor | None = None,
        speciality_index_grad: tf.Tensor | None = None,
    ) -> tf.Tensor:
        real_dtype = tf.math.real(weyl_i).dtype
        eps = tf.cast(self.epsilon, real_dtype)
        eps_weyl = tf.cast(self._EPS_WEYL, real_dtype)
        weyl_floor = tf.cast(self._WEYL_FLOOR, real_dtype)

        weyl_i_abs = tf.abs(weyl_i)  # (batch,)
        speciality_index = speciality_index_from_invariants(
            weyl_i, weyl_j, eps_weyl
        )

        # Mask near-flat points where S is numerically unreliable.
        valid = weyl_i_abs > weyl_floor  # (batch,) bool
        n_valid = tf.reduce_sum(tf.cast(valid, tf.int32))
        if tf.equal(n_valid, 0):
            return tf.cast(0.0, real_dtype)

        if self.mode == "value":
            target = tf.cast(_SPECIALITY_INDEX_SCHWARZSCHILD, weyl_i.dtype)
            # Repeller: large when |S - 1| is small.
            norm = eps / (tf.abs(speciality_index - target) + eps)  # (batch,)
            return self._weighted_valid_mean(x_vars, norm, valid, metric_pred_mat)

        if self.mode in ("profile", "gradient", "hybrid"):
            return self._profile_loss(
                x_vars, speciality_index, valid, metric_pred_mat
            )

        # Previous gradient-repeller objective, retained for reference while the
        # profile objective is evaluated:
        # if self.mode == "gradient":
        #     return self._speciality_index_gradient_repeller(
        #         x_vars, speciality_index_grad, metric_pred_mat, valid, eps
        #     )

        discriminant_loss = self._discriminant_repeller(
            x_vars, weyl_i, weyl_j, valid, eps, metric_pred_mat
        )
        if self.mode == "discriminant":
            return discriminant_loss

        # Previous hybrid objective:
        # gradient_loss = self._speciality_index_gradient_repeller(
        #     x_vars, speciality_index_grad, metric_pred_mat, valid, eps
        # )
        # return gradient_loss + discriminant_loss
        return discriminant_loss


class TotalSchwarzschildLoss:
    """
    Total training loss for the S^2-embedding Schwarzschild architecture.

    Contains an Einstein equation loss, Kretschmann scalar loss, and optional
    \\mathbb{R}^2 determinant finiteness barrier. Loss multipliers can be scheduled during
    training using FloatScheduler. If a scheduler is configured, call
    set_epoch() before computing the loss during training.
    """

    config: SchwarzschildConfig

    def __init__(self, config: SchwarzschildConfig) -> None:
        self.config = config
        self.lorentzian = getattr(config.model_specific, "lorentzian", False)

        # Store base multipliers
        self.einstein_multiplier_base = config.loss.einstein_multiplier
        self.kretschmann_multiplier_base = config.model_specific.kretschmann_multiplier
        self.r2_det_loss_multiplier_base = getattr(
            config.model_specific, "r2_det_loss_multiplier", 0.0
        )
        self.killing_symmetry_multiplier_base = getattr(
            config.model_specific, "killing_symmetry_multiplier", 0.0
        )
        self.k_repeller_multiplier_base = getattr(
            config.model_specific, "k_repeller_multiplier", 0.0
        )
        self.speciality_index_rprofile_multiplier_base = getattr(
            config.model_specific, "speciality_index_rprofile_multiplier", 0.0
        )
        self.horizon_anchor_multiplier_base = getattr(
            config.model_specific, "horizon_anchor_multiplier", 0.0
        )
        self.trapped_surface_multiplier_base = getattr(
            config.model_specific, "trapped_surface_multiplier", 0.0
        )

        # Current scheduled multipliers (updated by set_epoch())
        self.einstein_multiplier = self.einstein_multiplier_base
        self.kretschmann_multiplier = self.kretschmann_multiplier_base
        self.r2_det_loss_multiplier = self.r2_det_loss_multiplier_base
        self.killing_symmetry_multiplier = self.killing_symmetry_multiplier_base
        self.k_repeller_multiplier = self.k_repeller_multiplier_base
        self.speciality_index_rprofile_multiplier = (
            self.speciality_index_rprofile_multiplier_base
        )
        self.horizon_anchor_multiplier = self.horizon_anchor_multiplier_base
        self.trapped_surface_multiplier = self.trapped_surface_multiplier_base

        assert (
            abs(self.einstein_multiplier)
            + abs(self.kretschmann_multiplier)
            + abs(self.r2_det_loss_multiplier)
            + abs(self.killing_symmetry_multiplier)
            + abs(self.k_repeller_multiplier)
            + abs(self.speciality_index_rprofile_multiplier)
            + abs(self.horizon_anchor_multiplier)
            + abs(self.trapped_surface_multiplier)
            > 0.0
        ), (
            "All loss terms (einstein, kretschmann, r2_det, killing_symmetry, "
            "k_repeller, speciality_index_rprofile, horizon_anchor, "
            "trapped_surface) are turned off."
        )

        weighter = WeightSchwarzschild(config, weight=False)  # remove weighting for now
        self.weighter = weighter

        self.einstein_loss = EinsteinLossEmbed(config, weighter)
        self.kretschmann_loss = KretschmannLossEmbed(config, weighter)
        self.r2_det_loss = R2DetFinitenessLoss(config, weighter)
        self.killing_symmetry_loss = KillingSymmetryLossEmbed(config, weighter)
        self.k_repeller_loss = KRepellerLoss(config, weighter)
        self.speciality_index_rprofile_loss = SpecialityIndexRProfileLoss(
            config, weighter
        )
        self.horizon_anchor_loss = HorizonAnchorLoss(config, weighter)
        self.trapped_surface_loss = TrappedSurfaceLoss(config, weighter)

        kernel_name = getattr(config.model_specific, "ricci_kernel", "standard")
        try:
            self._ricci_kernel = _RICCI_KERNELS[kernel_name]
        except KeyError as e:
            raise ValueError(
                f"Unknown ricci_kernel '{kernel_name}'; expected one of "
                f"{sorted(_RICCI_KERNELS.keys())}."
            ) from e
        self._ricci_kernel_name = kernel_name
        self._kernel_fallback_warned = False

        # Initialize schedulers
        self._init_schedulers()

    def _init_schedulers(self):
        """Initialize multiplier schedulers from config."""
        self.einstein_scheduler = None
        self.kretschmann_scheduler = None
        self.r2_det_scheduler = None
        self.killing_symmetry_scheduler = None
        self.k_repeller_scheduler = None
        self.speciality_index_rprofile_scheduler = None
        self.horizon_anchor_scheduler = None
        self.trapped_surface_scheduler = None

        if self.config.loss.einstein_schedule is not None:
            self.einstein_scheduler = FloatScheduler(
                strategy=self.config.loss.einstein_schedule.strategy,
                init_value=self.einstein_multiplier_base,
                final_value=self.config.loss.einstein_schedule.final_value,
                warmup_epochs=self.config.loss.einstein_schedule.warmup_epochs,
                decay_rate=self.config.loss.einstein_schedule.decay_rate,
                steps=self.config.loss.einstein_schedule.steps,
            )

        if self.config.loss.kretschmann_schedule is not None:
            self.kretschmann_scheduler = FloatScheduler(
                strategy=self.config.loss.kretschmann_schedule.strategy,
                init_value=self.kretschmann_multiplier_base,
                final_value=self.config.loss.kretschmann_schedule.final_value,
                warmup_epochs=self.config.loss.kretschmann_schedule.warmup_epochs,
                decay_rate=self.config.loss.kretschmann_schedule.decay_rate,
                steps=self.config.loss.kretschmann_schedule.steps,
            )

        if self.config.loss.r2_det_schedule is not None:
            self.r2_det_scheduler = FloatScheduler(
                strategy=self.config.loss.r2_det_schedule.strategy,
                init_value=self.r2_det_loss_multiplier_base,
                final_value=self.config.loss.r2_det_schedule.final_value,
                warmup_epochs=self.config.loss.r2_det_schedule.warmup_epochs,
                decay_rate=self.config.loss.r2_det_schedule.decay_rate,
                steps=self.config.loss.r2_det_schedule.steps,
            )

        if self.config.loss.killing_symmetry_schedule is not None:
            self.killing_symmetry_scheduler = FloatScheduler(
                strategy=self.config.loss.killing_symmetry_schedule.strategy,
                init_value=self.killing_symmetry_multiplier_base,
                final_value=self.config.loss.killing_symmetry_schedule.final_value,
                warmup_epochs=self.config.loss.killing_symmetry_schedule.warmup_epochs,
                decay_rate=self.config.loss.killing_symmetry_schedule.decay_rate,
                steps=self.config.loss.killing_symmetry_schedule.steps,
            )

        if self.config.loss.k_repeller_schedule is not None:
            self.k_repeller_scheduler = FloatScheduler(
                strategy=self.config.loss.k_repeller_schedule.strategy,
                init_value=self.k_repeller_multiplier_base,
                final_value=self.config.loss.k_repeller_schedule.final_value,
                warmup_epochs=self.config.loss.k_repeller_schedule.warmup_epochs,
                decay_rate=self.config.loss.k_repeller_schedule.decay_rate,
                steps=self.config.loss.k_repeller_schedule.steps,
            )

        if self.config.loss.speciality_index_rprofile_schedule is not None:
            self.speciality_index_rprofile_scheduler = FloatScheduler(
                strategy=self.config.loss.speciality_index_rprofile_schedule.strategy,
                init_value=self.speciality_index_rprofile_multiplier_base,
                final_value=self.config.loss.speciality_index_rprofile_schedule.final_value,
                warmup_epochs=self.config.loss.speciality_index_rprofile_schedule.warmup_epochs,
                decay_rate=self.config.loss.speciality_index_rprofile_schedule.decay_rate,
                steps=self.config.loss.speciality_index_rprofile_schedule.steps,
            )

        if self.config.loss.horizon_anchor_schedule is not None:
            self.horizon_anchor_scheduler = FloatScheduler(
                strategy=self.config.loss.horizon_anchor_schedule.strategy,
                init_value=self.horizon_anchor_multiplier_base,
                final_value=self.config.loss.horizon_anchor_schedule.final_value,
                warmup_epochs=self.config.loss.horizon_anchor_schedule.warmup_epochs,
                decay_rate=self.config.loss.horizon_anchor_schedule.decay_rate,
                steps=self.config.loss.horizon_anchor_schedule.steps,
            )

        if self.config.loss.trapped_surface_schedule is not None:
            self.trapped_surface_scheduler = FloatScheduler(
                strategy=self.config.loss.trapped_surface_schedule.strategy,
                init_value=self.trapped_surface_multiplier_base,
                final_value=self.config.loss.trapped_surface_schedule.final_value,
                warmup_epochs=self.config.loss.trapped_surface_schedule.warmup_epochs,
                decay_rate=self.config.loss.trapped_surface_schedule.decay_rate,
                steps=self.config.loss.trapped_surface_schedule.steps,
            )

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update scheduled multipliers for the current epoch."""
        if self.einstein_scheduler is not None:
            self.einstein_multiplier = self.einstein_scheduler.get(epoch, total_epochs)

        if self.kretschmann_scheduler is not None:
            self.kretschmann_multiplier = self.kretschmann_scheduler.get(
                epoch, total_epochs
            )

        if self.r2_det_scheduler is not None:
            self.r2_det_loss_multiplier = self.r2_det_scheduler.get(epoch, total_epochs)

        if self.killing_symmetry_scheduler is not None:
            self.killing_symmetry_multiplier = self.killing_symmetry_scheduler.get(
                epoch, total_epochs
            )

        if self.k_repeller_scheduler is not None:
            self.k_repeller_multiplier = self.k_repeller_scheduler.get(
                epoch, total_epochs
            )

        if self.speciality_index_rprofile_scheduler is not None:
            self.speciality_index_rprofile_multiplier = (
                self.speciality_index_rprofile_scheduler.get(
                    epoch, total_epochs
                )
            )

        if self.horizon_anchor_scheduler is not None:
            self.horizon_anchor_multiplier = self.horizon_anchor_scheduler.get(
                epoch, total_epochs
            )

        if self.trapped_surface_scheduler is not None:
            self.trapped_surface_multiplier = self.trapped_surface_scheduler.get(
                epoch, total_epochs
            )

    def call(
        self, model, x_vars, metric_pred=None, return_constituents=False, val_print=True
    ):
        """
        Args:
            model:       SchwarzschildGlobalModel
            x_vars:      (batch, 5) = [T, X, q1, q2, patch_idx_float]
            metric_pred: unused — kept for API compatibility with BaseNetwork.evaluate_loss.
                         The metric is recomputed inside the double-tape pass so that
                         coordinate gradients are tracked correctly.
            return_constituents: if True, return dict of individual loss values.
            val_print:    if True and print_batch_losses, print each component.
        """
        need_einstein = self.einstein_multiplier > 0.0
        _einstein_mode = getattr(self.einstein_loss, "loss_mode", "volume_integral")
        # The curvature-normalized Einstein form needs K in its denominator.
        einstein_needs_K = (
            need_einstein
            and _einstein_mode
            in ("curvature_normalized", "curvature_normalized_capped")
        )
        # The "weyl_normalized" Einstein form needs the Weyl invariant I in its
        # denominator, which comes from the speciality-index branch of the kernel.
        einstein_needs_weyl = need_einstein and _einstein_mode == "weyl_normalized"
        # The horizon anchor needs |I| (weyl) or |K| (kretschmann), per its invariant.
        _ha_invariant = getattr(self.horizon_anchor_loss, "invariant", "weyl_i")
        horizon_needs_weyl = (
            self.horizon_anchor_multiplier > 0.0 and _ha_invariant == "weyl_i"
        )
        horizon_needs_K = (
            self.horizon_anchor_multiplier > 0.0 and _ha_invariant == "kretschmann"
        )
        need_kretschmann = (
            self.kretschmann_multiplier > 0.0
            or self.k_repeller_multiplier > 0.0
            or einstein_needs_K
            or horizon_needs_K
        )
        # The "weyl_invariant"(_volume / _alt) Kretschmann forms prescribe the Weyl
        # invariant I, which is produced by the speciality-index branch of the kernel.
        kretschmann_needs_weyl = (
            self.kretschmann_multiplier > 0.0
            and getattr(self.kretschmann_loss, "loss_mode", "log_profile")
            in ("weyl_invariant", "weyl_invariant_volume", "weyl_invariant_alt")
        )
        need_speciality_index = (
            self.speciality_index_rprofile_multiplier > 0.0
            or kretschmann_needs_weyl
            or einstein_needs_weyl
            or horizon_needs_weyl
            or return_constituents
        )
        need_speciality_index_gradient = (
            self.speciality_index_rprofile_multiplier > 0.0
            and self.speciality_index_rprofile_loss.needs_speciality_index_gradient
        )

        # Single pass (reverse-mode tapes for "standard", nested forward-mode
        # JVPs for "optimised").  Post-assembly of Riemann / Kretschmann
        # contraction and Ricci reduction is skipped for any term whose
        # multiplier is zero.
        try:
            if need_speciality_index_gradient:
                q_4d = x_vars[:, :4]
                with tf.GradientTape() as speciality_index_tape:
                    speciality_index_tape.watch(q_4d)
                    x_curvature = tf.concat([q_4d, x_vars[:, 4:5]], axis=1)
                    (
                        metric_pred_mat,
                        ricci_tensor,
                        kretschmann_scalar,
                        weyl_i,
                        weyl_j,
                    ) = self._ricci_kernel(
                        x_curvature,
                        model.submodel,
                        self.lorentzian,
                        need_ricci=need_einstein,
                        need_kretschmann=True,
                        need_speciality_index=True,
                    )
                    eps_weyl = tf.cast(
                        self.speciality_index_rprofile_loss._EPS_WEYL,
                        tf.math.real(weyl_i).dtype,
                    )
                    speciality_index_for_grad = speciality_index_from_invariants(
                        weyl_i, weyl_j, eps_weyl
                    )
                speciality_index_grad = tf.squeeze(
                    speciality_index_tape.batch_jacobian(
                        speciality_index_for_grad[:, tf.newaxis], q_4d
                    ),
                    axis=1,
                )
            else:
                speciality_index_grad = None
                (
                    metric_pred_mat,
                    ricci_tensor,
                    kretschmann_scalar,
                    weyl_i,
                    weyl_j,
                ) = self._ricci_kernel(
                    x_vars,
                    model.submodel,
                    self.lorentzian,
                    need_ricci=need_einstein,
                    need_kretschmann=need_kretschmann,
                    need_speciality_index=need_speciality_index,
                )
        except Exception as exc:
            # The forward-mode optimised kernel can fail on some TF/TFP builds
            # (observed as a staging-time IndexError around fill_triangular).
            # Fall back to the standard reverse-mode kernel for robustness.
            if self._ricci_kernel_name != "optimised":
                raise

            self._ricci_kernel = _RICCI_KERNELS["standard"]
            self._ricci_kernel_name = "standard"
            if not self._kernel_fallback_warned:
                print(
                    "Warning: optimised Ricci kernel failed at runtime; "
                    "falling back to standard kernel for this run. "
                    f"Original error: {exc}"
                )
                self._kernel_fallback_warned = True

            if need_speciality_index_gradient:
                q_4d = x_vars[:, :4]
                with tf.GradientTape() as speciality_index_tape:
                    speciality_index_tape.watch(q_4d)
                    x_curvature = tf.concat([q_4d, x_vars[:, 4:5]], axis=1)
                    (
                        metric_pred_mat,
                        ricci_tensor,
                        kretschmann_scalar,
                        weyl_i,
                        weyl_j,
                    ) = self._ricci_kernel(
                        x_curvature,
                        model.submodel,
                        self.lorentzian,
                        need_ricci=need_einstein,
                        need_kretschmann=True,
                        need_speciality_index=True,
                    )
                    eps_weyl = tf.cast(
                        self.speciality_index_rprofile_loss._EPS_WEYL,
                        tf.math.real(weyl_i).dtype,
                    )
                    speciality_index_for_grad = speciality_index_from_invariants(
                        weyl_i, weyl_j, eps_weyl
                    )
                speciality_index_grad = tf.squeeze(
                    speciality_index_tape.batch_jacobian(
                        speciality_index_for_grad[:, tf.newaxis], q_4d
                    ),
                    axis=1,
                )
            else:
                speciality_index_grad = None
                (
                    metric_pred_mat,
                    ricci_tensor,
                    kretschmann_scalar,
                    weyl_i,
                    weyl_j,
                ) = self._ricci_kernel(
                    x_vars,
                    model.submodel,
                    self.lorentzian,
                    need_ricci=need_einstein,
                    need_kretschmann=need_kretschmann,
                    need_speciality_index=need_speciality_index,
                )

        # Einstein loss (weyl_i is None unless the weyl_normalized mode requested it)
        if self.einstein_multiplier > 0.0:
            e_loss = self.einstein_loss.compute_from_precomputed(
                x_vars, metric_pred_mat, ricci_tensor, model.submodel,
                kretschmann_scalar, weyl_i=weyl_i,
            )
        else:
            e_loss = tf.cast(0.0, tf.float64)

        # Kretschmann loss (weyl_i is None unless the weyl_invariant mode requested it)
        if self.kretschmann_multiplier > 0.0:
            k_loss = self.kretschmann_loss.compute_from_precomputed(
                x_vars, kretschmann_scalar, metric_pred_mat=metric_pred_mat,
                weyl_i=weyl_i,
            )
        else:
            k_loss = tf.cast(0.0, tf.float64)

        # ℝ² determinant finiteness barrier
        if self.r2_det_loss_multiplier > 0.0:
            r2_det_loss = self.r2_det_loss.compute(x_vars, metric_pred_mat)
        else:
            r2_det_loss = tf.cast(0.0, tf.float64)

        # SO(3) spherical-symmetry loss.
        if self.killing_symmetry_multiplier > 0.0:
            killing_loss = self.killing_symmetry_loss.compute_from_precomputed(
                x_vars, metric_pred_mat, model.submodel
            )
        else:
            killing_loss = tf.cast(0.0, tf.float64)

        # K-repeller: epsilon/(|K|+epsilon)
        if self.k_repeller_multiplier > 0.0:
            kr_loss = self.k_repeller_loss.compute_from_precomputed(
                x_vars, kretschmann_scalar, metric_pred_mat=metric_pred_mat
            )
        else:
            kr_loss = tf.cast(0.0, tf.float64)

        # Speciality-index r-profile loss: ZV/gamma-inspired type-I target.
        if self.speciality_index_rprofile_multiplier > 0.0:
            srp_loss = self.speciality_index_rprofile_loss.compute_from_precomputed(
                x_vars,
                weyl_i,
                weyl_j,
                metric_pred_mat=metric_pred_mat,
                speciality_index_grad=speciality_index_grad,
            )
        else:
            srp_loss = tf.cast(0.0, tf.float64)

        # Horizon curvature anchor: fix |I| (or |K|) near r=2m -> regular horizon + scale.
        if self.horizon_anchor_multiplier > 0.0:
            ha_loss = self.horizon_anchor_loss.compute_from_precomputed(
                x_vars, weyl_i=weyl_i, kretschmann_scalar=kretschmann_scalar
            )
        else:
            ha_loss = tf.cast(0.0, tf.float64)

        # Trapped-surface (Misner-Sharp) prescriber: chi<0 inside r<2m, chi->0 at the horizon
        # (own GradientTape pass; supplies the black-hole causal structure).
        if self.trapped_surface_multiplier > 0.0:
            ts_loss = self.trapped_surface_loss.compute(x_vars, model.submodel)
        else:
            ts_loss = tf.cast(0.0, tf.float64)

        if self.config.logging.print_batch_losses and val_print:
            print(
                "Einstein:       {:.3g}\n"
                "Kretschmann:    {:.3g}\n"
                "R2DetBarrier:   {:.3g}\n"
                "KillingSym:     {:.3g}\n"
                "K-Repeller:     {:.3g}\n"
                "S-RProfile:     {:.3g}\n".format(
                    tf.get_static_value(e_loss),
                    tf.get_static_value(k_loss),
                    tf.get_static_value(r2_det_loss),
                    tf.get_static_value(killing_loss),
                    tf.get_static_value(kr_loss),
                    tf.get_static_value(srp_loss),
                )
            )

        if return_constituents:
            speciality_summary = (
                _speciality_index_summary(
                    weyl_i,
                    weyl_j,
                    self.speciality_index_rprofile_loss._WEYL_FLOOR,
                    self.speciality_index_rprofile_loss._EPS_WEYL,
                )
                if weyl_i is not None and weyl_j is not None
                else {}
            )

            loss_constituents = {
                "einstein_loss": tf.get_static_value(e_loss),
                "kretschmann_loss": tf.get_static_value(k_loss),
                "r2_det_loss": tf.get_static_value(r2_det_loss),
                "killing_symmetry_loss": tf.get_static_value(killing_loss),
                "k_repeller_loss": tf.get_static_value(kr_loss),
                "speciality_index_rprofile_loss": tf.get_static_value(srp_loss),
                "horizon_anchor_loss": tf.get_static_value(ha_loss),
                "trapped_surface_loss": tf.get_static_value(ts_loss),
                **speciality_summary,
            }
        else:
            loss_constituents = None

        # Weighted sum, normalised by the sum of active multipliers
        total_loss = 0.0
        norm_denom = 0.0
        if self.einstein_multiplier > 0.0:
            total_loss += self.einstein_multiplier * tf.math.abs(e_loss)
            norm_denom += self.einstein_multiplier
        if self.kretschmann_multiplier > 0.0:
            total_loss += self.kretschmann_multiplier * tf.math.abs(k_loss)
            norm_denom += self.kretschmann_multiplier
        if self.r2_det_loss_multiplier > 0.0:
            total_loss += self.r2_det_loss_multiplier * tf.math.abs(r2_det_loss)
            norm_denom += self.r2_det_loss_multiplier
        if self.killing_symmetry_multiplier > 0.0:
            total_loss += self.killing_symmetry_multiplier * tf.math.abs(killing_loss)
            norm_denom += self.killing_symmetry_multiplier
        if self.k_repeller_multiplier > 0.0:
            total_loss += self.k_repeller_multiplier * tf.math.abs(kr_loss)
            norm_denom += self.k_repeller_multiplier
        if self.speciality_index_rprofile_multiplier > 0.0:
            total_loss += self.speciality_index_rprofile_multiplier * tf.math.abs(
                srp_loss
            )
            norm_denom += self.speciality_index_rprofile_multiplier
        if self.horizon_anchor_multiplier > 0.0:
            total_loss += self.horizon_anchor_multiplier * tf.math.abs(ha_loss)
            norm_denom += self.horizon_anchor_multiplier
        if self.trapped_surface_multiplier > 0.0:
            total_loss += self.trapped_surface_multiplier * tf.math.abs(ts_loss)
            norm_denom += self.trapped_surface_multiplier

        total_loss /= norm_denom

        return total_loss, loss_constituents


class TotalSchwarzschildLocal2DLoss:
    """Einstein-only (plus optional 2D determinant barrier) loss for local 2D runs.

    This path is intentionally independent from the Schwarzschild embedding
    pipeline: no Penrose sampling, no S^2 embedding Jacobian, and no
    Schwarzschild-specific radial filter are used.
    """

    config: SchwarzschildConfig

    def __init__(self, config: SchwarzschildConfig) -> None:
        self.config = config
        self.lorentzian = getattr(config.model_specific, "lorentzian", False)
        self.einstein_constant = config.geometry.einstein_constant

        self.use_volume_scaling = _use_component_volume_scaling(config, "einstein")
        self.use_metric_contraction = _use_component_metric_contraction(
            config, "einstein"
        )
        self.use_r2_det_volume_scaling = _use_component_volume_scaling(
            config, "r2_det"
        )

        # Multipliers (Kretschmann is unsupported in this local 2D mode).
        self.einstein_multiplier_base = config.loss.einstein_multiplier
        self.r2_det_loss_multiplier_base = getattr(
            config.model_specific, "r2_det_loss_multiplier", 0.0
        )
        self.kretschmann_multiplier_base = getattr(
            config.model_specific, "kretschmann_multiplier", 0.0
        )

        if self.kretschmann_multiplier_base > 0.0:
            raise ValueError(
                "local_2d_mode does not support Kretschmann loss. Set "
                "model_specific.kretschmann_multiplier=0.0."
            )

        self.einstein_multiplier = self.einstein_multiplier_base
        self.r2_det_loss_multiplier = self.r2_det_loss_multiplier_base

        assert (
            abs(self.einstein_multiplier_base) + abs(self.r2_det_loss_multiplier_base)
            > 0.0
        ), "All local 2D loss terms are turned off."

        self.einstein_scheduler = None
        self.r2_det_scheduler = None
        if self.config.loss.einstein_schedule is not None:
            self.einstein_scheduler = FloatScheduler(
                strategy=self.config.loss.einstein_schedule.strategy,
                init_value=self.einstein_multiplier_base,
                final_value=self.config.loss.einstein_schedule.final_value,
                warmup_epochs=self.config.loss.einstein_schedule.warmup_epochs,
                decay_rate=self.config.loss.einstein_schedule.decay_rate,
                steps=self.config.loss.einstein_schedule.steps,
            )
        if self.config.loss.r2_det_schedule is not None:
            self.r2_det_scheduler = FloatScheduler(
                strategy=self.config.loss.r2_det_schedule.strategy,
                init_value=self.r2_det_loss_multiplier_base,
                final_value=self.config.loss.r2_det_schedule.final_value,
                warmup_epochs=self.config.loss.r2_det_schedule.warmup_epochs,
                decay_rate=self.config.loss.r2_det_schedule.decay_rate,
                steps=self.config.loss.r2_det_schedule.steps,
            )

    def set_epoch(self, epoch: int, total_epochs: int):
        if self.einstein_scheduler is not None:
            self.einstein_multiplier = self.einstein_scheduler.get(epoch, total_epochs)
        if self.r2_det_scheduler is not None:
            self.r2_det_loss_multiplier = self.r2_det_scheduler.get(epoch, total_epochs)

    def _einstein_loss(self, x_vars, metric_pred_mat, model, metric_pred):
        ricci_tensor = compute_ricci_tensor(x_vars, model.submodel)
        error = self.einstein_constant * metric_pred_mat - ricci_tensor

        if self.use_metric_contraction:
            g_inv = tf.linalg.inv(cholesky_from_vec(metric_pred, lorentzian=False))
            norm = tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error)
            norm = tf.abs(norm)
        else:
            norm = tf.norm(error, axis=(1, 2))

        norm = _apply_volume_scaling(norm, metric_pred_mat, self.use_volume_scaling)
        return tf.reduce_mean(norm)

    def _r2_det_barrier(self, metric_pred_mat):
        det_g = tf.abs(tf.linalg.det(metric_pred_mat))
        norm = 1.0 / (det_g + tf.cast(1e-30, det_g.dtype))
        norm = _apply_volume_scaling(
            norm, metric_pred_mat, self.use_r2_det_volume_scaling
        )
        return tf.reduce_mean(norm)

    def call(
        self, model, x_vars, metric_pred=None, return_constituents=False, val_print=True
    ):
        # metric_pred is the raw Cholesky vector output of the 2D local model.
        if metric_pred is None:
            metric_pred = model(x_vars)

        metric_pred_mat = cholesky_from_vec(metric_pred, lorentzian=self.lorentzian)

        if self.einstein_multiplier > 0.0:
            e_loss = self._einstein_loss(x_vars, metric_pred_mat, model, metric_pred)
        else:
            e_loss = tf.cast(0.0, tf.float64)

        if self.r2_det_loss_multiplier > 0.0:
            r2_det_loss = self._r2_det_barrier(metric_pred_mat)
        else:
            r2_det_loss = tf.cast(0.0, tf.float64)

        if self.config.logging.print_batch_losses and val_print:
            print(
                "Einstein:    {:.3g}\n"
                "R2DetBarrier:{:.3g}\n".format(
                    tf.get_static_value(e_loss),
                    tf.get_static_value(r2_det_loss),
                )
            )

        if return_constituents:
            loss_constituents = {
                "einstein_loss": tf.get_static_value(e_loss),
                "r2_det_loss": tf.get_static_value(r2_det_loss),
            }
        else:
            loss_constituents = None

        total_loss = 0.0
        norm_denom = 0.0
        if self.einstein_multiplier > 0.0:
            total_loss += self.einstein_multiplier * tf.math.abs(e_loss)
            norm_denom += self.einstein_multiplier
        if self.r2_det_loss_multiplier > 0.0:
            total_loss += self.r2_det_loss_multiplier * tf.math.abs(r2_det_loss)
            norm_denom += self.r2_det_loss_multiplier

        total_loss /= norm_denom

        return total_loss, loss_constituents
