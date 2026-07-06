from __future__ import annotations

"""
Optimised Ricci / Kretschmann computation for the Schwarzschild S^2-embedding
architecture, implemented with forward-mode automatic differentiation
(``tf.autodiff.ForwardAccumulator``) instead of the reverse-mode double
``tf.GradientTape`` used by ``geometry.schwarzschild``.

Rationale
---------
The standard embed kernel takes one forward pass of the network and then two
levels of reverse-mode tape (one to build the metric Jacobian, one to build
the Christoffel Jacobian).  The inner ``batch_jacobian`` cost scales with the
number of output components: for dim=4 that is 16 metric components and 64
Christoffel components, so the total cost is ~1 forward + O(80) reverse sweeps.

For small manifold dimension the same second-order information can be gathered
with only ``n * (n + 1) / 2`` nested forward JVPs over the unique symmetric
metric components.  For n=4 this is 10 nested-JVP passes (~20 forward-equivalent
evaluations), after which the Christoffel symbols, Ricci tensor and Kretschmann
scalar are assembled algebraically from ``g, g^{-1}, ∂g, ∂∂g``.

Public API
----------
``compute_ricci_and_kretschmann_embed_opt(x_vars, submodel, lorentzian, ...)``
mirrors the signature and return contract of
``geometry.schwarzschild.compute_ricci_and_kretschmann_embed`` and is a drop-in
replacement when ``model_specific.ricci_kernel == "optimised"``.
"""
import tensorflow as tf
import tensorflow_probability as tfp

tf.keras.backend.set_floatx("float64")

from geometry.schwarzschild import embed_S2_coords, embedding_jacobian_stereo
from helper_functions.helper_functions import cholesky_from_vec

# ---------------------------------------------------------------------------
# Auxiliary Cholesky / SPD utilities — retained from the original scaffolding.
# Not used by the forward-mode pipeline below, but kept available as general
# helpers for symmetric-positive-definite metric work.
# ---------------------------------------------------------------------------


def cholesky_factor_from_vec(lower_triangular_vector: tf.Tensor) -> tf.Tensor:
    return tfp.math.fill_triangular(lower_triangular_vector)


def spd_from_cholesky_factor(lower_triangular_matrix: tf.Tensor) -> tf.Tensor:
    return tf.matmul(lower_triangular_matrix, lower_triangular_matrix, transpose_b=True)


def spd_inverse(full_matrix: tf.Tensor) -> tf.Tensor:
    return spd_inverse_from_cholesky_factor(tf.linalg.cholesky(full_matrix))


def spd_inverse_from_cholesky_factor(lower_triangular_matrix: tf.Tensor) -> tf.Tensor:
    return tf.linalg.cholesky_solve(
        lower_triangular_matrix, _batched_identity_like(lower_triangular_matrix)
    )


def _batched_identity_like(matrix_or_factor: tf.Tensor) -> tf.Tensor:
    dim = tf.shape(matrix_or_factor)[-1]
    eye = tf.eye(dim, dtype=matrix_or_factor.dtype)
    target_shape = tf.concat(
        [tf.shape(matrix_or_factor)[:-2], tf.stack([dim, dim])], axis=0
    )
    return tf.broadcast_to(eye, target_shape)


# ---------------------------------------------------------------------------
# Note on metric symmetry.
#
# The pullback metric ``g = G J J`` is symmetric by construction (since
# ``cholesky_from_vec(..., lorentzian=True)`` returns a symmetric
# ``G = L · diag(η) · L^T``), so in principle we could JVP only the
# ``n(n+1)/2`` unique upper-triangular entries and mirror to rebuild the
# full tensor.  We instead propagate JVPs of the full (n, n) metric: the
# per-pass cost is dominated by the forward evaluation through the network,
# so the extra generality is effectively free, and it keeps the kernel
# robust to any future change in ``metric_fn`` that might break the
# symmetry assumption.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tensor-calculus assembly from metric + first derivs + Hessian.
# ---------------------------------------------------------------------------


def _christoffel_from_metric(
    metric_derivs: tf.Tensor, inv_metric: tf.Tensor
) -> tf.Tensor:
    """Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}).

    Conventions:
        metric_derivs[s, i, j, k] = ∂_k g_{ij}
        inv_metric[s, i, j]        = g^{ij}
        gamma[s, k, i, j]          = Γ^k_{ij}
    """
    gamma = tf.einsum("skl,sjli->skij", inv_metric, metric_derivs)
    gamma += tf.einsum("skl,silj->skij", inv_metric, metric_derivs)
    gamma -= tf.einsum("skl,sijl->skij", inv_metric, metric_derivs)
    gamma *= 0.5
    return gamma


def _christoffel_derivative_from_hessian(
    inv_metric: tf.Tensor,
    metric_derivs: tf.Tensor,
    metric_hessian: tf.Tensor,
    gamma: tf.Tensor,
) -> tf.Tensor:
    """d_gamma[s, m, j, k, i] = ∂_i Γ^m_{jk} expressed algebraically.

    Differentiating Γ^m_{jk} = ½ g^{mp} (∂_j g_{kp} + ∂_k g_{jp} - ∂_p g_{jk}),
    using ∂_i g^{mp} = -g^{ma} g^{pb} (∂_i g_{ab}) and the contraction
    g^{pb}(∂_j g_{kp} + ∂_k g_{jp} - ∂_p g_{jk}) = 2 Γ^b_{jk}:

        ∂_i Γ^m_{jk} = -g^{ma} D_{abi} Γ^b_{jk}
                       + ½ g^{mp} (H_{kp,ij} + H_{jp,ik} - H_{jk,ip})
    """
    term_a = -tf.einsum("sma,sabi,sbjk->smjki", inv_metric, metric_derivs, gamma)
    term_b1 = tf.einsum("smp,skpij->smjki", inv_metric, metric_hessian)
    term_b2 = tf.einsum("smp,sjpik->smjki", inv_metric, metric_hessian)
    term_b3 = tf.einsum("smp,sjkip->smjki", inv_metric, metric_hessian)
    return term_a + 0.5 * (term_b1 + term_b2 - term_b3)


def _ricci_from_d_gamma(d_gamma: tf.Tensor, gamma: tf.Tensor) -> tf.Tensor:
    """R_{jk} = ∂_i Γ^i_{jk} - ∂_j Γ^i_{ki} + Γ^i_{il} Γ^l_{jk} - Γ^i_{jl} Γ^l_{ik}."""
    ricci = tf.einsum("sijki->sjk", d_gamma)
    ricci -= tf.einsum("sikij->sjk", d_gamma)
    ricci += tf.einsum("siil,sljk->sjk", gamma, gamma)
    ricci -= tf.einsum("sijl,slik->sjk", gamma, gamma)
    return ricci


def _riemann_from_d_gamma(
    d_gamma: tf.Tensor,
    gamma: tf.Tensor,
) -> tf.Tensor:
    """R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj} + Γ^i_{km} Γ^m_{lj} - Γ^i_{lm} Γ^m_{kj}.

    Returns riemann (batch, 4, 4, 4, 4) with riemann[s, i, j, k, l] = R^i_{jkl}.
    """
    riemann = tf.einsum("siljk->sijkl", d_gamma)
    riemann -= tf.einsum("sikjl->sijkl", d_gamma)
    riemann += tf.einsum("sikm,smlj->sijkl", gamma, gamma)
    riemann -= tf.einsum("silm,smkj->sijkl", gamma, gamma)
    return riemann


def _kretschmann_from_d_gamma(
    d_gamma: tf.Tensor,
    gamma: tf.Tensor,
    metric: tf.Tensor,
    inv_metric: tf.Tensor,
) -> tf.Tensor:
    """K = R_{abcd} R^{abcd} via
    R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj} + Γ^i_{km} Γ^m_{lj} - Γ^i_{lm} Γ^m_{kj}.
    """
    riemann = _riemann_from_d_gamma(d_gamma, gamma)
    return tf.einsum(
        "sijkl,smnop,sim,sjn,sko,slp->s",
        riemann,
        riemann,
        metric,
        inv_metric,
        inv_metric,
        inv_metric,
    )


def _levi_civita_symbol_4(dtype: tf.dtypes.DType) -> tf.Tensor:
    """Return the 4D Levi-Civita symbol with epsilon_0123 = +1."""
    perms = [
        (0, 1, 2, 3, 1),
        (0, 1, 3, 2, -1),
        (0, 2, 1, 3, -1),
        (0, 2, 3, 1, 1),
        (0, 3, 1, 2, 1),
        (0, 3, 2, 1, -1),
        (1, 0, 2, 3, -1),
        (1, 0, 3, 2, 1),
        (1, 2, 0, 3, 1),
        (1, 2, 3, 0, -1),
        (1, 3, 0, 2, -1),
        (1, 3, 2, 0, 1),
        (2, 0, 1, 3, 1),
        (2, 0, 3, 1, -1),
        (2, 1, 0, 3, -1),
        (2, 1, 3, 0, 1),
        (2, 3, 0, 1, 1),
        (2, 3, 1, 0, -1),
        (3, 0, 1, 2, -1),
        (3, 0, 2, 1, 1),
        (3, 1, 0, 2, 1),
        (3, 1, 2, 0, -1),
        (3, 2, 0, 1, -1),
        (3, 2, 1, 0, 1),
    ]
    eps = tf.zeros((4, 4, 4, 4), dtype=dtype)
    indices = tf.constant([p[:4] for p in perms], dtype=tf.int32)
    updates = tf.constant([p[4] for p in perms], dtype=dtype)
    return tf.tensor_scatter_nd_update(eps, indices, updates)


def _weyl_ij_from_riemann(
    riemann: tf.Tensor,
    metric: tf.Tensor,
    inv_metric: tf.Tensor,
    ricci: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the complex Weyl invariants I and J in four dimensions.

    The invariants use the self-dual Weyl tensor
    Ctilde_abcd = C_abcd - i *C_abcd, with
    I = Ctilde_abcd Ctilde^abcd / 32 and
    J = Ctilde_abcd Ctilde^cd_mn Ctilde^mnab / 384.
    """
    R_low = tf.einsum("sai,sibcd->sabcd", metric, riemann)
    scalar_curvature = tf.einsum("sab,sab->s", inv_metric, ricci)

    ricci_part = (
        tf.einsum("sac,sbd->sabcd", metric, ricci)
        - tf.einsum("sad,sbc->sabcd", metric, ricci)
        - tf.einsum("sbc,sad->sabcd", metric, ricci)
        + tf.einsum("sbd,sac->sabcd", metric, ricci)
    )
    scalar_part = tf.einsum("s,sac,sbd->sabcd", scalar_curvature, metric, metric)
    scalar_part -= tf.einsum("s,sad,sbc->sabcd", scalar_curvature, metric, metric)

    weyl_low = R_low - 0.5 * ricci_part + scalar_part / 6.0

    eps_symbol = _levi_civita_symbol_4(metric.dtype)
    det_g = tf.linalg.det(metric)
    eps_low = tf.sqrt(tf.abs(det_g))[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    eps_low = eps_low * eps_symbol[tf.newaxis, ...]
    eps_ab_up_mn = tf.einsum("sabpq,spm,sqn->sabmn", eps_low, inv_metric, inv_metric)
    dual_weyl_low = 0.5 * tf.einsum("sabmn,smncd->sabcd", eps_ab_up_mn, weyl_low)

    complex_dtype = tf.complex128 if metric.dtype == tf.float64 else tf.complex64
    inv_metric_c = tf.cast(inv_metric, complex_dtype)
    weyl_self_dual = tf.complex(weyl_low, -dual_weyl_low)
    weyl_self_dual = tf.cast(weyl_self_dual, complex_dtype)

    I_raw = tf.einsum(
        "sabcd,sefgh,sae,sbf,scg,sdh->s",
        weyl_self_dual,
        weyl_self_dual,
        inv_metric_c,
        inv_metric_c,
        inv_metric_c,
        inv_metric_c,
    )
    weyl_mixed = tf.einsum(
        "sce,sdf,sabef->sabcd", inv_metric_c, inv_metric_c, weyl_self_dual
    )
    J_raw = tf.einsum("sabcd,scdef,sefab->s", weyl_mixed, weyl_mixed, weyl_mixed)
    return I_raw / 32.0, J_raw / 384.0


def speciality_index_from_invariants(
    weyl_i: tf.Tensor,
    weyl_j: tf.Tensor,
    eps: float | tf.Tensor = 1e-12,
) -> tf.Tensor:
    """Complex speciality index S = 27 J^2 / I^3."""
    eps = tf.cast(eps, tf.math.real(weyl_i).dtype)
    eps = tf.cast(eps, weyl_i.dtype)
    return 27.0 * tf.square(weyl_j) / (weyl_i ** 3 + eps)


# ---------------------------------------------------------------------------
# Core JVP-based pipeline: metric + first derivs + Hessian in one sweep.
# ---------------------------------------------------------------------------


def _metric_and_derivs_jvp(x_vars: tf.Tensor, metric_fn):
    """Metric, first derivatives, and Hessian w.r.t. ``x_vars`` via nested
    ``tf.autodiff.ForwardAccumulator``.

    The nested loop exploits only the (coordinate-index) Hessian symmetry
    ``∂_k ∂_l = ∂_l ∂_k`` — not any symmetry of ``g_{ij}``, which cannot be
    assumed given the asymmetric ``cholesky_from_vec`` output in Lorentzian
    mode (see the module-level note above).

    Parameters
    ----------
    x_vars : tf.Tensor (batch, n).
    metric_fn : callable (batch, n) -> (batch, n, n).

    Returns
    -------
    metric         : (batch, n, n)
    metric_derivs  : (batch, n, n, n)      with [s, i, j, k] = ∂_k g_{ij}
    metric_hessian : (batch, n, n, n, n)   with [s, i, j, k, l] = ∂_k ∂_l g_{ij}
    """
    n = x_vars.shape[-1]

    # Pre-build the basis of tangent vectors, broadcast to the batch.
    tangents = [
        tf.broadcast_to(
            tf.one_hot(d, n, dtype=x_vars.dtype)[tf.newaxis, :], tf.shape(x_vars)
        )
        for d in range(n)
    ]

    first_derivs: list[tf.Tensor | None] = [None] * n
    hessian_grid: list[list[tf.Tensor | None]] = [[None] * n for _ in range(n)]
    metric: tf.Tensor | None = None

    # Nested JVPs over coordinate pairs (k, l) with l >= k.
    #   - first-derivatives dg/dx_l are captured on the (0, l) pass;
    #   - second-derivatives d²g/(dx_k dx_l) on the (k, l) pass,
    #     mirrored across k↔l via partials-commute.
    for k in range(n):
        for l in range(k, n):
            with tf.autodiff.ForwardAccumulator(
                primals=x_vars, tangents=tangents[k]
            ) as acc_k:
                with tf.autodiff.ForwardAccumulator(
                    primals=x_vars, tangents=tangents[l]
                ) as acc_l:
                    m = metric_fn(x_vars)  # (batch, n, n)
                jvp_l = acc_l.jvp(m)  # dg / dx_l
            jvp_kl = acc_k.jvp(jvp_l)  # d²g / dx_k dx_l

            hessian_grid[k][l] = jvp_kl
            if k != l:
                hessian_grid[l][k] = jvp_kl  # ∂_k ∂_l = ∂_l ∂_k

            if first_derivs[l] is None:
                first_derivs[l] = jvp_l
            if metric is None:
                metric = m

    # (s, n, n, n): axis -1 = coordinate derivative.
    metric_derivs = tf.stack(first_derivs, axis=-1)

    # (s, n, n, n, n): axes 3, 4 = coordinate derivatives k, l.
    hessian_rows = [tf.stack(hessian_grid[k], axis=-1) for k in range(n)]
    metric_hessian = tf.stack(hessian_rows, axis=3)

    return metric, metric_derivs, metric_hessian


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for compute_ricci_and_kretschmann_embed.
# ---------------------------------------------------------------------------


@tf.function(reduce_retracing=True)
def compute_ricci_and_kretschmann_embed_opt(
    x_vars: tf.Tensor,
    submodel,
    lorentzian: bool = False,
    need_ricci: bool = True,
    need_kretschmann: bool = True,
    need_speciality_index: bool = False,
):
    """Forward-mode-autodiff analogue of
    ``geometry.schwarzschild.compute_ricci_and_kretschmann_embed``.

    Same inputs and return contract.  The pulled-back 4D metric, its first
    derivatives and its Hessian w.r.t. the 4D intrinsic coordinates are
    obtained in a single nested-JVP sweep; Christoffel, Ricci and Kretschmann
    are then assembled algebraically with ``need_ricci`` / ``need_kretschmann``
    gating the final post-assembly to avoid wasted einsums when a term is off.

    Parameters
    ----------
    x_vars           : tf.Tensor (batch, 5) = [T, X, q1, q2, patch_idx_float].
    submodel         : Keras model (batch, 5) ambient -> (batch, 15) Cholesky.
    lorentzian       : bool, passed through to ``cholesky_from_vec``.
    need_ricci       : if False, returns ``None`` for the Ricci tensor.
    need_kretschmann : if False, returns ``None`` for the Kretschmann scalar.
    need_speciality_index : if True, also returns the complex Weyl invariants
                            I and J used to form the speciality index; implies
                            Riemann assembly.

    Returns
    -------
    (metric (batch, 4, 4),
     ricci  (batch, 4, 4) or None,
     kretschmann (batch,) or None,
     weyl_i (batch,) or None,
     weyl_j (batch,) or None)
    """
    q_4d = x_vars[:, :4]
    patch_idx = tf.cast(x_vars[:, 4], tf.int32)

    def metric_fn(q: tf.Tensor) -> tf.Tensor:
        x_5d = embed_S2_coords(q, patch_idx)
        G_vec = submodel(x_5d)
        G_5d = cholesky_from_vec(G_vec, lorentzian=lorentzian)  # (s, 5, 5)
        J = embedding_jacobian_stereo(q, patch_idx)  # (s, 5, 4)
        return tf.einsum("sAB,sAm,sBn->smn", G_5d, J, J)  # (s, 4, 4)

    metric, metric_derivs, metric_hessian = _metric_and_derivs_jvp(q_4d, metric_fn)

    inv_metric = tf.linalg.inv(metric)
    gamma = _christoffel_from_metric(metric_derivs, inv_metric)

    ricci = None
    kretschmann = None
    weyl_i = None
    weyl_j = None
    if need_ricci or need_kretschmann or need_speciality_index:
        d_gamma = _christoffel_derivative_from_hessian(
            inv_metric, metric_derivs, metric_hessian, gamma
        )
        ricci_for_weyl = None
        if need_ricci or need_speciality_index:
            ricci_for_weyl = _ricci_from_d_gamma(d_gamma, gamma)
        if need_ricci:
            ricci = ricci_for_weyl
        if need_kretschmann or need_speciality_index:
            # Assemble Riemann once; share it for K and Weyl invariants.
            riemann = _riemann_from_d_gamma(d_gamma, gamma)
            if need_kretschmann:
                kretschmann = tf.einsum(
                    "sijkl,smnop,sim,sjn,sko,slp->s",
                    riemann, riemann, metric, inv_metric, inv_metric, inv_metric,
                )
            if need_speciality_index:
                weyl_i, weyl_j = _weyl_ij_from_riemann(
                    riemann, metric, inv_metric, ricci_for_weyl
                )

    return metric, ricci, kretschmann, weyl_i, weyl_j


@tf.function(reduce_retracing=True)
def compute_full_curvature_embed_opt(
    x_vars: tf.Tensor,
    submodel,
    lorentzian: bool = False,
):
    """Like ``compute_ricci_and_kretschmann_embed_opt`` but additionally returns
    the full Riemann tensor R^i_{jkl}, needed for higher-order invariants such
    as the Weyl scalars used by the speciality index.

    Parameters
    ----------
    x_vars    : tf.Tensor (batch, 5) = [T, X, q1, q2, patch_idx_float].
    submodel  : Keras model (batch, 5) -> (batch, 15) Cholesky vec.
    lorentzian: bool.

    Returns
    -------
    (metric      (batch, 4, 4),
     ricci       (batch, 4, 4),
     kretschmann (batch,),
     riemann     (batch, 4, 4, 4, 4))  riemann[s,i,j,k,l] = R^i_{jkl}
    """
    q_4d = x_vars[:, :4]
    patch_idx = tf.cast(x_vars[:, 4], tf.int32)

    def metric_fn(q: tf.Tensor) -> tf.Tensor:
        x_5d = embed_S2_coords(q, patch_idx)
        G_vec = submodel(x_5d)
        G_5d = cholesky_from_vec(G_vec, lorentzian=lorentzian)
        J = embedding_jacobian_stereo(q, patch_idx)
        return tf.einsum("sAB,sAm,sBn->smn", G_5d, J, J)

    metric, metric_derivs, metric_hessian = _metric_and_derivs_jvp(q_4d, metric_fn)
    inv_metric = tf.linalg.inv(metric)
    gamma = _christoffel_from_metric(metric_derivs, inv_metric)
    d_gamma = _christoffel_derivative_from_hessian(
        inv_metric, metric_derivs, metric_hessian, gamma
    )
    ricci = _ricci_from_d_gamma(d_gamma, gamma)
    riemann = _riemann_from_d_gamma(d_gamma, gamma)
    kretschmann = _kretschmann_from_d_gamma(d_gamma, gamma, metric, inv_metric)

    return metric, ricci, kretschmann, riemann
