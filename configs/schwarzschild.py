from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .base import (BaseConfig, FloatScheduleConfig, KerasSerialisableObject,
                   LossConfig, ModelSpecific, VisualisationConfig)


class SchwarzschildProperties(BaseModel, KerasSerialisableObject):
    lorentzian: bool = False
    embed: bool = True  # use global S^2 embedding architecture
    # Local 2D mode disables Schwarzschild embedding/Penrose pathways and uses
    # a direct 2D local patch metric model on ball coordinates.
    local_2d_mode: bool = False
    # In local_2d_mode, patch_width_S2 and density_power_S2 are used as the
    # single 2D patch sampling controls; R2 controls are ignored.
    local_2d_patch_width: float | None = None
    local_2d_density_power: float | None = None
    # Local-patch mode: sample a single stereographic S^2 chart only.
    # This is useful for local Einstein-geometry experiments where overlap is
    # intentionally disabled and one chart acts as the local patch domain.
    local_single_s2_patch: bool = False
    local_s2_patch_idx: Literal[0, 1] = 0
    density_power_R2: float = 1.0
    density_power_R2_schedule: FloatScheduleConfig | None = None
    density_power_S2: float = 1.0
    density_power_S2_schedule: FloatScheduleConfig | None = None
    patch_width_R2: float = 1.0
    patch_width_S2: float = 1.0
    use_penrose_region_curriculum: bool = False
    penrose_region_exterior_start: float = 0.55
    penrose_region_interior_start: float = 0.25
    penrose_region_horizon_start: float = 0.15
    penrose_region_singularity_start: float = 0.05
    penrose_region_exterior_end: float = 0.25
    penrose_region_interior_end: float = 0.55
    penrose_region_horizon_end: float = 0.15
    penrose_region_singularity_end: float = 0.05
    penrose_region_horizon_width: float = 0.06
    penrose_region_singularity_width: float = 0.10
    penrose_interior_only: bool = False
    kretschmann_multiplier: float = 1.0
    r2_det_loss_multiplier: float = 0.0  # weight for 1/|det(g_R2)| barrier loss
    speciality_index_multiplier: float = 0.0  # weight for S=1 type-D loss
    killing_symmetry_multiplier: float = 0.0  # weight for SO(3) Killing residual loss
    k_repeller_multiplier: float = 0.0  # weight for 1/(|K|+epsilon) Minkowski-repeller
    k_repeller_epsilon: float = 1e-4    # additive floor in K-repeller denominator
    speciality_index_rprofile_mode: Literal[
        "value", "profile", "gradient", "variance", "discriminant", "hybrid"
    ] = "profile"  # "variance" is accepted as a legacy alias for "gradient"
    speciality_index_rprofile_centre: float = 2.0  # mean target for S profile
    speciality_index_rprofile_multiplier: float = 0.0  # weight for S profile term
    speciality_index_rprofile_epsilon: float = 1e-2    # additive floor in repeller denominator
    # Horizon curvature anchor (Campaign 7, LC-0015): pin the curvature near the horizon
    # r=2m to fix the scale and enforce a regular (finite-curvature) horizon, while leaving
    # the bulk free to be type-I. Built on the same degree-2 quantity as weyl_invariant,
    # sqrt(|inv|+eps)*r^3 (Schwarzschild: =sqrt(c*48)*m, r-independent), Gaussian-weighted around
    # r=2m:  L = sum_s w_s (sqrt(|inv_s|+eps)*r_s^3 - target)^2 / sum_s w_s,
    #        w_s = exp(-((r_s-2m)/(band*2m))^2).  inv = |weyl_i| (Ricci-decoupled; default) or |K|.
    horizon_anchor_multiplier: float = 0.0  # weight for the horizon curvature anchor
    horizon_anchor_target: float = 1.7320508075688772  # sqrt(3): Schwarzschild sqrt(|I|)*r^3 value
    # (m=1, c=1/16). For invariant="kretschmann" use sqrt(48)=6.928203 instead.
    horizon_anchor_band: float = 0.1  # Gaussian sigma in r, as a fraction of 2m
    horizon_anchor_invariant: Literal["weyl_i", "kretschmann"] = "weyl_i"
    # Trapped-surface (Misner-Sharp) prescriber (Campaign 7b, LC-0017): enforce a black-hole
    # CAUSAL structure. chi = g^{mn} d_m R d_n R (R = areal radius from the angular 2-block;
    # = 1-2m/r for Schwarzschild): chi<0 TRAPPED, =0 marginal (horizon), >0 untrapped. The c7
    # type-I near-vacuum lacks trapped surfaces (chi>0 even inside); this term supplies the
    # missing causal/null structure WITHOUT pinning the radial profile (sign + horizon-zero only):
    #   L = mean_s [ relu(-chi_s*sign(r_s-2m) + margin)^2 + hor_weight * w_hor(r_s) * chi_s^2 ],
    #   w_hor = exp(-((r-2m)/(band*2m))^2).  chi via one GradientTape (cheaper than Ricci).
    trapped_surface_multiplier: float = 0.0  # weight for the trapped-surface prescriber
    trapped_surface_band: float = 0.1  # Gaussian sigma in r (fraction of 2m) for the chi->0 horizon term
    trapped_surface_horizon_weight: float = 1.0  # rel. weight of the chi->0 band term vs the sign hinge
    trapped_surface_margin: float = 0.0  # require chi<=-margin inside / >=+margin outside (0 = sign only)
    # margin_mode (LC-0019): "constant" = the original c7b term (band + (1-w_hor) sign-hinge, uses
    # trapped_surface_{band,horizon_weight,margin}); "profile" = a single r-dependent one-sided bound
    # margin(r)=margin_slope*max(0,2m-r)/(2m), penalty relu(margin(r)-sign(r-2m)*chi)^2, NO band/
    # suppression -> chi=0 at the horizon automatically, chi<0 growing inward (profile free above it).
    trapped_surface_margin_mode: Literal["constant", "profile"] = "constant"
    trapped_surface_margin_slope: float = 1.0  # alpha in the profile margin (1.0 ~ Schwarzschild slope)
    # profile margin(r) = (margin_floor + slope*max(0,2m-r)/(2m)) for r<2m, else 0. The FLOOR is a
    # constant near-horizon push (>0 keeps a live gradient just inside r=2m, where most interior
    # samples sit and where slope*depth->0 is too weak); chi=0 still holds AT r=2m by continuity.
    trapped_surface_margin_floor: float = 0.0
    m: float = 1.0  # Schwarzschild mass parameter
    use_volume_scaling: bool | None = None  # weight loss norms by sqrt(abs(det(g)))
    # Legacy alias for use_volume_scaling, retained for existing hyperparameter files.
    use_area_measure_weight: bool = False
    use_metric_contraction: bool = False  # contract tensor/vector norms with inverse metric
    volume_scaling_loss_components: dict[str, bool] | None = None
    metric_contraction_loss_components: dict[str, bool] | None = None
    # Ricci / Kretschmann computation kernel.
    #   "standard"  -> reverse-mode two-level GradientTape (default, the kernel
    #                  used for all published / existing runs).
    #   "optimised" -> forward-mode autodiff (nested ForwardAccumulator JVPs
    #                  over unique symmetric metric components); algebraically
    #                  equivalent, typically faster when dim is small.
    ricci_kernel: Literal["standard", "optimised"] = "standard"

    # --- Experimental loss-form variants (proposals 3-5). Defaults reproduce
    #     the original behaviour exactly; the new modes are opt-in. ---
    # (3) Einstein-residual form:
    #   "volume_integral"     -> original: ||C g - Ric||^2_contracted * sqrt|det g|
    #                            (extensive; gameable by shrinking the metric)
    #   "curvature_normalized"-> ||C g - Ric||^2_contracted / (|K| + eps)
    #                            (intensive, homothety-invariant; cannot be gamed
    #                            by global rescaling). No volume factor.
    #   "contracted_plain"    -> ||C g - Ric||^2_contracted with NO volume factor
    #                            and NO 1/K, i.e. UNIFORM weighting of the
    #                            tensorial Ricci norm over the sample. Homothety-
    #                            dependent, but safe once the det barrier blocks
    #                            collapse; isolates "regional weighting = uniform".
    #   "curvature_normalized_capped" -> ||E||^2_M / (min(|K|, K_cap) + eps),
    #                            K_cap = curvature_norm_cap_kappa * K_hor,
    #                            K_hor = 48 m^2 / (2m)^6 = 0.75/m^4 (the horizon
    #                            Kretschmann, the BH's physical curvature scale).
    #                            For |K|<K_cap identical to curvature_normalized;
    #                            for |K|>K_cap (strong-field interior, where 1/|K|
    #                            starves the loss) it reverts to a fixed reference
    #                            scale, restoring uniform Ricci-flatness pressure.
    #                            Mass-scaled BY DESIGN (the cap is the mass scale).
    #   "contracted_volume"   -> ||C g - Ric||^2_M * sqrt|det g| (the SPD-contracted
    #                            residual times the proper volume element
    #                            sqrt(-det g); NO 1/K). The covariant-integral
    #                            sibling of "contracted_plain"; the volume factor is
    #                            intrinsic to the mode (does NOT depend on the
    #                            use_volume_scaling flag). Extensive -- relies on the
    #                            det barrier to block the shrink-the-metric collapse.
    #                            (LC-0011)
    #   "weyl_normalized"     -> ||C g - Ric||^2_M / (|I| + eps), where |I| = |weyl_i|
    #                            is the self-dual Weyl invariant (Ricci-DECOUPLED).
    #                            Like "curvature_normalized" (homothety-invariant: both
    #                            numerator and |I| scale as lambda^-2 under g->lambda g,
    #                            so inflation cannot game it), but normalising by the
    #                            Weyl part alone -- the denominator cannot be lowered by
    #                            trading Ricci into K (K = C^2 + 2|Ric|^2 - R^2/3).
    #                            Needs the speciality-index branch of the kernel for
    #                            weyl_i. eps = curvature_norm_epsilon. (LC-0014)
    einstein_loss_mode: Literal[
        "volume_integral",
        "curvature_normalized",
        "curvature_normalized_capped",
        "contracted_plain",
        "contracted_volume",
        "weyl_normalized",
    ] = "volume_integral"
    curvature_norm_epsilon: float = 1e-3  # eps in ||Ric||^2 / (|K| + eps)
    # kappa in K_cap = kappa * K_hor for "curvature_normalized_capped" only.
    # kappa=1 caps at the horizon (r=2m); larger kappa caps deeper inside; a very
    # large value reproduces "curvature_normalized" (uncapped). Unused otherwise.
    curvature_norm_cap_kappa: float = 2.0
    # (4) Kretschmann "scale" form:
    #   "log_profile" -> original: (log1p|K| - log1p K_analytic)^2
    #   "sqrt_const"  -> (sqrt(|K|+eps) * r^3 - sqrt(48) m)^2, a degree-2 (Weyl-
    #                    scalar-like) quantity matched to a CONSTANT target,
    #                    removing the dynamic range and the quartic-g^{-1}
    #                    conditioning of K. No volume factor.
    #   "weyl_invariant" -> Ricci-DECOUPLED prescriber: same sqrt_const form but on
    #                    the (directly computed) Weyl invariant |I| instead of the
    #                    full Kretschmann K. Since K = C^2 + 2|Ric|^2 - R^2/3, the
    #                    Kretschmann couples to Ricci (it can be satisfied by trading
    #                    too-small Weyl against too-large Ricci, rewarding R!=0);
    #                    prescribing the Weyl invariant removes that coupling so the
    #                    K-term and the Ricci-flatness act on orthogonal curvature
    #                    pieces. Loss = (sqrt(|weyl_i|+eps) * r^3 - sqrt(c*48) m)^2,
    #                    with |I| = c * K for Schwarzschild; this code's I = Ctilde^2/32
    #                    gives c = 1/16 (verified: |I| = K/16 = 3 m^2/r^6, target
    #                    sqrt(3) m). Still pins the curvature scale everywhere, so it
    #                    does NOT permit the Minkowski (K=0) branch.
    #   "weyl_invariant_volume" -> identical to "weyl_invariant" but the per-point
    #                    residual is multiplied by the proper volume element
    #                    sqrt(|det g|) = sqrt(-det g) (covariant integral measure).
    #                    The volume factor is intrinsic to the mode (does NOT depend
    #                    on use_volume_scaling); plain "weyl_invariant" stays
    #                    unweighted. (LC-0011)
    #   "weyl_invariant_alt" -> Ricci-decoupled prescriber like "weyl_invariant" but
    #                    targeting a NON-Schwarzschild curvature profile:
    #                    (sqrt(|weyl_i|+eps) * r^p - T)^2, prescribing |I| = T^2 / r^(2p)
    #                    with p = weyl_alt_r_power, T = weyl_alt_target. Schwarzschild is
    #                    p=3, T=sqrt(c*48) m (=> |I| ~ r^-6); p != 3 prescribes a different
    #                    curvature falloff (a "reasonable alternative" scale). Still
    #                    Ricci-decoupled, so it does not reward R != 0. (LC-0014)
    kretschmann_loss_mode: Literal[
        "log_profile", "sqrt_const", "weyl_invariant", "weyl_invariant_volume",
        "weyl_invariant_alt",
    ] = "log_profile"
    kretschmann_sqrt_epsilon: float = 1e-12  # eps under the sqrt in sqrt_const / weyl_invariant
    # c in |weyl_i| = c * K for "weyl_invariant" only (target = sqrt(c*48) m).
    # Default 1/16 matches this code's I = Ctilde_abcd Ctilde^abcd / 32 convention
    # (empirically |I| = K/16 = 3 m^2/r^6 on the exact solution). Unused otherwise.
    weyl_invariant_norm: float = 0.0625
    # "weyl_invariant_alt" prescriber: sqrt(|weyl_i|) * r^(weyl_alt_r_power) -> weyl_alt_target.
    # Defaults (p=3, T=sqrt(3)) reproduce the Schwarzschild Weyl scale (|I| = 3 m^2/r^6 for
    # m=1, c=1/16). Change them for a non-Schwarzschild target. Used ONLY by that mode.
    weyl_alt_r_power: float = 3.0
    weyl_alt_target: float = 1.7320508075688772  # sqrt(3)
    # (5) Determinant / non-degeneracy barrier scope:
    #   "r2_only"     -> original: 1/|det g_R2| (T,X block only)
    #   "both_blocks" -> 1/|det g_R2| + 1/|det g_S2| (guards BOTH blocks against
    #                    collapse). Never volume-scaled (a barrier must not be
    #                    suppressed exactly where it should diverge).
    det_barrier_mode: Literal["r2_only", "both_blocks"] = "r2_only"


class SchwarzschildVisualisation(VisualisationConfig):
    num_samples: int = int(1e3)
    patch_width_R2: float = 0.8
    patch_width_S2: float = 1.0
    # Optional override for local single-chart visualisation. If null, the
    # value from model_specific is used so visualisation matches training.
    local_single_s2_patch: bool | None = None
    local_s2_patch_idx: Literal[0, 1] | None = None
    # ...the \alpha (>0) value in the beta
    # function, values < 1 skew to boundary, >1
    # skew to centre
    density_power_R2: float = 1.0
    # S^2 radial sampling bias for visualisation. Per StereoSampleHemisphere:
    #   <1 -> pole-biased, =1 -> UNIFORM on S^2 (full coverage incl. poles),
    #   >1 -> equator-biased (points pile up at the equator, leaving the poles
    #   empty). Default is 1.0 so plotted points cover the whole sphere.
    density_power_S2: float = 1.0


class SchwarzschildLossConfig(LossConfig, KerasSerialisableObject):
    """Schwarzschild-specific loss configuration with Kretschmann and R2Det schedulers."""

    kretschmann_schedule: FloatScheduleConfig | None = None
    r2_det_schedule: FloatScheduleConfig | None = None
    speciality_index_schedule: FloatScheduleConfig | None = None
    killing_symmetry_schedule: FloatScheduleConfig | None = None
    k_repeller_schedule: FloatScheduleConfig | None = None
    speciality_index_rprofile_schedule: FloatScheduleConfig | None = None
    horizon_anchor_schedule: FloatScheduleConfig | None = None
    trapped_surface_schedule: FloatScheduleConfig | None = None


class SchwarzschildTrainingStage(BaseModel, KerasSerialisableObject):
    """Optional staged-training override for Schwarzschild continuation phases."""

    name: str = "stage"
    epochs: int | None = None
    init_learning_rate: float | None = None
    min_learning_rate: float | None = None

    einstein_multiplier: float | None = None
    kretschmann_multiplier: float | None = None
    r2_det_loss_multiplier: float | None = None
    speciality_index_multiplier: float | None = None
    killing_symmetry_multiplier: float | None = None
    k_repeller_multiplier: float | None = None
    speciality_index_rprofile_multiplier: float | None = None

    einstein_schedule: FloatScheduleConfig | None = None
    kretschmann_schedule: FloatScheduleConfig | None = None
    r2_det_schedule: FloatScheduleConfig | None = None
    speciality_index_schedule: FloatScheduleConfig | None = None
    killing_symmetry_schedule: FloatScheduleConfig | None = None
    k_repeller_schedule: FloatScheduleConfig | None = None
    speciality_index_rprofile_schedule: FloatScheduleConfig | None = None

    use_area_measure_weight: bool | None = None
    use_volume_scaling: bool | None = None
    use_metric_contraction: bool | None = None
    volume_scaling_loss_components: dict[str, bool] | None = None
    metric_contraction_loss_components: dict[str, bool] | None = None

    density_power_R2: float | None = None
    density_power_R2_schedule: FloatScheduleConfig | None = None
    density_power_S2: float | None = None
    density_power_S2_schedule: FloatScheduleConfig | None = None
    patch_width_R2: float | None = None
    patch_width_S2: float | None = None

    use_penrose_region_curriculum: bool | None = None
    penrose_region_exterior_start: float | None = None
    penrose_region_interior_start: float | None = None
    penrose_region_horizon_start: float | None = None
    penrose_region_singularity_start: float | None = None
    penrose_region_exterior_end: float | None = None
    penrose_region_interior_end: float | None = None
    penrose_region_horizon_end: float | None = None
    penrose_region_singularity_end: float | None = None
    penrose_region_horizon_width: float | None = None
    penrose_region_singularity_width: float | None = None
    penrose_interior_only: bool | None = None


class SchwarzschildConfig(BaseConfig, ModelSpecific, KerasSerialisableObject):
    model_specific: SchwarzschildProperties = SchwarzschildProperties()
    visualisation: SchwarzschildVisualisation = SchwarzschildVisualisation()

    loss: SchwarzschildLossConfig = SchwarzschildLossConfig()
    training_stages: list[SchwarzschildTrainingStage] | None = None
