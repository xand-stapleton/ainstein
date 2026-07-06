from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf.keras.backend.set_floatx("float64")
from helper_functions.helper_functions import cholesky_from_vec


###############################################################################
# Functions to change between patches
def PatchChange_Coordinates_R2S2(coords):
    coords_R2, coords_S2 = tf.split(coords, num_or_size_splits=2, axis=1)

    # Compute the coordinate norm
    norm = tf.norm(coords_S2, axis=1)

    # Compute the patch transformation
    coords_S2_otherpatch = coords_S2 * tf.expand_dims(
        (norm - 1) / (norm * (norm + 1)), axis=-1
    )

    # Readd the R2 coordinates
    coords_otherpatch = tf.concat([coords_R2, coords_S2_otherpatch], axis=1)

    return coords_otherpatch


def PatchChange_Metric_R2S2(coords, metric_pred):
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_R2S2(coords)
    coords_otherpatch_S2 = coords_otherpatch[:, 2:]

    # Compute the coordinate norm
    norm = tf.norm(coords_otherpatch_S2, axis=1)

    # Compute the Jacobian
    # Use tf.shape for the batch dimension so this is safe under @tf.function
    # with dynamic (unknown-at-trace-time) batch sizes.
    _bs = tf.shape(coords_otherpatch_S2)[0]
    jacobian_term1 = tf.eye(
        coords_otherpatch_S2.shape[1],  # always 2 – static dim is fine
        batch_shape=[_bs],
        dtype=coords_otherpatch_S2.dtype,
    )
    jacobian_term1 *= tf.expand_dims(
        tf.expand_dims((norm - 1) / (norm * (norm + 1)), axis=-1), axis=-1
    )
    jacobian_term2 = tf.einsum("si,sj->sij", coords_otherpatch_S2, coords_otherpatch_S2)
    jacobian_term2 *= tf.expand_dims(
        tf.expand_dims(
            (1 + 2 * norm - tf.square(norm)) / (tf.pow(norm, 3) * tf.square(1 + norm)),
            axis=-1,
        ),
        axis=-1,
    )
    jacobian_S2 = jacobian_term1 + jacobian_term2

    # Embed the S2 jacobian in the full R2xS2 jacobian (R2 Jacobian is just identity)
    _zeros22 = tf.zeros(tf.stack([_bs, 2, 2]), dtype=coords.dtype)
    jacobian = tf.concat(
        [
            tf.concat(
                [tf.eye(2, batch_shape=[_bs], dtype=coords.dtype), _zeros22],
                axis=2,
            ),
            tf.concat(
                [_zeros22, jacobian_S2],
                axis=2,
            ),
        ],
        axis=1,
    )

    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sji,sjk,skl->sil", jacobian, metric_pred, jacobian)

    return metric_otherpatch


###############################################################################
def PenroseDiagram_Masks(coords):
    """Return boolean masks for B_I, B_II, B_III, B_IV given Penrose diagram coordinates."""
    T, X = coords[:, 0], coords[:, 1]
    pi = tf.constant(np.pi, dtype=coords.dtype)

    mask_BI = tf.logical_and(
        tf.logical_and(
            T > (tf.abs(X - pi / 4) - pi / 4), T < (-tf.abs(X - pi / 4) + pi / 4)
        ),
        tf.logical_and(X > 0, X < pi / 2),
    )

    mask_BII = tf.logical_and(
        tf.logical_and(T > tf.abs(X), T < pi / 4),
        tf.logical_and(X > -pi / 4, X < pi / 4),
    )

    mask_BIII = tf.logical_and(
        tf.logical_and(
            T > (tf.abs(X + pi / 4) - pi / 4), T < (-tf.abs(X + pi / 4) + pi / 4)
        ),
        tf.logical_and(X > -pi / 2, X < 0),
    )

    mask_BIV = tf.logical_and(
        tf.logical_and(T > -pi / 4, T < -tf.abs(X)),
        tf.logical_and(X > -pi / 4, X < pi / 4),
    )

    return mask_BI, mask_BII, mask_BIII, mask_BIV


def PenroseRadiusWeighting(coords, m=1.0):
    """
    Compute r(T,X) for batch coords using the piecewise domain definition.

    Args:
        coords: tf.Tensor, Penrose diagram coordinates
        m: scalar (float or tf.Tensor), mass
    Returns:
        tf.Tensor of shape, r(T,X) values.
    """
    T, X = coords[:, 0], coords[:, 1]

    # Define domain masks
    mask_BI, mask_BII, mask_BIII, mask_BIV = PenroseDiagram_Masks(coords)

    # Compute common trigonometric formulas
    cos_2T = tf.cos(2.0 * T)
    cos_2X = tf.cos(2.0 * X)

    # For BI and BIII: (note arccoth(z) = 0.5 * log((z+1)/(z-1)))
    arccoth_val = 0.5 * tf.math.log((cos_2T / cos_2X + 1.0) / (cos_2T / cos_2X - 1.0))
    val_BI_BIII = tfp.math.lambertw(tf.exp(-2.0 * arccoth_val - 1.0))

    # For BII and BIV:
    arctanh_val = tf.math.atanh(cos_2T / cos_2X)
    val_BII_BIV = tfp.math.lambertw(-tf.exp(-2.0 * arctanh_val - 1.0))

    # Combine according to masks
    r_values = tf.zeros_like(T, dtype=coords.dtype)
    r_values = tf.where(mask_BI | mask_BIII, val_BI_BIII, r_values)
    r_values = tf.where(mask_BII | mask_BIV, val_BII_BIV, r_values)

    # Final weighting values
    r_values = 2.0 * m * (1.0 + r_values)

    return r_values


def AnalyticMetric_R2S2(coords, identity=True, lorentzian=False, m=1.0):
    # Return the product metric Mink(1,1) x round-S^2 if requested.
    #
    # In stereographic coordinates the round S^2 metric is
    #   g_S2 = 4 / (1 + |q|^2)^2 * I_2,
    # so the full identity-target diagonal is (-1, 1, s2_f, s2_f) with
    #   s2_f = 4 / (1 + q1^2 + q2^2)^2.
    #
    # This equals the pullback of the flat 5D Lorentzian ambient metric
    # G = diag(-1,1,1,1,1) through the embedding Jacobian J (proven by the
    # identity (1+q2^2-q1^2)^2 + 4q1^2*q2^2 + 4*qi^2 = N^2), and it is
    # globally consistent across stereographic patches because the round-S^2
    # formula 4/N^2 is invariant under the inversion transition q' = q/|q|^2
    # (the inversion Jacobian satisfies J_inv^T J_inv = |q|^{-4} I_2, which
    # cancels (1+|q'|^2)^{-2} = |q|^4 (1+|q|^2)^{-2} exactly).
    # The natural Cholesky initialisation is therefore L = I_5.
    if identity:
        bs = tf.shape(coords)[0]
        dtype = coords.dtype
        # Round S^2 conformal factor in stereographic coordinates
        q_sq = tf.square(coords[:, 2]) + tf.square(coords[:, 3])  # (batch,)
        s2_f = 4.0 / tf.square(1.0 + q_sq)  # (batch,)
        if lorentzian:
            # Mink(1,1) x round-S^2: diag(-1, 1, s2_f, s2_f)
            diag_vals = tf.stack(
                [
                    tf.fill([bs], tf.cast(-1.0, dtype)),
                    tf.ones([bs], dtype=dtype),
                    s2_f,
                    s2_f,
                ],
                axis=1,
            )
        else:
            # Euclidean R^2 x round-S^2: diag(1, 1, s2_f, s2_f)
            diag_vals = tf.stack(
                [tf.ones([bs], dtype=dtype), tf.ones([bs], dtype=dtype), s2_f, s2_f],
                axis=1,
            )
        return tf.linalg.diag(diag_vals)  # (batch, 4, 4)

    else:
        # Ensure using Lorentzian signature
        assert lorentzian, (
            "Schwarzschild metric only implemented for Lorentzian signature"
        )

        # Separate the coordinates
        T, X = coords[:, 0], coords[:, 1]  # ...R2
        x, y = coords[:, 2], coords[:, 3]  # ...S2

        # Compute r(T, X)
        penrose_radii_weightings = PenroseRadiusWeighting(coords[:, :2], m)
        prw_sq = tf.square(penrose_radii_weightings)

        # Compute F(T, X)
        F_numerator = 32.0 * (m**3) * tf.exp(-penrose_radii_weightings / (2.0 * m))
        F_denom = penrose_radii_weightings * (tf.cos(T) ** 2 - tf.sin(X) ** 2) ** 2
        F = F_numerator / F_denom

        # S^2 terms.
        # The stereographic charts use the convention of stereo_to_cartesian:
        #   xc = 2q1/N,  yc = 2q2/N,  zc = ±(1-r²)/N,  N = 1+q1²+q2²
        # The pullback of the round metric dΩ² = dxc²+dyc²+dzc² to these
        # coordinates is the isotropic form:
        #   g_S² = 4/(1+q1²+q2²)² (dq1² + dq2²)
        # so the Schwarzschild S² contribution prw² dΩ² is:
        x_sq, y_sq = tf.square(x), tf.square(y)
        stereo_denom = tf.square(1.0 + x_sq + y_sq)  # (1 + r_stereo²)²
        s2_coeff = 4.0 * prw_sq / stereo_denom
        gs00 = s2_coeff
        gs01 = tf.zeros_like(s2_coeff)
        gs11 = s2_coeff

        # Combine into full metric
        metric = tf.stack(
            [
                tf.stack(
                    [-F, tf.zeros_like(F), tf.zeros_like(F), tf.zeros_like(F)], axis=1
                ),
                tf.stack(
                    [tf.zeros_like(F), F, tf.zeros_like(F), tf.zeros_like(F)], axis=1
                ),
                tf.stack([tf.zeros_like(F), tf.zeros_like(F), gs00, gs01], axis=1),
                tf.stack([tf.zeros_like(F), tf.zeros_like(F), gs01, gs11], axis=1),
            ],
            axis=1,
        )

        return metric


def AnalyticVielbein_R2S2(coords, m=1.0):
    # Separate the coordinates
    T, X = coords[:, 0], coords[:, 1]  # ...R2
    x, y = coords[:, 2], coords[:, 3]  # ...S2

    # Compute r(T, X)
    prw = PenroseRadiusWeighting(coords[:, :2], m)

    # Compute F(T, X)
    F_numerator = 32.0 * (m**3) * tf.exp(-prw / (2.0 * m))
    F_denom = prw * (tf.cos(T) ** 2 - tf.sin(X) ** 2) ** 2
    F = F_numerator / F_denom
    F_sqrt = tf.sqrt(F)

    # Compute S^2 vielbein in stereographic coordinates.
    # The round-sphere metric is g_S² = 4/(1+r²)² I₂ (per stereo_to_cartesian
    # convention), so the lower-triangular Cholesky factor is (2prw/(1+r²)) I₂.
    x_sq, y_sq = tf.square(x), tf.square(y)
    vs_coeff = 2.0 * prw / (1.0 + x_sq + y_sq)  # 2r / (1 + q1² + q2²)
    vs00 = vs_coeff
    vs10 = tf.zeros_like(vs_coeff)
    vs11 = vs_coeff

    # Combine into full lower-triangular vielbein
    vielbein = tf.stack(
        [
            tf.stack(
                [F_sqrt, tf.zeros_like(F), tf.zeros_like(F), tf.zeros_like(F)], axis=1
            ),
            tf.stack(
                [tf.zeros_like(F), F_sqrt, tf.zeros_like(F), tf.zeros_like(F)], axis=1
            ),
            tf.stack(
                [tf.zeros_like(F), tf.zeros_like(F), vs00, tf.zeros_like(F)], axis=1
            ),
            tf.stack([tf.zeros_like(F), tf.zeros_like(F), vs10, vs11], axis=1),
        ],
        axis=1,
    )

    return vielbein


def Analytic_Kretschmann(coords, m=1.0):
    """
    Compute the analytic Kretschmann invariant over the 4-part domain of the Penrose diagram.

    Args:
        coords: tf.Tensor, Penrose diagram coordinates
        m: scalar (float or tf.Tensor), mass
    Returns:
        tf.Tensor
    """
    # Separate the coordinates
    T, X = coords[:, 0], coords[:, 1]

    # Define domain masks
    mask_BI, mask_BII, mask_BIII, mask_BIV = PenroseDiagram_Masks(coords)

    # Compute common trigonometric formulas
    cos_2T = tf.cos(2.0 * T)
    cos_2X = tf.cos(2.0 * X)

    # For B_I and B_III: (note arccoth(z) = 0.5 * log((z+1)/(z-1)))
    arccoth_val = 0.5 * tf.math.log((cos_2T / cos_2X + 1.0) / (cos_2T / cos_2X - 1.0))
    W_BI_BIII = tfp.math.lambertw(tf.exp(-2.0 * arccoth_val - 1.0))
    expr_BI_BIII = 3.0 / (4.0 * m**4 * (W_BI_BIII + 1.0) ** 6)

    # For B_II and B_IV:
    arctanh_val = tf.math.atanh(cos_2T / cos_2X)
    W_BII_BIV = tfp.math.lambertw(-tf.exp(-2.0 * arctanh_val - 1.0))
    expr_BII_BIV = 3.0 / (4.0 * m**4 * (W_BII_BIV + 1.0) ** 6)

    # Combine to form full invariant
    kretschamnn = tf.zeros_like(T, dtype=coords.dtype)
    kretschamnn = tf.where(mask_BI | mask_BIII, expr_BI_BIII, kretschamnn)
    kretschamnn = tf.where(mask_BII | mask_BIV, expr_BII_BIV, kretschamnn)

    return kretschamnn


###############################################################################
# S^2 global embedding functions


def stereo_to_cartesian(coords_S2, patch_idx):
    """
    Inverse stereographic projection: 2D patch coords -> unit-sphere Cartesian.

    North chart (patch_idx=0) — north pole maps to the origin:
        (q1, q2) -> (2q1/(1+r^2),  2q2/(1+r^2),  (1-r^2)/(1+r^2))
    South chart (patch_idx=1) — south pole maps to the origin:
        (q1, q2) -> (2q1/(1+r^2),  2q2/(1+r^2),  (r^2-1)/(1+r^2))

    The unit disk ||q|| <= 1 maps to the closed hemisphere containing the
    pole that is near the origin in each chart.

    Args:
        coords_S2:  tf.Tensor (batch, 2), stereographic coordinates.
        patch_idx:  tf.Tensor (batch,) int32, 0 = north chart, 1 = south chart.
    Returns:
        tf.Tensor (batch, 3), Cartesian coordinates on the unit S^2.
    """
    q1, q2 = coords_S2[:, 0], coords_S2[:, 1]
    r_sq = tf.square(q1) + tf.square(q2)
    denom = 1.0 + r_sq

    x_cart = 2.0 * q1 / denom
    y_cart = 2.0 * q2 / denom
    z_north = (1.0 - r_sq) / denom
    z_south = (r_sq - 1.0) / denom

    z_cart = tf.where(tf.equal(patch_idx, 0), z_north, z_south)
    return tf.stack([x_cart, y_cart, z_cart], axis=1)


def embedding_jacobian_stereo(q_4d, patch_idx):
    """
    Analytic Jacobian of the embedding map (T, X, q1, q2) -> (T, X, Xc, Yc, Zc),
    shape (batch, 5, 4).

    The R^2 block is the 2x2 identity.  The S^2 block is the 3x2 derivative
    of the hemisphere-aware inverse stereographic projection:

        N = 1 + q1^2 + q2^2
        d_q1 Xc =  2(1 + q2^2 - q1^2) / N^2
        d_q2 Xc = -4 q1 q2 / N^2
        d_q1 Yc = -4 q1 q2 / N^2
        d_q2 Yc =  2(1 + q1^2 - q2^2) / N^2
        north:  d_qi Zc = -4 qi / N^2
        south:  d_qi Zc = +4 qi / N^2

    Args:
        q_4d:       tf.Tensor (batch, 4) = [T, X, q1, q2].
        patch_idx:  tf.Tensor (batch,) int32, 0 = north chart, 1 = south chart.
    Returns:
        tf.Tensor (batch, 5, 4).
    """
    # Handle unbatched (rank-1) input that arrives from JVP/ForwardAccumulator
    # tracing.  Promote to a 1-element batch, compute J, then squeeze back.
    unbatched = q_4d.shape.ndims == 1
    if unbatched:
        q_4d = tf.expand_dims(q_4d, 0)  # (1, 4)
        patch_idx = tf.expand_dims(tf.cast(patch_idx, tf.int32), 0)  # (1,)

    q1 = q_4d[:, 2]  # (batch,)
    q2 = q_4d[:, 3]
    N = 1.0 + tf.square(q1) + tf.square(q2)  # (batch,)
    N2 = tf.square(N)

    # --- S^2 block partial derivatives ---
    dXc_dq1 = 2.0 * (1.0 + tf.square(q2) - tf.square(q1)) / N2
    dXc_dq2 = -4.0 * q1 * q2 / N2
    dYc_dq1 = -4.0 * q1 * q2 / N2
    dYc_dq2 = 2.0 * (1.0 + tf.square(q1) - tf.square(q2)) / N2
    dZc_dq1_north = -4.0 * q1 / N2
    dZc_dq2_north = -4.0 * q2 / N2
    dZc_dq1_south = 4.0 * q1 / N2
    dZc_dq2_south = 4.0 * q2 / N2

    is_north = tf.equal(patch_idx, 0)  # (batch,)
    dZc_dq1 = tf.where(is_north, dZc_dq1_north, dZc_dq1_south)
    dZc_dq2 = tf.where(is_north, dZc_dq2_north, dZc_dq2_south)

    # Build the full (batch, 5, 4) Jacobian row-by-row.
    # Columns: [dT, dX, dq1, dq2].  Rows: [T, X, Xc, Yc, Zc].
    # Use arithmetic on q1 for batch-sized zero/one constants: tf.zeros_like /
    # tf.ones_like dispatch through dtensor inside ForwardAccumulator and hit a
    # bug in TF 2.16 (IndexError in call_with_layout).  Multiplication by a
    # scalar constant is a standard differentiable op whose JVP is handled
    # correctly: JVP(q1 * 0) = jvp(q1) * 0 = 0, JVP(q1 * 0 + 1) = 0.
    z = q1 * tf.constant(0.0, dtype=q1.dtype)  # (batch,) — zero, JVP = 0
    o = z + tf.constant(1.0, dtype=q1.dtype)  # (batch,) — one,  JVP = 0

    # Each row_* is (batch, 4)
    row_T = tf.stack([o, z, z, z], axis=1)
    row_X = tf.stack([z, o, z, z], axis=1)
    row_Xc = tf.stack([z, z, dXc_dq1, dXc_dq2], axis=1)
    row_Yc = tf.stack([z, z, dYc_dq1, dYc_dq2], axis=1)
    row_Zc = tf.stack([z, z, dZc_dq1, dZc_dq2], axis=1)

    J = tf.stack([row_T, row_X, row_Xc, row_Yc, row_Zc], axis=1)  # (batch, 5, 4)

    if unbatched:
        return J[0]  # (5, 4)
    return J


def embed_S2_coords(q_4d, patch_idx):
    """
    Embed 4D intrinsic R^2 x S^2 coordinates to 5D ambient space.

    The R^2 part (Penrose coords T, X) passes through unchanged.  The S^2
    part (q1, q2) is mapped to 3D Cartesian via the hemisphere-aware inverse
    stereographic projection.

    Args:
        q_4d:       tf.Tensor (batch, 4) = [T, X, q1, q2].
        patch_idx:  tf.Tensor (batch,) int32, 0 = north chart, 1 = south chart.
    Returns:
        tf.Tensor (batch, 5) = [T, X, X_cart, Y_cart, Z_cart].
    """
    coords_R2 = q_4d[:, :2]
    coords_S2 = q_4d[:, 2:]
    coords_cart = stereo_to_cartesian(coords_S2, patch_idx)
    return tf.concat([coords_R2, coords_cart], axis=1)


def riemannian_inverse_metric_embed(x_vars, submodel):
    """Inverse of the *Riemannian* pulled-back metric (eta -> I).

    Identical to the metric pullback used by the curvature kernels, except the
    5D ambient metric is built as the positive-definite G = L L^T
    (``lorentzian=False``) instead of the Lorentzian G = L eta L^T.  Returns
    g_R^{-1} = inv(J^T (L L^T) J), which is symmetric-positive-definite (hence
    always invertible / well conditioned).

    Args:
        x_vars:   tf.Tensor (batch, 5) = [T, X, q1, q2, patch_idx_float].
        submodel: Keras model: (batch, 5) embedded coords -> (batch, 15) Cholesky vec.
    Returns:
        tf.Tensor (batch, 4, 4): the inverse Riemannian pullback metric.
    """
    q_4d = x_vars[:, :4]
    patch_idx = tf.cast(x_vars[:, 4], tf.int32)
    x_5d = embed_S2_coords(q_4d, patch_idx)
    G_5d_vec = submodel(x_5d)
    G_5d_riem = cholesky_from_vec(G_5d_vec, lorentzian=False)  # L L^T (SPD)
    J = embedding_jacobian_stereo(q_4d, patch_idx)  # (batch, 5, 4)
    g_riem = tf.einsum("sAB,sAm,sBn->smn", G_5d_riem, J, J)  # (batch, 4, 4)
    return tf.linalg.inv(g_riem)


def compute_ricci_tensor_embed(x_vars, submodel, lorentzian=False):
    """
    Ricci tensor for the S^2-embedding architecture.

    The network is parameterised by 5D Cartesian-embedded coordinates
    (T, X, X_cart, Y_cart, Z_cart), but the metric lives on the 4D manifold
    R^2 x S^2.  GradientTapes watch the 4D intrinsic coordinates q so that
    TF autograd automatically contributes all chain-rule correction terms
    (including the position-dependent Jacobian of the stereographic map).

    Args:
        x_vars:    tf.Tensor (batch, 5) = [T, X, q1, q2, patch_idx_float].
        submodel:  Keras model: (batch, 5) embedded coords -> (batch, n_out).
        lorentzian: bool.
    Returns:
        Ricci tensor, tf.Tensor (batch, 4, 4).
    """
    q_4d = x_vars[:, :4]
    patch_idx = tf.cast(x_vars[:, 4], tf.int32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(q_4d)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(q_4d)

            x_5d = embed_S2_coords(q_4d, patch_idx)
            # Network outputs 15-component 5D Cholesky vector
            G_5d_vec = submodel(x_5d)
            G_5d = cholesky_from_vec(G_5d_vec, lorentzian=lorentzian)  # (batch, 5, 5)
            # Analytic Jacobian of the embedding map (inside the tape so that
            # tape1 captures dg/dq contributions from dJ/dq as well as dG/dq).
            # Provided analytically rather than via tape.batch_jacobian to avoid
            # a third-order tape nesting in the Kretschmann computation.
            J = embedding_jacobian_stereo(q_4d, patch_idx)  # (batch, 5, 4)
            # Pull back 5D ambient metric to 4D: g_{mn} = G_{AB} J^A_m J^B_n
            pred = tf.einsum("sAB,sAm,sBn->smn", G_5d, J, J)  # (batch, 4, 4)

        # Metric derivative w.r.t. 4D intrinsic coordinates
        d_g = tape1.batch_jacobian(pred, q_4d)  # (batch, 4, 4, 4)
        del tape1

        # Christoffel symbols
        g_inv = tf.linalg.inv(pred)
        gamma = tf.einsum("skl,sjli->skij", g_inv, d_g)
        gamma += tf.einsum("skl,silj->skij", g_inv, d_g)
        gamma -= tf.einsum("skl,sijl->skij", g_inv, d_g)
        gamma *= 0.5

    # Christoffel derivative w.r.t. 4D intrinsic coordinates
    d_gamma = tape2.batch_jacobian(gamma, q_4d)  # (batch, 4, 4, 4, 4)
    del tape2

    # R_{jk} = ∂_i Γ^i_{jk} - ∂_j Γ^i_{ki} + Γ^i_{il}Γ^l_{jk} - Γ^i_{jl}Γ^l_{ik}
    Ricci_tensor = tf.einsum("sijki->sjk", d_gamma)
    Ricci_tensor -= tf.einsum("sikij->sjk", d_gamma)
    Ricci_tensor += tf.einsum("siil,sljk->sjk", gamma, gamma)
    Ricci_tensor -= tf.einsum("sijl,slik->sjk", gamma, gamma)

    return Ricci_tensor


def compute_kretschmann_scalar_embed(x_vars, submodel, lorentzian=False):
    """
    Kretschmann scalar K = R_{abcd}R^{abcd} for the S^2-embedding architecture.

    Derivatives are taken w.r.t. 4D intrinsic coords q with the embedding
    map inside the tapes, ensuring all chain-rule correction terms are included.

    Args:
        x_vars:    tf.Tensor (batch, 5) = [T, X, q1, q2, patch_idx_float].
        submodel:  Keras model: (batch, 5) embedded coords -> (batch, n_out).
        lorentzian: bool.
    Returns:
        Kretschmann scalar, tf.Tensor (batch,).
    """
    q_4d = x_vars[:, :4]
    patch_idx = tf.cast(x_vars[:, 4], tf.int32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(q_4d)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(q_4d)

            x_5d = embed_S2_coords(q_4d, patch_idx)
            # Network outputs 15-component 5D Cholesky vector
            G_5d_vec = submodel(x_5d)
            G_5d = cholesky_from_vec(G_5d_vec, lorentzian=lorentzian)  # (batch, 5, 5)
            # Analytic Jacobian (inside the tape; provided analytically to avoid
            # a third-order tape nesting).  Shape: (batch, 5, 4).
            J = embedding_jacobian_stereo(q_4d, patch_idx)  # (batch, 5, 4)
            # Pull back 5D ambient metric to 4D: g_{mn} = G_{AB} J^A_m J^B_n
            pred = tf.einsum("sAB,sAm,sBn->smn", G_5d, J, J)  # (batch, 4, 4)

        # Metric derivative w.r.t. 4D intrinsic coordinates
        d_g = tape1.batch_jacobian(pred, q_4d)  # (batch, 4, 4, 4)
        del tape1

        # Christoffel symbols
        g_inv = tf.linalg.inv(pred)
        gamma = tf.einsum("skl,sjli->skij", g_inv, d_g)
        gamma += tf.einsum("skl,silj->skij", g_inv, d_g)
        gamma -= tf.einsum("skl,sijl->skij", g_inv, d_g)
        gamma *= 0.5

    # Christoffel derivative w.r.t. 4D intrinsic coordinates
    d_gamma = tape2.batch_jacobian(gamma, q_4d)  # (batch, 4, 4, 4, 4)
    del tape2

    # R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj} + Γ^i_{km}Γ^m_{lj} - Γ^i_{lm}Γ^m_{kj}
    Riemann_tensor = tf.einsum("siljk->sijkl", d_gamma)
    Riemann_tensor -= tf.einsum("sikjl->sijkl", d_gamma)
    Riemann_tensor += tf.einsum("sikm,smlj->sijkl", gamma, gamma)
    Riemann_tensor -= tf.einsum("silm,smkj->sijkl", gamma, gamma)

    # K = R_{abcd}R^{abcd} = R^i_{jkl}R^m_{nop} g_{im} g^{jn} g^{ko} g^{lp}
    Kretschmann_scalar = tf.einsum(
        "sijkl,smnop,sim,sjn,sko,slp->s",
        Riemann_tensor,
        Riemann_tensor,
        pred,  # first Riemann index is lowered
        g_inv,
        g_inv,
        g_inv,
    )

    return Kretschmann_scalar


def compute_ricci_and_kretschmann_embed(
    x_vars, submodel, lorentzian=False, need_ricci=True, need_kretschmann=True,
    need_speciality_index=False,
):
    """
    Compute the Ricci tensor and/or the Kretschmann scalar in a single
    double-tape pass.  Use the ``need_ricci`` / ``need_kretschmann`` flags to
    skip post-tape assembly for terms that are not required (e.g. when the
    corresponding loss multiplier is zero), avoiding wasted einsum work.

    Args:
        x_vars:           tf.Tensor (batch, 5) = [T, X, q1, q2, patch_idx_float].
        submodel:         Keras model: (batch, 5) embedded coords -> (batch, n_out).
        lorentzian:       bool.
        need_ricci:       if False, skip Ricci contraction; returns None for ricci_tensor.
        need_kretschmann: if False, skip Riemann assembly and Kretschmann contraction;
                          returns None for kretschmann_scalar.
        need_speciality_index: if True, also compute the complex Weyl
                               invariants I and J used to form the speciality
                               index; implies Riemann assembly.
    Returns:
        pred:               tf.Tensor (batch, 4, 4) — pulled-back intrinsic metric.
        ricci_tensor:       tf.Tensor (batch, 4, 4), or None if need_ricci=False.
        kretschmann_scalar: tf.Tensor (batch,), or None if need_kretschmann=False.
        weyl_i:             tf.Tensor (batch,), or None if need_speciality_index=False.
        weyl_j:             tf.Tensor (batch,), or None if need_speciality_index=False.
    """
    q_4d = x_vars[:, :4]
    patch_idx = tf.cast(x_vars[:, 4], tf.int32)

    with tf.GradientTape() as tape2:
        tape2.watch(q_4d)

        with tf.GradientTape() as tape1:
            tape1.watch(q_4d)

            x_5d = embed_S2_coords(q_4d, patch_idx)
            G_5d_vec = submodel(x_5d)
            G_5d = cholesky_from_vec(G_5d_vec, lorentzian=lorentzian)  # (batch, 5, 5)
            J = embedding_jacobian_stereo(q_4d, patch_idx)  # (batch, 5, 4)
            pred = tf.einsum("sAB,sAm,sBn->smn", G_5d, J, J)  # (batch, 4, 4)

        d_g = tape1.batch_jacobian(pred, q_4d)  # (batch, 4, 4, 4)
        del tape1

        g_inv = tf.linalg.inv(pred)
        gamma = tf.einsum("skl,sjli->skij", g_inv, d_g)
        gamma += tf.einsum("skl,silj->skij", g_inv, d_g)
        gamma -= tf.einsum("skl,sijl->skij", g_inv, d_g)
        gamma *= 0.5

    d_gamma = tape2.batch_jacobian(gamma, q_4d)  # (batch, 4, 4, 4, 4)
    del tape2

    # Ricci tensor: R_{jk} = ∂_i Γ^i_{jk} - ∂_j Γ^i_{ik} + Γ^i_{il}Γ^l_{jk} - Γ^i_{jl}Γ^l_{ik}
    if need_ricci:
        ricci_tensor = tf.einsum("sijki->sjk", d_gamma)
        ricci_tensor -= tf.einsum("sikij->sjk", d_gamma)
        ricci_tensor += tf.einsum("siil,sljk->sjk", gamma, gamma)
        ricci_tensor -= tf.einsum("sijl,slik->sjk", gamma, gamma)
    else:
        ricci_tensor = None

    # Riemann tensor: R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj} + Γ^i_{km}Γ^m_{lj} - Γ^i_{lm}Γ^m_{kj}
    if need_kretschmann or need_speciality_index:
        riemann = tf.einsum("siljk->sijkl", d_gamma)
        riemann -= tf.einsum("sikjl->sijkl", d_gamma)
        riemann += tf.einsum("sikm,smlj->sijkl", gamma, gamma)
        riemann -= tf.einsum("silm,smkj->sijkl", gamma, gamma)

        if need_kretschmann:
            # K = R_{abcd}R^{abcd}
            kretschmann_scalar = tf.einsum(
                "sijkl,smnop,sim,sjn,sko,slp->s",
                riemann,
                riemann,
                pred,  # lowers first Riemann index
                g_inv,
                g_inv,
                g_inv,
            )
        else:
            kretschmann_scalar = None

        if need_speciality_index:
            from geometry.ricci_opt import _weyl_ij_from_riemann

            if ricci_tensor is None:
                ricci_for_weyl = tf.einsum("sijki->sjk", d_gamma)
                ricci_for_weyl -= tf.einsum("sikij->sjk", d_gamma)
                ricci_for_weyl += tf.einsum("siil,sljk->sjk", gamma, gamma)
                ricci_for_weyl -= tf.einsum("sijl,slik->sjk", gamma, gamma)
            else:
                ricci_for_weyl = ricci_tensor
            weyl_i, weyl_j = _weyl_ij_from_riemann(
                riemann, pred, g_inv, ricci_for_weyl
            )
        else:
            weyl_i = None
            weyl_j = None
    else:
        kretschmann_scalar = None
        weyl_i = None
        weyl_j = None

    return pred, ricci_tensor, kretschmann_scalar, weyl_i, weyl_j
