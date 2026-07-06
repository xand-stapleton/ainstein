from __future__ import annotations

import tensorflow as tf

tf.keras.backend.set_floatx("float64")
from helper_functions.helper_functions import cholesky_from_vec


# Neural Network differential geometric functions
@tf.function
def compute_ricci_tensor(x_vars, model):
    # Set up the gradients for the Ricci tensor double derivates of the metric
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x_vars)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_vars)

            # Compute the metric at the datapoints: g_{ij}
            pred = model(x_vars)

            # Use the getattr here in case we're using a config which doesn't
            # have that attribute (e.g. lens). There it would default to False
            pred = cholesky_from_vec(
                pred,
                lorentzian=getattr(model.config.model_specific, "lorentzian", False),
            )

        # Compute the metric derivative
        d_g = tape1.batch_jacobian(pred, x_vars)  # ...source dim added to the end
        del tape1  # release persistent tape; no longer needed

        # Compute Christoffel symbols: (s is the batch dimension)
        # \Gamma^k_{ij} = 0.5 * g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})
        g_inv = tf.linalg.inv(pred)  # ...this raises both indices
        gamma = tf.einsum("skl,sjli->skij", g_inv, d_g)
        gamma += tf.einsum("skl,silj->skij", g_inv, d_g)
        gamma -= tf.einsum("skl,sijl->skij", g_inv, d_g)
        gamma *= 0.5

    # Christoffel derivative
    d_gamma = tape2.batch_jacobian(gamma, x_vars)  # ...source dim added to the end
    del tape2  # release persistent tape; no longer needed

    # Compute Ricci tensor: (s is the batch dimension)
    # R_{jk} = \partial_i \Gamma^i_{jk} - \partial_j \Gamma^i_{ki} + \Gamma^i_{il}\Gamma^l_{jk} - \Gamma^i_{jl}\Gamma^l_{ik}
    # Christoffel derivative terms
    Ricci_tensor = tf.einsum("sijki->sjk", d_gamma)
    Ricci_tensor -= tf.einsum("sikij->sjk", d_gamma)

    # Christoffel product terms
    Ricci_tensor += tf.einsum("siil,sljk->sjk", gamma, gamma)
    Ricci_tensor -= tf.einsum("sijl,slik->sjk", gamma, gamma)

    return Ricci_tensor


@tf.function
def compute_kretschmann_scalar(x_vars, model):
    # Set up the gradients for the Ricci tensor double derivates of the metric
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x_vars)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_vars)

            # Compute the metric at the datapoints: g_{ij}
            pred = model(x_vars)
            pred = cholesky_from_vec(
                pred,
                lorentzian=getattr(model.config.model_specific, "lorentzian", False),
            )  # ...pred shape: (batch, dim_x, dim_x)

        # Compute the metric derivative
        d_g = tape1.batch_jacobian(pred, x_vars)  # ...source dim added to the end
        del tape1  # release persistent tape; no longer needed

        # Compute Christoffel symbols: (s is the batch dimension)
        # \Gamma^k_{ij} = 0.5 * g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})
        g_inv = tf.linalg.inv(pred)  # ...this raises both indices
        gamma = tf.einsum("skl,sjli->skij", g_inv, d_g)
        gamma += tf.einsum("skl,silj->skij", g_inv, d_g)
        gamma -= tf.einsum("skl,sijl->skij", g_inv, d_g)
        gamma *= 0.5

    # Christoffel derivative
    d_gamma = tape2.batch_jacobian(gamma, x_vars)  # ...source dim added to the end
    del tape2  # release persistent tape; no longer needed

    # Compute Ricci tensor: ; s is the batch dimension
    # R^i_{jkl} = \partial_k \Gamma^i_{lj} - \partial_l \Gamma^i_{kj} + \Gamma^i_{km}\Gamma^m_{lj} - \Gamma^i_{lm}\Gamma^m_{kj}
    # Christoffel derivative terms
    Riemann_tensor = tf.einsum("siljk->sijkl", d_gamma)
    Riemann_tensor -= tf.einsum("sikjl->sijkl", d_gamma)

    # Christoffel product terms
    Riemann_tensor += tf.einsum("sikm,smlj->sijkl", gamma, gamma)
    Riemann_tensor -= tf.einsum("silm,smkj->sijkl", gamma, gamma)

    # Compute Kretschmann scalar: (s is the batch dimension)
    # K = R_{abcd} R^{abcd} = R^i_{jkl} R^m_{nop} g_{im} g^{jn} g^{ko} g^{lp}
    Kretschmann_scalar = tf.einsum(
        "sijkl,smnop,sim,sjn,sko,slp->s",
        Riemann_tensor,
        Riemann_tensor,
        pred,  # ...note the first riemann index is raised
        g_inv,
        g_inv,
        g_inv,
    )

    return Kretschmann_scalar


# Bonus function --> currently unused as above compute Christoffel symbols implicitly
@tf.function
def _compute_christoffel_symbols(model, x_vars):
    # Set up the gradient for the Christoffel symbols derivate of the metric
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x_vars)

        # Compute the metric at the datapoints: g_{ij}
        pred = model(x_vars)
        pred = cholesky_from_vec(
            pred,
            lorentzian=getattr(model.config.model_specific, "lorentzian", False),
        )  # ...pred shape: (batch, dim_x, dim_x)

    # Compute the metric derivative
    d_g = tape1.batch_jacobian(pred, x_vars)  # ...source dim added to the end

    # Compute Christoffel symbols: (s is the batch dimension)
    # \Gamma^k_{ij} = 0.5 * g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})
    g_inv = tf.linalg.inv(pred)
    gamma = tf.einsum("skl,sjli->skij", g_inv, d_g)
    gamma += tf.einsum("skl,silj->skij", g_inv, d_g)
    gamma -= tf.einsum("skl,sijl->skij", g_inv, d_g)
    gamma *= 0.5

    return gamma
