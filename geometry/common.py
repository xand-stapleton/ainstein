import tensorflow as tf

tf.keras.backend.set_floatx("float64")
from helper_functions.helper_functions import cholesky_from_vec


# Neural Network differential geometric functions
@tf.function
def compute_ricci_tensor(x_vars, model):
    # Set up the gradients for the Ricci tensor double derivates of the metric
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x_vars)  # ...gradient is metric shape x number of inputs

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_vars)

            # Compute the metric at the datapoints (in both patches)
            pred = model(x_vars)
            pred = cholesky_from_vec(pred)  # ...pred dim is batch x dim_x x dim_x

        # Compute the metric derivative
        di_dg = tape1.batch_jacobian(pred, x_vars)

        # Compute Christoffel symbols: \Gamma^c_{ab} is christoffel[a, b, c]; s is the batch dimension
        g_cd_up = tf.linalg.inv(pred)
        gamma_c_up_ab_down = tf.einsum("scd,sdab->scab", g_cd_up, di_dg)
        gamma_c_up_ab_down += tf.einsum("scd,sdba->scab", g_cd_up, di_dg)
        gamma_c_up_ab_down -= tf.einsum("scd,sabd->scab", g_cd_up, di_dg)
        gamma_c_up_ab_down *= 0.5

    # Christoffel derivative terms
    d_gamma = tape2.batch_jacobian(gamma_c_up_ab_down, x_vars)

    d_gamma_a_up_ij_down = tf.einsum("saija->sij", d_gamma)
    d_gamma_a_up_ai_down = tf.einsum("sajai->sij", d_gamma)
    R_ij = d_gamma_a_up_ij_down - d_gamma_a_up_ai_down

    # Christoffel product terms
    R_ij += tf.einsum("saab,sbij->sij", gamma_c_up_ab_down, gamma_c_up_ab_down)
    R_ij -= tf.einsum("saib,sbaj->sij", gamma_c_up_ab_down, gamma_c_up_ab_down)

    return R_ij


# Bonus function --> currently unused as above computes Christoffel symbols implicitly
@tf.function
def _compute_christoffel_symbols(model, x_vars):
    # Set up the gradient for the Christoffel symbols derivate of the metric
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x_vars)  # ...gradient is metric shape x number of inputs

        # Compute the metric at the datapoints
        pred = cholesky_from_vec(model(x_vars))  # ...pred dim is batch x dim_x x dim_x

    # Compute the metric derivative
    di_dg = tape1.batch_jacobian(pred, x_vars)

    # Compute Christoffel symbols: \Gamma^c_{ab} is christoffel[a, b, c]; s is the batch dimension
    g_cd_up = tf.linalg.inv(pred)
    gamma_c_up_ab_down = tf.einsum("scd,sdab->scab", g_cd_up, di_dg)
    gamma_c_up_ab_down += tf.einsum("scd,sdba->scab", g_cd_up, di_dg)
    gamma_c_up_ab_down -= tf.einsum("scd,sabd->scab", g_cd_up, di_dg)
    gamma_c_up_ab_down *= 0.5

    return gamma_c_up_ab_down
