import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from helper_functions.helper_functions import cholesky_from_vec

###############################################################################
# Functions to change between patches
# Ball coordinates
def PatchChange_Coordinates_Ball(coords):
    # Compute the coordinate norm
    norm = tf.norm(coords, axis=1)
    
    # Compute the patch transformation
    coords_otherpatch = coords * tf.expand_dims((norm - 1) / (norm * (norm + 1)), axis=-1)
    
    return coords_otherpatch

def PatchChange_Metric_Ball(coords, metric_pred):  
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_Ball(coords)

    # Compute the coordinate norm
    norm = tf.norm(coords_otherpatch, axis=1)
    
    # Compute the Jacobian
    jacobian_term1 = tf.eye(coords_otherpatch.shape[1], batch_shape=[coords_otherpatch.shape[0]], dtype=coords_otherpatch.dtype)
    jacobian_term1 *= tf.expand_dims(tf.expand_dims((norm - 1) / (norm * (norm + 1)), axis=-1), axis=-1)    
    jacobian_term2 = tf.einsum("si,sj->sij", coords_otherpatch, coords_otherpatch)
    jacobian_term2 *= tf.expand_dims(tf.expand_dims((1 + 2 * norm - norm**2) / (norm**3 * (1 + norm)**2), axis=-1), axis=-1)
    jacobian = jacobian_term1 + jacobian_term2  
    
    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sij,sjk,skl->sil", jacobian, metric_pred, jacobian)
    
    return metric_otherpatch

# Stereographic coordinates
def PatchChange_Coordinates_Stereo(coords):
    # Compute the coordinate norm
    norm = tf.norm(coords, axis=1)
    
    # Compute the patch transformation
    coords_otherpatch = coords / tf.expand_dims(norm**2, axis=-1) 
    
    return coords_otherpatch

def PatchChange_Metric_Stereo(coords, metric_pred): 
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_Stereo(coords)
    
    # Compute the coordinate norm
    norm = tf.norm(coords_otherpatch, axis=1)

    # Compute the Jacobian
    jacobian_term1 = tf.eye(coords_otherpatch.shape[1], batch_shape=[coords_otherpatch.shape[0]], dtype=coords_otherpatch.dtype)
    jacobian_term1 /= tf.expand_dims(tf.expand_dims(norm**2, axis=-1), axis=-1)
    jacobian_term2 = tf.einsum("si,sj->sij", coords_otherpatch, coords_otherpatch)
    jacobian_term2 *= tf.expand_dims(tf.expand_dims(-2/norm**4, axis=-1), axis=-1)
    jacobian = jacobian_term1 + jacobian_term2  
    
    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sij,sjk,skl->sil", jacobian, metric_pred, jacobian)
    
    return metric_otherpatch

###############################################################################
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


###############################################################################
# Define function to compute the analytic round metric at input ball points
def AnalyticMetric_Ball(coords, identity=False):
    # Return the identity function if requested
    if identity:
        return tf.eye(coords.shape[1], batch_shape=[coords.shape[0]], dtype=coords.dtype)
        
    # Otherwise compute the round metric
    norm = tf.norm(coords, axis=1)
    
    metric_term1 = tf.eye(coords.shape[1], batch_shape=[coords.shape[0]], dtype=coords.dtype)
    metric_term1 *= tf.expand_dims(tf.expand_dims(16 * (1 - norm**2)**2, axis=-1), axis=-1)
    metric_term2 = 64*tf.einsum("si,sj->sij", coords, coords)
    metric = metric_term1 + metric_term2
    metric /= tf.expand_dims(tf.expand_dims((1 + norm**2)**4, axis=-1), axis=-1)
    metric *= (coords.shape[1] - 1.0)

    return metric
