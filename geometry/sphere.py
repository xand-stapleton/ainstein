from __future__ import annotations

import tensorflow as tf

tf.keras.backend.set_floatx("float64")


###############################################################################
# Functions to change between patches
# Sphere coordinates
def PatchChange_Coordinates_Sphere(coords):
    # Compute the coordinate norm
    norm = tf.norm(coords, axis=1)

    # Compute the patch transformation
    coords_otherpatch = coords * tf.expand_dims(
        (norm - 1) / (norm * (norm + 1)), axis=-1
    )

    return coords_otherpatch


def PatchChange_Metric_Sphere(coords, metric_pred):
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_Sphere(coords)

    # Compute the coordinate norm
    norm = tf.norm(coords_otherpatch, axis=1)

    # Compute the Jacobian
    jacobian_term1 = tf.eye(
        coords_otherpatch.shape[1],
        batch_shape=[coords_otherpatch.shape[0]],
        dtype=coords_otherpatch.dtype,
    )
    jacobian_term1 *= tf.expand_dims(
        tf.expand_dims((norm - 1) / (norm * (norm + 1)), axis=-1), axis=-1
    )
    jacobian_term2 = tf.einsum("si,sj->sij", coords_otherpatch, coords_otherpatch)
    jacobian_term2 *= tf.expand_dims(
        tf.expand_dims(
            (1 + 2 * norm - tf.square(norm)) / (tf.pow(norm, 3) * tf.square(1 + norm)),
            axis=-1,
        ),
        axis=-1,
    )
    jacobian = jacobian_term1 + jacobian_term2

    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sij,sjk,skl->sil", jacobian, metric_pred, jacobian)

    return metric_otherpatch


# Stereographic coordinates
def PatchChange_Coordinates_Stereo(coords):
    # Compute the coordinate norm
    norm = tf.norm(coords, axis=1)

    # Compute the patch transformation
    coords_otherpatch = coords / tf.expand_dims(tf.square(norm), axis=-1)

    return coords_otherpatch


def PatchChange_Metric_Stereo(coords, metric_pred):
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_Stereo(coords)

    # Compute the coordinate norm
    norm = tf.norm(coords_otherpatch, axis=1)

    # Compute the Jacobian
    jacobian_term1 = tf.eye(
        coords_otherpatch.shape[1],
        batch_shape=[coords_otherpatch.shape[0]],
        dtype=coords_otherpatch.dtype,
    )
    jacobian_term1 /= tf.expand_dims(tf.expand_dims(tf.square(norm), axis=-1), axis=-1)
    jacobian_term2 = tf.einsum("si,sj->sij", coords_otherpatch, coords_otherpatch)
    jacobian_term2 *= tf.expand_dims(
        tf.expand_dims(-2 / tf.pow(norm, 4), axis=-1), axis=-1
    )
    jacobian = jacobian_term1 + jacobian_term2

    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sij,sjk,skl->sil", jacobian, metric_pred, jacobian)

    return metric_otherpatch


# Define function to compute the analytic round metric at input sphere points
def AnalyticMetric_Sphere(coords, identity=False, hyperbolic=False, lorentzian=False):
    assert not (hyperbolic and identity), (
        "If using the hyperbolic argument, identity must not be True"
    )
    assert identity or not lorentzian, "No analytic Lorentzian metric configured."

    dimension = coords.shape[1]
    # Return the identity function if requested
    if identity:
        if lorentzian:
            return tf.linalg.set_diag(
                tf.eye(dimension, batch_shape=[coords.shape[0]], dtype=coords.dtype),
                tf.concat(
                    [
                        tf.fill([coords.shape[0], 1], tf.cast(-1.0, coords.dtype)),
                        tf.ones([coords.shape[0], dimension - 1], dtype=coords.dtype),
                    ],
                    axis=1,
                ),
            )
        else:
            return tf.eye(dimension, batch_shape=[coords.shape[0]], dtype=coords.dtype)

    elif hyperbolic:
        norm = tf.norm(coords, axis=1)
        prefactor = tf.expand_dims(
            tf.expand_dims(
                4 * (dimension - 1) / tf.square(1 - tf.square(norm)), axis=-1
            ),
            axis=-1,
        )

        return prefactor * tf.eye(
            dimension, batch_shape=[coords.shape[0]], dtype=coords.dtype
        )

    # Otherwise compute the round metric
    norm = tf.norm(coords, axis=1)

    metric_term1 = tf.eye(dimension, batch_shape=[coords.shape[0]], dtype=coords.dtype)
    metric_term1 *= tf.expand_dims(
        tf.expand_dims(16 * tf.square(1 - tf.square(norm)), axis=-1), axis=-1
    )
    metric_term2 = 64 * tf.einsum("si,sj->sij", coords, coords)
    metric = metric_term1 + metric_term2
    metric /= tf.expand_dims(
        tf.expand_dims(tf.pow(1 + tf.square(norm), 4), axis=-1), axis=-1
    )
    metric *= dimension - 1.0

    return metric
