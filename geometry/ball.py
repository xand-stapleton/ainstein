import tensorflow as tf

tf.keras.backend.set_floatx("float64")


###############################################################################
# Functions to change between patches
# Ball coordinates
def PatchChange_Coordinates_Ball(coords):
    # Compute the coordinate norm
    norm = tf.norm(coords, axis=1)

    # Compute the patch transformation
    coords_otherpatch = coords * tf.expand_dims(
        (norm - 1) / (norm * (norm + 1)), axis=-1
    )

    return coords_otherpatch


def PatchChange_Metric_Ball(coords, metric_pred):
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_Ball(coords)

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
        tf.expand_dims((1 + 2 * norm - tf.square(norm)) / (tf.pow(norm, 3) * tf.square(1 + norm)), axis=-1),
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
    jacobian_term2 *= tf.expand_dims(tf.expand_dims(-2 / tf.pow(norm, 4), axis=-1), axis=-1)
    jacobian = jacobian_term1 + jacobian_term2

    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sij,sjk,skl->sil", jacobian, metric_pred, jacobian)

    return metric_otherpatch


# Define function to compute the analytic round metric at input ball points
def AnalyticMetric_Ball(coords, identity=False):
    # Return the identity function if requested
    if identity:
        return tf.eye(
            coords.shape[1], batch_shape=[coords.shape[0]], dtype=coords.dtype
        )

    # Otherwise compute the round metric
    norm = tf.norm(coords, axis=1)

    metric_term1 = tf.eye(
        coords.shape[1], batch_shape=[coords.shape[0]], dtype=coords.dtype
    )
    metric_term1 *= tf.expand_dims(
        tf.expand_dims(16 * tf.square(1 - tf.square(norm)), axis=-1), axis=-1
    )
    metric_term2 = 64 * tf.einsum("si,sj->sij", coords, coords)
    metric = metric_term1 + metric_term2
    metric /= tf.expand_dims(tf.expand_dims(tf.pow(1 + tf.square(norm), 4), axis=-1), axis=-1)
    metric *= coords.shape[1] - 1.0

    return metric
