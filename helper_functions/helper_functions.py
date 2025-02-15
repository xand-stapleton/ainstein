import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')

###############################################################################
# Functions to perform Cholesky decomposition
def cholesky_from_vec(lower_triangular_vector):
    lower_triangular_matrix = tfp.math.fill_triangular(lower_triangular_vector)
    full_matrix = tf.matmul(
        lower_triangular_matrix, lower_triangular_matrix, transpose_b=True
    )

    return full_matrix


def cholesky_to_vec(full_matrix):
    lower_triangular_matrices = tf.linalg.cholesky(full_matrix)
    lower_triangular_vector = tfp.math.fill_triangular_inverse(
        lower_triangular_matrices
    )

    return lower_triangular_vector


# Function to compute weights to scale contributions of points to the losses based on their radii 
def RadiusWeighting(pts, filter_width=0.5, filter_midpt=0.):
    radius = tf.sqrt(tf.reduce_sum(tf.square(pts), axis=1))
    radius_filter = tf.exp(-tf.pow((radius-filter_midpt)/filter_width, 20))
    
    return radius_filter


# Function to generate a random filepath 
def create_time_date_dir(base_path=None, run_name=""):
    # Get the current date and time
    current_time = datetime.now()
    # Format the current date and time
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # If the run name is blank, add a separator character
    if run_name != "":
        run_name += "_"

    save_path_tail = run_name + timestamp

    # Construct the directory name
    if base_path is not None:
        directory_name = os.path.join(base_path, save_path_tail)
    else:
        directory_name = save_path_tail

    # Create the directory
    os.makedirs(directory_name, exist_ok=True)

    return directory_name


# Generic 3d plotting function
def plot_fig(samples, z_values, title):
    # Make a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], z_values, c=z_values, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # plt.colorbar(scatter)
    plt.show()
