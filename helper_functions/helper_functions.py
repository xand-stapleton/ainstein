from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tf.keras.backend.set_floatx("float64")

import warnings

warnings.filterwarnings("once", category=UserWarning)


###############################################################################
# Functions to perform Cholesky decomposition
def cholesky_from_vec(lower_triangular_vector, lorentzian=False):
    # Normalise to rank-2 [batch, n_tril] so fill_triangular sees a stable
    # shape contract inside tf.function / ForwardAccumulator code paths.
    lower_triangular_vector = tf.convert_to_tensor(lower_triangular_vector)
    lower_triangular_vector = tf.reshape(
        lower_triangular_vector,
        [-1, tf.shape(lower_triangular_vector)[-1]],
    )

    lower_triangular_matrix = tfp.math.fill_triangular(lower_triangular_vector)
    if lorentzian:
        eta = tf.constant(
            [-1.0] + [1.0] * (lower_triangular_matrix.shape[-1] - 1),
            dtype=lower_triangular_matrix.dtype,
        )
        # Broadcast eta along the column axis so that L * eta = L · diag(eta),
        # giving the symmetric decomposition G = L · diag(eta) · L^T.
        eta = eta[tf.newaxis, tf.newaxis, :]
        lower_triangular_matrix_scaled = (
            lower_triangular_matrix * eta
        )  # ...L · diag(eta)
        full_matrix = tf.matmul(
            lower_triangular_matrix_scaled, lower_triangular_matrix, transpose_b=True
        )
    else:
        full_matrix = tf.matmul(
            lower_triangular_matrix, lower_triangular_matrix, transpose_b=True
        )

    return full_matrix


def cholesky_to_vec(full_matrix, lorentzian=False):
    if lorentzian:
        warnings.warn(
            "Assuming lower-triangular / Minkowski input (non-unique decomposition for indefinite matrices).",
            UserWarning,
        )
        lower_triangular_matrices = full_matrix
        """ 
        ###old code (didn't work)
        eta = tf.constant([-1.0] + [1.0] * (full_matrix.shape[-1] - 1), dtype=full_matrix.dtype)
        eta = eta[tf.newaxis, :, tf.newaxis] #...reshape with broadcasting
        full_matrix_positive_definite = eta * full_matrix #...compute positive-definite equivalent
        lower_triangular_matrices_positive_definite = tf.linalg.cholesky(full_matrix_positive_definite) #...cholesky decompose
        lower_triangular_matrices = lower_triangular_matrices_positive_definite / eta #...remove eta scaling (note eta = eta^T so no transposing needed)
        """
    else:
        lower_triangular_matrices = tf.linalg.cholesky(full_matrix)
    lower_triangular_vector = tfp.math.fill_triangular_inverse(
        lower_triangular_matrices
    )

    return lower_triangular_vector


###############################################################################
# Function to compute weights to scale contributions of points to the losses based on their radii
def RadiusWeighting(pts, filter_width=0.5, filter_midpt=0.0):
    radius = tf.sqrt(tf.reduce_sum(tf.square(pts), axis=1))
    radius_filter = tf.exp(-tf.pow((radius - filter_midpt) / filter_width, 20))

    return radius_filter


###############################################################################
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


###############################################################################
# Filepath functions
# Function to generate a random filepath
def create_time_date_dir(base_path: Path | None = None, run_name: str | None = ""):
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


# Function to list filepaths for trained models
def list_saved_models(import_from_seed_models: bool = False):
    """
    List either seed model files (if import_from_seed_models=True)
    or subfolders inside 'runs' (if import_from_seed_models=False).
    If listing seed models, strip the '.keras' extension when printing.
    """
    base_dir = os.getcwd()
    root_runs_path = os.path.join(
        base_dir, "..", "seed_models" if import_from_seed_models else "runs"
    )

    if not os.path.exists(root_runs_path):
        raise FileNotFoundError(f"Directory not found: {root_runs_path}")

    if import_from_seed_models:
        # List all files ending with '.keras'
        saved_runs = [
            f
            for f in os.listdir(root_runs_path)
            if os.path.isfile(os.path.join(root_runs_path, f)) and f.endswith(".keras")
        ]
        # Strip '.keras' for printing
        saved_runs = [os.path.splitext(f)[0] for f in saved_runs]
        label = "seed_models"
    else:
        # List all subfolders
        saved_runs = [
            run
            for run in os.listdir(root_runs_path)
            if os.path.isdir(os.path.join(root_runs_path, run))
        ]
        label = "runs"

    print(f"Available models in {label}:")
    for idx, name in enumerate(saved_runs):
        if not import_from_seed_models:
            name = "-".join(name.split("-")[:2])
        print(f"{idx}: {name}")

    return saved_runs
