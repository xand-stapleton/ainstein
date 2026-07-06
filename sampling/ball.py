from __future__ import annotations

"""Sampling Schemes for the Hyperball & Hypercube"""
# Import libraries
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


# Define sampling functions
def BallSample(
    num_pts,
    dimension=2,
    patch_width=1.0,
    density_power=1.0,
):
    """
    Code to sample the n-ball representation of S^n (for n geq 2),
    up to some boundary cutoff.

    Parameters
    ----------
    num_pts : int
        The number of sample points to generate.
    dimension : int, optional
        The dimension of the ball to be sampled. The default is 2.
    patch_width : float, optional
        The maximum sample value in the uniform sampling. The default 1.0.
    density_power : float, optional
        The power factor to skew the beta function sampling by
        (< 1 skews towards radial extremeties). The default is 1.

    Returns
    -------
    array
        The sample points for the ball (in Cartesian coordinates),
        shape (num_pts, dimension).

    """
    # Sample the radii (using a beta distribution)
    # Centred the beta distribution on radial midpoint, and symmetrise between the patches
    radii_p1 = np.random.beta(
        density_power,
        density_power / (np.sqrt(2.0) - 1.0) - density_power,
        size=int(num_pts / 2),
    )
    radii_p2 = np.random.beta(
        density_power,
        density_power / (np.sqrt(2.0) - 1.0) - density_power,
        size=int(num_pts / 2),
    )
    radii_p2_inp1 = (1 - radii_p2) / (1 + radii_p2)
    radii = np.concatenate((radii_p1, radii_p2_inp1))

    # Scale the radii to the maximum size
    radii *= patch_width

    # Sample the final angle
    angles = np.random.uniform(high=2 * np.pi, size=num_pts)
    # Sample the remaining spherical polar angles
    angles = np.hstack(
        (
            np.random.uniform(high=np.pi, size=(num_pts, dimension - 2)),
            angles.reshape(-1, 1),
        )
    )

    # Define the vector of (cos(\phi_1), cos(\phi_2), ..., cos(\phi_{n-1}), 1.)
    cc = np.hstack((np.cos(angles), np.ones(num_pts).reshape(-1, 1)))
    # Define the vector of (1., sin(\phi_1), sin(\phi_2), ..., sin(\phi_{n-1}))
    ss = np.hstack((np.ones(num_pts).reshape(-1, 1), np.sin(angles)))
    # Take the cumulative product to produce the vector (1., sin(\phi_1), sin(\phi_1)*sin(\phi_2), ..., sin(\phi_1)*...*sin(\phi_{n-1}))
    ss = np.cumprod(ss, axis=1)

    return radii.reshape(-1, 1) * cc * ss


def StereoSampleHemisphere(
    num_pts,
    patch_width=1.0,
    density_power=1.0,
):
    """
    Sample S^2 uniformly (or with a tunable pole/equator bias) using two
    stereographic hemisphere charts.

    Points are split exactly evenly between the northern hemisphere
    (pole at origin, patch_idx=0) and the southern hemisphere
    (pole at origin, patch_idx=1).

    The radial distribution uses the exact inverse-CDF of the spherical area
    element in stereographic coordinates,

        dA = 4r dr dφ / (1 + r²)²,

    so that ``density_power=1`` produces a **uniform distribution on S²**.
    Deviations from 1 skew the radial density via u → u^(1/density_power)
    applied before the inverse CDF:

        density_power < 1  →  pole-biased   (more points near r = 0)
        density_power = 1  →  uniform on S² (exact)
        density_power > 1  →  equator-biased (more points near r = patch_width)

    Parameters
    ----------
    num_pts : int
        Total number of sample points (rounded to the nearest even number).
    patch_width : float, optional
        Maximum radial extent in each chart.  1.0 reaches the equator.
    density_power : float, optional
        Exponent controlling the pole/equator bias.  1.0 = uniform on S².

    Returns
    -------
    coords : np.ndarray, shape (num_pts, 2)
        Stereographic coordinates expressed in each point's native chart.
    patch_idx : np.ndarray, shape (num_pts,), dtype int32
        0 = north chart, 1 = south chart.
    """
    n_half = num_pts // 2
    n_pts = 2 * n_half  # ensure exact even split

    # Uniform samples on [0, 1] for the inverse-CDF method
    u = np.random.uniform(0.0, 1.0, size=n_pts)

    # Apply density_power skew before the inverse CDF:
    #   density_power=1 → identity, exact uniform on S²
    #   density_power<1 → u^(1/α) with 1/α>1 maps toward 0 → small r → poles
    #   density_power>1 → u^(1/α) with 1/α<1 maps toward 1 → large r → equator
    v = u ** (1.0 / density_power)

    # Exact inverse CDF of the area-element-uniform radial distribution.
    # Derived from F(r) = r²(1+R²) / [R²(1+r²)] where R = patch_width:
    #   r = R * sqrt(v / (1 + R²(1 - v)))
    R2 = patch_width**2
    radii = patch_width * np.sqrt(v / (1.0 + R2 * (1.0 - v)))

    # Uniform azimuthal angles
    angles = np.random.uniform(0.0, 2.0 * np.pi, size=n_pts)

    coords = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    # First half = north chart (0), second half = south chart (1)
    patch_idx = np.concatenate(
        [np.zeros(n_half, dtype=np.int32), np.ones(n_half, dtype=np.int32)]
    )

    return coords, patch_idx


def StereoSampleSingleHemisphere(
    num_pts,
    patch_idx=0,
    patch_width=1.0,
    density_power=1.0,
):
    """Sample points from a single stereographic S^2 chart.

    This helper reuses ``StereoSampleHemisphere`` (which enforces an exact
    north/south split) and retains only the requested hemisphere, guaranteeing
    the returned batch has size ``num_pts``.

    Parameters
    ----------
    num_pts : int
        Number of points to return in the selected chart.
    patch_idx : int, optional
        0 for north chart, 1 for south chart.
    patch_width : float, optional
        Maximum radial extent in stereographic coordinates.
    density_power : float, optional
        Radial density skew parameter (same semantics as
        ``StereoSampleHemisphere``).

    Returns
    -------
    coords : np.ndarray, shape (num_pts, 2)
        Stereographic coordinates in the selected chart.
    patch_labels : np.ndarray, shape (num_pts,), dtype int32
        Patch labels (all equal to ``patch_idx``).
    """
    if patch_idx not in (0, 1):
        raise ValueError("patch_idx must be 0 (north) or 1 (south).")

    if num_pts <= 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int32)

    # Oversample in paired mode, then keep one chart.
    draw_n = 2 * int(num_pts)
    coords, labels = StereoSampleHemisphere(
        draw_n,
        patch_width=patch_width,
        density_power=density_power,
    )
    keep = labels == patch_idx
    kept_coords = coords[keep]
    kept_labels = labels[keep]

    # Very defensive top-up path for odd/edge cases.
    while kept_coords.shape[0] < num_pts:
        extra_coords, extra_labels = StereoSampleHemisphere(
            draw_n,
            patch_width=patch_width,
            density_power=density_power,
        )
        extra_keep = extra_labels == patch_idx
        kept_coords = np.concatenate([kept_coords, extra_coords[extra_keep]], axis=0)
        kept_labels = np.concatenate([kept_labels, extra_labels[extra_keep]], axis=0)

    return kept_coords[:num_pts], kept_labels[:num_pts]

def BallSample_Normal(num_samples, stddev, radius, dtype=tf.float64):
    """
    Generates 2D samples from a normal distribution, constrained to lie within
    the unit circle.

    This function draws `num_samples` 2D vectors from a normal distribution
    with the specified standard deviation and dtype. Any samples that fall
    outside the unit circle (i.e., with squared L2 norm greater than 1) are
    iteratively resampled until all samples lie within the circle.

    Parameters:
        num_samples (int): Number of 2D samples to generate.
        stddev (float): Standard deviation for the normal distribution.
        radius (float): Sample radius
        dtype (tf.DType, optional): Data type of the generated samples.
            Defaults to tf.float64.

    Returns:
        tf.Tensor: A tensor of shape (num_samples, 2) containing the valid
            samples within the unit circle.
    """
    samples = tf.random.normal(shape=(num_samples, 2), stddev=stddev, dtype=dtype)
    norms = tf.sqrt(tf.reduce_sum(samples**2, axis=1))
    mask = norms > radius

    while tf.reduce_any(mask):
        num_violations = tf.reduce_sum(tf.cast(mask, tf.int32))
        replacements = tf.random.normal(
            shape=(num_violations, 2), stddev=stddev, dtype=dtype
        )
        # Replace only the violating samples
        samples = tf.tensor_scatter_nd_update(
            samples,
            indices=tf.where(mask),
            updates=replacements,
        )
        norms = tf.sqrt(tf.reduce_sum(samples**2, axis=1))
        mask = norms > radius

    return samples


###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from geometry.sphere import PatchChange_Coordinates_Sphere

    # Sampling hyperparameters
    num_samples = int(1e4)
    patch_width = 1.0
    scaling_power = 4.0

    # Test the BallSample
    test_ball_sample = BallSample(
        num_samples, patch_width=patch_width, density_power=scaling_power
    )
    plt.figure()
    plt.title("Patch 1")
    plt.scatter(test_ball_sample[:, 0], test_ball_sample[:, 1], alpha=0.1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()

    test_ball_sample_p2 = PatchChange_Coordinates_Sphere(test_ball_sample)
    plt.figure()
    plt.title("Patch 2")
    plt.scatter(test_ball_sample_p2[:, 0], test_ball_sample_p2[:, 1], alpha=0.1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()

    """
    # Test the CubeSample
    test_cube_sample = CubeSample(
        num_samples, width=patch_width, density_power=scaling_power
    )
    plt.figure()
    plt.title("Cube Sample")
    plt.scatter(test_cube_sample[:, 0], test_cube_sample[:, 1], alpha=0.1)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.tight_layout()
    """
