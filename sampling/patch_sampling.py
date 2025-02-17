"""Sampling Schemes for the Hyperball & Hypercube"""
# Import libraries
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

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

    # Sample the radii (using a beta distribution, centred on radial midpoint and symmetrised between the patches)
    radii_p1 = np.random.beta(density_power, density_power/(np.sqrt(2.)-1.)-density_power, size=int(num_pts/2)) 
    radii_p2 = np.random.beta(density_power, density_power/(np.sqrt(2.)-1.)-density_power, size=int(num_pts/2)) 
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


def CubeSample(num_pts, dimension=2, width=1.0, density_power=1.0):
    """
    Code to sample the n-cube representation of I^n (for n geq 2),
    up to some boundary cutoff.

    Parameters
    ----------
    num_pts : int
        The number of sample points to generate.
    dimension : int, optional
        The dimension of the cube to be sampled. The default is 2.
    width : float, optional
        The width of the cube from the origin. The default is 1.
    density_power : float, optional
        The power factor to skew the buniform sampling by
        (< 1 skews towards boundary). The default is 1.

    Returns
    -------
    array
        The sample points for the cube, shape (num_pts, dimension).

    """
    # Sample in the [0,1] range
    sample = np.random.uniform(low=0.0, high=1.0, size=(num_pts, dimension))

    # Skew the sample
    sample = sample**density_power

    # Transform to the cube range
    sample = sample * np.random.choice([-1, 1], size=(num_pts, dimension)) * width

    return sample


###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry.geometry import PatchChange_Coordinates_Ball

    # Sampling hyperparameters
    num_samples = int(1e4)
    patch_width = 1.
    scaling_power = 4.

    # Test the BallSample
    test_ball_sample = BallSample(
        num_samples, patch_width=patch_width, density_power=scaling_power
    )
    plt.figure()
    plt.title("Patch 1")
    plt.scatter(test_ball_sample[:, 0], test_ball_sample[:, 1], alpha=0.1)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.tight_layout()
    
    test_ball_sample_p2 = PatchChange_Coordinates_Ball(test_ball_sample)
    plt.figure()
    plt.title("Patch 2")
    plt.scatter(test_ball_sample_p2[:, 0], test_ball_sample_p2[:, 1], alpha=0.1)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.tight_layout()
    
    
    '''
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
    '''
