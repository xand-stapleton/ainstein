from __future__ import annotations

import numpy as np
import tensorflow as tf


def check_set_random_seeds(
    np_seed: int | None = None, tf_seed: int | None = None, print_keys: bool = True
) -> tuple[int, int]:
    """
    Ensures reproducibility by checking and setting NumPy and TensorFlow seeds
        in the provided config dictionary. If seeds are not set, random ones are
        generated and stored back into the config.

        Also initializes the random states for NumPy and TensorFlow using these seeds,
        and optionally prints a sample of generated values to verify deterministic behavior.

        Args:
            config_dict: A dictionary containing model configuration. It must include
                         the keys 'model' -> 'np_seed' and 'tf_seed', which can be
                         either integers or None.
            print_keys: If True, prints random values from NumPy and TensorFlow to help
                        verify that the seeds were applied correctly.

        Returns:
            A tuple containing:
            - The NumPy seed used (int)
            - The TensorFlow seed used (int)
    """

    # Check and set seeds for reproducibility
    rng = np.random.default_rng()
    # ...for NumPy
    if np_seed is None:
        np_seed = int(rng.integers(2**32 - 2))
    # ...for TensorFlow
    if tf_seed is None:
        tf_seed = int(rng.integers(2**32 - 2))

    np.random.seed(np_seed)
    tf.random.set_seed(tf_seed)
    tf.keras.utils.set_random_seed(tf_seed)

    np_key = [int(i) for i in np.random.randint(1, np.iinfo(np.int32).max, size=6)]
    tf_key = [float(i) for i in tf.random.uniform(shape=[6])]
    if print_keys:
        # Print some random characters to check seed applied correctly
        print("TF random key: ", tf_key)
        print("NP random key: ", np_key)
    return np_seed, tf_seed
