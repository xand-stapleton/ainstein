import numpy as np


def get_model_weights_stats(model):
    """
    Calculate the standard deviation, minimum, maximum, and average of the weights for all layers in a TensorFlow/Keras model.

    Parameters:
    - model: A compiled TensorFlow/Keras model.

    Returns:
    - stats: A dictionary with the standard deviation, minimum, maximum, and average of all weights.
    """
    all_weights = []
    for layer in model.layers:
        # Get weights of the current layer
        layer_weights = layer.get_weights()
        for weights in layer_weights:
            all_weights.extend(weights.flatten())  # Flatten and add to the list

    # Compute statistics
    std = np.std(all_weights)
    min_val = np.min(all_weights)
    max_val = np.max(all_weights)
    avg = np.mean(all_weights)

    return {
        "Parameters std": std,
        "Parameters min": min_val,
        "Parameters max": max_val,
        "Parameters avg": avg,
    }
