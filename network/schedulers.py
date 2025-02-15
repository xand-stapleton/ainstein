import math


# Cosine annealing scheduler
def cosine_annealing(epoch, total_epochs, lr_init=1e-3, lr_min=0.0):
    """
    Compute the learning rate using cosine annealing.

    Parameters:
    - epoch: Current epoch number
    - total_epochs: Total number of epochs
    - lr_init: Maximum initialisation learning rate (default 1e-3)
    - lr_min: Minimum learning rate to which it will decay (default 0)

    Returns:
    - Adjusted learning rate
    """
    lr_max = lr_init
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * epoch / total_epochs)
    )


# Exponential decay scheduler
def exponential_decay(epoch, lr_init=0.001, decay_rate=0.96, decay_steps=10):
    """
    Compute the learning rate using exponential decay.

    Parameters:
    - epoch: Current epoch number
    - lr_init: Maximum initialisation learning rate (default 1e-3)
    - decay_rate: The rate at which the learning rate decays (default 0.96)
    - decay_steps: Number of epochs after which the learning rate decays (default 10)

    Returns:
    - Adjusted learning rate
    """
    return lr_init * (decay_rate ** (epoch // decay_steps))
