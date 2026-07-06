from __future__ import annotations

import math
from typing import Literal


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


class FloatScheduler:
    """
    General-purpose scheduler for floating-point schedule values (einstein, overlap, finiteness, density powers).

    Supports multiple scheduling strategies:
    - 'constant': Keep the initial value unchanged
    - 'linear': Linear increase/decrease from initial to final value
    - 'exponential': Exponential decay/growth
    - 'cosine': Cosine annealing between initial and final values
    - 'step': Piecewise-constant with N decay steps evenly distributed over training

    The scheduler is step/epoch-agnostic; call get() with the current progress.
    """

    def __init__(
        self,
        strategy: Literal["constant", "linear", "exponential", "cosine", "step"] = "constant",
        init_value: float = 1.0,
        final_value: float | None = None,
        warmup_epochs: int = 0,
        decay_rate: float = 0.96,
        steps: int = 1,
    ):
        """
        Initialize a float scheduler.

        Args:
            strategy: Scheduling strategy type
            init_value: Initial multiplier value
            final_value: Final value (for linear/cosine); defaults to init_value
            warmup_epochs: Number of epochs to linearly ramp from 0 to init_value
            decay_rate: Decay factor per step for exponential/step strategies (default 0.96)
            steps: For 'step' strategy, number of decay steps (default 1 for Heaviside)
        """
        self.strategy = strategy
        self.init_value = init_value
        self.final_value = final_value if final_value is not None else init_value
        self.warmup_epochs = max(0, warmup_epochs)
        self.decay_rate = decay_rate
        self.steps = max(1, steps)

    def get(self, epoch: int, total_epochs: int) -> float:
        """
        Compute the multiplier value for a given epoch.

        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs

        Returns:
            Scheduled multiplier value
        """
        # Handle warmup phase
        if epoch < self.warmup_epochs:
            return self.init_value * (epoch / self.warmup_epochs)

        progress = (epoch - self.warmup_epochs) / max(
            1, total_epochs - self.warmup_epochs
        )
        progress = min(1.0, progress)  # Clamp to [0, 1]
        step_progress = max(0, epoch - self.warmup_epochs)

        match self.strategy:
            case "constant":
                return self.init_value
            case "linear":
                return self.init_value + (self.final_value - self.init_value) * progress
            case "exponential":
                return self.init_value * (self.decay_rate**progress)
            case "cosine":
                factor = 0.5 * (1 + math.cos(math.pi * progress))
                return self.final_value + (self.init_value - self.final_value) * factor
            case "step":
                # Divide training into (steps + 1) phases, decay at each boundary
                # Example: steps=1 (Heaviside) → single jump at epoch total_epochs/2
                actual_step_epoch_interval = max(1, total_epochs // (self.steps + 1))
                step_index = step_progress // actual_step_epoch_interval
                return self.init_value * (self.decay_rate ** step_index)
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")


# Backwards compatibility for older imports.
MultiplierScheduler = FloatScheduler
