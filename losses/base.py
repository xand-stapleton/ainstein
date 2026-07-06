from __future__ import annotations

import tensorflow as tf

from configs.base import BaseConfig, FinitenessConfig
from geometry.base import compute_ricci_tensor
from helper_functions.helper_functions import RadiusWeighting
from network.schedulers import FloatScheduler

tf.keras.backend.set_floatx("float64")


class WeightBase:
    def __init__(self, config: BaseConfig, weight: bool = True) -> None:
        self.config = config
        self.weight = weight
        self.overlap_upperwidth = self.config.geometry.overlap_upperwidth
        self.lorentzian = getattr(self.config.model_specific, "lorentzian", False)

    def _weight_patch(self, x_vars, norm):
        # Apply weighting
        if self.weight:
            radial_midpoint = tf.cast(tf.sqrt(2.0) - 1.0, tf.float64)
            filter_width = radial_midpoint + self.overlap_upperwidth
            radial_weights = RadiusWeighting(x_vars, filter_width)
            norm = norm * radial_weights
        return norm


class TotalLossBase:
    """
    Represents a class for computing the total training loss, which has
    contributions from solving the Einstein equation, from satisfying overlap
    conditions of patches, and from finiteness of the metric components.

    Loss multipliers can be scheduled using FloatScheduler. If a scheduler
    is configured, call set_epoch() before computing the loss during training.
    """

    def __init__(self, config: BaseConfig, weighter: WeightBase):
        self.config = config
        self.weighter = weighter

        self.dim = self.config.geometry.dim
        self.n_patches = self.config.geometry.n_patches
        self.overlap_upperwidth = self.config.geometry.overlap_upperwidth

        # Check if lorentzian is defined. If not, set it as false
        self.lorentzian = getattr(self.config.model_specific, "lorentzian", False)

        # Einstein constant, $\lambda$ in the Einstein equation: $R_{ij} = \lambda g_{ij}$
        self.einstein_constant = self.config.geometry.einstein_constant

        # Base loss multipliers
        self.einstein_multiplier_base = self.config.loss.einstein_multiplier
        self.overlap_multiplier_base = self.config.loss.overlap_multiplier
        self.finiteness_multiplier_base = self.config.loss.finiteness_multiplier

        # Current scheduled multipliers (updated by set_epoch())
        self.einstein_multiplier = self.einstein_multiplier_base
        self.overlap_multiplier = self.overlap_multiplier_base
        self.finiteness_multiplier = self.finiteness_multiplier_base

        # Initialize schedulers
        self._init_schedulers()

        # Einstein Loss
        self.einstein_losses = [
            EinsteinLoss(
                self.config,
                weighter=self.weighter,
            )
            for patch in range(int(self.n_patches))
        ]

        self.finite_losses = [
            FiniteLoss(
                self.config,
                self.config.finiteness,
                weighter=self.weighter,
            )
            for patch in range(int(self.n_patches))
        ]

    def _init_schedulers(self):
        """Initialize multiplier schedulers from config."""
        self.einstein_scheduler = None
        self.overlap_scheduler = None
        self.finiteness_scheduler = None

        if self.config.loss.einstein_schedule is not None:
            self.einstein_scheduler = FloatScheduler(
                strategy=self.config.loss.einstein_schedule.strategy,
                init_value=self.einstein_multiplier_base,
                final_value=self.config.loss.einstein_schedule.final_value,
                warmup_epochs=self.config.loss.einstein_schedule.warmup_epochs,
                decay_rate=self.config.loss.einstein_schedule.decay_rate,
                steps=self.config.loss.einstein_schedule.steps,
            )

        if self.config.loss.overlap_schedule is not None:
            self.overlap_scheduler = FloatScheduler(
                strategy=self.config.loss.overlap_schedule.strategy,
                init_value=self.overlap_multiplier_base,
                final_value=self.config.loss.overlap_schedule.final_value,
                warmup_epochs=self.config.loss.overlap_schedule.warmup_epochs,
                decay_rate=self.config.loss.overlap_schedule.decay_rate,
                steps=self.config.loss.overlap_schedule.steps,
            )

        if self.config.loss.finiteness_schedule is not None:
            self.finiteness_scheduler = FloatScheduler(
                strategy=self.config.loss.finiteness_schedule.strategy,
                init_value=self.finiteness_multiplier_base,
                final_value=self.config.loss.finiteness_schedule.final_value,
                warmup_epochs=self.config.loss.finiteness_schedule.warmup_epochs,
                decay_rate=self.config.loss.finiteness_schedule.decay_rate,
                steps=self.config.loss.finiteness_schedule.steps,
            )

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update scheduled multipliers for the current epoch."""
        if self.einstein_scheduler is not None:
            self.einstein_multiplier = self.einstein_scheduler.get(epoch, total_epochs)

        if self.overlap_scheduler is not None:
            self.overlap_multiplier = self.overlap_scheduler.get(epoch, total_epochs)

        if self.finiteness_scheduler is not None:
            self.finiteness_multiplier = self.finiteness_scheduler.get(
                epoch, total_epochs
            )


class EinsteinLoss:
    r"""
    Represents a class for computing the Einstein loss, which measures the
    difference between the Ricci tensor and the predicted metric tensor
    (scaled by the Einstein constant $\lambda$).
    """

    def __init__(
        self,
        config: BaseConfig,
        weighter: WeightBase,
    ) -> None:
        self.config = config
        self.weighter = weighter

        self.einstein_constant = self.config.geometry.einstein_constant

    def compute(self, x_vars, metric_pred, model):
        # Compute the Ricci tensor
        ricci_tensor = compute_ricci_tensor(x_vars, model)

        # Compute the loss from the Einstein equation
        if not self.config.model_specific.lorentzian:
            # Use Riemannian L2 norm
            g_inv = tf.linalg.inv(metric_pred)
            error = self.einstein_constant * metric_pred - ricci_tensor
            norm = tf.einsum("sij,sik,sjl,skl->s", error, g_inv, g_inv, error)
        else:
            # Use Euclidean L2 norm
            norm = tf.norm(
                self.einstein_constant * metric_pred - ricci_tensor, axis=(1, 2)
            )

        # Apply weighting and sum over batch
        ###norm *= tf.sqrt(tf.abs(tf.linalg.det(metric_pred))) #...weight by \sqrt(|g|)
        norm = self.weighter._weight_patch(x_vars, norm)  # ...radial weighting
        einstein_loss = tf.reduce_mean(norm)

        return einstein_loss


class FiniteLoss:
    """
    Represents a class for computing the finiteness loss, which measures the
    norm of the metric components and weights according to a predefined filter.
    This loss component ensures the zero metric is avoided as an attractor point
    of the learning.
    """

    def __init__(
        self,
        config: BaseConfig,
        filter_config: FinitenessConfig,
        weighter: WeightBase,
        dim: int | None = None,
        submanifold_is_R2: bool = True,
    ) -> None:
        self.config = config
        self.filter_config = filter_config
        self.weighter = weighter
        self.submanifold_is_R2 = submanifold_is_R2

        if dim:
            self.dim = dim
        else:
            self.dim = self.config.geometry.dim

    def compute(self, x_vars, metric_pred):
        ### HACKY WORKAROUND for Schwarzschild ###
        if self.config.model.experiment == "schwarzschild":
            param_a, param_b = 2, 0.1
            # Reduce to submanifold, and take \sum|g|
            if self.submanifold_is_R2:
                metric_pred_submanifold = metric_pred[:, :2, :2]
            else:
                metric_pred_submanifold = metric_pred[:, 2:, 2:]
            sum_metric_pred = (
                tf.reduce_sum(abs(metric_pred_submanifold), axis=[1, 2], keepdims=True)
                * 2
                / ((2) * (2 - 1))
            )
            # Apply function
            # finite_weight = (1 / (sum_metric_pred * param_a + param_b)) + 1.0
            finite_weight = 1 / (sum_metric_pred * param_a + param_b)
            finite_loss = tf.square(tf.reduce_mean(finite_weight))
        ##########################################

        else:
            # Compute the norm of the metric components
            sum_metric_pred = (
                tf.reduce_sum(abs(metric_pred), axis=[1, 2], keepdims=True)
                * 2
                / ((self.dim) * (self.dim - 1))
            )
            # Define the finiteness filter weighting function
            finite_weight = (
                (
                    tf.square(
                        self.filter_config.finite_height
                        * tf.exp(
                            -tf.pow(
                                (
                                    (sum_metric_pred - self.filter_config.finite_centre)
                                    / self.filter_config.finite_width
                                ),
                                self.filter_config.finite_sharpness,
                            )
                        )
                        - self.filter_config.finite_height
                    )
                    + 1
                )
                + (
                    sum_metric_pred / self.filter_config.finite_slope
                    - (
                        self.filter_config.finite_centre
                        + self.filter_config.finite_width
                    )
                    / self.filter_config.finite_slope
                )
                * (
                    1
                    + tf.math.tanh(
                        sum_metric_pred / 2
                        - (
                            self.filter_config.finite_centre
                            + self.filter_config.finite_width
                        )
                        / 2
                    )
                )
                / 2
                + (
                    -sum_metric_pred / self.filter_config.finite_slope
                    + (
                        self.filter_config.finite_centre
                        - self.filter_config.finite_width
                    )
                    / self.filter_config.finite_slope
                )
                * (
                    1
                    + tf.math.tanh(
                        -sum_metric_pred / 2
                        + (
                            self.filter_config.finite_centre
                            - self.filter_config.finite_width
                        )
                        / 2
                    )
                )
                / 2
            )
            # Apply weighting and sum over batch
            finite_weight = self.weighter._weight_patch(x_vars, finite_weight)
            finite_loss = tf.square(1 - tf.reduce_mean(finite_weight))

        return finite_loss
