from __future__ import annotations

import tensorflow as tf

tf.keras.backend.set_floatx("float64")

from configs.sphere import SphereConfig
from geometry.sphere import (PatchChange_Coordinates_Sphere,
                             PatchChange_Metric_Sphere)
from helper_functions.helper_functions import (RadiusWeighting,
                                               cholesky_from_vec)
from losses.base import EinsteinLoss, TotalLossBase, WeightBase


class WeightSphere(WeightBase):
    def __init__(self, config: SphereConfig, weight: bool = True) -> None:
        super().__init__(config, weight)

    # _weight_patch comes from the base class -- for
    # spheres it's default

    def _weight_overlap(self, x_vars: tf.Tensor, overlap_loss: tf.Tensor) -> tf.Tensor:
        # Apply weighting
        if self.weight:
            radial_midpoint = tf.cast(tf.sqrt(2.0) - 1.0, tf.float64)
            filter_lower_bound = (1 - (radial_midpoint + self.overlap_upperwidth)) / (
                1 + (radial_midpoint + self.overlap_upperwidth)
            )
            filter_midpoint = (
                (radial_midpoint + self.overlap_upperwidth) + filter_lower_bound
            ) / 2.0
            filter_width = radial_midpoint + self.overlap_upperwidth - filter_midpoint
            radial_weights = RadiusWeighting(
                x_vars, filter_width=filter_width, filter_midpt=filter_midpoint
            )
            overlap_loss = radial_weights * overlap_loss
        return overlap_loss


class TotalSphereLoss(TotalLossBase):
    """
    Represents a class for computing the total training loss, which has
    contributions from solving the Einstein equation, from satisfying overlap
    conditions of patches, and from finiteness of the metric components.
    """

    def __init__(self, config: SphereConfig):
        weighter = WeightSphere(config, weight=True)
        super().__init__(config, weighter)
        self.weighter = weighter

        assert (
            abs(self.einstein_multiplier)
            + abs(self.overlap_multiplier)
            + abs(self.finiteness_multiplier)
            > 0.0
        ), "All loss terms turned off..."

        # Overlap Loss
        if self.n_patches == 1:
            self.overlap_loss = tf.cast(0.0, tf.float64)
        elif self.n_patches == 2:
            self.overlap_loss = OverlapLossSphere(
                config=self.config, weighter=self.weighter
            )
        else:
            self.overlap_loss = tf.cast(0.0, tf.float64)
            raise RuntimeError(
                f"Overlap loss not configured for {self.n_patches} patches..."
            )

    def call(
        self, model, x_vars, metric_pred, return_constituents=False, val_print=True
    ):
        # Set up the network inputs & outputs
        patch_inputs = [x_vars]
        metric_preds_vecs, metric_preds = [], []
        if self.n_patches > 1:
            # Compute the input coordinates in the second patch
            patch_inputs.append(model.patch_transform_layer(x_vars))

            # Split the output into the metrics in each patch
            patch_1_output, patch_2_output = tf.split(
                metric_pred, num_or_size_splits=2, axis=-1
            )
            metric_preds_vecs.append(patch_1_output)
            metric_preds_vecs.append(patch_2_output)
        else:
            metric_preds_vecs.append(metric_pred)

        # Convert from vielbein vector to symmetric matrix
        metric_preds = [
            cholesky_from_vec(patch_output, lorentzian=self.lorentzian)
            for patch_output in metric_preds_vecs
        ]

        # Compute the loss components
        # Einstein
        if self.einstein_multiplier > 0.0:
            e_losses = [
                self.einstein_losses[patch_idx].compute(
                    patch_inputs[patch_idx],
                    metric_preds[patch_idx],
                    model.patch_submodels[patch_idx],
                )
                for patch_idx in range(int(self.n_patches))
            ]
        else:
            e_losses = [
                tf.cast(0.0, tf.float64) for patch_idx in range(int(self.n_patches))
            ]

        # Overlap
        if self.overlap_multiplier > 0.0 and self.n_patches == 2:
            overlap_loss = self.overlap_loss.compute(
                x_vars, [metric_preds[0], metric_preds[1]]
            )
        else:
            overlap_loss = tf.cast(0.0, tf.float64)

        # Finiteness
        if self.finiteness_multiplier > 0.0:
            f_losses = [
                tf.math.log(
                    self.finite_losses[patch_idx].compute(
                        x_vars, metric_preds[patch_idx]
                    )
                    + 1.0
                )
                for patch_idx in range(int(self.n_patches))
            ]
        else:
            f_losses = [
                tf.cast(0.0, tf.float64) for patch_idx in range(int(self.n_patches))
            ]

        # Print the batch loss values
        if self.config.logging.print_batch_losses and val_print:
            print(
                f"Einstein: {[f'{tf.get_static_value(e_loss):.3g}' for e_loss in e_losses]}\n"
                f"Overlap: {tf.get_static_value(overlap_loss):.3g}\n"
                f"Finite: {[f'{tf.get_static_value(f_loss):.3g}' for f_loss in f_losses]}\n"
            )

        # Initialise the constituent losses dictionary (holds each of the
        # loss components pre-sum)
        if return_constituents:
            loss_constituents = {
                "einstein_losses": [tf.get_static_value(e_loss) for e_loss in e_losses],
                "overlap_loss": tf.get_static_value(overlap_loss),
                "finiteness_losses": [
                    tf.get_static_value(f_loss) for f_loss in f_losses
                ],
            }
        else:
            loss_constituents = None

        # Compute the total loss (accounting for multipliers)
        total_loss = 0.0
        if self.einstein_multiplier > 0.0:
            total_loss += self.einstein_multiplier * tf.reduce_sum(
                tf.math.abs(e_losses)
            )
        if self.overlap_multiplier > 0.0:
            total_loss += self.overlap_multiplier * tf.math.abs(overlap_loss)
        if self.finiteness_multiplier > 0.0:
            total_loss += self.finiteness_multiplier * tf.reduce_sum(
                tf.math.abs(f_losses)
            )
        # Normalise by the multiplier factors
        total_loss /= (
            self.einstein_multiplier
            + self.overlap_multiplier
            + self.finiteness_multiplier
        )

        return total_loss, loss_constituents


class OverlapLossSphere:
    """
    Represents a class for computing the overlap loss, which measures the
    difference between agreement of the metric predictions between the patches.
    This uses symmetric contributions from the difference between the metric
    prediction in patch 1 and the metric prediction for patch 2 transformed
    into patch 1, and equivalently the difference between the metric prediction
    in patch 2 and the metric prediction for patch 1 transformed into patch 2.
    The contributions are weighted by the points radial positions, prioirtising
    points within the overlap region which is an annulus about the radial midpoint.
    """

    def __init__(
        self,
        config: SphereConfig,
        weighter: WeightSphere,
    ) -> None:
        self.config = config
        self.weighter = weighter

    def compute(self, x_vals, metric_preds):
        # Compute the patch changes of the both outputs
        patch_2_metrics_from_patch_1 = PatchChange_Metric_Sphere(
            x_vals, metric_preds[0]
        )
        patch_1_metrics_from_patch_2 = PatchChange_Metric_Sphere(
            PatchChange_Coordinates_Sphere(x_vals), metric_preds[1]
        )

        # Take the total difference in both patches between the metrics in both patches
        if not self.config.model_specific.lorentzian:
            # Use Riemannian L2 norm
            g_inv_p1 = tf.linalg.inv(metric_preds[0])
            error_p1 = patch_1_metrics_from_patch_2 - metric_preds[0]
            overlap_loss_p1 = tf.einsum(
                "sij,sik,sjl,skl->s", error_p1, g_inv_p1, g_inv_p1, error_p1
            )

            g_inv_p2 = tf.linalg.inv(metric_preds[1])
            error_p2 = patch_2_metrics_from_patch_1 - metric_preds[1]
            overlap_loss_p2 = tf.einsum(
                "sij,sik,sjl,skl->s", error_p2, g_inv_p2, g_inv_p2, error_p2
            )
        else:
            # Use Euclidean L2 norm
            overlap_loss = tf.reduce_mean(
                tf.square(patch_2_metrics_from_patch_1 - metric_preds[1]), axis=(1, 2)
            ) + tf.reduce_mean(
                tf.square(patch_1_metrics_from_patch_2 - metric_preds[0]), axis=(1, 2)
            )

        # Apply weighting and sum over batch
        ###overlap_loss_p1 *= tf.sqrt(tf.abs(tf.linalg.det(metric_preds[0]))) #...weight by \sqrt(|g|)
        ###overlap_loss_p2 *= tf.sqrt(tf.abs(tf.linalg.det(metric_preds[1]))) #...weight by \sqrt(|g|)
        overlap_loss = overlap_loss_p1 + overlap_loss_p2
        overlap_loss = self.weighter._weight_overlap(
            x_vals, overlap_loss
        )  # ...radial weighting
        overlap_loss = tf.reduce_mean(overlap_loss)

        return overlap_loss


class GlobalLossSphere:
    """
    Represents a class for computing the global test loss, which has
    contributions from solving the Einstein equation and from satisfying overlap
    conditions of patches. The patches are restricted to points within the radial
    limit, and the overlap region is an annulus which spans either side of the
    radial midpoint and runs up to the radial limit, such that it is symmetric
    under the patch transform function.
    """

    def __init__(self, config: SphereConfig, radial_limit: float | None = None):
        self.config = config
        assert radial_limit is not None
        self.radial_limit: float = radial_limit
        self.weighter = WeightSphere(config, weight=False)
        self.radial_midpoint = tf.cast(tf.sqrt(2.0) - 1.0, tf.float64)
        # Ensure the patching conditions are consistently defined
        if self.radial_limit:
            assert self.radial_limit > self.radial_midpoint, "Patches do not overlap..."

        self.dim = self.config.geometry.dim
        self.n_patches = self.config.geometry.n_patches

        # Einstein constant, $\lambda$ in the Einstein equation: $R_{ij} = \lambda g_{ij}$
        self.einstein_constant = self.config.geometry.einstein_constant

        # Check if lorentzian is defined. If not, set it as false
        self.lorentzian = getattr(self.config.model_specific, "lorentzian", False)

        # Loss multipliers
        self.einstein_multiplier = self.config.loss.einstein_multiplier
        self.overlap_multiplier = self.config.loss.overlap_multiplier
        if self.n_patches == 1:
            self.overlap_multiplier = tf.cast(0.0, tf.float64)

        # Einstein Loss
        self.einstein_losses = [
            EinsteinLoss(self.config, weighter=self.weighter)
            for _ in range(int(self.n_patches))
        ]

        # Overlap Loss
        if self.n_patches == 1:
            self.overlap_loss = tf.cast(0.0, tf.float64)
        elif self.n_patches == 2:
            self.overlap_loss = OverlapLossSphere(self.config, weighter=self.weighter)
        else:
            self.overlap_loss = 0.0
            print(
                f"Overlap loss not yet configured for {self.n_patches} patches...",
                flush=True,
            )

    def call(self, model, x_vars, metric_pred):
        # Set up the network inputs & outputs
        patch_inputs = [x_vars]
        metric_preds_vecs = []
        if self.n_patches > 1:
            # Compute the input coordinates in the second patch
            patch_inputs.append(model.patch_transform_layer(x_vars))

            # Split the output into the metrics in each patch
            patch_1_output, patch_2_output = tf.split(
                metric_pred, num_or_size_splits=2, axis=-1
            )
            metric_preds_vecs.append(patch_1_output)
            metric_preds_vecs.append(patch_2_output)
        else:
            metric_preds_vecs.append(metric_pred)

        # Convert from vielbein vector to symmetric matrix
        metric_preds = [
            cholesky_from_vec(patch_output, lorentzian=self.lorentzian)
            for patch_output in metric_preds_vecs
        ]

        # Compute data limited to each patch
        if self.radial_limit and self.radial_limit > 0:
            # Patches
            norms = [
                tf.sqrt(tf.reduce_sum(tf.square(p_pts), axis=1))
                for p_pts in patch_inputs
            ]
            masks = [
                norm < self.radial_limit for norm in norms
            ]  # ...find points within the radial limit
            pts_limited = [
                tf.boolean_mask(patch_inputs[p_idx], masks[p_idx])
                for p_idx in range(int(self.n_patches))
            ]
            metrics_limited = [
                tf.boolean_mask(metric_preds[p_idx], masks[p_idx])
                for p_idx in range(int(self.n_patches))
            ]

            # Overlap Region
            mask_overlap = tf.logical_and(
                norms[0] >= (1 - self.radial_limit) / (1 + self.radial_limit),
                norms[0] <= self.radial_limit,
            )  # ...find points within the overlap region
            pts_overlap = tf.boolean_mask(patch_inputs[0], mask_overlap)
            metrics_overlap = [
                tf.boolean_mask(metric_preds[p_idx], mask_overlap)
                for p_idx in range(int(self.n_patches))
            ]
        else:
            # ...otherwise use the full patches in each case
            pts_limited, metrics_limited, pts_overlap, metrics_overlap = (
                patch_inputs,
                metric_preds,
                patch_inputs[0],
                metric_preds,
            )

        # Compute the number of points in each region
        sample_sizes = [[p_pts.shape[0] for p_pts in pts_limited], pts_overlap.shape[0]]

        # Compute the loss components
        if self.einstein_multiplier > 0.0:
            e_losses = [
                self.einstein_losses[patch_idx].compute(
                    pts_limited[patch_idx],
                    metrics_limited[patch_idx],
                    model.patch_submodels[patch_idx],
                )
                for patch_idx in range(int(self.n_patches))
            ]
        else:
            e_losses = [
                tf.cast(0.0, tf.float64) for patch_idx in range(int(self.n_patches))
            ]
        if self.overlap_multiplier > 0.0 and self.n_patches > 1:
            overlap_loss = self.overlap_loss.compute(pts_overlap, metrics_overlap)
        else:
            overlap_loss = tf.cast(0.0, tf.float64)

        # Return loss components
        loss_constituents = {
            "einstein_losses": [tf.get_static_value(e_loss) for e_loss in e_losses],
            "overlap_loss": tf.get_static_value(overlap_loss),
        }

        # Compute the total loss (accounting for multiplier)
        global_loss = 0.0
        if self.einstein_multiplier > 0.0:
            global_loss += self.einstein_multiplier * tf.reduce_sum(
                tf.math.abs(e_losses)
            )
        if self.overlap_multiplier > 0.0:
            global_loss += self.overlap_multiplier * tf.math.abs(overlap_loss)
        global_loss /= self.einstein_multiplier + self.overlap_multiplier

        return global_loss, loss_constituents, sample_sizes
