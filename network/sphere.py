from __future__ import annotations

import tensorflow as tf

tfk = tf.keras
tfk.backend.set_floatx("float64")
from keras.saving import register_keras_serializable

from configs.sphere import SphereConfig
from geometry.sphere import PatchChange_Coordinates_Sphere
from losses.sphere import TotalSphereLoss
from network.base import BaseGlobalModel, BaseNetwork, BasePatchSubmodel


@register_keras_serializable()
class SpherePatchSubModel(BasePatchSubmodel):
    def __init__(self, config: SphereConfig, n_out, **kwargs):
        super().__init__(config, n_out, **kwargs)


@register_keras_serializable()
class SphereGlobalModel(BaseGlobalModel):
    config: SphereConfig

    def __init__(self, config: SphereConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.n_patches = config.geometry.n_patches

        # Compute the number of independent metric entries, this is the number
        # of vielbein entries used as the model outputs for each patch
        n_out = int(0.5 * self.dim * (self.dim + 1))

        # Define submodels for each patch
        self.patch_submodels = [
            SpherePatchSubModel(self.config, n_out) for _ in range(int(self.n_patches))
        ]
        if self.n_patches == 2:
            self.patch_transform_layer = tfk.layers.Lambda(
                PatchChange_Coordinates_Sphere, dtype=tf.float64
            )
        elif self.n_patches > 2:
            raise NotImplementedError("Codebase not yet configured for >2 patches...")

    def call(self, inputs):
        # Transform input data to all patches
        patch_inputs = [inputs]
        if self.n_patches > 1:
            patch_inputs.append(self.patch_transform_layer(inputs))
        # Compute the outputs for all patches
        concatenated_output = tfk.layers.Concatenate()(
            [
                self.patch_submodels[patch_idx](patch_inputs[patch_idx])
                for patch_idx in range(int(self.n_patches))
            ]
        )

        return concatenated_output


class SphereNetwork(BaseNetwork):
    """
    Represents a class for the machine learning processes used in training the
    global metric function across the patches. This object contains the metric
    neural network models as an attribute subclass via BSphereGlobalModel, otherwise
    containing functionality for training, validating, saving, logging.
    """

    config: SphereConfig

    def __init__(self, config: SphereConfig, restore_hps: bool = False):
        super().__init__(config, restore_hps)

        # Build the model
        if not hasattr(self, "model"):
            self.model = SphereGlobalModel(self.config)

        # Define the loss
        self.loss = TotalSphereLoss(config=self.config)

    def evaluate_loss(
        self, x, training=True, return_constituents=False, val_print=True
    ):
        metric_pred = self.model(x, training=training)
        return self.loss.call(
            self.model, x, metric_pred, return_constituents, val_print
        )
