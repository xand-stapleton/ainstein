import tensorflow as tf
from losses.ball import TotalBallLoss
from network.base import BaseNetwork

tfk = tf.keras
tfk.backend.set_floatx("float64")
from geometry.ball import PatchChange_Coordinates_Ball
from keras.saving import register_keras_serializable
from network.base import BaseGlobalModel, BasePatchSubmodel


@register_keras_serializable()
class BallPatchSubModel(BasePatchSubmodel):
    def __init__(self, hp, n_out, **kwargs):
        super().__init__(hp, n_out, **kwargs)


@register_keras_serializable()
class BallGlobalModel(BaseGlobalModel):
    def __init__(self, hp, **kwargs):
        super().__init__(hp, **kwargs)
        self.n_patches = hp["n_patches"]

        # Compute the number of independent metric entries, this is the number
        # of vielbein entries used as the model outputs for each patch
        n_out = int(0.5 * self.dim * (self.dim + 1))

        # Define submodels for each patch
        self.patch_submodels = [
            BallPatchSubModel(self.hp, n_out) for _ in range(int(self.n_patches))
        ]
        if self.n_patches == 2:
            self.patch_transform_layer = tfk.layers.Lambda(
                PatchChange_Coordinates_Ball, dtype=tf.float64
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


class BallNetwork(BaseNetwork):
    """
    Represents a class for the machine learning processes used in training the
    global metric function across the patches. This object contains the metric
    neural network models as an attribute subclass via BallGlobalModel, otherwise
    containing functionality for training, validating, saving, logging.

    Attributes:
    - hp (dict): Dictionary of the training hyperparameters.
    - val_print (bool): Whether to print the validation measures with each call.
    - best_loss (float): The best epoch training loss value across previous epochs.
    - model (tfk.Model): The global neural network model for the metric.
    - loss (float): The training loss of the neural network model.
    - log_dir (str): The filepath for the models to be saved too.
    - optimiser (tfk.optimizers): The optimiser used for neural network training.

    Methods:
    - __init__(self, hp, print_losses, restore_hps):
      Initializes the Network class with the respective training hyperparameters,
      calling the BallGlobalModel class to define the neural network architectures
      for the metric function, allowing pre-trained model imports. Sets up the
      training loss, and the logging.

    - evaluate_loss(self, x, training=True, return_constituents, val_print):
      Compute the model-predicted metric values for input points on the manifold
      (in patch 1 coordinates), and the respective training loss value.

    """

    def __init__(self, hp, print_losses=False, restore_hps=False):
        super().__init__(hp, print_losses, restore_hps)

        # Build the model
        if not hasattr(self, "model"):
            self.model = BallGlobalModel(self.hp)

        # Define the loss
        self.loss = TotalBallLoss(hp=self.hp, print_losses=print_losses)

    def evaluate_loss(
        self, x, training=True, return_constituents=False, val_print=True
    ):
        metric_pred = self.model(x, training=training)
        return self.loss.call(
            self.model, x, metric_pred, return_constituents, val_print
        )
