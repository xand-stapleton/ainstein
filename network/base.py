import tensorflow as tf

tfk = tf.keras
tfk.backend.set_floatx("float64")
import signal
import warnings
from sys import exit

import numpy as np

import wandb
from helper_functions import helper_functions
from losses.ball import TotalBallLoss
from network import network_analysis, schedulers
from keras.saving import register_keras_serializable


@register_keras_serializable()
class BasePatchSubmodel(tf.keras.Model):
    """
    Represents a class for the neural network model which represents the metric
    function in a patch, these are trained across the patches to satify the Einstein equation.
    Inherits from the tf.keras.Model class.

    Attributes:
    - hp (dict): Dictionary of the training hyperparameters.
    - serializable_hp (dict): Dictionary of the training hyperparameters which can be saved with the model.
    - dim (int): Number of dimensions of the manifold.
    - n_layers (int): The number of layers in the neural network model.
    - n_hidden (int): The number of neurons in each neural network layer.
    - activations (str): The name of the neural network activation function, recognisable by the tf functionality.
    - use_bias (bool): Whether the neural network layers use biases.
    - submodel (tfk.Model): The neural network model for the metric function on the patch.

    Methods:
    - __init__(self, hp, n_out):
      Initializes the PatchSubModel class with the respective model hyperparameters,
      initialising the neural network architecture for the metric function.

    - call(self, inputs):
      Computes the model-predicted metric components for input points on the patch.

    - _is_serializable(self, value):
      Identify whether a given hyperparameter is serialisable, such that it can be saved with the model.

    - set_serializable_hp(self):
      Set the hyperparameters to be saved with the model.

    - get_config(self):
      Set up the model saving configuration to allow reloading.

    - from_config(self, config)
      Extract the config to load the neural network model for the patch.
    """

    def __init__(self, hp, n_out, **kwargs):
        super().__init__(**kwargs)
        # Define hyperparameters
        self.hp = hp
        self.serializable_hp = None
        self.set_serializable_hp()
        self.dim = self.hp["dim"]
        self.n_layers = self.hp["n_layers"]
        self.n_hidden = self.hp["n_hidden"]
        self.activations = self.hp["activations"]
        self.use_bias = self.hp["use_bias"]

        # Define subnetwork architecture
        inputs = tfk.layers.Input(shape=(self.dim,), dtype=tf.float64)
        x = tfk.layers.Dense(
            self.n_hidden, activation=self.activations, use_bias=self.use_bias
        )(inputs)
        for _ in range(self.n_layers - 2):
            x = tfk.layers.Dense(
                self.n_hidden, activation=self.activations, use_bias=self.use_bias
            )(x)
        outputs = tfk.layers.Dense(n_out, activation=None, use_bias=False)(x)

        self.submodel = tfk.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.submodel(inputs)

    def _is_serializable(self, value):
        try:
            tf.keras.utils.serialize_keras_object(value)
            return True
        except (TypeError, ValueError):
            return False

    def set_serializable_hp(self):
        self.serializable_hp = {
            key: value for key, value in self.hp.items() if self._is_serializable(value)
        }

    def get_config(self):
        # Return the configuration necessary to recreate this model
        config = super().get_config()
        config.update({"hp": self.serializable_hp, "n_out": self.n_out})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class BaseGlobalModel(tf.keras.Model):
    """
    Represents a class for the global model of the metric function across the
    patches, these are trained to satify the Einstein equation. Inherits from
    the tf.keras.Model class.

    Attributes:
    - hp (dict): Dictionary of the training hyperparameters.
    - serializable_hp (dict): Dictionary of the training hyperparameters which can be saved with the model.
    - set_serializable_hp (): ...
    - dim (int): Number of dimensions of the manifold.
    - n_patches (int): Number of patches in the manifold definition (1 or 2).
    - patch_submodels (list): List of the metric-predicting neural network models for each patch.
    - patch_transform_layer (tfk layer): Function which transforms the input data between patches. Used for transforming patch 1 to 2, but is symmetric for 2 to 1 also.

    Methods:
    - __init__(self, hp):
      Initializes the GlobalModel class with the respective model hyperparameters,
      calling the PatchSubModel class to define the neural network architectures
      for the metric function in each patch.

    - call(self, inputs):
      Computes the model-predicted metric components for input points in each patch.

    - _is_serializable(self, value):
      Identify whether a given hyperparameter is serialisable, such that it can be saved with the model.

    - set_serializable_hp(self):
      Set the hyperparameters to be saved with the model.

    - get_config(self):
      Set up the model saving configuration to allow reloading.

    - from_config(self, config)
      Extract the config to load the neural network model for the patch.
    """

    def __init__(self, hp, **kwargs):
        super().__init__(**kwargs)
        # Define hyperparameters
        self.hp = hp
        self.serializable_hp = None
        self.set_serializable_hp()
        self.dim = self.hp["dim"]

    def _is_serializable(self, value):
        try:
            tf.keras.utils.serialize_keras_object(value)
            return True
        except (TypeError, ValueError):
            return False

    def set_serializable_hp(self):
        self.serializable_hp = {
            key: value for key, value in self.hp.items() if self._is_serializable(value)
        }

    def get_config(self):
        # Return the configuration necessary to recreate this model
        config = super().get_config()
        config.update({"hp": self.serializable_hp})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BaseNetwork:
    """
    Represents a class for the machine learning processes used in training the
    global metric function across the patches. This object contains the metric
    neural network models as an attribute subclass via GlobalModel, otherwise
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
      calling the GlobalModel class to define the neural network architectures
      for the metric function, allowing pre-trained model imports. Sets up the
      training loss, and the logging.

    - evaluate_loss(self, x, training=True, return_constituents, val_print):
      Compute the model-predicted metric values for input points on the manifold
      (in patch 1 coordinates), and the respective training loss value.

    - grad(self, x):
      Compute the training loss gradient with respect to the neural network
      model parameters, used in training.

    - train(self, x_train, validate, x_val):
      Performs the training loop for the neural network global model of the
      metric function. Setting up batching, and performing logging and
      intermediate saving.

    - save(self, epoch, x_train, x_val, overwrite_old):
      Save the neural network metric model in its current form, along with the
      most recent training and validation data. Functionality to overwrite
      previous saves to reduce output memory requirements.

    - validate(self, validation_set):
      Perform the in training validation, using the independent validation data.
      Functionality included for identifying data batches which lead to numerical
      overflows and training failure.

    - check_log_to_wandb(self, epoch_loss_avg, val_losses, batch_idx, is_epoch_end):
      Set up the 'Weights & Biases' logging.
    """

    def __init__(self, hp, print_losses=False, restore_hps=False):
        self.hp = hp
        self.val_print = self.hp["val_print"]

        # Create a class variable for tracking the interim best loss in the
        # training process. Prevents extraneous saving.
        self.best_loss = None

        # Import the model
        if hp["saved_model"]:
            self.model = tfk.models.load_model(hp["saved_model_path"])
            # Overwrite the model's hyperparameters to the new ones
            if not restore_hps:
                # Update imported model implicit hps
                wandb.config.update(
                    {
                        "dim": self.model.hp["dim"],
                        "n_patches": self.model.hp["n_patches"],
                        "n_hidden": self.model.hp["n_hidden"],
                        "n_layers": self.model.hp["n_layers"],
                        "use_bias": self.model.hp["use_bias"],
                    },
                    allow_val_change=True,
                )  # ...these are overwritten by the imported model
                self.model.hp = hp
                self.model.set_serializable_hp()

        # Print model summary
        # print('Summary:',self.model.summary())
        # print('Submodel summary:',[psm.summary() for psm in self.patch_submodels])

        # Define the loss
        self.loss = None

        # Initialise the log dir
        self.log_dir = None

        # If log_interim true, set the log dir
        if self.hp["log_interim"]:
            # Append the WandB unique name for tracability
            log_dir = helper_functions.create_time_date_dir(
                base_path=self.hp["log_dir"],
                run_name=self.hp.run_identifiers[0],
            )
            # Add the log dir to the class scope
            self.log_dir = log_dir
            print("Logging to: ", self.log_dir)

    def evaluate_loss(
        self, x, training=True, return_constituents=False, val_print=True
    ):
        metric_pred = self.model(x, training=training)
        return self.loss.call(
            self.model, x, metric_pred, return_constituents, val_print
        )

    def grad(self, x):
        with tf.GradientTape() as tape:
            # Take the 0th element to filter out the loss constituents which
            # aren't used for training
            loss_value = self.evaluate_loss(
                x, training=True, return_constituents=False
            )[0]

        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, x_train, validate=True, x_val=None):
        def _handle_sigint(signal_received, frame):
            print("SIGINT received! Saving model and exiting...")
            if self.log_dir is not None:
                self.save(epoch, x_train, x_val, overwrite_old=True)
            exit(0)

        # Register the signal handler
        signal.signal(signal.SIGINT, _handle_sigint)

        # Batching
        # Calculate split sizes
        split_size = self.hp["batch_size"]
        remainder = len(x_train) % self.hp["batch_size"]
        split_sizes = [split_size] * (len(x_train) // self.hp["batch_size"])
        if remainder:
            split_sizes[-1] += remainder

        # Split the tensor
        batched_x_train = tf.split(x_train, num_or_size_splits=split_sizes, axis=0)

        self.optimiser = tfk.optimizers.Adam(
            learning_rate=self.hp["init_learning_rate"]
        )
        lr_schedule = schedulers.cosine_annealing

        # Keep results for plotting
        train_loss_results = []

        # Run the training loop
        for epoch in range(self.hp["epochs"]):
            epoch_loss_avg = tfk.metrics.Mean()

            # Adjust learning rate (scheduled)
            new_lr = lr_schedule(
                epoch,
                total_epochs=self.hp["epochs"],
                lr_init=self.hp["init_learning_rate"],
                lr_min=self.hp["min_learning_rate"],
            )
            self.optimiser.learning_rate.assign(new_lr)

            # Initialise the the number of batches skipped due to inv. error
            skip_number = 0
            for batch_idx, batch in enumerate(batched_x_train):
                # Training loop
                try:
                    loss_value, grads = self.grad(batch)

                except tf.errors.InvalidArgumentError as _:
                    skip_number += 1
                    if skip_number >= 10:
                        raise RuntimeError("Skipped too many batches!")
                    else:
                        # Raise a warning
                        warnings.warn(
                            f"Warning: skipping (training) batch due to inversion error. Number of skipped batches: {skip_number}",
                            RuntimeWarning,
                        )

                        if self.log_dir is not None:
                            # Convert the tensor to a NumPy array and then to a string
                            tensor_string = str(batch.numpy())

                            # Open a file in append mode and write the tensor to the file
                            with open(
                                f"{self.log_dir}/train_inv_error_batches.txt", "a+"
                            ) as f:
                                # Add a newline character for readability
                                f.write(tensor_string + "\n")

                            if (
                                self.hp["log_errors"] is not None
                                and epoch % self.hp["log_interval"] == 0
                            ):
                                self.model.save(
                                    f"{self.log_dir}/TRAIN_INV_ERROR_MODEL_DUMP_epoch_{epoch}_skip{skip_number}.keras"
                                )

                        # Advance to the next batch
                        continue

                self.optimiser.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss

                # End epoch
                train_loss_results.append(epoch_loss_avg.result())

                # Log every "wandb_log_freq" batches/epoch. (We need the +1 to
                # protect against div by 0)
                self.check_log_to_wandb(
                    epoch_loss_avg,
                    val_losses=None,
                    batch_idx=batch_idx,
                    is_epoch_end=False,
                )

            # Perform the validation
            if validate:
                val_losses = self.validate(x_val)
            else:
                val_losses = {}

            # Print the validation measures at each specified interval
            if epoch % self.hp["verbosity"] == 0:
                print(
                    "Epoch {:03d}: Loss: {:.3f}\n".format(
                        epoch + 1, epoch_loss_avg.result()
                    ),
                    flush=True,
                )

            # Logging
            if self.log_dir is not None and self.hp["log_interim"]:
                if (
                    self.hp["log_interval"] is not None
                    and epoch % self.hp["log_interval"] == 0
                ):
                    self.save(epoch, x_train, x_val)
                elif self.hp["track_best"] and (
                    self.best_loss is None
                    or epoch_loss_avg.result().numpy() <= self.best_loss
                ):
                    overwrite_old = not (self.hp["save_best_hist"])
                    self.best_loss = epoch_loss_avg.result().numpy()
                    self.save(epoch, x_train, x_val, overwrite_old=overwrite_old)

            # Log to WandB (if it's enabled in hyperparameters) -- always log
            # at end of epoch if logging enabled
            self.check_log_to_wandb(
                epoch_loss_avg, val_losses, batch_idx=None, is_epoch_end=True
            )
        return train_loss_results

    def save(self, epoch, x_train, x_val, overwrite_old=False):
        """
        Save training data, predictions, and model at a specific epoch.

        Parameters:
            epoch (int): The current epoch number (ignored if `overwrite_old` is True).
            x_train (tf.Tensor): The training data to save.
            overwrite_old (bool): If True, overwrite old files with fixed names.
            verbose (bool): If True, print log messages during saving.

        Raises:
            ValueError: If `x_train` is not provided or `epoch` is invalid.
        """
        # Validate `x_train`
        if x_train is None:
            raise ValueError("`x_train` must be provided.")

        if overwrite_old:
            base_name = "final"
        else:
            if epoch is None or epoch < 0:
                raise ValueError(
                    "Epoch must be a non-negative integer when `overwrite_old` is False."
                )
            base_name = f"epoch_{epoch}"

        file_paths = {
            "batch": f"{self.log_dir}/{base_name}_batch.npy",
            "val_batch": f"{self.log_dir}/{base_name}_val_batch.npy",
            "predictions": f"{self.log_dir}/{base_name}_batch_pred.npy",
            "model": f"{self.log_dir}/{base_name}_model.keras",
        }

        # Save training and validation batches
        np.save(file_paths["batch"], x_train.numpy(), allow_pickle=True)

        if x_val is not None:
            np.save(file_paths["val_batch"], x_val.numpy(), allow_pickle=True)

        # Save predictions
        predictions = self.model(x_train).numpy()
        np.save(file_paths["predictions"], predictions, allow_pickle=True)

        # Save model
        self.model.save(file_paths["model"])

    def validate(self, validation_set):
        keys = None
        values = []

        # Calculate split sizes
        split_size = self.hp["val_batch_size"]
        remainder = len(validation_set) % self.hp["val_batch_size"]
        split_sizes = [split_size] * (len(validation_set) // self.hp["val_batch_size"])
        if remainder:
            split_sizes[-1] += remainder

        # Split the tensor
        batched_x_val = tf.split(validation_set, num_or_size_splits=split_sizes, axis=0)

        # Iterate through all validation batches and find the loss constituents.
        val_batch_errors = 0
        for validation_batch in batched_x_val:
            try:
                constituent_batch_loss = self.evaluate_loss(
                    validation_batch,
                    training=False,
                    return_constituents=True,
                    val_print=self.val_print,
                )
            except tf.errors.InvalidArgumentError as _:
                val_batch_errors += 1
                if val_batch_errors > 10:
                    raise RuntimeError("Too many validation batches failed")
                warnings.warn(
                    f"Validation batch inversion failure. Skipping batch. Validation batch errors: {val_batch_errors}",
                    RuntimeWarning,
                )
                if self.log_dir is not None:
                    # Convert the tensor to a NumPy array and then to a string
                    tensor_string = str(validation_batch.numpy())

                    # Open a file in append mode and write the tensor to the file
                    with open(f"{self.log_dir}/val_inv_error_batches.txt", "a+") as f:
                        # Add a newline character for readability
                        f.write(tensor_string + "\n")
                    self.model.save(
                        f"{self.log_dir}/VAL_INV_ERROR_MODEL_DUMP_{val_batch_errors}.keras"
                    )
                continue

            if keys is None:
                # Take the first element of the evaluated batch loss tuple
                # (the 0th is simply the combined network loss)
                keys = constituent_batch_loss[1].keys()
                # Initialiise a subloss list per subloss key
                constituent_values = [[] for _ in range(len(keys))]

            # Append the values of the total losss, and batch average of each consistent losses to a list
            values.append(constituent_batch_loss[0])
            for key_idx, key in enumerate(keys):
                constituent_values[key_idx].append(
                    np.mean(constituent_batch_loss[1][key])
                )

        if len(values) == 0:
            raise RuntimeError("Not enough invertible validation batches!")

        # Check if keys are still None
        if keys is None:
            raise RuntimeError("The constituent losses cannot be None.")

        # Compute the mean total loss
        total_avg_loss = np.mean(values)
        # Compute the mean loss components, and zip with the keys
        constituent_avg_loss = zip(keys, np.mean(constituent_values, axis=1))

        return {"total_avg_val_loss": total_avg_loss} | dict(constituent_avg_loss)

    def check_log_to_wandb(
        self, epoch_loss_avg, val_losses=None, batch_idx=None, is_epoch_end=False
    ):
        """
        Logs metrics to Weights & Biases at specified intervals or at the
        end of an epoch.

        Parameters:
           - epoch_loss_avg: The average loss for the current epoch.
           - val_losses: Validation losses (optional, defaults to None).
           - batch_idx: The current batch index (optional, required for
                 batch-based logging).
           - is_epoch_end: Whether this is the end of the epoch (defaults to
                 False).

        Returns:
            None
        """
        if not self.hp.get("log_wandb", False):
            return  # Exit early if logging is disabled

        # Determine if we should log based on batch frequency
        log_because_batch = (
            (batch_idx + 1) % self.hp["wandb_log_freq"] == 0
            if batch_idx is not None
            else False
        )

        if log_because_batch or is_epoch_end:
            model_param_stats = network_analysis.get_model_weights_stats(self.model)

            # Prepare logs, adding `val_losses` if provided
            wandb_logs = (
                {"avg_train_loss": epoch_loss_avg.result()}
                | (val_losses or {})  # Use empty dict if `val_losses` is None
                | model_param_stats
            )

            wandb.log(wandb_logs)
