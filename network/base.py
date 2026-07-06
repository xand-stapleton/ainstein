from __future__ import annotations

import sys

import tensorflow as tf
import yaml

tfk = tf.keras
tfk.backend.set_floatx("float64")
import signal
import warnings
from pathlib import Path
from sys import exit
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar

import numpy as np
import tensorflow as tf

import wandb
from configs.base import BaseConfig
from configs.loader import find_config_class
from helper_functions import helper_functions
from network import network_analysis, schedulers

T = TypeVar("T", bound="SerialisableModel")


class SerialisableModel(tf.keras.Model):
    """
    A TensorFlow Keras Model subclass that supports serialisation of a
    Pydantic-based configuration (`BaseConfig`) along with additional model
    parameters like `n_out`.

    This class overrides the standard Keras `get_config` and `from_config`
    methods to ensure the config is safely converted into TensorFlow Keras
    serialisable types for saving and loading.
    """

    config: BaseConfig
    n_out: Optional[int]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serialisable configuration of the model.

        Extends the base `get_config` by including a serialised version of the
        Pydantic config (`self.config`) and the optional `n_out` attribute,
        ensuring all values are TensorFlow Keras serialisable types.
        """
        network_config = super().get_config()

        # serialise the config safely to TF Keras serialisable types
        model_config = self._make_serialisable(self.config.model_dump())
        network_config.update({"model_config": model_config})

        if hasattr(self, "n_out") and self.n_out is not None:
            network_config.update({"n_out": self.n_out})

        return network_config

    @staticmethod
    def _serialise_value(value: Any) -> Any:
        """
        Convert a value into a TensorFlow Keras serialisable form.

        Handles common non-serialisable types such as pathlib.Path by
        converting them to strings, recursively processes dicts and lists, and
        converts other objects with a __str__ method to their string
        representation if they are not basic serialisable types.
        """
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, dict):
            return SerialisableModel._make_serialisable(value)
        elif isinstance(value, list):
            return [SerialisableModel._serialise_value(v) for v in value]
        elif hasattr(value, "__str__") and not isinstance(
            value, (str, int, float, bool, type(None))
        ):
            return str(value)
        else:
            return value

    @staticmethod
    def _make_serialisable(d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a dictionary's values to TensorFlow Keras serialisable types.
        """
        return {k: SerialisableModel._serialise_value(v) for k, v in d.items()}

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        """
        Reconstruct the model from a serialised config dictionary.

        Used by Keras during model loading to recreate the model
        from the saved configuration.
        """
        # Fallback to sphere to allow backward-compatibility
        experiment_name = config["model_config"]["model"].get("experiment", "sphere")
        experiment_config_class = find_config_class(experiment_name)
        model_config = experiment_config_class(**config["model_config"])
        return cls(
            config=model_config,
            **{
                **({"n_out": config["n_out"]} if "n_out" in config else {}),
            },
        )


@tfk.utils.register_keras_serializable()
class BasePatchSubmodel(SerialisableModel):
    """
    Represents a class for the neural network model which represents the metric
    function in a patch, these are trained across the patches to satify the Einstein equation.
    Inherits from the tf.keras.Model class.
    """

    def __init__(
        self,
        config: BaseConfig,
        n_out: int,
        input_dim: int | None = None,
        output_bias_init=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_out = n_out
        # Define hyperparameters
        self.config = config
        self.dim = self.config.geometry.dim
        self.n_layers = self.config.model.n_layers
        self.n_hidden = self.config.model.n_hidden
        self.activations = self.config.model.activations
        self.use_bias = self.config.model.use_bias

        # Define subnetwork architecture
        # input_dim overrides config dim (e.g. for embedding architectures)
        in_dim = input_dim if input_dim is not None else self.dim
        inputs = tfk.layers.Input(shape=(in_dim,), dtype=tf.float64)
        x = tfk.layers.Dense(
            self.n_hidden, activation=self.activations, use_bias=self.use_bias
        )(inputs)
        for _ in range(self.n_layers - 2):
            x = tfk.layers.Dense(
                self.n_hidden, activation=self.activations, use_bias=self.use_bias
            )(x)
        # output_bias_init: optional Keras initialiser for a trainable output bias
        # (e.g. to break symmetry at initialisation without a permanent offset).
        if output_bias_init is not None:
            outputs = tfk.layers.Dense(
                n_out, activation=None, use_bias=True, bias_initializer=output_bias_init
            )(x)
        else:
            outputs = tfk.layers.Dense(n_out, activation=None, use_bias=False)(x)

        self.submodel = tfk.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.submodel(inputs)


@tfk.utils.register_keras_serializable()
class BaseGlobalModel(SerialisableModel):
    """
    Represents a class for the global model of the metric function across the
    patches, these are trained to satify the Einstein equation. Inherits from
    the tf.keras.Model class.
    """

    def __init__(self, config: BaseConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.dim = self.config.geometry.dim


class BaseNetwork:
    """
    Represents a class for the machine learning processes used in training the
    global metric function across the patches. This object contains the metric
    neural network models as an attribute subclass via GlobalModel, otherwise
    containing functionality for training, validating, saving, logging.
    """

    def __init__(self, config: BaseConfig, restore_hps=False) -> None:
        self.config = config
        self.val_print = self.config.training.val_print

        # Create a class variable for tracking the interim best loss in the
        # training process. Prevents extraneous saving.
        self.best_loss = None

        # Track current epoch for multiplier scheduling
        self.current_epoch = 0
        self.total_epochs = self.config.training.epochs

        # Import the model
        if config.model.saved_model:
            assert config.model.saved_model_path is not None, (
                "Saved model is set to true, but no path provided"
            )
            self.model = tfk.models.load_model(config.model.saved_model_path)
            weight_noise = float(
                getattr(config.model, "saved_model_weight_noise", 0.0) or 0.0
            )
            if weight_noise > 0.0:
                for weight in self.model.weights:
                    weight_dtype = tf.as_dtype(weight.dtype)
                    if weight_dtype.is_floating:
                        weight.assign_add(
                            tf.random.normal(
                                tf.shape(weight),
                                mean=tf.cast(0.0, weight_dtype),
                                stddev=tf.cast(weight_noise, weight_dtype),
                                dtype=weight_dtype,
                            )
                        )
            # Overwrite the model's hyperparameters to the new ones
            if restore_hps:
                self._warn_if_mismatch(
                    "geometry.dim",
                    self.config.geometry.dim,
                    self.model.config.geometry.dim,
                )
                self.config.geometry.dim = self.model.config.geometry.dim

                self._warn_if_mismatch(
                    "geometry.n_patches",
                    self.config.geometry.n_patches,
                    self.model.config.geometry.n_patches,
                )
                self.config.geometry.n_patches = self.model.config.geometry.n_patches

                self._warn_if_mismatch(
                    "model.n_hidden",
                    self.config.model.n_hidden,
                    self.model.config.model.n_hidden,
                )
                self.config.model.n_hidden = self.model.config.model.n_hidden

                self._warn_if_mismatch(
                    "model.n_layers",
                    self.config.model.n_layers,
                    self.model.config.model.n_layers,
                )
                self.config.model.n_layers = self.model.config.model.n_layers

                self._warn_if_mismatch(
                    "model.use_bias",
                    self.config.model.use_bias,
                    self.model.config.model.use_bias,
                )
                self.config.model.use_bias = self.model.config.model.use_bias

            # Re-synchronise both configs
            self.model.config = self.config

            if self.config.logging.log_wandb:
                wandb.config.update(
                    self.config.model_dump(),
                    allow_val_change=True,
                )

        # Define the loss
        self.loss = None

        # Initialise the log dir
        self.log_dir = None

        # If log_interim true, set the log dir
        if self.config.logging.log_interim:
            # Append the WandB unique name for tracability
            self.log_dir = helper_functions.create_time_date_dir(
                base_path=self.config.logging.log_dir,
                run_name=self.config.metadata.run_name,
            )
            print("Logging to: ", self.log_dir)

            # Save the effective hyperparameters (after all overrides) to the
            # run directory so it is always clear exactly what was used.
            def _to_yaml_serializable(obj):
                if isinstance(obj, dict):
                    return {k: _to_yaml_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_to_yaml_serializable(v) for v in obj]
                if isinstance(obj, Path):
                    return str(obj)
                return obj

            hps_path = Path(self.log_dir) / "hps_used.yaml"
            with open(hps_path, "w") as _f:
                yaml.dump(
                    _to_yaml_serializable(self.config.model_dump()),
                    _f,
                    default_flow_style=False,
                    sort_keys=False,
                )

    def evaluate_loss(
        self, x, training=True, return_constituents=False, val_print=True
    ):
        """Evaluate loss, updating scheduled multipliers if configured."""
        # Update scheduled multipliers if any are active
        if hasattr(self.loss, "set_epoch") and not tf.inside_function():
            self.loss.set_epoch(self.current_epoch, self.total_epochs)

        metric_pred = self.model(x, training=training)
        return self.loss.call(
            self.model, x, metric_pred, return_constituents, val_print
        )

    @staticmethod
    def _project_gradients_to_variable_dtype(grads, variables):
        """Project complex autodiff gradients onto the real trainable variables."""
        projected = []
        for grad, variable in zip(grads, variables):
            if grad is None:
                projected.append(None)
                continue

            variable_dtype = tf.as_dtype(variable.dtype)
            if grad.dtype.is_complex and not variable_dtype.is_complex:
                grad = tf.math.real(grad)
            projected.append(grad)
        return projected

    def grad(self, x):
        with tf.GradientTape() as tape:
            # Take the 0th element to filter out the loss constituents which
            # aren't used for training
            loss_value = self.evaluate_loss(
                x, training=True, return_constituents=False
            )[0]

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        grads = self._project_gradients_to_variable_dtype(
            grads, self.model.trainable_variables
        )
        return loss_value, grads

    def train(
        self,
        x_train,
        validate=True,
        x_val=None,
        refresh_fn: Optional[
            Callable[[int], Tuple[tf.Tensor, Optional[tf.Tensor]]]
        ] = None,
    ):
        print("Training...")

        def _handle_sigint(signal_received, frame):
            print("SIGINT received! Saving model and exiting...")
            if self.log_dir is not None:
                # attempt to save the most recent epoch's data if available
                try:
                    self.save(epoch, last_x_train, last_x_val, overwrite_old=True)
                except Exception:
                    pass
            exit(0)

        # Register the signal handler
        signal.signal(signal.SIGINT, _handle_sigint)

        # Keep track of the last seen x_train/x_val for safe saving on SIGINT
        last_x_train = x_train
        last_x_val = x_val

        self.optimiser = tfk.optimizers.Adam(
            learning_rate=self.config.training.init_learning_rate,
            clipnorm=1.0,  # prevent gradient explosions (e.g. Kretschmann log-ratio near K→0)
        )
        lr_schedule = schedulers.cosine_annealing

        # JIT-compile the full forward + backward + optimiser update into a
        # single TF graph, eliminating Python overhead on every batch.
        #
        # ``@tf.function`` is skipped only when the Apple Metal GPU is visible,
        # because Metal's graph optimiser crashes on float64 ops.  The standard
        # macOS setup in ``run.py`` hides the Metal GPU (float64 ⇒ CPU), in
        # which case graph compilation is safe and gives a large speedup —
        # especially for the forward-mode ``ricci_kernel="optimised"`` path,
        # whose nested ``ForwardAccumulator`` contexts carry significant
        # Python overhead when run eagerly.
        _visible_gpus = tf.config.get_visible_devices("GPU")
        _on_apple_metal = sys.platform == "darwin" and any(
            "Metal" in getattr(d, "name", "") for d in _visible_gpus
        )
        _loss_has_schedules = any(
            name.endswith("_scheduler") and getattr(self.loss, name) is not None
            for name in dir(self.loss)
        )

        def _train_step_impl(batch):
            with tf.GradientTape() as tape:
                loss_value = self.evaluate_loss(
                    batch, training=True, return_constituents=False
                )[0]
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            grads = self._project_gradients_to_variable_dtype(
                grads, self.model.trainable_variables
            )
            self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss_value

        if _on_apple_metal or _loss_has_schedules:
            compiled_train_step = _train_step_impl
        else:
            compiled_train_step = tf.function(_train_step_impl)

        # Keep results for plotting
        train_loss_results = []

        # Run the training loop
        for epoch in range(self.config.training.epochs):
            # If a refresh function is provided, call it to obtain per-epoch data
            if refresh_fn is not None:
                cur_x_train, cur_x_val = refresh_fn(epoch)
            else:
                cur_x_train, cur_x_val = x_train, x_val

            # If the refresh function returns None for either dataset, keep
            # the last-known value so validation can be kept static.
            if cur_x_train is None:
                cur_x_train = last_x_train
            if cur_x_val is None:
                cur_x_val = last_x_val

            if cur_x_train is None:
                raise ValueError(
                    "Training data is None after refresh. Check the Schwarzschild "
                    "refresh callback and density schedule."
                )
            if len(cur_x_train) == 0:
                raise ValueError(
                    "Training data is empty after refresh. Check the Schwarzschild "
                    "sampling parameters and density schedule."
                )
            if validate and cur_x_val is None:
                raise ValueError(
                    "Validation data is None but validation is enabled. Ensure the "
                    "initial validation set is passed into train()."
                )
            if validate and len(cur_x_val) == 0:
                raise ValueError(
                    "Validation data is empty. Check the initial Schwarzschild "
                    "validation sample generation."
                )

            last_x_train = cur_x_train
            last_x_val = cur_x_val

            # Dataset pipeline: sample-level shuffle + batch + GPU prefetch.
            # Replaces manual tf.split + Python random.sample, keeping the GPU fed
            # while the CPU prepares the next batch in parallel.
            train_dataset = (
                tf.data.Dataset.from_tensor_slices(cur_x_train)
                .shuffle(buffer_size=len(cur_x_train), reshuffle_each_iteration=True)
                .batch(self.config.training.batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )
            # Update current epoch for multiplier scheduling
            self.current_epoch = epoch
            if hasattr(self.loss, "set_epoch"):
                self.loss.set_epoch(self.current_epoch, self.total_epochs)

            epoch_loss_avg = tfk.metrics.Mean()

            # Adjust learning rate (scheduled)
            new_lr = lr_schedule(
                epoch,
                total_epochs=self.config.training.epochs,
                lr_init=self.config.training.init_learning_rate,
                lr_min=self.config.training.min_learning_rate,
            )
            self.optimiser.learning_rate.assign(new_lr)

            # Initialise the the number of batches skipped due to inv. error
            skip_number = 0
            for batch_idx, batch in enumerate(train_dataset):
                # Training loop
                try:
                    loss_value = compiled_train_step(batch)

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

                            if self.config.logging.log_errors is not None:
                                self.model.save(
                                    f"{self.log_dir}/TRAIN_INV_ERROR_MODEL_DUMP_epoch_{epoch}_skip{skip_number}.keras"
                                )

                        # Advance to the next batch
                        continue

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss

                # Log every "wandb_log_freq" batches/epoch. (We need the +1 to
                # protect against div by 0)
                self.check_log_to_wandb(
                    epoch_loss_avg,
                    val_losses=None,
                    batch_idx=batch_idx,
                    is_epoch_end=False,
                )

            # Record one loss value per epoch (the mean over all batches)
            train_loss_results.append(epoch_loss_avg.result())

            # Perform the validation on the current epoch's validation set
            if validate:
                val_losses = self.validate(cur_x_val)
            else:
                val_losses = {}

            # Print the validation measures at each specified interval
            if (
                self.config.training.verbosity != 0
                and epoch % self.config.training.verbosity == 0
            ):
                print(
                    "Epoch {:03d}: Loss: {:.3g}\n".format(
                        epoch + 1, epoch_loss_avg.result()
                    ),
                    flush=True,
                )

            # Logging
            if self.log_dir is not None and self.config.logging.log_interim:
                if (
                    self.config.logging.log_interval is not None
                    and epoch % self.config.logging.log_interval == 0
                ):
                    self.save(epoch, cur_x_train, cur_x_val)
                elif self.config.logging.track_best and (
                    self.best_loss is None
                    or epoch_loss_avg.result().numpy() <= self.best_loss
                ):
                    overwrite_old = not (self.config.logging.save_best_hist)
                    self.best_loss = epoch_loss_avg.result().numpy()
                    self.save(
                        epoch, cur_x_train, cur_x_val, overwrite_old=overwrite_old
                    )

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
        split_size = self.config.training.val_batch_size
        remainder = len(validation_set) % self.config.training.val_batch_size
        split_sizes = [split_size] * (
            len(validation_set) // self.config.training.val_batch_size
        )
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
        if not self.config.logging.log_wandb:
            return  # Exit early if logging is disabled

        # Determine if we should log based on batch frequency
        log_because_batch = (
            (batch_idx + 1) % self.config.logging.wandb_log_freq == 0
            if batch_idx is not None
            else False
        )

        if log_because_batch or is_epoch_end:
            # Only pull weight stats at epoch end – get_weights() forces a
            # GPU->CPU sync; doing it every wandb_log_freq batches wastes ~90%
            # of those syncs since weights barely change mid-epoch.
            model_param_stats = (
                network_analysis.get_model_weights_stats(self.model)
                if is_epoch_end
                else {}
            )

            # Prepare logs, adding `val_losses` if provided
            wandb_logs = (
                {"avg_train_loss": epoch_loss_avg.result()}
                | (val_losses or {})  # Use empty dict if `val_losses` is None
                | model_param_stats
            )

            wandb.log(wandb_logs)

    @staticmethod
    def _warn_if_mismatch(name, old_value, new_value):
        if old_value != new_value:
            warnings.warn(
                f"Saved model hyperparameter mismatch for '{name}': existing value {old_value!r} "
                f"differs from restored value {new_value!r}. Overwriting.",
                RuntimeWarning,
                stacklevel=2,
            )
