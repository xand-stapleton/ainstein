from __future__ import annotations

import warnings
from copy import deepcopy as dc
from pathlib import Path

import tensorflow as tf

import wandb
from configs.base import BaseConfig
from configs.loader import apply_runtime_overrides, load_config
from helper_functions import wandb_helper
from helper_functions.random import check_set_random_seeds
from network.base import BaseGlobalModel

tfk = tf.keras
tfk.backend.set_floatx("float64")


class BaseTrainerRunner:
    def __init__(
        self,
        config_file: Path,
        config_class: type[BaseConfig],
        runtime_config: dict,
        wandb_id: str | None = None,
    ):
        self.config_file = config_file
        self.config_class = config_class
        self.runtime_config = runtime_config
        self.wandb_id = wandb_id
        self.config_dict = load_config(self.config_file)

        apply_runtime_overrides(self.config_dict, self.runtime_config)
        self.config = config_class(**self.config_dict)
        self.config.model.np_seed, self.config.model.tf_seed = check_set_random_seeds(
            self.config.model.np_seed, self.config.model.tf_seed
        )

        self.train_sample = None
        self.val_sample = None
        self.train_sample_tf = None
        self.val_sample_tf = None
        self.train_sample_metrics_vecs = None
        self.val_sample_metrics_vecs = None

        self.network = None

    def _initialize_wandb(self, wandb_project_name: str):
        # If the original and the new seeds are the same, don't need to re-set
        # them.
        original_seed_np, original_seed_tf = (
            dc(self.config.model.np_seed),
            dc(self.config.model.tf_seed),
        )

        if self.wandb_id is not None:
            self.config_dict, self.train_sample = wandb_helper.restore_wandb(
                self.config_dict, self.wandb_id
            )

        # Try online first with a short timeout so HPC nodes without internet
        # fail fast.  If that fails, try offline mode so results are saved
        # locally (run `wandb sync` afterwards to upload).  If even that
        # fails, disable wandb entirely so training can still proceed.
        _wandb_ok = False
        try:
            wandb.init(
                project=wandb_project_name,
                config=self.config.model_dump(),
                id=self.wandb_id,
                resume="allow",
                settings=wandb.Settings(init_timeout=60),
            )
            _wandb_ok = True
        except Exception as e:
            warnings.warn(
                f"wandb online init failed ({e}); falling back to offline mode. "
                "Run `wandb sync` after the job completes to upload results."
            )
            try:
                wandb.init(
                    project=wandb_project_name,
                    config=self.config.model_dump(),
                    id=self.wandb_id,
                    resume="allow",
                    mode="offline",
                )
                _wandb_ok = True
            except Exception as e2:
                warnings.warn(
                    f"wandb offline init also failed ({e2}); disabling wandb for "
                    "this run. Training will proceed without wandb logging."
                )

        if _wandb_ok:
            self.config_dict = dict(wandb.config)
            self.config = self.config_class(**self.config_dict)
            self.config.metadata.run_name = wandb.run.name or ""
            self.config.metadata.run_id = wandb.run.id or "42"
        else:
            self.config.logging.log_wandb = False

        if (
            original_seed_np != self.config.model.np_seed
            or original_seed_tf != self.config.model.tf_seed
        ):
            self.config.model.np_seed, self.config.model.tf_seed = (
                check_set_random_seeds(
                    self.config.model.np_seed, self.config.model.tf_seed
                )
            )

    def _setup_supervised_network(self, model: BaseGlobalModel):
        if (
            self.config.training.init_learning_rate
            == self.config.training.min_learning_rate
        ):
            optimiser = tfk.optimizers.Adam(
                learning_rate=self.config.training.init_learning_rate
            )
        else:
            steps_per_epoch = max(
                1, self.config.data.num_samples // self.config.training.batch_size
            )
            total_steps = self.config.training.epochs * steps_per_epoch
            lr_schedule = tfk.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.config.training.init_learning_rate,
                decay_steps=total_steps,
                end_learning_rate=self.config.training.min_learning_rate,
                power=1.0,
            )
            optimiser = tfk.optimizers.Adam(learning_rate=lr_schedule)

        # Import the model
        if self.config.model.saved_model:
            self.supervised_model = tfk.models.load_model(
                self.config.model.saved_model_path
            )
            self.supervised_model.compile(optimizer=optimiser, loss="MSE")
            # Update imported model implicit hps
            self.config.geometry.dim = self.supervised_model.config.geometry.dim
            self.config.geometry.n_patches = (
                self.supervised_model.config.geometry.n_patches
            )
            self.config.model.n_hidden = self.supervised_model.config.model.n_hidden
            self.config.model.n_layers = self.supervised_model.config.model.n_layers
            self.config.model.activations = (
                self.supervised_model.config.model.activations
            )
            self.config.model.use_bias = self.supervised_model.config.model.use_bias
            self.supervised_model.config = self.config

        # Build the model
        else:
            self.supervised_model = model(self.config)
            self.supervised_model.compile(optimizer=optimiser, loss="MSE")

    def _tensorise_data(self):
        # Assumes the setup_data method of child class has
        # run first (if not restoring from WandB)
        assert self.train_sample is not None, (
            "Training sample has not been initialised for the specific problem yet!"
        )

        self.train_sample_tf = tf.convert_to_tensor(
            self.train_sample, dtype=tf.dtypes.float64
        )
        self.val_sample_tf = None
        if self.config.training.use_validation:
            self.val_sample_tf = tf.convert_to_tensor(
                self.val_sample, dtype=tf.dtypes.float64
            )

    def _tensorise_supervised_data(self):
        # Convert to tf objects
        self.train_sample_tf = tf.convert_to_tensor(
            self.train_sample, dtype=tf.dtypes.float64
        )
        self.train_sample_metrics_tf = tf.convert_to_tensor(
            tf.concat(self.train_sample_metrics_vecs, axis=1), dtype=tf.dtypes.float64
        )

        self.val_sample_tf = None
        self.val_sample_metrics_tf = None

        if self.val_sample is not None:
            self.val_sample_tf = tf.convert_to_tensor(
                self.val_sample, dtype=tf.dtypes.float64
            )
            self.val_sample_metrics_tf = tf.convert_to_tensor(
                tf.concat(self.val_sample_metrics_vecs, axis=1), dtype=tf.dtypes.float64
            )

        self.val_data = (self.val_sample_tf, self.val_sample_metrics_tf)

    def _train_model(self):
        """Train the model for non-specialised runners."""
        return self.network.train(
            x_train=self.train_sample_tf,
            validate=self.config.training.use_validation,
            x_val=self.val_sample_tf,
        )

    def run(self):
        assert self.network is not None, (
            "Network is not initialised. Did you mean to run runner._setup_network()?"
        )
        assert self.train_sample_tf is not None, (
            "Train samples are None. Did you mean to run runner.setup_data()?"
        )
        loss_hist = self._train_model()

        if self.config.logging.log_wandb:
            wandb.finish()

        return loss_hist, self.train_sample_tf, self.val_sample_tf

    def run_supervised(self, save: bool = True):
        assert self.supervised_model is not None, (
            "Supervised model not initialised. Did you mean to run runner._setup_supervised_network?"
        )

        # Check if validation data is None (need 2nd statement for more general experiments)
        validate = False if self.val_data is None or self.val_data[0] is None else True

        loss_hist = self.supervised_model.fit(
            self.train_sample_tf,
            self.train_sample_metrics_tf,
            batch_size=self.config.training.batch_size,
            epochs=self.config.training.epochs,
            verbose=self.config.training.verbosity,
            validation_data=self.val_data if validate else None,
            shuffle=True,
        )

        if save:
            expt = Path(
                f"runs/supervised_{self.config.metadata.run_name}_{self.config.metadata.run_id}/"
            )
            expt.mkdir(exist_ok=True, parents=True)
            self.supervised_model.save(f"{expt}/final_model.keras")
        return (
            self.supervised_model,
            loss_hist,
            self.train_sample_tf,
            self.train_sample_metrics_tf,
            self.val_data,
        )

    def visualise(self, train_losses=None) -> None:
        """
        Abstract class which will eventually be removed. It means
        runner.visualise() can be written in general in the run file without
        raising exception where specific visualisation is not implemented.
        """
        warnings.warn(
            "Visualisation is not implemented for this experiment yet!",
            RuntimeWarning,
        )
        return None
