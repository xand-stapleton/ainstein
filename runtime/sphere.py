from __future__ import annotations

import tensorflow as tf

from configs.sphere import SphereConfig
from helper_functions.helper_functions import cholesky_to_vec
from network.sphere import SphereGlobalModel, SphereNetwork
from runtime.base import BaseTrainerRunner
from sampling.ball import BallSample, CubeSample

tfk = tf.keras
tfk.backend.set_floatx("float64")


class SphereTrainerRunner(BaseTrainerRunner):
    config: SphereConfig

    def __init__(
        self,
        supervised=False,
        identity=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, config_class=SphereConfig)
        self._initialize_wandb("Ainstein_sphere")
        self.supervised_model = None
        self._setup_data()
        if not supervised:
            self._tensorise_data()
            self._setup_network()
        else:
            self._setup_supervised_data(identity)
            self._tensorise_supervised_data()
            self._setup_supervised_network(SphereGlobalModel)

    def _setup_data(self):
        if self.train_sample is None:
            if self.config.model_specific.ball:
                # Ball patch sampling
                self.train_sample = BallSample(
                    self.config.data.num_samples,
                    dimension=self.config.geometry.dim,
                    patch_width=self.config.data.patch_width,
                    density_power=self.config.model_specific.density_power,
                )
            else:
                # Cube patch sampling (full functionality unlikely enitrely compatible at present)
                self.train_sample = CubeSample(
                    self.config.data.num_samples,
                    dimension=self.config.geometry.dim,
                    width=self.config.data.patch_width,
                    density_power=self.config.model_specific.density_power,
                )

            if self.config.training.use_validation and self.val_sample is None:
                if self.config.model_specific.ball:
                    self.val_sample = BallSample(
                        self.config.data.num_samples,
                        dimension=self.config.geometry.dim,
                        patch_width=self.config.data.patch_width,
                        density_power=self.config.model_specific.density_power,
                    )
                else:
                    self.val_sample = CubeSample(
                        self.config.data.num_samples,
                        dimension=self.config.geometry.dim,
                        width=self.config.data.patch_width,
                        density_power=self.config.model_specific.density_power,
                    )

    def _setup_supervised_data(self, identity):
        from geometry.sphere import (AnalyticMetric_Sphere,
                                     PatchChange_Coordinates_Sphere)

        lorentzian = getattr(self.config.model_specific, "lorentzian", False)

        # Compute the sample analytic outputs
        train_sample_inputs = [self.train_sample]
        if self.config.geometry.n_patches == 2:
            train_sample_inputs.append(
                PatchChange_Coordinates_Sphere(self.train_sample)
            )
        elif self.config.geometry.n_patches > 2:
            raise SystemExit("codebase not yet configured for >2 patches...")
        train_sample_metrics = [
            AnalyticMetric_Sphere(
                ts,
                identity=identity,
                lorentzian=lorentzian,
            )
            for ts in train_sample_inputs
        ]

        # Convert to Cholesky vectors (vielbeins)
        self.train_sample_metrics_vecs = [
            cholesky_to_vec(tsm, lorentzian=lorentzian) for tsm in train_sample_metrics
        ]

        # Generate validation data if required
        if self.config.training.use_validation:
            val_sample_inputs = [self.val_sample]
            if self.config.geometry.n_patches > 1:
                val_sample_inputs.append(
                    PatchChange_Coordinates_Sphere(self.val_sample)
                )

            val_sample_metrics = [
                AnalyticMetric_Sphere(vs, identity=identity) for vs in val_sample_inputs
            ]
            # Convert to Cholesky vectors (vielbeins)
            self.val_sample_metrics_vecs = [
                cholesky_to_vec(vsm, lorentzian=lorentzian)
                for vsm in val_sample_metrics
            ]

    def _setup_network(self):
        self.network = SphereNetwork(
            config=self.config,
        )
