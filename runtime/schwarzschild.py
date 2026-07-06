from __future__ import annotations

import numpy as np
import tensorflow as tf

from configs.schwarzschild import SchwarzschildConfig
from losses.schwarzschild import (
    TotalSchwarzschildLoss,
    make_supervised_metric_loss,
    relative_mse,
)
from network.schedulers import FloatScheduler
from network.schwarzschild import (SchwarzschildGlobalModel,
                                   SchwarzschildLocal2DNetwork,
                                   SchwarzschildNetwork,
                                   SchwarzschildSupervisedWrapper)
from runtime.base import BaseTrainerRunner
from sampling.ball import (BallSample, StereoSampleHemisphere,
                           StereoSampleSingleHemisphere)
from sampling.penrose import PenroseRegionMixtureSample, PenroseSample
from visualisation.schwarzschild import SchwarzschildVisualiser

tfk = tf.keras
tfk.backend.set_floatx("float64")


class SchwarzschildTrainerRunner(BaseTrainerRunner):
    config: SchwarzschildConfig

    def __init__(
        self,
        supervised=False,
        identity=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, config_class=SchwarzschildConfig)
        self._initialize_wandb("Ainstein_schwarzschild")
        self.supervised_model = None

        if supervised and self._is_local_2d_mode():
            raise NotImplementedError(
                "Supervised mode is not configured for local_2d_mode yet. "
                "Use unsupervised local runs."
            )

        self._reset_density_schedulers()
        self._sampling_epoch = 0
        self._setup_data()
        if not supervised:
            self._tensorise_data()
            self._setup_network()
        else:
            self._setup_supervised_data(identity)
            self._tensorise_supervised_data()
            self._setup_supervised_network(SchwarzschildGlobalModel)

        if hasattr(self.config, "model_specific") and hasattr(
            self.config.model_specific, "lorentzian"
        ):
            self.config.model_specific.lorentzian = (
                self.config.model_specific.lorentzian
            )
        else:
            self.config.model_specific.lorentzian = False  # or some other default

    @staticmethod
    def _make_density_scheduler(schedule_config, init_value: float):
        if schedule_config is None:
            return None
        return FloatScheduler(
            strategy=schedule_config.strategy,
            init_value=init_value,
            final_value=schedule_config.final_value,
            warmup_epochs=schedule_config.warmup_epochs,
            decay_rate=schedule_config.decay_rate,
            steps=schedule_config.steps,
        )

    def _reset_density_schedulers(self) -> None:
        self._density_power_R2_base = self.config.model_specific.density_power_R2
        self._density_power_S2_base = self.config.model_specific.density_power_S2
        self._density_power_R2_scheduler = self._make_density_scheduler(
            self.config.model_specific.density_power_R2_schedule,
            self._density_power_R2_base,
        )
        self._density_power_S2_scheduler = self._make_density_scheduler(
            self.config.model_specific.density_power_S2_schedule,
            self._density_power_S2_base,
        )

    def _apply_density_schedule(self, epoch: int):
        self._sampling_epoch = epoch
        if self._density_power_R2_scheduler is not None:
            self.config.model_specific.density_power_R2 = (
                self._density_power_R2_scheduler.get(epoch, self.config.training.epochs)
            )
        else:
            self.config.model_specific.density_power_R2 = self._density_power_R2_base

        if self._density_power_S2_scheduler is not None:
            self.config.model_specific.density_power_S2 = (
                self._density_power_S2_scheduler.get(epoch, self.config.training.epochs)
            )
        else:
            self.config.model_specific.density_power_S2 = self._density_power_S2_base

    def _is_local_2d_mode(self) -> bool:
        return bool(getattr(self.config.model_specific, "local_2d_mode", False))

    def _get_local_2d_patch_width(self) -> float:
        local_width = getattr(self.config.model_specific, "local_2d_patch_width", None)
        if local_width is not None:
            return float(local_width)
        # Backward-compatible default for existing local YAMLs.
        return float(self.config.model_specific.patch_width_S2)

    def _get_local_2d_density_power(self) -> float:
        local_density = getattr(
            self.config.model_specific, "local_2d_density_power", None
        )
        if local_density is not None:
            return float(local_density)
        # Backward-compatible default for existing local YAMLs.
        return float(self.config.model_specific.density_power_S2)

    def _sample_s2(self, num_samples: int, patch_width: float, density_power: float):
        """Sample S^2 coordinates in either global or local single-chart mode."""
        if getattr(self.config.model_specific, "local_single_s2_patch", False):
            local_patch_idx = int(
                getattr(self.config.model_specific, "local_s2_patch_idx", 0)
            )
            return StereoSampleSingleHemisphere(
                num_samples,
                patch_idx=local_patch_idx,
                patch_width=patch_width,
                density_power=density_power,
            )

        return StereoSampleHemisphere(
            num_samples,
            patch_width=patch_width,
            density_power=density_power,
        )

    def _sample_r2(self, num_samples: int):
        """Sample Penrose (T, X), optionally using the region curriculum."""
        if not getattr(
            self.config.model_specific, "use_penrose_region_curriculum", False
        ):
            return PenroseSample(
                num_samples,
                patch_width=self.config.model_specific.patch_width_R2,
                density_power=self.config.model_specific.density_power_R2,
            )

        progress = self._sampling_epoch / max(1, self.config.training.epochs - 1)
        fractions = self._penrose_curriculum_fractions(progress)
        return PenroseRegionMixtureSample(
            num_samples,
            patch_width=self.config.model_specific.patch_width_R2,
            density_power=self.config.model_specific.density_power_R2,
            exterior_fraction=fractions["exterior"],
            interior_fraction=fractions["interior"],
            horizon_fraction=fractions["horizon"],
            singularity_fraction=fractions["singularity"],
            horizon_width=self.config.model_specific.penrose_region_horizon_width,
            singularity_width=(
                self.config.model_specific.penrose_region_singularity_width
            ),
            interior_only=self.config.model_specific.penrose_interior_only,
        )

    def _penrose_curriculum_fractions(self, progress: float):
        progress = float(np.clip(progress, 0.0, 1.0))

        def interp(start_attr: str, end_attr: str) -> float:
            start = float(getattr(self.config.model_specific, start_attr))
            end = float(getattr(self.config.model_specific, end_attr))
            return start + (end - start) * progress

        return {
            "exterior": interp(
                "penrose_region_exterior_start", "penrose_region_exterior_end"
            ),
            "interior": interp(
                "penrose_region_interior_start", "penrose_region_interior_end"
            ),
            "horizon": interp(
                "penrose_region_horizon_start", "penrose_region_horizon_end"
            ),
            "singularity": interp(
                "penrose_region_singularity_start",
                "penrose_region_singularity_end",
            ),
        }

    def _setup_data(self, refresh: bool = False):
        if refresh or self.train_sample is None:
            if self._is_local_2d_mode():
                # Local 2D mode: direct ball-coordinate patch sampling.
                self.train_sample = BallSample(
                    self.config.data.num_samples,
                    dimension=self.config.geometry.dim,
                    patch_width=self._get_local_2d_patch_width(),
                    density_power=self._get_local_2d_density_power(),
                )
            else:
                # Elliptic sampling of the Penrose diagram
                train_sample_R2 = self._sample_r2(self.config.data.num_samples)
                # S^2 sampling: global mode keeps both hemispheres; local mode keeps
                # one selected stereographic chart only.
                train_sample_S2, train_patch_idx = self._sample_s2(
                    self.config.data.num_samples,
                    patch_width=self.config.model_specific.patch_width_S2,
                    density_power=self.config.model_specific.density_power_S2,
                )
                # Append patch_idx as a float column so it can be batched uniformly
                train_patch_idx_col = train_patch_idx.reshape(-1, 1).astype(np.float64)
                self.train_sample = np.concatenate(
                    [train_sample_R2, train_sample_S2, train_patch_idx_col], axis=1
                )  # shape (N, 5): [T, X, q1, q2, patch_idx]

        if self.config.training.use_validation:
            if refresh or self.val_sample is None:
                if self._is_local_2d_mode():
                    self.val_sample = BallSample(
                        self.config.training.num_val_samples,
                        dimension=self.config.geometry.dim,
                        patch_width=self._get_local_2d_patch_width(),
                        density_power=self._get_local_2d_density_power(),
                    )
                else:
                    val_sample_R2 = self._sample_r2(
                        self.config.training.num_val_samples
                    )
                    val_sample_S2, val_patch_idx = self._sample_s2(
                        self.config.training.num_val_samples,
                        patch_width=self.config.model_specific.patch_width_S2,
                        density_power=self.config.model_specific.density_power_S2,
                    )
                    val_patch_idx_col = val_patch_idx.reshape(-1, 1).astype(np.float64)
                    self.val_sample = np.concatenate(
                        [val_sample_R2, val_sample_S2, val_patch_idx_col], axis=1
                    )
        else:
            self.val_sample = None

    def _setup_supervised_data(self, identity):
        from geometry.schwarzschild import AnalyticMetric_R2S2

        # 4D intrinsic coords [T, X, q1, q2] — no patch transform needed
        train_coords_4d = self.train_sample[:, :4]

        # The supervised model (SchwarzschildSupervisedWrapper) outputs the
        # pulled-back 4D metric flattened to 16 components.  Targets must match.
        if identity:
            train_metric = AnalyticMetric_R2S2(
                train_coords_4d,
                identity=True,
                lorentzian=self.config.model_specific.lorentzian,
            )
        elif self.config.model_specific.lorentzian:
            train_metric = AnalyticMetric_R2S2(
                train_coords_4d,
                identity=False,
                lorentzian=True,
                m=self.config.model_specific.m,
            )
        else:
            raise ValueError(
                "Euclidean Schwarzschild metric not configured: "
                "cannot have identity=False and lorentzian=False."
            )
        self.train_sample_metrics_vecs = [tf.reshape(train_metric, [-1, 16])]

        if self.config.training.use_validation:
            val_coords_4d = self.val_sample[:, :4]
            if identity:
                val_metric = AnalyticMetric_R2S2(
                    val_coords_4d,
                    identity=True,
                    lorentzian=self.config.model_specific.lorentzian,
                )
            else:
                val_metric = AnalyticMetric_R2S2(
                    val_coords_4d,
                    identity=False,
                    lorentzian=True,
                    m=self.config.model_specific.m,
                )
            self.val_sample_metrics_vecs = [tf.reshape(val_metric, [-1, 16])]
        else:
            self.val_sample_metrics_tf = None
            self.val_data = None

    def _setup_supervised_network(self, model=None):
        """
        Override: wrap SchwarzschildGlobalModel with SchwarzschildSupervisedWrapper
        so that fit() trains MSE directly on the pulled-back 4D metric targets
        (16 flattened components) rather than on the raw 5D Cholesky output.

        If ``use_volume_scaling`` / ``use_area_measure_weight`` or
        ``use_metric_contraction`` is enabled in
        the config, the loss is upgraded to mirror the unsupervised Einstein loss:
        differences are weighted by sqrt(|det(g_analytic)|) and/or contracted with
        the inverse analytic metric respectively.  Otherwise the default
        ``relative_mse`` is used.
        """
        # Let the base class build / load / compile the raw SchwarzschildGlobalModel.
        super()._setup_supervised_network(SchwarzschildGlobalModel)
        lorentzian = getattr(self.config.model_specific, "lorentzian", False)
        use_volume_scaling = getattr(
            self.config.model_specific, "use_volume_scaling", None
        )
        use_area = (
            bool(use_volume_scaling)
            if use_volume_scaling is not None
            else bool(
                getattr(self.config.model_specific, "use_area_measure_weight", False)
            )
        )
        use_contraction = getattr(
            self.config.model_specific, "use_metric_contraction", False
        )
        raw = self.supervised_model
        wrapper = SchwarzschildSupervisedWrapper(raw, lorentzian=lorentzian)
        if use_area or use_contraction:
            loss_fn = make_supervised_metric_loss(
                use_area_measure_weight=use_area,
                use_metric_contraction=use_contraction,
            )
        else:
            # Default: scale-invariant relative MSE, no geometric weighting.
            loss_fn = relative_mse
        wrapper.compile(optimizer=raw.optimizer, loss=loss_fn)
        self.supervised_model = wrapper

    def run_supervised(self, save: bool = True):
        """
        Override: after training, save only the base SchwarzschildGlobalModel
        (not the pullback wrapper) so the saved file can be loaded directly for
        main (unsupervised) training.
        """
        result = super().run_supervised(save=False)
        if save:
            from pathlib import Path

            expt = Path(
                f"runs/supervised_{self.config.metadata.run_name}_"
                f"{self.config.metadata.run_id}/"
            )
            expt.mkdir(exist_ok=True, parents=True)
            self.supervised_model.base_model.save(f"{expt}/final_model.keras")
        return result

    def _setup_network(self):
        if self._is_local_2d_mode():
            self.network = SchwarzschildLocal2DNetwork(
                config=self.config, restore_hps=True
            )
        else:
            self.network = SchwarzschildNetwork(config=self.config, restore_hps=True)

    def _train_model(self):
        if self.config.training_stages:
            return self._train_model_staged()

        return self._train_current_config()

    def _apply_training_stage(self, stage) -> None:
        training_fields = ("epochs", "init_learning_rate", "min_learning_rate")
        for field in training_fields:
            value = getattr(stage, field)
            if value is not None:
                setattr(self.config.training, field, value)

        loss_fields = (
            "einstein_multiplier",
            "einstein_schedule",
            "kretschmann_schedule",
            "r2_det_schedule",
            "speciality_index_schedule",
            "killing_symmetry_schedule",
            "k_repeller_schedule",
            "speciality_index_rprofile_schedule",
        )
        for field in loss_fields:
            value = getattr(stage, field)
            if value is not None:
                setattr(self.config.loss, field, value)

        model_specific_fields = (
            "kretschmann_multiplier",
            "r2_det_loss_multiplier",
            "speciality_index_multiplier",
            "killing_symmetry_multiplier",
            "k_repeller_multiplier",
            "speciality_index_rprofile_multiplier",
            "use_volume_scaling",
            "use_area_measure_weight",
            "use_metric_contraction",
            "volume_scaling_loss_components",
            "metric_contraction_loss_components",
            "density_power_R2",
            "density_power_R2_schedule",
            "density_power_S2",
            "density_power_S2_schedule",
            "patch_width_R2",
            "patch_width_S2",
            "use_penrose_region_curriculum",
            "penrose_region_exterior_start",
            "penrose_region_interior_start",
            "penrose_region_horizon_start",
            "penrose_region_singularity_start",
            "penrose_region_exterior_end",
            "penrose_region_interior_end",
            "penrose_region_horizon_end",
            "penrose_region_singularity_end",
            "penrose_region_horizon_width",
            "penrose_region_singularity_width",
            "penrose_interior_only",
        )
        for field in model_specific_fields:
            value = getattr(stage, field)
            if value is not None:
                setattr(self.config.model_specific, field, value)

        self._reset_density_schedulers()

        assert self.network is not None, "Network not initialised"
        self.network.config = self.config
        self.network.model.config = self.config
        self.network.total_epochs = self.config.training.epochs
        self.network.current_epoch = 0
        self.network.best_loss = None
        self.network.loss = TotalSchwarzschildLoss(config=self.config)

    def _train_model_staged(self):
        assert self.network is not None, "Network not initialised"

        all_losses = []
        stages = self.config.training_stages or []
        for stage_idx, stage in enumerate(stages):
            print(
                f"Starting training stage {stage_idx + 1}/{len(stages)}: {stage.name}",
                flush=True,
            )
            self._apply_training_stage(stage)
            self._setup_data(refresh=True)
            self._tensorise_data()
            stage_losses = self._train_current_config()
            all_losses.extend(stage_losses)

        return all_losses

    def _train_current_config(self):
        # Provide a refresh callback so the shared network.train loop can
        # regenerate samples each epoch using the Schwarzschild scheduler.
        assert self.network is not None, "Network not initialised"

        def _refresh_fn(epoch: int):
            # apply schedule, regenerate numpy samples, then tensorise
            self._apply_density_schedule(epoch)
            self._setup_data(refresh=True)
            self._tensorise_data()
            # Return only the refreshed training set; keep validation static
            return self.train_sample_tf, None

        return self.network.train(
            x_train=None,
            validate=self.config.training.use_validation,
            x_val=self.val_sample_tf,
            refresh_fn=_refresh_fn,
        )

    def visualise(self, train_losses=None) -> None:
        if self._is_local_2d_mode():
            print(
                "Local 2D mode: skipping Schwarzschild embedding visualiser. "
                "Use visualisation/report_schwarzschild_local_lambda.py for summary outputs."
            )
            return None

        visualiser = SchwarzschildVisualiser(model_parent=self.network.log_dir)

        # Run computations
        visualiser.compute_quantities()
        visualiser.plot_points()
        visualiser.plot_all_pairs()
        visualiser.plot_analytic_metrics()
        visualiser.plot_metric_determinant()
        visualiser.plot_analytic_metric_determinant()
        visualiser.plot_kretschmann_scalar()
        visualiser.plot_analytic_kretschmann_scalar()
        visualiser.plot_kretschmann_r6()
        visualiser.plot_speciality_index()
        visualiser.evaluate_and_save_losses(train_losses=train_losses)

        return None
