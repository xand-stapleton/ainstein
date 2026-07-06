from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import keras
from pydantic import BaseModel


@keras.saving.register_keras_serializable()
class KerasSerialisableObject:
    def get_config(self):
        # We must change any PathLib paths to strings at the point of
        # serialisation to stop TF complaining
        config_dict = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }
        return config_dict


class GeometryConfig(BaseModel, KerasSerialisableObject):
    dim: int = 4
    n_patches: int = 1
    overlap_upperwidth: float = 0.1
    einstein_constant: float = 0.0


class DataConfig(BaseModel, KerasSerialisableObject):
    patch_width: float | None = None
    num_samples: int = 1000


class FloatScheduleConfig(BaseModel, KerasSerialisableObject):
    """Configuration for scheduling a single loss multiplier."""

    strategy: Literal["constant", "linear", "exponential", "cosine", "step"] = "constant"
    final_value: float | None = None
    warmup_epochs: int = 0
    decay_rate: float = 0.96
    steps: int = 1


class LossConfig(BaseModel, KerasSerialisableObject):
    """Loss function configuration with optional per-multiplier scheduling."""

    einstein_multiplier: float = 1.0
    overlap_multiplier: float = 1.0
    finiteness_multiplier: float = 1.0

    # Optional schedulers for each multiplier (None = no scheduling)
    einstein_schedule: FloatScheduleConfig | None = None
    overlap_schedule: FloatScheduleConfig | None = None
    finiteness_schedule: FloatScheduleConfig | None = None


class FinitenessConfig(BaseModel, KerasSerialisableObject):
    finite_centre: float = 140
    finite_width: float = 160
    finite_sharpness: float = 8
    finite_height: float = 100
    finite_slope: float = 0.2


class ModelConfig(BaseModel, KerasSerialisableObject):
    experiment: Literal["sphere", "schwarzschild", "lens"] = "sphere"
    saved_model: bool = False
    saved_model_path: Optional[str] = None
    saved_model_weight_noise: float = 0.0
    n_hidden: int = 128
    n_layers: int = 4
    activations: str = "gelu"
    use_bias: bool = True
    np_seed: Optional[int] = None
    tf_seed: Optional[int] = None
    dtype: str = "float64"


class TrainingConfig(BaseModel, KerasSerialisableObject):
    epochs: int = 100
    batch_size: int = 100
    init_learning_rate: float = 0.005
    min_learning_rate: float = 0.0005
    use_validation: bool = True
    val_print: bool = False
    num_val_samples: int = 500
    val_batch_size: int = 100
    verbosity: int = 1


class LoggingConfig(BaseModel, KerasSerialisableObject):
    log_wandb: bool = False
    wandb_log_freq: int = 10
    log_interim: bool = True
    log_dir: Path = Path("runs")
    log_interval: int | None = None
    track_best: bool = True
    save_best_hist: bool = False
    log_errors: bool = True
    print_batch_losses: bool = False


class MetadataConfig(BaseModel, KerasSerialisableObject):
    run_name: str | None = ""
    run_id: str | None = None
    misc: dict | None = {}


class ModelSpecific(BaseModel):
    """
    Just a type helper for an arbitrary model specific class...
    """

    pass


class VisualisationConfig(BaseModel):
    """
    Class to hold the optional visualisation information
    """

    visualise: bool = True


class BaseConfig(BaseModel, KerasSerialisableObject):
    geometry: GeometryConfig = GeometryConfig()
    data: DataConfig = DataConfig()
    loss: LossConfig = LossConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    finiteness: FinitenessConfig = FinitenessConfig()
    logging: LoggingConfig = LoggingConfig()
    metadata: MetadataConfig = MetadataConfig()
    model_specific: ModelSpecific = ModelSpecific()
    visualisation: VisualisationConfig = VisualisationConfig()

    # This is a special PyDantic flag. We do this so that the model is mutable
    # and we can get validation when changing attributes without using
    # model = self.model_copy({foo: "bar"})
    model_config = {"validate_assignment": True}
