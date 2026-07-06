from __future__ import annotations

import argparse
import json
from copy import deepcopy


def parse_bool_map(value: str) -> dict[str, bool]:
    """Parse JSON or comma-separated key=value booleans for CLI overrides."""
    def parse_bool(raw):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int) and raw in (0, 1):
            return bool(raw)
        if isinstance(raw, str):
            raw_lower = raw.strip().lower()
            if raw_lower in ("true", "1", "yes"):
                return True
            if raw_lower in ("false", "0", "no"):
                return False
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got {raw!r}.")

    if value.strip().startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError("Expected a JSON object.")
        return {str(key): parse_bool(val) for key, val in parsed.items()}

    result = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "Expected comma-separated key=value pairs."
            )
        key, raw_val = item.split("=", 1)
        result[key.strip()] = parse_bool(raw_val)
    return result


def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Add argument for hyperparameters file
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        help="Path to the hyperparameters YAML file (default: hyperparameters/hps.yaml)",
        required=True,
    )

    # Supervised runs additional hyperparameters
    parser.add_argument(
        "--supervised",
        help="Train a supervised model.",
        action="store_true",
    )
    parser.add_argument(
        "--identity",
        help="In the supervised model, use the identity target",
        action="store_true",
    )

    # Logging and sweep arguments
    parser.add_argument(
        "--log_wandb",
        type=bool,
        help="Enable or disable logging to Weights and Biases (WandB)",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Weights and biases ID. If None, start a fresh WandB run.",
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        help="Path to saved model. If None, initialize a new one",
    )

    # Data and patch sampling configuration
    parser.add_argument(
        "--dim",
        type=int,
        help="Dimensionality of the data",
    )
    parser.add_argument(
        "--ball",
        type=bool,
        help="True for ball sampling shape, False for cube",
    )
    parser.add_argument(
        "--patch_width",
        type=float,
        help="1D size of the patch (radius for ball, width for cube)",
    )
    parser.add_argument(
        "--density_power",
        type=float,
        help="Skew factor for sampling",
    )
    parser.add_argument(
        "--density_power_R2",
        type=float,
        help="Skew factor for sampling",
    )
    parser.add_argument(
        "--density_power_S2",
        type=float,
        help="Skew factor for sampling",
    )

    # Training and validation samples
    parser.add_argument(
        "--num_samples", type=int, help="Number of samples for training"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--validate", type=bool, help="Enable validation")
    parser.add_argument("--val_print", type=bool, help="Print validation results")
    parser.add_argument(
        "--num_val_samples",
        type=int,
        help="Number of samples for validation",
    )
    parser.add_argument("--val_batch_size", type=int, help="Batch size for validation")

    # Loss and metric configuration
    parser.add_argument(
        "--einstein_constant",
        type=float,
        help="Multiplier for the metric in the Einstein loss",
    )
    parser.add_argument(
        "--use_volume_scaling",
        type=bool,
        help="Weight Schwarzschild per-sample loss norms by sqrt(abs(det(g)))",
    )
    parser.add_argument(
        "--use_area_measure_weight",
        type=bool,
        help="Legacy alias for --use_volume_scaling",
    )
    parser.add_argument(
        "--use_metric_contraction",
        type=bool,
        help="Contract Schwarzschild tensor/vector loss norms with the inverse metric",
    )
    parser.add_argument(
        "--volume_scaling_loss_components",
        type=parse_bool_map,
        help=(
            "Per-loss overrides for volume scaling, e.g. "
            "'einstein=true,r2_det=false'"
        ),
    )
    parser.add_argument(
        "--metric_contraction_loss_components",
        type=parse_bool_map,
        help=(
            "Per-loss overrides for metric contraction, e.g. "
            "'einstein=true,killing_symmetry=false'"
        ),
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")

    # Network architecture
    parser.add_argument(
        "--n_hidden", type=int, help="Number of hidden units in each layer"
    )
    parser.add_argument("--n_layers", type=int, help="Number of layers in the network")
    parser.add_argument("--activations", type=str, help="Activation function to use")
    parser.add_argument("--use_bias", type=bool, help="Use bias in network layers")

    # Learning parameters
    parser.add_argument(
        "--init_learning_rate",
        type=float,
        help="Initial learning rate for optimizer",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        help="Minimum learning rate",
    )

    # Logging parameters
    parser.add_argument("--verbosity", type=int, help="Logging verbosity level")
    parser.add_argument(
        "--log_interim",
        type=bool,
        help="Log interim results during training",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Directory for saving logs and outputs",
    )
    parser.add_argument(
        "--print_batch_losses",
        type=bool,
        help="Print breakdown of loss terms",
    )
    # Loss multiplier factors
    parser.add_argument(
        "--einstein_multiplier",
        type=float,
        help="Multiplier factor for Einstein loss",
    )
    parser.add_argument(
        "--kretschmann_multiplier",
        type=float,
        help="Multiplier factor for Kretschmann loss",
    )

    parser.add_argument(
        "--r2_det_loss_multiplier",
        type=float,
        help="Multiplier factor for the R2 determinant barrier loss",
    )
    parser.add_argument(
        "--speciality_index_multiplier",
        type=float,
        help="Multiplier factor for the speciality-index S=1 type-D loss",
    )
    parser.add_argument(
        "--k_repeller_multiplier",
        type=float,
        help="Multiplier factor for the K-repeller loss",
    )
    parser.add_argument(
        "--k_repeller_epsilon",
        type=float,
        help="Denominator epsilon for the K-repeller loss",
    )
    parser.add_argument(
        "--speciality_index_rprofile_mode",
        type=str,
        choices=["value", "profile", "gradient", "variance", "discriminant", "hybrid"],
        help=(
            "Speciality-index profile mode; 'profile' fits a smooth non-constant "
            "Zipoy-Voorhees/gamma-inspired S profile, 'gradient' uses clipped "
            "full-coordinate S-gradient power ('variance' is a legacy alias)"
        ),
    )
    parser.add_argument(
        "--speciality_index_rprofile_multiplier",
        type=float,
        help="Multiplier factor for the speciality-index r-profile loss",
    )
    parser.add_argument(
        "--speciality_index_rprofile_centre",
        type=float,
        help="Mean target value for the speciality-index r-profile",
    )
    parser.add_argument(
        "--speciality_index_rprofile_epsilon",
        type=float,
        help="Denominator epsilon for the speciality-index r-profile loss",
    )

    parser.add_argument(
        "--overlap_multiplier",
        type=float,
        help="Multiplier factor for overlap loss",
    )
    parser.add_argument(
        "--finiteness_multiplier",
        type=float,
        help="Multiplier factor for finiteness loss",
    )

    # Loss multiplier schedules
    parser.add_argument(
        "--einstein_schedule.strategy",
        type=str,
        dest="einstein_schedule_strategy",
        help="Schedule strategy for Einstein multiplier (constant, linear, exponential, cosine, step)",
    )
    parser.add_argument(
        "--einstein_schedule.final_value",
        type=float,
        dest="einstein_schedule_final_value",
        help="Final value for Einstein multiplier schedule",
    )
    parser.add_argument(
        "--einstein_schedule.warmup_epochs",
        type=int,
        dest="einstein_schedule_warmup_epochs",
        help="Warmup epochs for Einstein multiplier schedule",
    )
    parser.add_argument(
        "--einstein_schedule.decay_rate",
        type=float,
        dest="einstein_schedule_decay_rate",
        help="Decay rate for Einstein multiplier schedule",
    )
    parser.add_argument(
        "--einstein_schedule.steps",
        type=int,
        dest="einstein_schedule_steps",
        help="Number of steps for Einstein multiplier schedule",
    )

    parser.add_argument(
        "--overlap_schedule.strategy",
        type=str,
        dest="overlap_schedule_strategy",
        help="Schedule strategy for overlap multiplier",
    )
    parser.add_argument(
        "--overlap_schedule.final_value",
        type=float,
        dest="overlap_schedule_final_value",
        help="Final value for overlap multiplier schedule",
    )
    parser.add_argument(
        "--overlap_schedule.warmup_epochs",
        type=int,
        dest="overlap_schedule_warmup_epochs",
        help="Warmup epochs for overlap multiplier schedule",
    )
    parser.add_argument(
        "--overlap_schedule.decay_rate",
        type=float,
        dest="overlap_schedule_decay_rate",
        help="Decay rate for overlap multiplier schedule",
    )
    parser.add_argument(
        "--overlap_schedule.steps",
        type=int,
        dest="overlap_schedule_steps",
        help="Number of steps for overlap multiplier schedule",
    )

    parser.add_argument(
        "--finiteness_schedule.strategy",
        type=str,
        dest="finiteness_schedule_strategy",
        help="Schedule strategy for finiteness multiplier",
    )
    parser.add_argument(
        "--finiteness_schedule.final_value",
        type=float,
        dest="finiteness_schedule_final_value",
        help="Final value for finiteness multiplier schedule",
    )
    parser.add_argument(
        "--finiteness_schedule.warmup_epochs",
        type=int,
        dest="finiteness_schedule_warmup_epochs",
        help="Warmup epochs for finiteness multiplier schedule",
    )
    parser.add_argument(
        "--finiteness_schedule.decay_rate",
        type=float,
        dest="finiteness_schedule_decay_rate",
        help="Decay rate for finiteness multiplier schedule",
    )
    parser.add_argument(
        "--finiteness_schedule.steps",
        type=int,
        dest="finiteness_schedule_steps",
        help="Number of steps for finiteness multiplier schedule",
    )

    # Schwarzschild-specific multiplier schedules
    parser.add_argument(
        "--kretschmann_schedule.strategy",
        type=str,
        dest="kretschmann_schedule_strategy",
        help="Schedule strategy for Kretschmann multiplier",
    )
    parser.add_argument(
        "--kretschmann_schedule.final_value",
        type=float,
        dest="kretschmann_schedule_final_value",
        help="Final value for Kretschmann multiplier schedule",
    )
    parser.add_argument(
        "--kretschmann_schedule.warmup_epochs",
        type=int,
        dest="kretschmann_schedule_warmup_epochs",
        help="Warmup epochs for Kretschmann multiplier schedule",
    )
    parser.add_argument(
        "--kretschmann_schedule.decay_rate",
        type=float,
        dest="kretschmann_schedule_decay_rate",
        help="Decay rate for Kretschmann multiplier schedule",
    )
    parser.add_argument(
        "--kretschmann_schedule.steps",
        type=int,
        dest="kretschmann_schedule_steps",
        help="Number of steps for Kretschmann multiplier schedule",
    )

    parser.add_argument(
        "--r2_det_schedule.strategy",
        type=str,
        dest="r2_det_schedule_strategy",
        help="Schedule strategy for R2 determinant multiplier",
    )
    parser.add_argument(
        "--r2_det_schedule.final_value",
        type=float,
        dest="r2_det_schedule_final_value",
        help="Final value for R2 determinant multiplier schedule",
    )
    parser.add_argument(
        "--r2_det_schedule.warmup_epochs",
        type=int,
        dest="r2_det_schedule_warmup_epochs",
        help="Warmup epochs for R2 determinant multiplier schedule",
    )
    parser.add_argument(
        "--r2_det_schedule.decay_rate",
        type=float,
        dest="r2_det_schedule_decay_rate",
        help="Decay rate for R2 determinant multiplier schedule",
    )
    parser.add_argument(
        "--r2_det_schedule.steps",
        type=int,
        dest="r2_det_schedule_steps",
        help="Number of steps for R2 determinant multiplier schedule",
    )

    parser.add_argument(
        "--speciality_index_schedule.strategy",
        type=str,
        dest="speciality_index_schedule_strategy",
        help="Schedule strategy for speciality-index multiplier",
    )
    parser.add_argument(
        "--speciality_index_schedule.final_value",
        type=float,
        dest="speciality_index_schedule_final_value",
        help="Final value for speciality-index multiplier schedule",
    )
    parser.add_argument(
        "--speciality_index_schedule.warmup_epochs",
        type=int,
        dest="speciality_index_schedule_warmup_epochs",
        help="Warmup epochs for speciality-index multiplier schedule",
    )
    parser.add_argument(
        "--speciality_index_schedule.decay_rate",
        type=float,
        dest="speciality_index_schedule_decay_rate",
        help="Decay rate for speciality-index multiplier schedule",
    )
    parser.add_argument(
        "--speciality_index_schedule.steps",
        type=int,
        dest="speciality_index_schedule_steps",
        help="Number of steps for speciality-index multiplier schedule",
    )

    parser.add_argument(
        "--k_repeller_schedule.strategy",
        type=str,
        dest="k_repeller_schedule_strategy",
        help="Schedule strategy for K-repeller multiplier",
    )
    parser.add_argument(
        "--k_repeller_schedule.final_value",
        type=float,
        dest="k_repeller_schedule_final_value",
        help="Final value for K-repeller multiplier schedule",
    )
    parser.add_argument(
        "--k_repeller_schedule.warmup_epochs",
        type=int,
        dest="k_repeller_schedule_warmup_epochs",
        help="Warmup epochs for K-repeller multiplier schedule",
    )
    parser.add_argument(
        "--k_repeller_schedule.decay_rate",
        type=float,
        dest="k_repeller_schedule_decay_rate",
        help="Decay rate for K-repeller multiplier schedule",
    )
    parser.add_argument(
        "--k_repeller_schedule.steps",
        type=int,
        dest="k_repeller_schedule_steps",
        help="Number of steps for K-repeller multiplier schedule",
    )

    parser.add_argument(
        "--speciality_index_rprofile_schedule.strategy",
        type=str,
        dest="speciality_index_rprofile_schedule_strategy",
        help="Schedule strategy for speciality-index r-profile multiplier",
    )
    parser.add_argument(
        "--speciality_index_rprofile_schedule.final_value",
        type=float,
        dest="speciality_index_rprofile_schedule_final_value",
        help="Final value for speciality-index r-profile multiplier schedule",
    )
    parser.add_argument(
        "--speciality_index_rprofile_schedule.warmup_epochs",
        type=int,
        dest="speciality_index_rprofile_schedule_warmup_epochs",
        help="Warmup epochs for speciality-index r-profile multiplier schedule",
    )
    parser.add_argument(
        "--speciality_index_rprofile_schedule.decay_rate",
        type=float,
        dest="speciality_index_rprofile_schedule_decay_rate",
        help="Decay rate for speciality-index r-profile multiplier schedule",
    )
    parser.add_argument(
        "--speciality_index_rprofile_schedule.steps",
        type=int,
        dest="speciality_index_rprofile_schedule_steps",
        help="Number of steps for speciality-index r-profile multiplier schedule",
    )

    # Density power schedules (Schwarzschild)
    parser.add_argument(
        "--density_power_R2_schedule.strategy",
        type=str,
        dest="density_power_R2_schedule_strategy",
        help="Schedule strategy for density_power_R2",
    )
    parser.add_argument(
        "--density_power_R2_schedule.final_value",
        type=float,
        dest="density_power_R2_schedule_final_value",
        help="Final value for density_power_R2 schedule",
    )
    parser.add_argument(
        "--density_power_R2_schedule.warmup_epochs",
        type=int,
        dest="density_power_R2_schedule_warmup_epochs",
        help="Warmup epochs for density_power_R2 schedule",
    )
    parser.add_argument(
        "--density_power_R2_schedule.decay_rate",
        type=float,
        dest="density_power_R2_schedule_decay_rate",
        help="Decay rate for density_power_R2 schedule",
    )
    parser.add_argument(
        "--density_power_R2_schedule.steps",
        type=int,
        dest="density_power_R2_schedule_steps",
        help="Number of steps for density_power_R2 schedule",
    )

    parser.add_argument(
        "--density_power_S2_schedule.strategy",
        type=str,
        dest="density_power_S2_schedule_strategy",
        help="Schedule strategy for density_power_S2",
    )
    parser.add_argument(
        "--density_power_S2_schedule.final_value",
        type=float,
        dest="density_power_S2_schedule_final_value",
        help="Final value for density_power_S2 schedule",
    )
    parser.add_argument(
        "--density_power_S2_schedule.warmup_epochs",
        type=int,
        dest="density_power_S2_schedule_warmup_epochs",
        help="Warmup epochs for density_power_S2 schedule",
    )
    parser.add_argument(
        "--density_power_S2_schedule.decay_rate",
        type=float,
        dest="density_power_S2_schedule_decay_rate",
        help="Decay rate for density_power_S2 schedule",
    )
    parser.add_argument(
        "--density_power_S2_schedule.steps",
        type=int,
        dest="density_power_S2_schedule_steps",
        help="Number of steps for density_power_S2 schedule",
    )

    # Seed values
    parser.add_argument("--np_seed", type=int, help="Seed for numpy")
    parser.add_argument("--tf_seed", type=int, help="Seed for TensorFlow")

    # Weight initialization
    parser.add_argument(
        "--weights_init_mean",
        type=float,
        help="Mean for weight initialization",
    )
    parser.add_argument(
        "--weights_init_std",
        type=float,
        help="Standard deviation for weight initialization",
    )

    return parser.parse_args()


def prune_none_args(args):
    trainable_args = vars(deepcopy(args))
    keys_to_remove = [arg for arg, val in trainable_args.items() if val is None]

    # Remove the keys from the dictionary
    for key in keys_to_remove:
        trainable_args.pop(key)

    return trainable_args
