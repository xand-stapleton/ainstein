import argparse
from copy import deepcopy


def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Add argument for hyperparameters file
    parser.add_argument(
        "--hyperparams",
        type=str,
        help="Path to the hyperparameters YAML file (default: hyperparameters/patch.yaml)",
        required=True,
    )

    # Logging and sweep arguments
    parser.add_argument(
        "--log_wandb",
        type=bool,
        help="Enable or disable logging to Weights and Biases (WandB)",
    )

    # Model configuration
    parser.add_argument(
        "--saved_model_path",
        type=str,
        help="Path to saved model. If None, initialize a new one",
    )

    # WandB ID
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Weights and biases ID. If None, start a fresh WandB run.",
    )

    # Data and patch sampling configuration
    parser.add_argument("--dim", type=int, help="Dimensionality of the data")
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
    parser.add_argument("--density_power", type=float, help="Skew factor for sampling")

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
        "--einstein_metric_multiplier",
        type=float,
        help="Multiplier for the metric in the Einstein loss",
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
    parser.add_argument("--min_learning_rate", type=float, help="Minimum learning rate")

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
        "--print_losses", type=bool, help="Print breakdown of loss terms"
    )

    # Damping factors
    parser.add_argument(
        "--overlap_damper",
        type=float,
        help="Damping factor for overlap loss",
    )
    parser.add_argument(
        "--finiteness_damper",
        type=float,
        help="Damping factor for finiteness loss",
    )
    parser.add_argument(
        "--initial_einstein_damper",
        type=float,
        help="Initial damping factor for Einstein loss",
    )

    # Short-range configuration
    parser.add_argument(
        "--short_range_threshold",
        type=float,
        help="Threshold for undamped Einstein loss",
    )
    parser.add_argument(
        "--short_range_einstein_damper",
        type=float,
        help="Damping factor for Einstein loss at short range",
    )

    # Filter hyperparameters
    parser.add_argument(
        "--einsteinfilter_centre",
        type=int,
        help="Center for Einstein filter",
    )
    parser.add_argument(
        "--einsteinfilter_width", type=int, help="Width for Einstein filter"
    )
    parser.add_argument(
        "--einsteinfilter_sharpness",
        type=int,
        help="Sharpness for Einstein filter",
    )
    parser.add_argument(
        "--originfilter_cutoff",
        type=float,
        help="Cutoff for origin filter",
    )
    parser.add_argument(
        "--originfilter_sharpness",
        type=int,
        help="Sharpness for origin filter",
    )
    parser.add_argument(
        "--boundaryfilter_centre",
        type=int,
        help="Center for boundary filter",
    )
    parser.add_argument(
        "--boundaryfilter_sharpness",
        type=int,
        help="Sharpness for boundary filter",
    )
    parser.add_argument(
        "--boundaryfilter_height",
        type=float,
        help="Height for boundary filter",
    )
    parser.add_argument(
        "--uniqueness_weightfactor",
        type=float,
        help="Weight factor for uniqueness",
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
