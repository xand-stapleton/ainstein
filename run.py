import tensorflow as tf

tfk = tf.keras
tfk.backend.set_floatx("float64")
import numpy as np
import yaml

import wandb
from helper_functions import argument_parser, wandb_helper
from network.ball import BallNetwork
from sampling.ball import BallSample, CubeSample


# Main body function for performing the metric training
def main(hyperparameters_file, runtime_args, wandb_id=None):
    ###########################################################################
    ### Training & Logging set-up ###
    # Load the hyperparameters YAML file
    with open(hyperparameters_file, "r") as file:
        args = yaml.safe_load(file)

    for arg, arg_val in args.items():
        if arg in runtime_args:
            args[arg] = runtime_args[arg]

    # Check if restoring WandB
    if wandb_id is not None:
        args, x_train = wandb_helper.restore_wandb(args, wandb_id)
    else:
        # Initialise the training and val batches
        x_train, x_val = None, None

    # Check and set seeds for reproducibility
    rng = np.random.default_rng()
    # ...for NumPy
    if args["np_seed"] is None:
        args["np_seed"] = int(rng.integers(2**32 - 2))
    # ...for TensorFlow
    if args["tf_seed"] is None:
        args["tf_seed"] = int(rng.integers(2**32 - 2))

    # Make sure the config things are ints
    args["np_seed"] = int(args["np_seed"])
    args["tf_seed"] = int(args["tf_seed"])
    np.random.seed(args["np_seed"])
    tf.random.set_seed(args["tf_seed"])
    tfk.utils.set_random_seed(args["tf_seed"])

    # Print some random characters to check seed applied correctly
    print("TF random key: ", tf.random.uniform(shape=[6]))
    print("NP random key: ", np.random.randint(1, np.iinfo(int).max, size=6))

    # Start a WeightsandBiases session, and allow resuming from checkpoint
    wandb.init(project="Exotric", config=args, id=wandb_id, resume="allow")

    # Allow WandB to control the hyperparameters for sweeps (amounts to
    # over-writing the hyperparameters file with new values).
    hp = wandb.config

    # Add run identifiers for saving tracability
    hp["run_identifiers"] = (wandb.run.name, wandb.run.id)

    ###########################################################################
    ### Data set-up ###
    # Create training and validation samples
    if wandb_id is None:
        # Ball patch sampling
        if hp["ball"]:
            train_sample = BallSample(
                hp.num_samples,
                dimension=hp.dim,
                patch_width=hp["patch_width"],
                density_power=hp["density_power"],
            )
            if hp["validate"]:
                val_sample = BallSample(
                    hp.num_val_samples,
                    dimension=hp.dim,
                    patch_width=hp["patch_width"],
                    density_power=hp["density_power"],
                )
        # Cube patch sampling (full functionality unlikely entirely compatible at present)
        else:
            assert hp["n_patches"] == 1, "Cube sampling only suitable for local geometries where don't need the ball structure for patching (set n_patches = 1)"
            train_sample = CubeSample(
                hp.num_samples,
                dimension=hp.dim,
                width=hp["patch_width"],
                density_power=hp["density_power"],
            )
            if hp["validate"]:
                val_sample = CubeSample(
                    hp.num_val_samples,
                    dimension=hp.dim,
                    width=hp["patch_width"],
                    density_power=hp["density_power"],
                )
    # If wandb_id is not None
    else:
        train_sample = x_train
        if hp["validate"]:
            val_sample = x_val

    # Convert to tf objects
    train_sample_tf = tf.convert_to_tensor(train_sample, dtype=tf.dtypes.float64)
    val_sample_tf = None
    if hp["validate"]:
        val_sample_tf = tf.convert_to_tensor(val_sample, dtype=tf.dtypes.float64)

    ###########################################################################
    ### Run ML ###
    # Instantiate the network
    network = BallNetwork(hp=hp, print_losses=hp.print_losses)

    # Train!
    loss_hist = network.train(
        x_train=train_sample_tf, validate=hp["validate"], x_val=val_sample_tf
    )

    # Close the WandB session
    wandb.finish()

    return loss_hist, train_sample_tf, val_sample_tf


###############################################################################
if __name__ == "__main__":
    # Extract the runtime args
    args = argument_parser.get_args()

    # Extract any specified WandB id passed to the run
    wandb_id = args.wandb_id

    # Create a dict of the training arguments
    runtime_args = argument_parser.prune_none_args(args)

    # Perform the training
    lh, train_data, val_data = main(args.hyperparams, runtime_args, wandb_id)
