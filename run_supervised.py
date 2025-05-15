import tensorflow as tf

tfk = tf.keras
tfk.backend.set_floatx("float64")
import yaml

from geometry.ball import AnalyticMetric_Ball, PatchChange_Coordinates_Ball
from helper_functions.helper_functions import cholesky_to_vec
from network.ball import BallGlobalModel
from sampling.ball import BallSample, CubeSample


# Main body function for performing the metric training
def main(identity_bool=False):
    ###########################################################################
    ### Training set-up ###
    # Load the hyperparameters YAML file
    with open("hyperparameters/hps.yaml", "r") as file:
        hp = yaml.safe_load(file)

    ###########################################################################
    ### Data set-up ###
    # Create training and validation manifold samples
    # Ball patch sampling
    if hp["ball"]:
        train_sample = BallSample(
            hp["num_samples"],
            dimension=hp["dim"],
            patch_width=hp["patch_width"],
            density_power=hp["density_power"],
        )
        if hp["validate"]:
            val_sample = BallSample(
                hp["num_val_samples"],
                dimension=hp["dim"],
                patch_width=hp["patch_width"],
                density_power=hp["density_power"],
            )
    # Cube patch sampling
    else:
        assert hp["n_patches"] == 1, "Cube sampling only suitable for local geometries where don't need the ball structure for patching (set n_patches = 1)"
        train_sample = CubeSample(
            hp["num_samples"],
            dimension=hp["dim"],
            width=hp["patch_width"],
            density_power=hp["density_power"],
        )
        if hp["validate"]:
            val_sample = CubeSample(
                hp["num_val_samples"],
                dimension=hp["dim"],
                width=hp["patch_width"],
                density_power=hp["density_power"],
            )

    # Compute the sample analytic outputs
    train_sample_inputs = [train_sample]
    if hp["n_patches"] == 2:
        train_sample_inputs.append(PatchChange_Coordinates_Ball(train_sample))
    elif hp["n_patches"] > 2:
        raise SystemExit("codebase not yet configured for >2 patches...")
    train_sample_metrics = [
        AnalyticMetric_Ball(ts, identity=identity_bool) for ts in train_sample_inputs
    ]

    # Generate validation data if required
    if hp["validate"]:
        val_sample_inputs = [val_sample]
        if hp["n_patches"] > 1:
            val_sample_inputs.append(PatchChange_Coordinates_Ball(val_sample))
        val_sample_metrics = [
            AnalyticMetric_Ball(vs, identity=identity_bool) for vs in val_sample_inputs
        ]

    # Convert to Cholesky vectors (vielbeins)
    train_sample_metrics_vecs = [cholesky_to_vec(tsm) for tsm in train_sample_metrics]
    if hp["validate"]:
        val_sample_metrics_vecs = [cholesky_to_vec(vsm) for vsm in val_sample_metrics]

    # Convert to tf objects
    train_sample_tf = tf.convert_to_tensor(train_sample, dtype=tf.dtypes.float64)
    train_sample_metrics_tf = tf.convert_to_tensor(
        tf.concat(train_sample_metrics_vecs, axis=1), dtype=tf.dtypes.float64
    )
    if hp["validate"]:
        val_sample_tf = tf.convert_to_tensor(val_sample, dtype=tf.dtypes.float64)
        val_sample_metrics_tf = tf.convert_to_tensor(
            tf.concat(val_sample_metrics_vecs, axis=1), dtype=tf.dtypes.float64
        )
        val_data = (val_sample_tf, val_sample_metrics_tf)
    else:
        val_sample_tf = None
        val_sample_metrics_tf = None
        val_data = None

    ###########################################################################
    ### Model set-up ###
    # Set up optimiser
    if hp["init_learning_rate"] == hp["min_learning_rate"]:
        optimiser = tfk.optimizers.Adam(learning_rate=hp["init_learning_rate"])
    else:
        lr_schedule = tfk.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=hp["init_learning_rate"],
            decay_steps=1000,
            end_learning_rate=hp["min_learning_rate"],
            power=1.0,
        )
        optimiser = tfk.optimizers.Adam(learning_rate=lr_schedule)

    # Import the model
    if hp["saved_model"]:
        model = tfk.models.load_model(hp["saved_model_path"])
        model.compile(optimizer=optimiser, loss="MSE")
        # Update imported model implicit hps
        hp["dim"] = model.hp["dim"]
        hp["n_patches"] = model.hp["n_patches"]
        hp["n_hidden"] = model.hp["n_hidden"]
        hp["n_layers"] = model.hp["n_layers"]
        hp["activations"] = model.hp["activations"]
        hp["use_bias"] = model.hp["use_bias"]  # ...these are overwritten by the import
        model.hp = hp
        model.set_serializable_hp()
    # Build the model
    else:
        model = BallGlobalModel(hp)
        model.compile(optimizer=optimiser, loss="MSE")

    ###########################################################################
    ### Run ML ###
    # Train!
    loss_hist = model.fit(
        train_sample_tf,
        train_sample_metrics_tf,
        batch_size=hp["batch_size"],
        epochs=hp["epochs"],
        verbose=hp["verbosity"],
        validation_data=val_data,
        shuffle=True,
    )

    return (
        model,
        loss_hist,
        train_sample_tf,
        train_sample_metrics_tf,
        val_data,
    )


###############################################################################
if __name__ == "__main__":
    # Supervised run hyperparameters
    identity_bool = True  # ...select whether to train against the identity function metric (True) or round metric (False)
    save = True  # ...whether to save the trained supervised model
    save_flag = "test"  # ...the filename extension for the trained supervised model

    # Define and train the model
    network, lh, train_coords, train_metrics, val_data = main(identity_bool)
    print("trained.....")

    # Save the model
    if save == True:
        network.save(f"runs/supervised_model_{save_flag}.keras")
