from __future__ import annotations

import os
import sys

# --- GPU / device setup ---
# Apple Metal (macOS, tensorflow-metal) only supports float32 and will crash
# when its graph-optimizer intercepts float64 tensors.  CUDA / ROCm GPUs
# (Linux HPC clusters) support float64 natively and should be used freely.
# We therefore suppress the GPU only on macOS.
#
# Some PBS schedulers set CUDA_VISIBLE_DEVICES to "" (empty string) rather
# than leaving it unset, which causes TensorFlow to find no GPUs even when
# GPUs have been allocated to the job.  Unset it here – before TF is imported
# and CUDA initialises – so TF can discover available GPUs automatically.
if sys.platform != "darwin":
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if _cvd == "":
        # Some PBS schedulers set this to "" rather than leaving it unset,
        # which prevents TF from finding any GPUs.
        del os.environ["CUDA_VISIBLE_DEVICES"]
        print(
            "CUDA_VISIBLE_DEVICES was empty – unset so TF can discover all allocated GPUs."
        )
    elif _cvd is not None and _cvd.startswith("MIG-"):
        # PBS allocates MIG (Multi-Instance GPU) slices and sets
        # CUDA_VISIBLE_DEVICES to the MIG UUID (e.g. "MIG-<uuid>").
        # TensorFlow's CUDA backend cannot parse this UUID format and reports
        # CUDA_ERROR_NO_DEVICE even though a GPU slice is allocated.  Unsetting
        # the variable lets CUDA auto-discover the device; PBS cgroup isolation
        # ensures the job can only see its own MIG slice.
        del os.environ["CUDA_VISIBLE_DEVICES"]
        print(
            f"CUDA_VISIBLE_DEVICES was a MIG UUID ({_cvd!r}) – unset so TF can discover the MIG device."
        )
    elif _cvd is not None:
        print(f"CUDA_VISIBLE_DEVICES={_cvd!r}")

import tensorflow as tf

_physical_gpus = tf.config.list_physical_devices("GPU")
if _physical_gpus:
    try:
        for _gpu in _physical_gpus:
            tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError as e:
        print(f"GPU setup warning: {e}")

    _is_apple_metal = sys.platform == "darwin"
    if _is_apple_metal:
        # Hide the Metal GPU — float64 is incompatible with Apple Metal.
        tf.config.set_visible_devices([], "GPU")
        print(
            f"GPU(s) detected ({[d.name for d in _physical_gpus]}) but hidden: "
            "Apple Metal does not support float64. "
            "Switch to float32 (remove set_floatx call) to use the GPU on macOS."
        )
    else:
        # CUDA/ROCm — float64 is supported; use all visible GPUs.
        print(f"GPU(s) available and enabled: {[d.name for d in _physical_gpus]}.")
else:
    print("No GPU found – running on CPU.")
# ---

tfk = tf.keras
import yaml

from helper_functions import argument_parser


def main():
    args = argument_parser.get_args()
    wandb_id = args.wandb_id
    runtime_config = argument_parser.prune_none_args(args)

    # Load the YAML config and check the "experiment" value
    with open(args.config_file, "r") as f:
        config_yaml = yaml.safe_load(f)

    # Set float dtype before any TF model/tensor creation.
    _dtype = (
        config_yaml.get("dtype")
        or config_yaml.get("model", {}).get("dtype")
        or "float64"
    )
    tfk.backend.set_floatx(_dtype)

    # Fallback to sphere if not provided.
    # Allows legacy hyperparameters files to be used.
    experiment_name = (
        config_yaml.get("experiment")
        or config_yaml.get("model", {}).get("experiment")
        or "schwarzschild"
    )

    match experiment_name.lower():
        case "sphere":
            print("Running sphere experiment...")
            from runtime.sphere import SphereTrainerRunner

            runner = SphereTrainerRunner(
                supervised=args.supervised,
                identity=args.identity,
                config_file=args.config_file,
                runtime_config=runtime_config,
                wandb_id=wandb_id,
            )

        case "schwarzschild":
            print("Running Schwarzschild experiment...")
            from runtime.schwarzschild import SchwarzschildTrainerRunner

            runner = SchwarzschildTrainerRunner(
                supervised=args.supervised,
                identity=args.identity,
                config_file=args.config_file,
                runtime_config=runtime_config,
                wandb_id=wandb_id,
            )
        case "lens":
            print("Running lens experiment...")
            from runtime.lens import LensTrainerRunner

            runner = LensTrainerRunner(
                supervised=args.supervised,
                identity=args.identity,
                config_file=args.config_file,
                runtime_config=runtime_config,
                wandb_id=wandb_id,
            )
        case _:
            raise ValueError(...)

    if not args.supervised:
        train_sample_metric = None
        loss_hist, train_data, val_data = runner.run()
        if runner.config.visualisation.visualise:
            runner.visualise(train_losses=loss_hist)
    else:
        _, loss_hist, train_data, train_sample_metric, val_data = (
            runner.run_supervised()
        )

    return loss_hist, train_data, train_sample_metric, val_data


if __name__ == "__main__":
    main()
