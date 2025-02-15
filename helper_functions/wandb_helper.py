import numpy as np
from pathlib import Path

def restore_wandb(args, wandb_id):
    save_location = Path(args["saved_model_path"])

    # If the user has specified a file rather than a directory, take the parent
    # directory
    if not save_location.is_dir():
        save_location = save_location.parent

    file_paths = {
        "batch": f"{save_location}/final_batch.npy",
        "model": f"{save_location}/final_model.keras",
    }

    # Save training batch
    x_train = np.load(file_paths["batch"])

    # If the weights and biases ID is not None, restore from the terminal model
    if wandb_id is not None:
        args["saved_model"] = True
        args["saved_model_path"] = file_paths["model"]

    return args, x_train
