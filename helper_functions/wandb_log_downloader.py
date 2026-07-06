from __future__ import annotations

from pathlib import Path

# Thanks to fmamberti-wandb for this code
# https://community.wandb.ai/t/feature-request-bulk-download-logs/6467/2
import wandb

entity = "logml"
project = "Ainstein_schwarzschild"
dst_folder = f"{project}-failed_runs_logs"

# Initialize a W&B API client
api = wandb.Api()

# Get the failed runs from the project
failed_runs = api.runs(path=f"{entity}/{project}", filters={"state": "finished"})

# Create a directory for the logs
dst_folder_path = Path(dst_folder)
dst_folder_path.mkdir(exist_ok=True, parents=True)

# Download the logs for each failed run
for run in failed_runs:
    log_files = run.files()
    for file in log_files:
        if file.name.endswith(".log"):
            file_name = file.name
            file.download(root=f"{dst_folder}/wandb-{run.id}", replace=True)
