from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml

# Mapping of flat YAML keys to nested config groups
SECTION_MAP = {
    "dim": "geometry",
    "n_patches": "geometry",
    "overlap_upperwidth": "geometry",
    "einstein_constant": "geometry",
    "patch_width": "data",
    "num_samples": "data",
    "einstein_multiplier": "loss",
    "overlap_multiplier": "loss",
    "finiteness_multiplier": "loss",
    "kretschmann_multiplier": "model_specific",
    "k_repeller_multiplier": "model_specific",
    "k_repeller_epsilon": "model_specific",
    "speciality_index_multiplier": "model_specific",
    "speciality_index_rprofile_mode": "model_specific",
    "speciality_index_rprofile_centre": "model_specific",
    "speciality_index_rprofile_multiplier": "model_specific",
    "speciality_index_rprofile_epsilon": "model_specific",
    "use_volume_scaling": "model_specific",
    "use_area_measure_weight": "model_specific",
    "use_metric_contraction": "model_specific",
    "volume_scaling_loss_components": "model_specific",
    "metric_contraction_loss_components": "model_specific",
    "finite_centre": "finiteness",
    "finite_width": "finiteness",
    "finite_sharpness": "finiteness",
    "finite_height": "finiteness",
    "finite_slope": "finiteness",
    "saved_model": "model",
    "saved_model_path": "model",
    "saved_model_weight_noise": "model",
    "n_hidden": "model",
    "n_layers": "model",
    "activations": "model",
    "use_bias": "model",
    "np_seed": "model",
    "tf_seed": "model",
    "dtype": "model",
    "epochs": "training",
    "batch_size": "training",
    "init_learning_rate": "training",
    "min_learning_rate": "training",
    "use_validation": "training",
    "val_print": "training",
    "num_val_samples": "training",
    "val_batch_size": "training",
    "verbosity": "training",
    "log_wandb": "logging",
    "wandb_log_freq": "logging",
    "log_interim": "logging",
    "log_dir": "logging",
    "log_interval": "logging",
    "track_best": "logging",
    "save_best_hist": "logging",
    "log_errors": "logging",
    "print_batch_losses": "logging",
    "experiment": "model",
}


def nest_flat_config(flat: dict) -> dict:
    """
    Convert a flat configuration dictionary into a nested dictionary based on
    predefined sections.

    Keys are grouped into sub-dictionaries (e.g., 'geometry', 'data', 'model')
    using SECTION_MAP.

    Args:
        flat (dict): Flat dictionary of configuration values (e.g., from a YAML
        file).

    Returns:
        dict: Nested dictionary where keys are grouped by their section.
    """
    nested = defaultdict(dict)
    for key, value in flat.items():
        section = SECTION_MAP.get(key)
        if section:
            nested[section][key] = value
        else:
            # Assume it's top-level if no section is mapped
            nested["model_specific"][key] = value
    return dict(nested)


def unnest_config(nested: dict) -> dict:
    """
    Flatten a nested configuration dictionary into a single-level dictionary.

    This is the inverse of `nest_flat_config()`, useful for saving configs back
    to flat YAML files.

    Args:
        nested (dict): Nested dictionary (with sections like 'geometry',
        'model', etc.).

    Returns:
        dict: Flat dictionary with all key-value pairs moved to the top level.
    """
    flat = {}
    for section, contents in nested.items():
        if isinstance(contents, dict):
            for key, value in contents.items():
                flat[key] = value
        else:
            # Top-level entries that weren't nested (e.g. metadata)
            flat[section] = contents
    return flat


def load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file and return a nested dictionary.

    Automatically detects whether the config is already nested or flat.
    If flat, it will be nested using `nest_flat_config()`.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: A nested dictionary representing the configuration.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # If already nested, just try parsing it directly
    if any(k in raw for k in ("geometry", "data", "loss", "training", "model")):
        return raw

    # Else, assume flat and convert to nested
    nested = nest_flat_config(raw)
    return nested


def apply_runtime_overrides(config_dict, runtime_config):
    """
    Apply runtime overrides to the configuration dictionary.

    Parameters:
    - config_dict: The main configuration dictionary to be updated.
    - runtime_config: Dictionary of runtime overrides (flat or with underscore-delimited keys).

    Modifies:
    - config_dict in-place, applying overrides based on SECTION_MAP and nested key reconstruction.
    
    Handles nested keys with underscore separators (e.g., einstein_schedule_strategy maps to 
    loss.einstein_schedule.strategy).
    """
    # Schedule parameter mappings: keys with these prefixes become nested dicts
    SCHEDULE_PREFIXES = {
        "einstein_schedule": "loss",
        "overlap_schedule": "loss",
        "finiteness_schedule": "loss",
        "kretschmann_schedule": "loss",
        "r2_det_schedule": "loss",
        "k_repeller_schedule": "loss",
        "speciality_index_schedule": "loss",
        "speciality_index_rprofile_schedule": "loss",
        "density_power_R2_schedule": "model_specific",
        "density_power_S2_schedule": "model_specific",
    }
    
    # Build nested schedule dictionaries from flattened keys
    schedules_to_set = defaultdict(dict)  # {"schedule_name": {"section": section, "params": {}}}
    schedule_param_map = {}  # Maps flattened keys to schedule name
    
    for prefix, section in SCHEDULE_PREFIXES.items():
        schedules_to_set[prefix]["section"] = section
        schedules_to_set[prefix]["params"] = {}
    
    # Process each runtime override
    for key, val in runtime_config.items():
        # Check if this is a nested schedule parameter (e.g., einstein_schedule_strategy)
        found = False
        for schedule_name in SCHEDULE_PREFIXES.keys():
            if key.startswith(schedule_name + "_"):
                # Extract the parameter name (e.g., "strategy" from "einstein_schedule_strategy")
                param_name = key[len(schedule_name) + 1:]  # +1 for the underscore
                schedules_to_set[schedule_name]["params"][param_name] = val
                found = True
                break
        
        if not found:
            # Regular flat key
            section = SECTION_MAP.get(key)
            if section:
                config_dict.setdefault(section, {})[key] = val
            else:
                config_dict.setdefault("model_specific", {})[key] = val
    
    # Now apply the reconstructed schedules
    for schedule_name, schedule_info in schedules_to_set.items():
        if schedule_info["params"]:  # Only if there are parameters to set
            section = schedule_info["section"]
            config_dict.setdefault(section, {})[schedule_name] = schedule_info["params"]


def find_config_class(name: Literal["sphere", "schwarzschild", "lens"]):
    match name.lower():
        case "schwarzschild":
            from configs.schwarzschild import SchwarzschildConfig

            return SchwarzschildConfig
        case "lens":
            from configs.lens import LensConfig

            return LensConfig
        case "sphere":
            from configs.sphere import SphereConfig

            return SphereConfig
        case _:
            raise ValueError()
