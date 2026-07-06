from __future__ import annotations

import argparse
from pathlib import Path

from configs.loader import load_config
from visualisation.schwarzschild import SchwarzschildVisualiser


def _experiment_name(run_dir: Path) -> str:
    config_path = run_dir / "hps_used.yaml"
    if not config_path.exists():
        config_path = run_dir / "hps.yaml"
    config = load_config(config_path)
    return (
        config.get("model", {}).get("experiment")
        or config.get("experiment")
        or "schwarzschild"
    ).lower()


def _run_schwarzschild(run_dir: Path) -> None:
    visualiser = SchwarzschildVisualiser(run_dir)
    visualiser.run_all()


def visualise_directory(root: Path | str) -> None:
    root = Path(root)
    model_paths = sorted(root.rglob("final_model.keras"))
    if not model_paths:
        raise FileNotFoundError(f"No final_model.keras files found under {root}")

    for model_path in model_paths:
        run_dir = model_path.parent
        try:
            experiment = _experiment_name(run_dir)
            print(f"Processing {run_dir} ({experiment})")

            if experiment == "schwarzschild":
                _run_schwarzschild(run_dir)
            else:
                print(f"Skipping {run_dir}: no batch visualiser registered for {experiment}")
        except FileNotFoundError as e:
            print(f"Warning: Skipping {run_dir} - missing config file")
        except Exception as e:
            print(f"Warning: Skipping {run_dir} - {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run visualisation over saved runs.")
    parser.add_argument("root", type=Path, help="Directory containing run folders")
    args = parser.parse_args()
    visualise_directory(args.root)


if __name__ == "__main__":
    main()
