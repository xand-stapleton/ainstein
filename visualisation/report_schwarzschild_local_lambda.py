from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import tensorflow as tf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.base import compute_ricci_tensor
from geometry.schwarzschild import embedding_jacobian_stereo
from helper_functions.helper_functions import cholesky_from_vec
from losses.schwarzschild import (TotalSchwarzschildLocal2DLoss,
                                  TotalSchwarzschildLoss)
from network.schwarzschild import (SchwarzschildGlobalModel,
                                   SchwarzschildLocal2DModel,
                                   SchwarzschildPatchSubModel)
from sampling.ball import (BallSample, StereoSampleHemisphere,
                           StereoSampleSingleHemisphere)
from sampling.penrose import PenroseSample
from visualisation.schwarzschild import SchwarzschildVisualiser


tf.keras.backend.set_floatx("float64")


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _is_local_schwarzschild_config(hps: dict) -> bool:
    model = hps.get("model", {})
    if model.get("experiment") not in ("blackhole", "schwarzschild"):
        return False
    model_specific = hps.get("model_specific", {})
    return bool(
        model_specific.get("local_2d_mode", False)
        or model_specific.get("local_single_s2_patch", False)
    )


def _is_local_2d_mode(hps: dict) -> bool:
    return bool(hps.get("model_specific", {}).get("local_2d_mode", False))


def _resolve_local_visualisation_mode(hps: dict) -> tuple[bool, int]:
    model_specific = hps.get("model_specific", {})
    visualisation = hps.get("visualisation", {})

    local_single = visualisation.get("local_single_s2_patch", None)
    if local_single is None:
        local_single = model_specific.get("local_single_s2_patch", False)

    local_patch_idx = visualisation.get("local_s2_patch_idx", None)
    if local_patch_idx is None:
        local_patch_idx = model_specific.get("local_s2_patch_idx", 0)

    return bool(local_single), int(local_patch_idx)


def _sampling_signature(hps: dict, num_samples: int, test_seed: int) -> tuple:
    model_specific = hps.get("model_specific", {})
    visualisation = hps.get("visualisation", {})

    if _is_local_2d_mode(hps):
        patch_width = model_specific.get("local_2d_patch_width", None)
        if patch_width is None:
            patch_width = visualisation.get(
                "patch_width_S2", model_specific.get("patch_width_S2", 1.0)
            )

        density = model_specific.get("local_2d_density_power", None)
        if density is None:
            density = visualisation.get(
                "density_power_S2", model_specific.get("density_power_S2", 1.0)
            )

        return (
            "local2d",
            int(num_samples),
            int(test_seed),
            float(patch_width),
            float(density),
        )

    local_single, local_patch_idx = _resolve_local_visualisation_mode(hps)

    patch_width_r2 = visualisation.get(
        "patch_width_R2", model_specific.get("patch_width_R2", 1.0)
    )
    patch_width_s2 = visualisation.get(
        "patch_width_S2", model_specific.get("patch_width_S2", 1.0)
    )
    density_r2 = visualisation.get(
        "density_power_R2", model_specific.get("density_power_R2", 1.0)
    )
    density_s2 = visualisation.get(
        "density_power_S2", model_specific.get("density_power_S2", 1.0)
    )

    return (
        int(num_samples),
        int(test_seed),
        float(patch_width_r2),
        float(patch_width_s2),
        float(density_r2),
        float(density_s2),
        bool(local_single),
        int(local_patch_idx),
    )


def _build_test_sample(hps: dict, num_samples: int, test_seed: int) -> tf.Tensor:
    model_specific = hps.get("model_specific", {})
    visualisation = hps.get("visualisation", {})

    if _is_local_2d_mode(hps):
        patch_width = model_specific.get("local_2d_patch_width", None)
        if patch_width is None:
            patch_width = visualisation.get(
                "patch_width_S2", model_specific.get("patch_width_S2", 1.0)
            )

        density = model_specific.get("local_2d_density_power", None)
        if density is None:
            density = visualisation.get(
                "density_power_S2", model_specific.get("density_power_S2", 1.0)
            )

        np.random.seed(test_seed)
        sample = BallSample(
            num_samples,
            dimension=2,
            patch_width=float(patch_width),
            density_power=float(density),
        )
        return tf.constant(sample, dtype=tf.float64)

    patch_width_r2 = visualisation.get(
        "patch_width_R2", model_specific.get("patch_width_R2", 1.0)
    )
    patch_width_s2 = visualisation.get(
        "patch_width_S2", model_specific.get("patch_width_S2", 1.0)
    )
    density_r2 = visualisation.get(
        "density_power_R2", model_specific.get("density_power_R2", 1.0)
    )
    density_s2 = visualisation.get(
        "density_power_S2", model_specific.get("density_power_S2", 1.0)
    )

    local_single, local_patch_idx = _resolve_local_visualisation_mode(hps)

    np.random.seed(test_seed)

    sample_r2 = PenroseSample(
        num_samples,
        patch_width=patch_width_r2,
        density_power=density_r2,
    )

    if local_single:
        sample_s2, patch_idx = StereoSampleSingleHemisphere(
            num_samples,
            patch_idx=local_patch_idx,
            patch_width=patch_width_s2,
            density_power=density_s2,
        )
    else:
        sample_s2, patch_idx = StereoSampleHemisphere(
            num_samples,
            patch_width=patch_width_s2,
            density_power=density_s2,
        )

    patch_idx_col = patch_idx.reshape(-1, 1).astype(np.float64)
    x = np.concatenate([sample_r2, sample_s2, patch_idx_col], axis=1)
    return tf.constant(x, dtype=tf.float64)


def _load_model(model_path: Path):
    custom_objects = {
        "GlobalModel": SchwarzschildGlobalModel,
        "PatchSubModel": SchwarzschildPatchSubModel,
        "SchwarzschildGlobalModel": SchwarzschildGlobalModel,
        "SchwarzschildLocal2DModel": SchwarzschildLocal2DModel,
        "SchwarzschildPatchSubModel": SchwarzschildPatchSubModel,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def _compute_einstein_loss(model, x_test: tf.Tensor) -> float:
    if getattr(model.config.model_specific, "local_2d_mode", False):
        loss_fn = TotalSchwarzschildLocal2DLoss(model.config)
        metric_pred = model(x_test, training=False)
        _, constituents = loss_fn.call(
            model,
            x_test,
            metric_pred=metric_pred,
            return_constituents=True,
            val_print=False,
        )
    else:
        loss_fn = TotalSchwarzschildLoss(model.config)
        _, constituents = loss_fn.call(
            model,
            x_test,
            metric_pred=None,
            return_constituents=True,
            val_print=False,
        )
    return float(constituents["einstein_loss"])


def _find_candidate_runs(runs_root: Path) -> list[dict]:
    candidates: list[dict] = []

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        hps_path = run_dir / "hps_used.yaml"
        model_path = run_dir / "final_model.keras"
        if not hps_path.exists() or not model_path.exists():
            continue

        hps = _load_yaml(hps_path)
        if not _is_local_schwarzschild_config(hps):
            continue

        try:
            lambda_value = float(hps.get("geometry", {}).get("einstein_constant"))
        except (TypeError, ValueError):
            continue

        seed_value = hps.get("model", {}).get("np_seed")

        candidates.append(
            {
                "run_dir": run_dir,
                "hps": hps,
                "lambda": lambda_value,
                "seed": seed_value,
            }
        )

    return candidates


def _candidate_from_run_dir(run_dir: Path) -> dict:
    hps_path = run_dir / "hps_used.yaml"
    model_path = run_dir / "final_model.keras"

    if not hps_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Run directory must contain hps_used.yaml and final_model.keras: {run_dir}"
        )

    hps = _load_yaml(hps_path)
    if not _is_local_schwarzschild_config(hps):
        raise ValueError(
            "The selected run is not a local Schwarzschild run. "
            "Expected model_specific.local_2d_mode=true or "
            "model_specific.local_single_s2_patch=true."
        )

    try:
        lambda_value = float(hps.get("geometry", {}).get("einstein_constant"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse geometry.einstein_constant in run: {run_dir}"
        ) from exc

    seed_value = hps.get("model", {}).get("np_seed")

    return {
        "run_dir": run_dir,
        "hps": hps,
        "lambda": lambda_value,
        "seed": seed_value,
    }


def _write_run_test_einstein_json(
    run_dir: Path,
    lambda_value: float,
    seed_value,
    einstein_loss: float,
    det_g_mean: float,
    det_g_std: float,
    num_test_samples: int,
    test_seed: int,
) -> None:
    payload = {
        "run_dir": str(run_dir.resolve()),
        "lambda": float(lambda_value),
        "seed": int(seed_value) if seed_value is not None else None,
        "einstein_loss": float(einstein_loss),
        "det_g_mean": float(det_g_mean),
        "det_g_std": float(det_g_std),
        "num_test_samples": int(num_test_samples),
        "test_seed": int(test_seed),
    }
    with open(run_dir / "test_einstein_loss.json", "w") as f:
        json.dump(payload, f, indent=2)


def _compute_metric_det_stats(model, x_test: tf.Tensor) -> tuple[float, float]:
    lorentzian = bool(getattr(model.config.model_specific, "lorentzian", False))

    if getattr(model.config.model_specific, "local_2d_mode", False):
        metric_vec = model(x_test, training=False)
        metric_pred = cholesky_from_vec(metric_vec, lorentzian=lorentzian)
    else:
        g_5d_vec = model(x_test, training=False)
        q_4d = x_test[:, :4]
        patch_idx = tf.cast(x_test[:, 4], tf.int32)
        g_5d = cholesky_from_vec(g_5d_vec, lorentzian=lorentzian)
        jacobian = embedding_jacobian_stereo(q_4d, patch_idx)
        metric_pred = tf.einsum("sAB,sAm,sBn->smn", g_5d, jacobian, jacobian)

    det_g = tf.linalg.det(metric_pred)
    det_g_mean = float(tf.reduce_mean(det_g).numpy())
    det_g_std = float(tf.math.reduce_std(det_g).numpy())
    return det_g_mean, det_g_std


def _generate_local2d_representative_plots(
    lambda_value: float,
    run_dir: Path,
    num_test_samples: int,
    test_seed: int,
) -> None:
    hps_path = run_dir / "hps_used.yaml"
    model_path = run_dir / "final_model.keras"

    hps = _load_yaml(hps_path)
    plot_num_samples = int(num_test_samples)
    if plot_num_samples <= 0:
        raise ValueError(
            f"num_test_samples must be positive for plotting, got {plot_num_samples}."
        )

    x_test = _build_test_sample(
        hps,
        num_samples=plot_num_samples,
        test_seed=test_seed,
    )
    model = _load_model(model_path)

    metric_vec = model(x_test, training=False)
    metric_pred = cholesky_from_vec(
        metric_vec,
        lorentzian=bool(getattr(model.config.model_specific, "lorentzian", False)),
    ).numpy()
    ricci_pred = compute_ricci_tensor(x_test, model.submodel).numpy()

    coords = x_test.numpy()
    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    is_zero_lambda = np.isclose(float(lambda_value), 0.0)

    for i in range(2):
        for j in range(2):
            z_metric = metric_pred[:, i, j]
            metric_min = float(np.min(z_metric))
            metric_max = float(np.max(z_metric))
            metric_span = not np.isclose(metric_min, metric_max)

            fig_metric = plt.figure(figsize=(8, 6))
            ax_metric = fig_metric.add_subplot(1, 1, 1, projection="3d")
            sc_metric = ax_metric.scatter(
                coords[:, 0],
                coords[:, 1],
                z_metric,
                c=z_metric,
                cmap="viridis",
                alpha=0.9,
            )
            if metric_span:
                ax_metric.set_zlim(metric_min, metric_max)
            ax_metric.set_xlabel(r"$x_1$")
            ax_metric.set_ylabel(r"$x_2$")
            ax_metric.set_zlabel(rf"$g_{{{i},{j}}}$")
            plt.colorbar(sc_metric, ax=ax_metric, shrink=0.75, pad=0.1)
            plt.tight_layout()
            plt.savefig(output_dir / f"local2d_g_ij_{i}_{j}.pdf", bbox_inches="tight")
            plt.close(fig_metric)

            z_ricci = ricci_pred[:, i, j]
            fig_ricci = plt.figure(figsize=(8, 6))
            ax_ricci = fig_ricci.add_subplot(1, 1, 1, projection="3d")
            if is_zero_lambda and metric_span:
                ricci_mean = float(np.mean(z_ricci))
                half_metric_range = 0.5 * (metric_max - metric_min)
                ricci_zmin = ricci_mean - half_metric_range
                ricci_zmax = ricci_mean + half_metric_range
                ricci_cmap = plt.get_cmap("viridis").copy()
                ricci_cmap.set_under("black")
                ricci_cmap.set_over("black")
                ricci_norm = mcolors.Normalize(
                    vmin=metric_min,
                    vmax=metric_max,
                    clip=False,
                )

                sc_ricci = ax_ricci.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    z_ricci,
                    c=z_ricci,
                    cmap=ricci_cmap,
                    norm=ricci_norm,
                    alpha=0.9,
                )
                ax_ricci.set_zlim(ricci_zmin, ricci_zmax)
                colorbar_extend = "both"
            else:
                sc_ricci = ax_ricci.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    z_ricci,
                    c=z_ricci,
                    cmap="viridis",
                    alpha=0.9,
                )
                colorbar_extend = "neither"
            ax_ricci.set_xlabel(r"$x_1$")
            ax_ricci.set_ylabel(r"$x_2$")
            ax_ricci.set_zlabel(rf"$R_{{{i},{j}}}$")
            plt.colorbar(
                sc_ricci,
                ax=ax_ricci,
                shrink=0.75,
                pad=0.1,
                extend=colorbar_extend,
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"local2d_R_ij_{i}_{j}.pdf", bbox_inches="tight")
            plt.close(fig_ricci)


def _generate_representative_plots(
    representative_runs: dict[float, Path],
    num_test_samples: int,
    test_seed: int,
) -> None:
    for lambda_value, run_dir in representative_runs.items():
        hps_path = run_dir / "hps_used.yaml"
        if hps_path.exists() and _is_local_2d_mode(_load_yaml(hps_path)):
            print(
                f"Generating local_2d_mode component plots for "
                f"(lambda={lambda_value:.1f}): {run_dir}"
            )
            _generate_local2d_representative_plots(
                lambda_value,
                run_dir,
                num_test_samples,
                test_seed,
            )
            continue

        print(f"Generating plots for lambda={lambda_value:.1f} from {run_dir}")
        visualiser = SchwarzschildVisualiser(model_parent=run_dir)
        visualiser.compute_quantities()
        visualiser.plot_points()
        visualiser.plot_all_pairs()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local Lorentzian Schwarzschild runs and write per-run "
            "test Einstein metrics JSON into each run directory."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root folder containing run subdirectories (default: runs).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Specific completed run directory to evaluate. "
            "If set, only this run is processed and --runs-root is ignored."
        ),
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=10000,
        help="Number of test samples used to evaluate Einstein loss per run.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=1234,
        help="Random seed used for deterministic test-sample generation.",
    )
    parser.add_argument(
        "--skip-representative-plots",
        action="store_true",
        help="Skip plot generation for representative runs.",
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        candidates = [_candidate_from_run_dir(run_dir)]
        print(f"Evaluating specific run: {run_dir}")
    else:
        runs_root = args.runs_root
        if not runs_root.exists():
            raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

        candidates = _find_candidate_runs(runs_root)
        if not candidates:
            raise RuntimeError(
                "No local Schwarzschild runs found. Ensure runs contain hps_used.yaml, "
                "final_model.keras, and either model_specific.local_2d_mode=true "
                "or model_specific.local_single_s2_patch=true."
            )

        print(f"Found {len(candidates)} candidate local runs under {runs_root}")

    sample_cache: dict[tuple, tf.Tensor] = {}
    per_run_rows: list[dict] = []
    errors: list[dict] = []

    for item in candidates:
        run_dir: Path = item["run_dir"]
        hps: dict = item["hps"]

        try:
            sig = _sampling_signature(hps, args.num_test_samples, args.test_seed)
            if sig not in sample_cache:
                sample_cache[sig] = _build_test_sample(
                    hps,
                    num_samples=args.num_test_samples,
                    test_seed=args.test_seed,
                )
            x_test = sample_cache[sig]

            model = _load_model(run_dir / "final_model.keras")
            einstein_loss = _compute_einstein_loss(model, x_test)
            det_g_mean, det_g_std = _compute_metric_det_stats(model, x_test)

            _write_run_test_einstein_json(
                run_dir=run_dir,
                lambda_value=float(item["lambda"]),
                seed_value=item["seed"],
                einstein_loss=einstein_loss,
                det_g_mean=det_g_mean,
                det_g_std=det_g_std,
                num_test_samples=args.num_test_samples,
                test_seed=args.test_seed,
            )

            per_run_rows.append(
                {
                    "lambda": item["lambda"],
                    "seed": item["seed"],
                    "run_dir": str(run_dir),
                    "einstein_loss": einstein_loss,
                }
            )
        except Exception as exc:
            errors.append({"run_dir": str(run_dir), "error": str(exc)})

    if not per_run_rows:
        raise RuntimeError("No runs could be evaluated successfully.")

    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in per_run_rows:
        grouped[float(row["lambda"])].append(row)

    representative_runs: dict[float, Path] = {}
    print("Lambda summary (test Einstein loss over selected test points):")

    for lambda_value in sorted(grouped.keys()):
        rows = grouped[lambda_value]
        losses = [float(r["einstein_loss"]) for r in rows]
        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0

        sorted_rows = sorted(rows, key=lambda r: float(r["einstein_loss"]))
        rep_row = sorted_rows[len(sorted_rows) // 2]

        representative_runs[lambda_value] = Path(rep_row["run_dir"])
        print(
            f"  lambda={lambda_value:.1f}: n_runs={len(rows)}, "
            f"mean={mean_loss:.6g}, std={std_loss:.6g}"
        )

    if not args.skip_representative_plots:
        _generate_representative_plots(
            representative_runs,
            num_test_samples=args.num_test_samples,
            test_seed=args.test_seed,
        )

    print("Report complete.")
    print("Per-run JSON written to each run directory as test_einstein_loss.json")
    if errors:
        print(f"Warning: {len(errors)} run(s) failed during evaluation.")


if __name__ == "__main__":
    main()
