"""Paper-style final evaluation plots for trained Schwarzschild-family runs.

The output format is deliberately strict:
  * one Penrose-diagram field per PDF,
  * no plot titles,
  * a colourbar on every figure,
  * concise filenames that identify the quantity.

The same regular Penrose grid is used for all plotted quantities and printed
summary statistics.  Output goes to ``<run_dir>/plots_paper/`` by default.

Run from the repository root, e.g.

    conda run --no-capture-output -n exotric python -m visualisation.full_testing \
        --run-dir runs/<run_name>
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, SymLogNorm, TwoSlopeNorm
from matplotlib.tri import Triangulation

tf.keras.backend.set_floatx("float64")

from configs.loader import load_config
from configs.schwarzschild import SchwarzschildConfig
from geometry.ricci_opt import speciality_index_from_invariants
from geometry.schwarzschild import (
    AnalyticMetric_R2S2,
    Analytic_Kretschmann,
    PenroseRadiusWeighting,
    compute_ricci_and_kretschmann_embed,
    embedding_jacobian_stereo,
    riemannian_inverse_metric_embed,
    stereo_to_cartesian,
)
from helper_functions.helper_functions import cholesky_from_vec
from losses.schwarzschild import KillingSymmetryLossEmbed, WeightSchwarzschild
from network.schwarzschild import SchwarzschildGlobalModel, SchwarzschildPatchSubModel
from sampling.ball import StereoSampleHemisphere, StereoSampleSingleHemisphere
from sampling.penrose import PenroseSample, disc_to_penrose, draw_penrose

try:
    from geometry.ricci_opt import compute_ricci_and_kretschmann_embed_opt
except Exception:  # pragma: no cover - optional fast path may not exist in old runs
    compute_ricci_and_kretschmann_embed_opt = None


RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
}

RHO_TYPE_D_ABS = 1.0 / np.sqrt(12.0)
COMPONENT_CMAP = "turbo"
# Fixed signed-log floor for the shared metric/Ricci component scale.  Values
# with |x| below this sit in the central near-zero colour band.
COMPONENT_SYMLOG_LINTHRESH = 1e-1
COMPONENT_AXIS_LABELSIZE = 18
COMPONENT_AXIS_TICKSIZE = 16
COMPONENT_COLORBAR_LABELSIZE = 18
COMPONENT_COLORBAR_TICKSIZE = 15

# Edit these defaults directly for local paper-plot batches.  CLI arguments
# override them.
DEFAULT_RUN_DIR = Path(
    "runs/c2_weyl_k1_s42/c2_weyl_k1_s42_2026-06-22_03-11-30"
)
DEFAULT_NUM_POINTS = 2000


def slug(text: str) -> str:
    replacements = {
        r"\mathrm": "",
        r"\det": "det",
        r"\xi": "xi",
        r"\pi": "pi",
        r"\|": "norm",
        "≈": "approx",
        "→": "to",
        "–": " ",
        "—": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower() or "plot"


def finite_stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def print_stats(label: str, values: np.ndarray) -> dict[str, float | int]:
    stats = finite_stats(values)
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    n_finite = int(np.sum(np.isfinite(arr)))
    print(
        f"{label}: n={n_finite}  min={stats['min']:.8g}  "
        f"mean={stats['mean']:.8g}  max={stats['max']:.8g}  std={stats['std']:.8g}"
    )
    return stats


def symmetric_norm(values: np.ndarray, percentile: float = 98.0):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0), "RdBu_r"
    vmax = float(np.percentile(np.abs(vals), percentile))
    vmax = max(vmax, 1e-12)
    pos = np.abs(vals[np.abs(vals) > 0])
    linthresh = max(float(np.percentile(pos, 25)), vmax * 1e-4) if pos.size else vmax * 1e-4
    return SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax), "RdBu_r"


def linear_norm(values: np.ndarray, percentile=(2.0, 98.0)):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Normalize(vmin=0.0, vmax=1.0), "viridis"
    lo, hi = np.percentile(vals, percentile)
    if lo == hi:
        hi = lo + 1e-12
    return Normalize(vmin=float(lo), vmax=float(hi)), "viridis"


def component_norm_from_fields(*fields: np.ndarray):
    vals = np.concatenate([np.asarray(field).ravel() for field in fields])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0), COMPONENT_CMAP
    max_abs = max(float(np.max(np.abs(vals))), 1e-12)
    linthresh = min(COMPONENT_SYMLOG_LINTHRESH, max_abs * 0.1)
    linthresh = max(linthresh, max_abs * 1e-8)
    return (
        SymLogNorm(linthresh=float(linthresh), vmin=-max_abs, vmax=max_abs),
        COMPONENT_CMAP,
    )


def positive_log_norm(values: np.ndarray, cmap: str = "magma"):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return Normalize(vmin=0.0, vmax=1.0), cmap
    vmin, vmax = np.percentile(vals, [2.0, 98.0])
    vmin = max(float(vmin), float(vmax) * 1e-8, 1e-30)
    vmax = max(float(vmax), vmin * 10.0)
    return LogNorm(vmin=vmin, vmax=vmax), cmap


def target_norm(values: np.ndarray, target: float):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Normalize(vmin=target - 1.0, vmax=target + 1.0), "RdBu_r"
    lo, hi = np.percentile(vals, [2.0, 98.0])
    vmin = min(float(lo), target - 1e-9)
    vmax = max(float(hi), target + 1e-9)
    return TwoSlopeNorm(vmin=vmin, vcenter=target, vmax=vmax), "RdBu_r"


def even_numbered_abs_norm(values: np.ndarray, target: float | None = None, n_ticks: int = 5):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if target is not None:
            vmin = min(vmin, float(target))
            vmax = max(vmax, float(target))
        if vmin == vmax:
            pad = max(abs(vmin) * 0.05, 1e-6)
            vmin -= pad
            vmax += pad
    ticks = np.linspace(vmin, vmax, n_ticks)
    return Normalize(vmin=vmin, vmax=vmax), "viridis", ticks


def grid_shape_from_num_points(num_points: int) -> tuple[int, int]:
    n_r = max(2, int(np.floor(np.sqrt(num_points))))
    n_theta = max(3, int(np.ceil(num_points / n_r)))
    return n_r, n_theta


def penrose_grid(config: SchwarzschildConfig, n_r: int, n_theta: int, num_points: int | None = None):
    patch_width = float(getattr(config.visualisation, "patch_width_R2", 0.8))
    rr = np.linspace(1e-3, patch_width, n_r)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    R, TH = np.meshgrid(rr, th, indexing="ij")
    disc = np.stack([(R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()], axis=1)
    tx = disc_to_penrose(disc, inverse=False)
    if num_points is not None:
        tx = tx[:num_points]
    x5d = np.zeros((tx.shape[0], 5), dtype=np.float64)
    x5d[:, 0] = tx[:, 0]
    x5d[:, 1] = tx[:, 1]
    # Scalars are evaluated at the S^2 north pole.  In a good spherical run
    # this is representative; non-spherical angular variation should be tested
    # separately.
    return tx[:, 0], tx[:, 1], x5d


def resolve_local_single_s2_patch(config: SchwarzschildConfig) -> bool:
    local_vis = getattr(config.visualisation, "local_single_s2_patch", None)
    if local_vis is None:
        return bool(getattr(config.model_specific, "local_single_s2_patch", False))
    return bool(local_vis)


def resolve_local_s2_patch_idx(config: SchwarzschildConfig) -> int:
    local_idx = getattr(config.visualisation, "local_s2_patch_idx", None)
    if local_idx is None:
        local_idx = getattr(config.model_specific, "local_s2_patch_idx", 0)
    return int(local_idx)


def visualisation_sample(config: SchwarzschildConfig, n: int):
    r2 = PenroseSample(
        n,
        patch_width=float(getattr(config.visualisation, "patch_width_R2", 0.8)),
        density_power=float(getattr(config.visualisation, "density_power_R2", 1.0)),
    )
    if resolve_local_single_s2_patch(config):
        s2_stereo, patch_idx = StereoSampleSingleHemisphere(
            n,
            patch_idx=resolve_local_s2_patch_idx(config),
            patch_width=float(getattr(config.visualisation, "patch_width_S2", 1.0)),
            density_power=float(getattr(config.visualisation, "density_power_S2", 1.0)),
        )
    else:
        s2_stereo, patch_idx = StereoSampleHemisphere(
            n,
            patch_width=float(getattr(config.visualisation, "patch_width_S2", 1.0)),
            density_power=float(getattr(config.visualisation, "density_power_S2", 1.0)),
        )
    s2_cart = stereo_to_cartesian(
        tf.constant(s2_stereo, dtype=tf.float64),
        tf.constant(patch_idx, dtype=tf.int32),
    ).numpy()
    return r2, s2_stereo, patch_idx, s2_cart


def draw_penrose_3d_overlay(ax, z: float, zorder: int = 1000):
    pi = np.pi

    def plot_line(xs, ys):
        zs = np.full(len(xs), z, dtype=float)
        ax.plot(xs, ys, zs, color="black", linewidth=1.6, zorder=zorder)

    def singularity_line(x0, y0, x1, y1, waves=10, amp=0.02, pts=200):
        t = np.linspace(0, 1, pts)
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        px, py = -dy / length, dx / length
        wave = amp * np.sin(2 * np.pi * waves * t)
        return x + px * wave, y + py * wave

    top_wx, top_wy = singularity_line(-pi / 4, pi / 4, pi / 4, pi / 4)
    bot_wx, bot_wy = singularity_line(-pi / 4, -pi / 4, pi / 4, -pi / 4)
    plot_line(top_wx, top_wy)
    plot_line(bot_wx, bot_wy)
    plot_line([-pi / 2, -pi / 4], [0, pi / 4])
    plot_line([pi / 4, pi / 2], [pi / 4, 0])
    plot_line([-pi / 2, -pi / 4], [0, -pi / 4])
    plot_line([pi / 4, pi / 2], [-pi / 4, 0])
    plot_line([-pi / 4, pi / 4], [-pi / 4, pi / 4])
    plot_line([-pi / 4, pi / 4], [pi / 4, -pi / 4])
    plot_line([-pi / 2, pi / 2], [0, 0])
    plot_line([0, 0], [-pi / 4, pi / 4])
    ax.set_xticks(pi / 4 * np.array(range(-2, 3)))
    ax.set_yticks(pi / 4 * np.array(range(-1, 2)))
    ax.set_xticklabels(
        [
            r"$-\frac{\pi}{2}$",
            r"$-\frac{\pi}{4}$",
            r"$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
        ]
    )
    ax.set_yticklabels([r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$"])


def load_run_config(run_dir: Path) -> SchwarzschildConfig:
    for name in ("config.yaml", "hps_used.yaml"):
        path = run_dir / name
        if path.exists():
            return SchwarzschildConfig(**load_config(path))
    raise FileNotFoundError(f"No config.yaml or hps_used.yaml found in {run_dir}")


def assign_archived_weights(model: tf.keras.Model, model_path: Path) -> bool:
    """Fallback for older .keras files whose nested weights do not deserialize."""
    try:
        with zipfile.ZipFile(model_path) as zf:
            weights_blob = zf.read("model.weights.h5")
    except Exception:
        return False

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        tmp.write(weights_blob)
        tmp.flush()
        with h5py.File(tmp.name, "r") as h5:
            paths: list[str] = []
            h5.visititems(
                lambda name, obj: paths.append(name)
                if isinstance(obj, h5py.Dataset) and name.endswith(("/0", "/1"))
                else None
            )
            paths = sorted(paths, key=lambda p: (p.count("/"), p))
            if len(paths) != len(model.weights):
                return False
            for weight, h5_path in zip(model.weights, paths):
                arr = h5[h5_path][()]
                if tuple(weight.shape) != tuple(arr.shape):
                    return False
                weight.assign(arr)
    return True


def load_model(run_dir: Path, config: SchwarzschildConfig):
    model_path = run_dir / "final_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing final_model.keras in {run_dir}")
    custom_objects = {
        "GlobalModel": SchwarzschildGlobalModel,
        "PatchSubModel": SchwarzschildPatchSubModel,
        "SchwarzschildGlobalModel": SchwarzschildGlobalModel,
        "SchwarzschildPatchSubModel": SchwarzschildPatchSubModel,
    }
    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as exc:
        print(f"[warn] Keras load_model failed; trying archived-weight fallback: {exc}")
        model = SchwarzschildGlobalModel(config)
        _ = model(tf.zeros((1, 5), dtype=tf.float64))
        if not assign_archived_weights(model, model_path):
            raise
        return model


def is_supervised_run(run_dir: Path, config: SchwarzschildConfig) -> bool:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            if metadata.get("mode") == "supervised_mse":
                return True
        except Exception:
            pass

    multipliers = [
        getattr(config.loss, "einstein_multiplier", 0.0),
        getattr(config.model_specific, "kretschmann_multiplier", 0.0),
        getattr(config.model_specific, "speciality_index_multiplier", 0.0),
        getattr(config.model_specific, "killing_symmetry_multiplier", 0.0),
        getattr(config.model_specific, "k_repeller_multiplier", 0.0),
        getattr(config.model_specific, "speciality_index_rprofile_multiplier", 0.0),
    ]
    return all(float(value or 0.0) == 0.0 for value in multipliers)


def resolve_mode(
    config: SchwarzschildConfig,
    mode: Literal["auto", "schwarzschild", "general", "supervised"],
):
    if mode != "auto":
        if mode == "supervised":
            return "schwarzschild"
        return mode
    ms = config.model_specific
    if (
        float(getattr(ms, "speciality_index_rprofile_multiplier", 0.0) or 0.0) > 0.0
        and float(getattr(ms, "kretschmann_multiplier", 0.0) or 0.0) == 0.0
    ):
        return "general"
    return "schwarzschild"


def speciality_target(config: SchwarzschildConfig, mode: str) -> float:
    if mode == "general":
        return float(getattr(config.model_specific, "speciality_index_rprofile_centre", 1.0))
    return 1.0


def rho_from_weyl_j_and_k(weyl_j: np.ndarray, k_scalar: np.ndarray):
    k_abs = np.abs(np.asarray(k_scalar, dtype=np.float64))
    j_real = np.real(weyl_j)
    out = np.full_like(k_abs, np.nan, dtype=np.float64)
    valid = np.isfinite(k_abs) & np.isfinite(j_real) & (k_abs > 1e-12)
    out[valid] = -96.0 * j_real[valid] / np.power(k_abs[valid], 1.5)
    return out


def k_constant_values(k_scalar: np.ndarray, r_vals: np.ndarray, m: float):
    denom = max(abs(m) ** 2, 1e-30)
    return np.abs(k_scalar) * np.power(r_vals, 6) / denom


class PaperPlotter:
    def __init__(
        self,
        run_dir: Path,
        output_subdir: str = "plots_paper",
        mode: Literal["auto", "schwarzschild", "general", "supervised"] = "auto",
        num_points: int = DEFAULT_NUM_POINTS,
        n_r: int | None = None,
        n_theta: int | None = None,
        chunk: int = 1024,
    ):
        self.run_dir = run_dir
        self.config = load_run_config(run_dir)
        self.supervised = mode == "supervised" or is_supervised_run(run_dir, self.config)
        self.mode = resolve_mode(self.config, mode)
        self.loss_selection_mode = "supervised" if self.supervised else self.mode
        self.target_s_abs = speciality_target(self.config, self.mode)
        self.model = load_model(run_dir, self.config)
        self.lorentzian = bool(getattr(self.model.config.model_specific, "lorentzian", True))
        self.m = float(getattr(self.model.config.model_specific, "m", 1.0))
        self.num_points = int(num_points)
        inferred_n_r, inferred_n_theta = grid_shape_from_num_points(self.num_points)
        if n_r is None:
            n_r = inferred_n_r
        if n_theta is None:
            n_theta = inferred_n_theta
        self.n_r = n_r
        self.n_theta = n_theta
        self.chunk = chunk
        self.out_dir = run_dir / output_subdir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stats: dict[str, dict[str, float | int]] = {}
        self.selected_losses: dict[str, float | str] = {}
        self._random_visualisation_sample = None

    def path(self, name: str) -> Path:
        return self.out_dir / f"{slug(name)}.pdf"

    def get_random_visualisation_sample(self):
        if self._random_visualisation_sample is None:
            n = int(
                getattr(
                    self.config.visualisation,
                    "num_samples",
                    self.x5d.shape[0] if hasattr(self, "x5d") else DEFAULT_NUM_POINTS,
                )
                or (self.x5d.shape[0] if hasattr(self, "x5d") else DEFAULT_NUM_POINTS)
            )
            r2, s2_stereo, patch_idx, s2_cart = visualisation_sample(self.config, n)
            x5d = np.concatenate(
                [r2, s2_stereo, patch_idx.reshape(-1, 1).astype(np.float64)],
                axis=1,
            )
            self._random_visualisation_sample = (r2, s2_stereo, patch_idx, s2_cart, x5d)
        return self._random_visualisation_sample

    def metric_on_samples(self, x5d: np.ndarray):
        metrics = []
        for start in range(0, x5d.shape[0], self.chunk):
            xb = tf.constant(x5d[start : start + self.chunk], dtype=tf.float64)
            g_vec = self.model(xb, training=False)
            q_4d = xb[:, :4]
            patch_idx = tf.cast(xb[:, 4], tf.int32)
            g_5d = cholesky_from_vec(g_vec, lorentzian=self.lorentzian)
            jacobian = embedding_jacobian_stereo(q_4d, patch_idx)
            metric = tf.einsum("sAB,sAm,sBn->smn", g_5d, jacobian, jacobian)
            metrics.append(metric.numpy())
        return np.concatenate(metrics, axis=0)

    def evaluate(self):
        self.T, self.X, self.x5d = penrose_grid(
            self.config, self.n_r, self.n_theta, num_points=self.num_points
        )
        sub = self.model.submodel
        kernel = compute_ricci_and_kretschmann_embed
        kernel_name = "standard"
        if (
            getattr(self.config.model_specific, "ricci_kernel", "standard") == "optimised"
            and compute_ricci_and_kretschmann_embed_opt is not None
        ):
            kernel = compute_ricci_and_kretschmann_embed_opt
            kernel_name = "optimised"
        need_kretschmann = self.mode == "schwarzschild"

        while True:
            metrics, riccis, ks, wis, wjs = [], [], [], [], []
            try:
                for start in range(0, self.x5d.shape[0], self.chunk):
                    xb = tf.constant(self.x5d[start : start + self.chunk], dtype=tf.float64)
                    g, ric, k, wi, wj = kernel(
                        xb,
                        sub,
                        self.lorentzian,
                        need_ricci=True,
                        need_kretschmann=need_kretschmann,
                        need_speciality_index=True,
                    )
                    metrics.append(g.numpy())
                    riccis.append(ric.numpy())
                    if k is not None:
                        ks.append(k.numpy())
                    wis.append(wi.numpy())
                    wjs.append(wj.numpy())
                break
            except Exception as exc:
                if kernel_name != "optimised":
                    raise
                print(
                    "Warning: optimised Ricci kernel failed while plotting; "
                    "falling back to standard kernel for this run. "
                    f"Original error: {exc}"
                )
                kernel = compute_ricci_and_kretschmann_embed
                kernel_name = "standard"

        self.metric = np.concatenate(metrics, axis=0)
        self.ricci = np.concatenate(riccis, axis=0)
        self.metric_abs_component_mean = np.mean(np.abs(self.metric), axis=(1, 2))
        self.ricci_abs_component_mean = np.mean(np.abs(self.ricci), axis=(1, 2))
        self.metric_det = np.linalg.det(self.metric)
        self.kretschmann = np.concatenate(ks, axis=0) if ks else None
        self.weyl_i = np.concatenate(wis, axis=0)
        self.weyl_j = np.concatenate(wjs, axis=0)
        s = speciality_index_from_invariants(
            tf.constant(self.weyl_i), tf.constant(self.weyl_j), tf.constant(1e-12, dtype=tf.float64)
        ).numpy()
        self.speciality = np.where(np.abs(self.weyl_i) > 1e-6, s, np.nan + 0j)
        self.speciality_abs = np.abs(self.speciality)
        self.r = PenroseRadiusWeighting(tf.constant(self.x5d[:, :2], dtype=tf.float64), m=self.m).numpy()

        if self.mode == "schwarzschild":
            self.rho = rho_from_weyl_j_and_k(self.weyl_j, self.kretschmann)
            self.k_r6_over_m2 = k_constant_values(self.kretschmann, self.r, self.m)
            coords4d = tf.constant(self.x5d[:, :4], dtype=tf.float64)
            self.metric_analytic = AnalyticMetric_R2S2(
                coords4d, identity=False, lorentzian=True, m=self.m
            ).numpy()
            self.kretschmann_analytic = Analytic_Kretschmann(coords4d, m=self.m).numpy()
        else:
            self.rho = None
            self.k_r6_over_m2 = None
            self.metric_analytic = None
            self.kretschmann_analytic = None

    def plot_field(
        self,
        name: str,
        values: np.ndarray,
        norm,
        cmap: str,
        cbar_label: str,
        cbar_ticks=None,
        show_colorbar: bool = True,
        axis_labelsize: int | None = None,
        axis_ticksize: int | None = None,
    ):
        vals = np.asarray(values, dtype=float)
        tri = Triangulation(self.X, self.T)
        bad = ~np.isfinite(vals)
        if bad.any():
            tri.set_mask(np.any(bad[tri.triangles], axis=1))
        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(6.3, 4.9))
            pc = ax.tripcolor(tri, vals, norm=norm, cmap=cmap, shading="gouraud", rasterized=True)
            draw_penrose(ax)
            ax.set_xlim(-np.pi / 2 * 1.05, np.pi / 2 * 1.05)
            ax.set_ylim(-np.pi / 4 * 1.15, np.pi / 4 * 1.15)
            ax.set_aspect("equal")
            if axis_labelsize is not None:
                ax.xaxis.label.set_size(axis_labelsize)
                ax.yaxis.label.set_size(axis_labelsize)
            if axis_ticksize is not None:
                ax.tick_params(axis="both", labelsize=axis_ticksize)
            if show_colorbar:
                fig.colorbar(pc, ax=ax, shrink=0.82, label=cbar_label, ticks=cbar_ticks)
            fig.savefig(self.path(name))
            plt.close(fig)
        self.stats[name] = print_stats(name, vals)

    def plot_component_colorbar(self, norm, cmap: str):
        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(6.8, 0.525))
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=COMPONENT_COLORBAR_TICKSIZE)
            fig.savefig(self.path("metric_ricci_component_shared_colorbar"))
            plt.close(fig)

    def plot_sample_overview(self):
        r2, s2_stereo, patch_idx, s2_cart, _ = self.get_random_visualisation_sample()
        north = patch_idx == 0
        south = ~north

        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(6.3, 4.9))
            draw_penrose(ax)
            ax.scatter(r2[:, 1], r2[:, 0], alpha=0.8, s=4)
            ax.grid()
            fig.savefig(self.path("sample_penrose"))
            plt.close(fig)

        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            if np.any(north):
                ax.scatter(
                    s2_stereo[north, 0],
                    s2_stereo[north, 1],
                    alpha=0.8,
                    s=4,
                    label="north",
                )
            if np.any(south):
                ax.scatter(
                    s2_stereo[south, 0],
                    s2_stereo[south, 1],
                    alpha=0.8,
                    s=4,
                    label="south",
                )
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$q_1$")
            ax.set_ylabel(r"$q_2$")
            ax.legend()
            ax.grid()
            fig.savefig(self.path("sample_s2_stereographic_patches"))
            plt.close(fig)

        with plt.rc_context(RC):
            fig = plt.figure(figsize=(5.6, 5.2))
            ax = fig.add_subplot(111, projection="3d")
            if np.any(north):
                ax.scatter(
                    s2_cart[north, 0],
                    s2_cart[north, 1],
                    s2_cart[north, 2],
                    alpha=0.8,
                    s=4,
                    label="north",
                )
            if np.any(south):
                ax.scatter(
                    s2_cart[south, 0],
                    s2_cart[south, 1],
                    s2_cart[south, 2],
                    alpha=0.8,
                    s=4,
                    label="south",
                )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend()
            fig.savefig(self.path("sample_s2_embedded_r3"))
            plt.close(fig)

    def plot_metric_component_3d(
        self,
        name: str,
        t_values: np.ndarray,
        x_values: np.ndarray,
        values: np.ndarray,
        norm,
        cmap: str,
        zlim: tuple[float | None, float | None],
        elev: float = 30.0,
        azim: float = 45.0,
    ):
        vals = np.asarray(values, dtype=float)
        t_values = np.asarray(t_values, dtype=float)
        x_values = np.asarray(x_values, dtype=float)
        bottom, top = zlim
        finite = np.isfinite(vals) & np.isfinite(t_values) & np.isfinite(x_values)
        if bottom is not None:
            finite &= vals >= bottom
        if top is not None:
            finite &= vals <= top
        vals_plot = vals[finite]
        t_plot = t_values[finite]
        x_plot = x_values[finite]
        with plt.rc_context(RC):
            fig = plt.figure(figsize=(6.3, 4.9))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                x_plot,
                t_plot,
                vals_plot,
                c=vals_plot,
                cmap=cmap,
                norm=norm,
                s=5,
                depthshade=False,
                zorder=1,
            )
            ax.set_xlim(-np.pi / 2 * 1.1, np.pi / 2 * 1.1)
            ax.set_ylim(-np.pi / 4 * 1.1, np.pi / 4 * 1.1)
            if bottom is not None:
                ax.set_zlim(bottom=bottom)
            if top is not None:
                ax.set_zlim(top=top)
            z_extent = max(abs(bottom or 0.0), abs(top or 0.0), 1.0)
            overlay_above_points = not (
                bottom is not None and top is not None and bottom >= 0.0
            )
            if overlay_above_points:
                overlay_z = top if top is not None else float(np.nanmax(vals_plot))
                if hasattr(sc, "set_sort_zpos") and bottom is not None:
                    sc.set_sort_zpos(bottom - z_extent)
                draw_penrose_3d_overlay(ax, overlay_z, zorder=1000)
            else:
                overlay_z = bottom
                if hasattr(sc, "set_sort_zpos") and top is not None:
                    sc.set_sort_zpos(top + z_extent)
                draw_penrose_3d_overlay(ax, overlay_z, zorder=0)
            ax.set_xlabel(r"$X$")
            ax.set_ylabel(r"$T$")
            ax.view_init(elev=elev, azim=azim)
            fig.savefig(self.path(name))
            plt.close(fig)

    def plot_metric_component_3d_pairs(self):
        if self.metric_analytic is None:
            return
        r2, _, _, _, x5d = self.get_random_visualisation_sample()
        random_metric = self.metric_on_samples(x5d)
        random_metric_analytic = AnalyticMetric_R2S2(
            tf.constant(x5d[:, :4], dtype=tf.float64),
            identity=False,
            lorentzian=True,
            m=self.m,
        ).numpy()
        t_values = r2[:, 0]
        x_values = r2[:, 1]
        for i, j, zlim in [(0, 0, (-80.0, 0.0)), (1, 1, (0.0, 80.0))]:
            predicted = random_metric[:, i, j]
            analytic = random_metric_analytic[:, i, j]
            vals = np.concatenate([predicted, analytic])
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                norm = Normalize(vmin=-1.0, vmax=1.0)
            else:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                if vmin == vmax:
                    pad = max(abs(vmin) * 0.05, 1e-12)
                    vmin -= pad
                    vmax += pad
                norm = Normalize(vmin=vmin, vmax=vmax)
            self.plot_metric_component_3d(
                f"g_{i}{j}_3d_predicted",
                t_values,
                x_values,
                predicted,
                norm,
                "viridis",
                zlim,
            )
            self.plot_metric_component_3d(
                f"g_{i}{j}_3d_analytic",
                t_values,
                x_values,
                analytic,
                norm,
                "viridis",
                zlim,
            )

    def plot_components(self):
        norm, cmap = component_norm_from_fields(self.metric[:, :4, :4], self.ricci[:, :4, :4])
        self.plot_component_colorbar(norm, cmap)
        for i in range(4):
            for j in range(4):
                g = self.metric[:, i, j]
                r = self.ricci[:, i, j]
                self.plot_field(
                    f"g_{i}{j}",
                    g,
                    norm,
                    cmap,
                    rf"$g_{{{i}{j}}}$",
                    show_colorbar=False,
                    axis_labelsize=COMPONENT_AXIS_LABELSIZE,
                    axis_ticksize=COMPONENT_AXIS_TICKSIZE,
                )
                self.plot_field(
                    f"ricci_{i}{j}",
                    r,
                    norm,
                    cmap,
                    rf"$R_{{{i}{j}}}$",
                    show_colorbar=False,
                    axis_labelsize=COMPONENT_AXIS_LABELSIZE,
                    axis_ticksize=COMPONENT_AXIS_TICKSIZE,
                )

    def plot_kretschmann(self):
        if self.mode != "schwarzschild":
            return
        if self.mode == "schwarzschild" and self.kretschmann_analytic is not None:
            norm, cmap = positive_log_norm(
                np.concatenate([np.abs(self.kretschmann), np.abs(self.kretschmann_analytic)])
            )
            self.plot_field("kretschmann_predicted", np.abs(self.kretschmann), norm, cmap, r"$|K|$")
            self.plot_field(
                "kretschmann_analytic",
                np.abs(self.kretschmann_analytic),
                norm,
                cmap,
                r"$K_{\mathrm{analytic}}$",
            )
            residual = np.abs(self.kretschmann - self.kretschmann_analytic)
            rn, rc = positive_log_norm(residual, cmap="viridis")
            self.plot_field("kretschmann_abs_residual_predicted_minus_analytic", residual, rn, rc, r"$|K-K_{\mathrm{analytic}}|$")
        else:
            norm, cmap = positive_log_norm(np.abs(self.kretschmann))
            self.plot_field("kretschmann", np.abs(self.kretschmann), norm, cmap, r"$|K|$")

    def plot_speciality(self):
        norm, cmap, ticks = even_numbered_abs_norm(self.speciality_abs, self.target_s_abs)
        self.plot_field("speciality_index_abs", self.speciality_abs, norm, cmap, r"$|S|$", cbar_ticks=ticks)
        residual = np.abs(self.speciality_abs - self.target_s_abs)
        rn, rc = positive_log_norm(residual, cmap="viridis")
        self.plot_field(
            "speciality_index_abs_residual",
            residual,
            rn,
            rc,
            rf"$||S|-{self.target_s_abs:g}|$",
        )

    def plot_rho(self):
        if self.mode != "schwarzschild":
            return
        norm, cmap, ticks = even_numbered_abs_norm(np.abs(self.rho), RHO_TYPE_D_ABS)
        self.plot_field("rho_constant_abs", np.abs(self.rho), norm, cmap, r"$|\rho|$", cbar_ticks=ticks)
        residual = np.abs(np.abs(self.rho) - RHO_TYPE_D_ABS)
        rn, rc = positive_log_norm(residual, cmap="viridis")
        self.plot_field("rho_constant_abs_residual", residual, rn, rc, r"$||\rho|-1/\sqrt{12}|$")

    def print_invariant_summaries(self):
        print("\n--- Invariant summaries over plotted Penrose grid ---")
        self.stats["speciality_real"] = print_stats("speciality_index_real", np.real(self.speciality))
        self.stats["speciality_imag"] = print_stats("speciality_index_imag", np.imag(self.speciality))
        self.stats["speciality_abs"] = print_stats("speciality_index_abs", self.speciality_abs)
        if self.mode == "schwarzschild":
            self.stats["rho_signed"] = print_stats("rho_constant_signed", self.rho)
            self.stats["rho_abs"] = print_stats("rho_constant_abs", np.abs(self.rho))
            self.stats["k_r6_over_m2"] = print_stats("K_r6_over_m2", self.k_r6_over_m2)
        self.stats["metric_abs_component_mean"] = print_stats(
            "metric_abs_component_mean", self.metric_abs_component_mean
        )
        self.stats["ricci_abs_component_mean"] = print_stats(
            "ricci_abs_component_mean", self.ricci_abs_component_mean
        )
        self.stats["metric_det"] = print_stats("metric_det", self.metric_det)
        if self.mode == "schwarzschild":
            print(
                f"speciality |S| target: {self.target_s_abs:.8g}; "
                f"rho |rho| type-D target: {RHO_TYPE_D_ABS:.8g}; "
                "Schwarzschild K*r^6/m^2 target: 48"
            )
        else:
            print(f"speciality |S| target: {self.target_s_abs:.8g}")
        print("-----------------------------------------------\n")

    def riemannian_inverse_metric_on_grid(self) -> np.ndarray:
        if not hasattr(self, "_riemannian_inverse_metric"):
            chunks = []
            for start in range(0, self.x5d.shape[0], self.chunk):
                xb = tf.constant(self.x5d[start : start + self.chunk], dtype=tf.float64)
                inv = riemannian_inverse_metric_embed(xb, self.model.submodel)
                chunks.append(inv.numpy())
            self._riemannian_inverse_metric = np.concatenate(chunks, axis=0)
        return self._riemannian_inverse_metric

    def spd_contracted_rank2_norm_squared(self, tensor: np.ndarray) -> np.ndarray:
        inv = self.riemannian_inverse_metric_on_grid()
        tensor = np.asarray(tensor)
        return np.abs(np.einsum("sij,sik,sjl,skl->s", tensor, inv, inv, tensor))

    def finite_mean(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    def schwarzschild_curvature_normalized_einstein_loss(self) -> float:
        if self.kretschmann is None:
            return float("nan")
        ms = self.model.config.model_specific
        eps = float(getattr(ms, "curvature_norm_epsilon", 1e-3))
        kappa = float(getattr(ms, "curvature_norm_cap_kappa", 1.0))
        k_horizon = 48.0 * self.m**2 / max((2.0 * abs(self.m)) ** 6, 1e-30)
        k_cap = kappa * k_horizon
        ricci_norm = self.spd_contracted_rank2_norm_squared(self.ricci)
        denom = np.minimum(np.abs(self.kretschmann), k_cap) + eps
        return self.finite_mean(ricci_norm / denom)

    def general_weyl_normalized_einstein_loss(self) -> float:
        ms = self.model.config.model_specific
        eps = float(getattr(ms, "curvature_norm_epsilon", 1e-3))
        ricci_norm = self.spd_contracted_rank2_norm_squared(self.ricci)
        denom = np.abs(self.weyl_i) + eps
        return self.finite_mean(ricci_norm / denom)

    def schwarzschild_weyl_invariant_loss(self) -> float:
        if self.weyl_i is None:
            return float("nan")
        ms = self.model.config.model_specific
        eps = float(getattr(ms, "kretschmann_sqrt_epsilon", 1e-12))
        invariant_norm = float(getattr(ms, "weyl_invariant_norm", 1.0 / 16.0))
        target = np.sqrt(invariant_norm * 48.0) * self.m
        values = np.square(np.sqrt(np.abs(self.weyl_i) + eps) * self.r**3 - target)
        return self.finite_mean(values)

    def speciality_index_profile_loss(self) -> float:
        valid = np.abs(self.weyl_i) > 1e-6
        if not np.any(valid):
            return 0.0

        x_vars = self.x5d
        x_penrose = x_vars[:, 1]
        q1 = x_vars[:, 2]
        q2 = x_vars[:, 3]
        patch_idx = x_vars[:, 4].astype(np.int32)

        radial = x_penrose / (0.5 * np.pi)
        radial_modulation = 1.0 + 0.25 * radial
        r_sq = q1**2 + q2**2
        z_north = (1.0 - r_sq) / (1.0 + r_sq)
        z_south = (r_sq - 1.0) / (1.0 + r_sq)
        z_axis = np.where(patch_idx == 0, z_north, z_south)
        p2_axis = 0.5 * (3.0 * z_axis**2 - 1.0)
        raw_profile = radial_modulation * p2_axis

        profile_valid = raw_profile[valid]
        profile = (raw_profile - np.mean(profile_valid)) / (
            np.std(profile_valid) + 1e-12
        )
        amplitude = 0.25
        centre = float(
            getattr(
                self.model.config.model_specific,
                "speciality_index_rprofile_centre",
                2.0,
            )
        )
        target = centre + amplitude * profile
        values = np.square(np.abs(self.speciality - target) / amplitude)
        return self.finite_mean(values[valid])

    def killing_symmetry_loss_value(self) -> float:
        config = copy.deepcopy(self.model.config)
        weighter = WeightSchwarzschild(config, weight=False)
        loss_fn = KillingSymmetryLossEmbed(config, weighter)
        weighted_sum = 0.0
        n_total = 0
        for start in range(0, self.x5d.shape[0], self.chunk):
            stop = min(start + self.chunk, self.x5d.shape[0])
            xb = tf.constant(self.x5d[start:stop], dtype=tf.float64)
            metric = tf.constant(self.metric[start:stop], dtype=tf.float64)
            value = loss_fn.compute_from_precomputed(xb, metric, self.model.submodel)
            n = stop - start
            weighted_sum += float(value) * n
            n_total += n
        if n_total == 0:
            return float("nan")
        return weighted_sum / n_total

    def compute_selected_training_losses(self):
        if self.supervised:
            einstein_loss = self.schwarzschild_curvature_normalized_einstein_loss()
            killing_loss = self.killing_symmetry_loss_value()
            weyl_loss = self.schwarzschild_weyl_invariant_loss()
            total_loss = np.mean([einstein_loss, killing_loss, weyl_loss])
            selected: dict[str, float | str] = {
                "total_loss_normalised": float(total_loss),
                "loss_selection_mode": self.loss_selection_mode,
                "einstein_loss": einstein_loss,
                "killing_symmetry_loss": killing_loss,
                "weyl_kretschmann_loss": weyl_loss,
            }
            if self.metric_analytic is not None:
                diff = self.metric - self.metric_analytic
                per_point_mse = np.mean(np.square(diff), axis=(1, 2))
                analytic_norm = np.linalg.norm(self.metric_analytic, axis=(1, 2))
                diff_norm = np.linalg.norm(diff, axis=(1, 2))
                rel_frob = diff_norm / np.maximum(analytic_norm, 1e-30)
                selected.update(
                    {
                        "supervised_metric_mse": float(np.mean(per_point_mse)),
                        "supervised_metric_mse_std": float(np.std(per_point_mse)),
                        "supervised_metric_rel_frobenius_mean": float(np.mean(rel_frob)),
                        "supervised_metric_rel_frobenius_median": float(np.median(rel_frob)),
                        "supervised_metric_rel_frobenius_max": float(np.max(rel_frob)),
                    }
                )
            else:
                selected.update(
                    {
                        "supervised_metric_mse": float("nan"),
                        "supervised_metric_mse_std": float("nan"),
                        "supervised_metric_rel_frobenius_mean": float("nan"),
                        "supervised_metric_rel_frobenius_median": float("nan"),
                        "supervised_metric_rel_frobenius_max": float("nan"),
                    }
                )
            self.selected_losses = selected

            print("--- Selected supervised/physics diagnostics on plotted test grid ---")
            for key, value in selected.items():
                if key == "loss_selection_mode":
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value:.8g}")
            print("--------------------------------------------------------------\n")
            return

        selected: dict[str, float | str] = {
            "loss_selection_mode": self.mode,
        }
        if self.mode == "general":
            einstein_loss = self.general_weyl_normalized_einstein_loss()
            killing_loss = self.killing_symmetry_loss_value()
            srp_loss = self.speciality_index_profile_loss()
            srp_mu = float(
                getattr(
                    self.model.config.model_specific,
                    "speciality_index_rprofile_multiplier",
                    1.0,
                )
                or 1.0
            )
            total_loss = (einstein_loss + killing_loss + srp_mu * srp_loss) / (
                2.0 + srp_mu
            )
            selected.update(
                {
                    "total_loss_normalised": float(total_loss),
                    "einstein_loss": einstein_loss,
                    "killing_symmetry_loss": killing_loss,
                    "speciality_index_rprofile_loss": srp_loss,
                }
            )
        else:
            einstein_loss = self.schwarzschild_curvature_normalized_einstein_loss()
            killing_loss = self.killing_symmetry_loss_value()
            weyl_loss = self.schwarzschild_weyl_invariant_loss()
            ms = self.model.config.model_specific
            e_mu = float(
                getattr(self.model.config.loss, "einstein_multiplier", 0.0) or 0.0
            )
            w_mu = float(getattr(ms, "kretschmann_multiplier", 0.0) or 0.0)
            k_mu = float(getattr(ms, "killing_symmetry_multiplier", 0.0) or 0.0)
            denom = e_mu + w_mu + k_mu
            if denom <= 0.0:
                total_loss = float("nan")
            else:
                total_loss = (
                    e_mu * einstein_loss + w_mu * weyl_loss + k_mu * killing_loss
                ) / denom
            selected.update(
                {
                    "total_loss_normalised": float(total_loss),
                    "einstein_loss": einstein_loss,
                    "killing_symmetry_loss": killing_loss,
                    "weyl_kretschmann_loss": weyl_loss,
                }
            )
        self.selected_losses = selected

        print("--- Selected training losses on plotted test grid ---")
        for key, value in selected.items():
            if key == "loss_selection_mode":
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value:.8g}")
        print("----------------------------------------------------\n")

    def write_testdata(self):
        def is_metric_key(key: str) -> bool:
            return bool(re.fullmatch(r"g_[0-3][0-3]", key))

        def is_ricci_key(key: str) -> bool:
            return bool(re.fullmatch(r"ricci_[0-3][0-3]", key))

        metric_components = {
            key: self.stats[key] for key in sorted(self.stats) if is_metric_key(key)
        }
        ricci_components = {
            key: self.stats[key] for key in sorted(self.stats) if is_ricci_key(key)
        }
        scalar_and_invariant_fields = {
            key: self.stats[key]
            for key in sorted(self.stats)
            if not is_metric_key(key) and not is_ricci_key(key)
            and key != "speciality_index_abs"
        }

        constants = {
            "speciality_abs_target": self.target_s_abs,
            "mass_m": self.m,
        }
        if self.mode == "schwarzschild":
            constants.update(
                {
                    "rho_abs_type_d_target": RHO_TYPE_D_ABS,
                    "k_r6_over_m2_schwarzschild_target": 48.0,
                }
            )

        payload = {
            "metadata": {
                "run_dir": str(self.run_dir),
                "mode": self.mode,
                "loss_selection_mode": self.loss_selection_mode,
                "supervised": self.supervised,
                "num_points": int(self.x5d.shape[0]),
                "requested_num_points": self.num_points,
                "n_r": int(self.n_r),
                "n_theta": int(self.n_theta),
                "output_dir": str(self.out_dir),
            },
            "selected_training_losses": self.selected_losses,
            "constants": constants,
            "field_statistics": {
                "scalar_and_invariant_fields": scalar_and_invariant_fields,
                "metric_components": metric_components,
                "ricci_components": ricci_components,
            },
        }
        path = self.out_dir / "testdata.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote test data JSON: {path}")

    def run(self):
        print(f"Run directory: {self.run_dir}")
        print(f"Output directory: {self.out_dir}")
        print(
            f"Mode: {self.mode}; Penrose grid: n_r={self.n_r}, "
            f"n_theta={self.n_theta}, num_points={self.num_points}"
        )
        self.evaluate()
        self.plot_sample_overview()
        self.plot_components()
        self.plot_metric_component_3d_pairs()
        self.plot_kretschmann()
        self.plot_speciality()
        self.plot_rho()
        self.print_invariant_summaries()
        self.compute_selected_training_losses()
        self.write_testdata()


def main():
    parser = argparse.ArgumentParser(description="Final paper-style Penrose plots for a trained run.")
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        type=Path,
        help="Run directory containing final_model.keras",
    )
    parser.add_argument("--output-subdir", default="plots_paper")
    parser.add_argument("--mode", choices=["auto", "schwarzschild", "general", "supervised"], default="auto")
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--n-r", type=int, default=None)
    parser.add_argument("--n-theta", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    PaperPlotter(
        run_dir=args.run_dir,
        output_subdir=args.output_subdir,
        mode=args.mode,
        num_points=args.num_points,
        n_r=args.n_r,
        n_theta=args.n_theta,
        chunk=args.chunk,
    ).run()


if __name__ == "__main__":
    main()
