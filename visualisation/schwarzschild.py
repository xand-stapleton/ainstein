from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import SymLogNorm
from numpy.typing import NDArray

tf.keras.backend.set_floatx("float64")

import argparse
from typing import Optional, Sequence, Tuple

from configs.schwarzschild import SchwarzschildConfig
from geometry.ricci_opt import speciality_index_from_invariants
from geometry.schwarzschild import (Analytic_Kretschmann, AnalyticMetric_R2S2,
                                    PenroseRadiusWeighting,
                                    compute_kretschmann_scalar_embed,
                                    compute_ricci_and_kretschmann_embed,
                                    compute_ricci_tensor_embed,
                                    embedding_jacobian_stereo,
                                    stereo_to_cartesian)
from helper_functions.helper_functions import cholesky_from_vec
from losses.schwarzschild import (
    TotalSchwarzschildLoss,
    _rho_constant_summary,
    _speciality_index_summary,
)
from network.schwarzschild import (SchwarzschildGlobalModel,
                                   SchwarzschildPatchSubModel)
from sampling.ball import StereoSampleHemisphere, StereoSampleSingleHemisphere
from sampling.penrose import PenroseSample, draw_penrose
from visualisation.base import BaseVisualiser


class SchwarzschildVisualiser(BaseVisualiser):
    """
    Visualisation class for inspecting trained Schwarzschild neural network models.
    Generates test samples, predicts metrics and Ricci tensors, and produces 3D visualisations.
    """

    config: SchwarzschildConfig

    def __init__(self, model_parent: Path | str | None = None):
        network_custom_objects = {
            "GlobalModel": SchwarzschildGlobalModel,
            "PatchSubModel": SchwarzschildPatchSubModel,
        }
        super().__init__(model_parent, network_custom_objects)

        self.test_samples_5d: Optional[tf.Tensor] = None
        self.test_sample_R2: Optional[NDArray] = None
        # Cartesian S^2 coords (N, 3) for sphere-surface plots
        self.test_sample_S2_cart: Optional[NDArray] = None
        # Native stereo coords (N, 2) and hemisphere labels (N,)
        self.test_sample_S2_stereo: Optional[NDArray] = None
        self.test_sample_patch_idx: Optional[NDArray] = None

        self.predicted_metrics: Optional[NDArray] = None
        self.predicted_riccis: Optional[NDArray] = None

        self.lorentzian = self.config.model_specific.lorentzian
        self.generate_test_samples()

    def _use_local_single_s2_patch(self) -> bool:
        """Resolve whether visualisation should sample a single S^2 chart."""
        local_vis = getattr(self.config.visualisation, "local_single_s2_patch", None)
        if local_vis is None:
            return bool(getattr(self.config.model_specific, "local_single_s2_patch", False))
        return bool(local_vis)

    def _get_local_s2_patch_idx(self) -> int:
        """Resolve local chart index from visualisation or model-specific config."""
        local_idx = getattr(self.config.visualisation, "local_s2_patch_idx", None)
        if local_idx is None:
            local_idx = getattr(self.config.model_specific, "local_s2_patch_idx", 0)
        return int(local_idx)

    # ---------------- Sampling ----------------

    def generate_test_samples(self) -> None:
        """Generate a single (N, 5) test batch: [T, X, q1, q2, patch_idx_float]."""
        n = self.config.visualisation.num_samples

        self.test_sample_R2 = PenroseSample(
            n,
            patch_width=self.config.visualisation.patch_width_R2,
            density_power=self.config.visualisation.density_power_R2,
        )
        if self._use_local_single_s2_patch():
            s2_stereo, patch_idx = StereoSampleSingleHemisphere(
                n,
                patch_idx=self._get_local_s2_patch_idx(),
                patch_width=self.config.visualisation.patch_width_S2,
                density_power=self.config.visualisation.density_power_S2,
            )
        else:
            s2_stereo, patch_idx = StereoSampleHemisphere(
                n,
                patch_width=self.config.visualisation.patch_width_S2,
                density_power=self.config.visualisation.density_power_S2,
            )
        self.test_sample_S2_stereo = s2_stereo
        self.test_sample_patch_idx = patch_idx

        # Cartesian S^2 coords for sphere-surface plots
        self.test_sample_S2_cart = stereo_to_cartesian(
            tf.constant(s2_stereo, dtype=tf.float64),
            tf.constant(patch_idx, dtype=tf.int32),
        ).numpy()

        patch_idx_col = patch_idx.reshape(-1, 1).astype(np.float64)
        coords_4d = np.concatenate([self.test_sample_R2, s2_stereo], axis=1)
        self.test_samples_5d = tf.constant(
            np.concatenate([coords_4d, patch_idx_col], axis=1), dtype=tf.float64
        )

    def plot_points(self) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Visualise sampled points: Penrose diagram and S^2 (by hemisphere)."""
        if self.test_samples_5d is None:
            self.generate_test_samples()

        fig = plt.figure(figsize=(18, 5))

        # Penrose diagram
        ax0 = fig.add_subplot(1, 3, 1)
        ax0.set_title(r"$\mathbb{R}^2$ Penrose Diagram")
        draw_penrose(ax0)
        ax0.scatter(
            self.test_sample_R2[:, 1],
            self.test_sample_R2[:, 0],
            alpha=0.8,
            s=4,
        )
        ax0.grid()

        # S^2 stereo coords coloured by hemisphere
        ax1 = fig.add_subplot(1, 3, 2)
        ax1.set_title(r"$S^2$ stereographic")
        if self._use_local_single_s2_patch():
            local_patch_idx = self._get_local_s2_patch_idx()
            local_label = "north" if local_patch_idx == 0 else "south"
            ax1.scatter(
                self.test_sample_S2_stereo[:, 0],
                self.test_sample_S2_stereo[:, 1],
                alpha=0.8,
                s=4,
                label=local_label,
            )
        else:
            north = self.test_sample_patch_idx == 0
            south = ~north
            ax1.scatter(
                self.test_sample_S2_stereo[north, 0],
                self.test_sample_S2_stereo[north, 1],
                alpha=0.8,
                s=4,
                label="north",
            )
            ax1.scatter(
                self.test_sample_S2_stereo[south, 0],
                self.test_sample_S2_stereo[south, 1],
                alpha=0.8,
                s=4,
                label="south",
            )
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlabel(r"$q_1$")
        ax1.set_ylabel(r"$q_2$")
        ax1.legend()
        ax1.grid()

        # S^2 Cartesian, 3D
        ax2 = fig.add_subplot(1, 3, 3, projection="3d")
        C = self.test_sample_S2_cart
        ax2.scatter(
            C[:, 0], C[:, 1], C[:, 2], c=C[:, 2], cmap="coolwarm", alpha=0.8, s=4
        )
        ax2.set_title(r"$S^2$ Cartesian embedding")
        ax2.set_xlabel(r"$X_c$")
        ax2.set_ylabel(r"$Y_c$")
        ax2.set_zlabel(r"$Z_c$")

        plt.tight_layout()
        output_path = Path(self.model_parent) / "plots" / "sampled_points.pdf"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)
        return fig, [ax0, ax1, ax2]

    # ---------------- Computation ----------------

    def compute_quantities(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Predict the pulled-back 4D metric and Ricci tensor on the test batch."""
        if self.test_samples_5d is None:
            self.generate_test_samples()

        # Model outputs 15 Cholesky components of the 5D ambient metric G_{AB}.
        # Pull back to the 4D intrinsic metric: g_{mn} = G_{AB} J^A_m J^B_n.
        G_5d_vec = self.loaded_model(self.test_samples_5d, training=False)
        q_4d = self.test_samples_5d[:, :4]
        patch_idx = tf.cast(self.test_samples_5d[:, 4], tf.int32)
        G_5d = cholesky_from_vec(G_5d_vec, lorentzian=self.lorentzian)  # (N, 5, 5)
        J = embedding_jacobian_stereo(q_4d, patch_idx)  # (N, 5, 4)
        g_4d = tf.einsum("sAB,sAm,sBn->smn", G_5d, J, J)  # (N, 4, 4)
        self.predicted_metrics = g_4d.numpy()

        self.predicted_riccis = compute_ricci_tensor_embed(
            self.test_samples_5d, self.loaded_model.submodel, self.lorentzian
        ).numpy()
        return G_5d_vec.numpy(), self.predicted_metrics, self.predicted_riccis

    # ---------------- Plotting Helpers ----------------

    @staticmethod
    def _plot_component(
        ax: plt.Axes,
        x: NDArray,
        y: NDArray,
        z: NDArray,
        values: NDArray,
        title: str,
        xlab: str,
        ylab: str,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        elev: float = 30,
        azim: float = 45,
        cmap: str = "viridis",
    ) -> None:
        sc = ax.scatter(x, y, z, c=values, cmap=cmap)
        ax.set_title(title)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.view_init(elev=elev, azim=azim)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

    def _plot_s2_patches(
        self,
        ax_north: plt.Axes,
        ax_south: plt.Axes,
        values: NDArray,
        title_prefix: str,
        cmap: str = "viridis",
        norm=None,
        zscale: Optional[Tuple] = None,
    ) -> None:
        """Plot a scalar field on both S² stereographic patches as 3D scatter.

        Parameters
        ----------
        norm    : optional matplotlib Normalize instance for colour mapping.
        zscale  : optional (scale_name, linthresh) tuple, e.g. ('symlog', 1e-3).
        """
        north = self.test_sample_patch_idx == 0
        south = ~north
        for ax, mask, patch_name in [
            (ax_north, north, "north"),
            (ax_south, south, "south"),
        ]:
            if not np.any(mask):
                ax.set_title(rf"{title_prefix} ($S^2$, {patch_name}, no samples)")
                ax.set_axis_off()
                continue
            q1 = self.test_sample_S2_stereo[mask, 0]
            q2 = self.test_sample_S2_stereo[mask, 1]
            v = values[mask]
            finite = np.isfinite(v)
            q1 = q1[finite]
            q2 = q2[finite]
            v = v[finite]
            if len(v) == 0:
                ax.set_title(rf"{title_prefix} ($S^2$, {patch_name}, no finite samples)")
                ax.set_axis_off()
                continue
            sc = ax.scatter(q1, q2, v, c=v, cmap=cmap, norm=norm)
            ax.set_title(rf"{title_prefix} ($S^2$, {patch_name})")
            ax.set_xlabel(r"$q_1$")
            ax.set_ylabel(r"$q_2$")
            ax.set_zlabel("")
            if zscale is not None:
                ax.set_zscale(zscale[0], linthresh=zscale[1])
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

    # ---------------- Metric & Ricci Plots ----------------

    def _plot_pair_on_axes(
        self,
        ax_r2: plt.Axes,
        ax_s2_north: plt.Axes,
        ax_s2_south: plt.Axes,
        values: NDArray,
        i: int,
        j: int,
        symbol: str,
        elev: float,
        azim: float,
    ) -> None:
        """Plot one (i,j) component on Penrose (R²) and both S² patch axes."""
        z = values[:, i, j]
        T = self.test_samples_5d[:, 0].numpy()
        X = self.test_samples_5d[:, 1].numpy()

        self._plot_component(
            ax_r2,
            X,
            T,
            z,
            z,
            rf"${symbol}_{{{i},{j}}}$ ($\mathbb{{R}}^2$)",
            r"$X$",
            r"$T$",
            xlim=(-np.pi / 2 * 1.1, np.pi / 2 * 1.1),
            ylim=(-np.pi / 4 * 1.1, np.pi / 4 * 1.1),
            elev=elev,
            azim=azim,
        )
        draw_penrose(ax_r2)

        self._plot_s2_patches(
            ax_s2_north,
            ax_s2_south,
            z,
            rf"${symbol}_{{{i},{j}}}$",
        )

    def plot_all_pairs(self, elev: float = 30, azim: float = 45) -> None:
        """Plot predicted metric and Ricci tensor for all 4×4 index pairs."""
        if self.test_samples_5d is None:
            self.generate_test_samples()
        if self.predicted_metrics is None:
            self.compute_quantities()

        output_dir = Path(self.model_parent) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(4):
            for j in range(4):
                fig, axes = plt.subplots(
                    2, 3, figsize=(21, 12), subplot_kw={"projection": "3d"}
                )
                self._plot_pair_on_axes(
                    axes[0, 0],
                    axes[0, 1],
                    axes[0, 2],
                    self.predicted_metrics,
                    i,
                    j,
                    "g",
                    elev,
                    azim,
                )
                self._plot_pair_on_axes(
                    axes[1, 0],
                    axes[1, 1],
                    axes[1, 2],
                    self.predicted_riccis,
                    i,
                    j,
                    "R",
                    elev,
                    azim,
                )
                plt.tight_layout()
                plt.savefig(
                    output_dir / f"metric_pair_{i}_{j}.pdf", bbox_inches="tight"
                )
                plt.close(fig)

    # ---------------- Analytic Metrics ----------------

    def plot_analytic_metrics(
        self,
        identity_bool: bool = False,
        lorentzian_bool: bool = True,
        metric_index_1: list[int] | int = [0, 1, 2, 3],
        metric_index_2: list[int] | int = [0, 1, 2, 3],
        elev: float = 30,
        azim: float = 45,
    ) -> None:
        """Plot the analytic metric on Penrose diagram and Cartesian S^2."""
        if self.test_samples_5d is None:
            self.generate_test_samples()

        analytic_metric = AnalyticMetric_R2S2(
            self.test_samples_5d[:, :4],
            identity=identity_bool,
            lorentzian=lorentzian_bool,
            m=self.loaded_model.config.model_specific.m,
        ).numpy()

        if not isinstance(metric_index_1, list):
            metric_index_1 = [metric_index_1]
        if not isinstance(metric_index_2, list):
            metric_index_2 = [metric_index_2]

        index_pairs = [(i, j) for i in metric_index_1 for j in metric_index_2 if i <= j]

        output_dir = Path(self.model_parent) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        T = self.test_samples_5d[:, 0].numpy()
        X = self.test_samples_5d[:, 1].numpy()

        for i_idx, j_idx in index_pairs:
            z = analytic_metric[:, i_idx, j_idx]
            fig, axes = plt.subplots(
                1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"}
            )

            sc0 = axes[0].scatter(X, T, z, c=z, cmap="viridis")
            draw_penrose(axes[0])
            axes[0].set_title(rf"$g_{{{i_idx},{j_idx}}}$ Analytic ($\mathbb{{R}}^2$)")
            axes[0].set_xlim(-np.pi / 2 * 1.1, np.pi / 2 * 1.1)
            axes[0].set_ylim(-np.pi / 4 * 1.1, np.pi / 4 * 1.1)
            axes[0].set_xlabel(r"$X$")
            axes[0].set_ylabel(r"$T$")
            axes[0].view_init(elev=elev, azim=azim)
            plt.colorbar(sc0, ax=axes[0], shrink=0.6, pad=0.1)

            self._plot_s2_patches(
                axes[1],
                axes[2],
                z,
                rf"$g_{{{i_idx},{j_idx}}}$ Analytic",
            )

            plt.tight_layout()
            plt.savefig(
                output_dir / f"analytic_metrics_{i_idx}_{j_idx}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

    # ---------------- Determinant ----------------

    def plot_metric_determinant(self, elev: float = 30, azim: float = 45) -> None:
        """Plot det(g) with symlog z-scale (values are negative; log-scaled display)."""
        if self.predicted_metrics is None:
            self.compute_quantities()

        dets = np.linalg.det(self.predicted_metrics)
        linthresh = np.abs(dets).min() * 0.5
        norm = SymLogNorm(linthresh=linthresh, vmin=dets.min(), vmax=dets.max())
        T = self.test_samples_5d[:, 0].numpy()
        X = self.test_samples_5d[:, 1].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"})

        sc0 = axes[0].scatter(X, T, dets, c=dets, cmap="viridis", norm=norm)
        draw_penrose(axes[0])
        axes[0].set_title(r"$\det(g)$ ($\mathbb{R}^2$)")
        axes[0].set_xlim(-np.pi / 2 * 1.1, np.pi / 2 * 1.1)
        axes[0].set_ylim(-np.pi / 4 * 1.1, np.pi / 4 * 1.1)
        axes[0].set_xlabel(r"$X$")
        axes[0].set_ylabel(r"$T$")
        axes[0].set_zscale("symlog", linthresh=linthresh)
        axes[0].view_init(elev=elev, azim=azim)
        plt.colorbar(sc0, ax=axes[0], shrink=0.6, pad=0.1)

        self._plot_s2_patches(
            axes[1],
            axes[2],
            dets,
            r"$\det(g)$",
            cmap="plasma",
            norm=norm,
            zscale=("symlog", linthresh),
        )

        plt.tight_layout()
        output_dir = Path(self.model_parent) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "metric_determinant.pdf", bbox_inches="tight")
        plt.close(fig)

    def plot_analytic_metric_determinant(
        self,
        identity_bool: bool = False,
        lorentzian_bool: bool = True,
        elev: float = 30,
        azim: float = 45,
    ) -> None:
        """Plot det(g_analytic) over the Penrose diagram and on S² patches."""
        if self.test_samples_5d is None:
            self.generate_test_samples()

        analytic_metric = AnalyticMetric_R2S2(
            self.test_samples_5d[:, :4],
            identity=identity_bool,
            lorentzian=lorentzian_bool,
            m=self.loaded_model.config.model_specific.m,
        ).numpy()

        dets = np.linalg.det(analytic_metric)
        linthresh = np.abs(dets).min() * 0.5
        norm = SymLogNorm(linthresh=linthresh, vmin=dets.min(), vmax=dets.max())
        T = self.test_samples_5d[:, 0].numpy()
        X = self.test_samples_5d[:, 1].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"})

        sc0 = axes[0].scatter(X, T, dets, c=dets, cmap="viridis", norm=norm)
        draw_penrose(axes[0])
        axes[0].set_title(r"$\det(g_\mathrm{analytic})$ ($\mathbb{R}^2$)")
        axes[0].set_xlim(-np.pi / 2 * 1.1, np.pi / 2 * 1.1)
        axes[0].set_ylim(-np.pi / 4 * 1.1, np.pi / 4 * 1.1)
        axes[0].set_xlabel(r"$X$")
        axes[0].set_ylabel(r"$T$")
        axes[0].set_zscale("symlog", linthresh=linthresh)
        axes[0].view_init(elev=elev, azim=azim)
        plt.colorbar(sc0, ax=axes[0], shrink=0.6, pad=0.1)

        self._plot_s2_patches(
            axes[1],
            axes[2],
            dets,
            r"$\det(g_\mathrm{analytic})$",
            cmap="viridis",
            norm=norm,
            zscale=("symlog", linthresh),
        )

        plt.tight_layout()
        output_dir = Path(self.model_parent) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "metric_determinant_analytic.pdf", bbox_inches="tight")
        plt.close(fig)

    # ---------------- Kretschmann Scalar ----------------

    def _plot_scalar_on_domains(
        self,
        values: NDArray,
        title_suffix: str,
        filename: str,
        symbol: str,
        cmap: str = "inferno",
        elev: float = 30,
        azim: float = 45,
    ) -> None:
        T = self.test_samples_5d[:, 0].numpy()
        X = self.test_samples_5d[:, 1].numpy()
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"})

        sc0 = axes[0].scatter(
            X[finite],
            T[finite],
            values[finite],
            c=values[finite],
            cmap=cmap,
        )
        draw_penrose(axes[0])
        axes[0].set_title(rf"{symbol} {title_suffix} ($\mathbb{{R}}^2$)")
        axes[0].set_xlim(-np.pi / 2 * 1.1, np.pi / 2 * 1.1)
        axes[0].set_ylim(-np.pi / 4 * 1.1, np.pi / 4 * 1.1)
        axes[0].set_xlabel(r"$X$")
        axes[0].set_ylabel(r"$T$")
        axes[0].set_zlabel(symbol)
        axes[0].view_init(elev=elev, azim=azim)
        plt.colorbar(sc0, ax=axes[0], shrink=0.6, pad=0.1)

        self._plot_s2_patches(
            axes[1], axes[2], values, rf"{symbol} {title_suffix}", cmap=cmap
        )

        plt.tight_layout()
        output_dir = Path(self.model_parent) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    def _plot_kretschmann(
        self,
        K: NDArray,
        title_suffix: str,
        filename: str,
        symbol: str = r"$K$",
        cmap: str = "inferno",
        elev: float = 30,
        azim: float = 45,
    ) -> None:
        self._plot_scalar_on_domains(
            K,
            title_suffix,
            filename,
            symbol=symbol,
            cmap=cmap,
            elev=elev,
            azim=azim,
        )

    def plot_kretschmann_scalar(self, elev: float = 30, azim: float = 45) -> None:
        """Plot the Kretschmann scalar computed via automatic differentiation."""
        if self.test_samples_5d is None:
            self.generate_test_samples()
        K = compute_kretschmann_scalar_embed(
            self.test_samples_5d, self.loaded_model.submodel, self.lorentzian
        ).numpy()
        self._plot_kretschmann(K, "(predicted)", "kretschmann_scalar.pdf", elev=elev, azim=azim)

    def plot_analytic_kretschmann_scalar(
        self, elev: float = 30, azim: float = 45
    ) -> None:
        """Plot the analytic Kretschmann scalar."""
        if self.test_samples_5d is None:
            self.generate_test_samples()
        m = self.loaded_model.config.model_specific.m
        # Analytic K depends only on Penrose (T,X) coords; q1,q2 enter trivially
        K = Analytic_Kretschmann(self.test_samples_5d[:, :4], m=m).numpy()
        self._plot_kretschmann(
            K, "(analytic)", "kretschmann_scalar_analytic.pdf", elev=elev, azim=azim
        )

    @staticmethod
    def _plot_scalar_2d(
        values: NDArray,
        target: Optional[float],
        title: str,
        ylabel: str,
        filename: str,
        output_dir: Path,
        target_label: Optional[str] = None,
        yscale: str = "linear",
        ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> None:
        """2-D scatter: sample index (x) vs scalar value (y) with optional target line."""
        idx = np.arange(len(values))
        fig, ax = plt.subplots(figsize=(10, 4))
        valid = ~np.isnan(values)
        ax.scatter(idx[valid], values[valid], s=4, alpha=0.6)
        if target is not None:
            label = target_label if target_label is not None else f"target = {target:.4g}"
            ax.axhline(target, color="red", linewidth=1.2, linestyle="--", label=label)
            ax.legend()
        ax.set_xlabel("sample index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    def plot_kretschmann_r6(self, elev: float = 30, azim: float = 45) -> None:
        """Plot K·r⁶ (predicted). Should be constant = 48M² for Schwarzschild."""
        if self.test_samples_5d is None:
            self.generate_test_samples()
        m = self.loaded_model.config.model_specific.m
        K = compute_kretschmann_scalar_embed(
            self.test_samples_5d, self.loaded_model.submodel, self.lorentzian
        ).numpy()
        r_vals = PenroseRadiusWeighting(self.test_samples_5d[:, :2], m=m).numpy()
        K_r6 = np.abs(K) * r_vals ** 6
        target = 48.0 * m ** 2
        output_dir = Path(self.model_parent) / "plots"
        self._plot_scalar_2d(
            K_r6,
            target,
            r"$Kr^6$ (predicted)",
            r"$Kr^6$",
            "kretschmann_r6.pdf",
            output_dir,
            target_label=rf"target $= {target:.4g}$",
            yscale="log",
        )

    def plot_speciality_index(self, elev: float = 30, azim: float = 45) -> None:
        """Plot the real speciality index S = 27J²/I³ (predicted).

        Points with near-zero I are set to NaN and rendered as the per-cmap
        missing-data colour, since S is not meaningful there.
        """
        if self.test_samples_5d is None:
            self.generate_test_samples()
        _weyl_floor = 1e-6
        _eps_weyl = 1e-12
        _, _, _, weyl_i, weyl_j = compute_ricci_and_kretschmann_embed(
            self.test_samples_5d,
            self.loaded_model.submodel,
            self.lorentzian,
            need_ricci=False,
            need_kretschmann=False,
            need_speciality_index=True,
        )
        speciality_index = speciality_index_from_invariants(
            weyl_i, weyl_j, tf.cast(_eps_weyl, tf.math.real(weyl_i).dtype)
        )
        s_arr = speciality_index.numpy()
        i_arr = weyl_i.numpy()
        valid = np.abs(i_arr) > _weyl_floor
        speciality_index_vals = np.where(valid, np.real(s_arr), np.nan)
        output_dir = Path(self.model_parent) / "plots"
        self._plot_scalar_on_domains(
            speciality_index_vals,
            "(predicted)",
            "speciality_index.pdf",
            symbol=r"$S = 27J^2/I^3$",
            cmap="viridis",
            elev=elev,
            azim=azim,
        )
        self._plot_scalar_2d(
            speciality_index_vals,
            1.0,
            r"$S = 27J^2/I^3$ (predicted)",
            r"$S$",
            "speciality_index_by_sample.pdf",
            output_dir,
            target_label=r"type-D reference $= 1$",
            ylim=(0, None),
        )

    # ---------------- Spherical Symmetry ----------------

    def _killing_residual_stats(self) -> dict:
        """Compute ||L_ξ g||_F and ||L_ξ g||_F / ||g||_F for the three SO(3) generators.

        Uses the pullback definition:
            L_ξ g ≈ (φ_ε* g - g) / ε,   φ_ε(T,X,q1,q2) = (T, X, q1+ε·ξ², q2+ε·ξ³)
        where (φ_ε* g)_{mn} = g_{ab}(φ_ε) J^a_m J^b_n, J = I + ε·Dξ.

        North-chart SO(3) Killing vectors (used for all test points; south-chart
        L_x and L_y are the negatives of the north-chart formulae, so norms match):
            ξ^(z) = (-q2,  q1)
            ξ^(x) = (-q1·q2,  -½(1 - q1²+q2²))
            ξ^(y) = ( ½(1 + q1²-q2²),  q1·q2)

        Returns dict with keys "killing_z", "killing_x", "killing_y"; each a
        sub-dict with mean_abs, std_abs, mean_rel, std_rel.
        """
        if self.test_samples_5d is None:
            self.generate_test_samples()

        x_np = self.test_samples_5d.numpy()   # (N, 5)
        q1 = x_np[:, 2]
        q2 = x_np[:, 3]

        # Killing vector components (xi^2, xi^3) — north-chart formulae for all points
        killing_vecs = {
            "z": (-q2,                        q1                         ),
            "x": (-q1 * q2,                   -0.5 * (1 - q1**2 + q2**2)),
            "y": ( 0.5 * (1 + q1**2 - q2**2), q1 * q2                   ),
        }
        # Analytic Jacobians  d_{q_i} ξ^a  for north-chart formulae
        killing_jacobians = {
            "z": {"d1xi2": np.zeros_like(q1), "d2xi2": -np.ones_like(q1),
                  "d1xi3": np.ones_like(q1),  "d2xi3":  np.zeros_like(q1)},
            "x": {"d1xi2": -q2,  "d2xi2": -q1,
                  "d1xi3":  q1,  "d2xi3": -q2},
            "y": {"d1xi2":  q1,  "d2xi2": -q2,
                  "d1xi3":  q2,  "d2xi3":  q1},
        }

        def _metric_np(x_arr):
            x_tf = tf.constant(x_arr, dtype=tf.float64)
            G_vec = self.loaded_model.submodel(x_tf, training=False)
            q4d = x_tf[:, :4]
            pidx = tf.cast(x_tf[:, 4], tf.int32)
            G_5d = cholesky_from_vec(G_vec, lorentzian=self.lorentzian)
            J_emb = embedding_jacobian_stereo(q4d, pidx)
            return tf.einsum("sAB,sAm,sBn->smn", G_5d, J_emb, J_emb).numpy()

        g0 = _metric_np(x_np)     # (N, 4, 4)
        EPS = 1e-4
        N_pts = x_np.shape[0]

        results = {}
        for name, (xi2, xi3) in killing_vecs.items():
            jac = killing_jacobians[name]

            # Shifted coordinates: φ_EPS(x)
            x_shifted = x_np.copy()
            x_shifted[:, 2] += EPS * xi2
            x_shifted[:, 3] += EPS * xi3
            g_shifted = _metric_np(x_shifted)   # g(φ_EPS(x))

            # Flow Jacobian: J_flow = I + EPS · Dξ  (N, 4, 4)
            J_flow = np.tile(np.eye(4), (N_pts, 1, 1)).copy()
            J_flow[:, 2, 2] += EPS * jac["d1xi2"]
            J_flow[:, 2, 3] += EPS * jac["d2xi2"]
            J_flow[:, 3, 2] += EPS * jac["d1xi3"]
            J_flow[:, 3, 3] += EPS * jac["d2xi3"]

            # Pullback: (φ_EPS* g)_{mn} = g_{ab}(φ_EPS) J^a_m J^b_n
            g_pb = np.einsum("sab,sam,sbn->smn", g_shifted, J_flow, J_flow)

            # Lie derivative approximation
            lie_g = (g_pb - g0) / EPS              # (N, 4, 4)

            frob_lie = np.sqrt(np.sum(lie_g ** 2, axis=(1, 2)))   # (N,)
            frob_g   = np.sqrt(np.sum(g0   ** 2, axis=(1, 2)))    # (N,)
            rel_frob = frob_lie / np.maximum(frob_g, 1e-30)        # (N,)

            results[f"killing_{name}"] = {
                "mean_abs": float(np.mean(frob_lie)),
                "std_abs":  float(np.std(frob_lie)),
                "mean_rel": float(np.mean(rel_frob)),
                "std_rel":  float(np.std(rel_frob)),
                "abs_vals": frob_lie,
                "rel_vals": rel_frob,
            }

        return results

    # ---------------- Loss Evaluation ----------------

    def evaluate_and_save_losses(self, train_losses=None) -> dict:
        """
        Evaluate all loss components on the visualisation test sample, print a
        summary, and save everything to ``losses.json`` in the run directory.

        Parameters
        ----------
        train_losses : list | None
            Per-batch training losses returned by ``network.train()``.
            If ``None``, no training-loss entry is written to the JSON.

        Returns
        -------
        dict
            The computed loss data written to the JSON file.
        """
        loss_fn = TotalSchwarzschildLoss(self.config)
        if self.test_samples_5d is None:
            self.generate_test_samples()
        x_vars = self.test_samples_5d
        metric_pred = self.loaded_model(x_vars, training=False)
        total_loss, constituents = loss_fn.call(
            self.loaded_model,
            x_vars,
            metric_pred,
            return_constituents=True,
            val_print=False,
        )
        total_val = float(total_loss)

        # ---- Compute speciality index over valid (non-flat) test points ----
        if getattr(self.config.model_specific, "speciality_index_rprofile_multiplier", 0.0) > 0.0:
            _speciality_index_target = getattr(
                self.config.model_specific, "speciality_index_rprofile_centre", 1.0
            )
            _speciality_index_target_label = "profile centre"
        else:
            _speciality_index_target = 1.0
            _speciality_index_target_label = "type-D reference"
        (
            metric_pred_mat,
            ricci_tensor,
            k_scalar,
            weyl_i,
            weyl_j,
        ) = loss_fn._ricci_kernel(
            x_vars,
            self.loaded_model.submodel,
            self.lorentzian,
            need_ricci=True,
            need_kretschmann=True,
            need_speciality_index=True,
        )
        mean_abs_metric = float(tf.reduce_mean(tf.abs(metric_pred_mat)))
        mean_abs_ricci = float(tf.reduce_mean(tf.abs(ricci_tensor)))
        ricci_metric_abs_ratio = mean_abs_ricci / max(mean_abs_metric, 1e-30)
        ricci_frobenius_relative_mean = float(
            tf.reduce_mean(
                tf.norm(ricci_tensor, axis=(1, 2))
                / (tf.norm(metric_pred_mat, axis=(1, 2)) + tf.cast(1e-12, tf.float64))
            )
        )
        speciality_summary = _speciality_index_summary(weyl_i, weyl_j)
        rho_summary = _rho_constant_summary(weyl_j, k_scalar)

        # ---- Compute K·r⁶ ----
        m_val = self.loaded_model.config.model_specific.m
        r_vals = PenroseRadiusWeighting(x_vars[:, :2], m=m_val)
        K_r6_vals = tf.abs(k_scalar) * r_vals ** 6
        K_r6_mean = float(tf.reduce_mean(K_r6_vals))
        K_r6_std = float(tf.math.reduce_std(K_r6_vals))
        K_r6_target = 48.0 * m_val ** 2

        # ---- Killing vector residuals ----
        killing_stats = self._killing_residual_stats()

        # ---- Print ----
        print("\n--- Test-sample loss breakdown ---")
        for key, val in constituents.items():
            if isinstance(val, list):
                fmt = ", ".join(f"{v:.6g}" for v in val)
                print(f"  {key}: [{fmt}]")
            else:
                print(f"  {key}: {val:.6g}")
        print(f"  total_loss (normalised): {total_val:.6g}")
        print(
            "  speciality index S.real: "
            f"median = {speciality_summary['speciality_index_real_median']:.6g},"
            f" trimmed_mean = {speciality_summary['speciality_index_real_trimmed_mean']:.6g},"
            f" mean = {speciality_summary['speciality_index_real_mean']:.6g},"
            f" std = {speciality_summary['speciality_index_real_std']:.6g},"
            f" trimmed_std = {speciality_summary['speciality_index_real_trimmed_std']:.6g}"
            f"  [{_speciality_index_target_label}"
            f" {_speciality_index_target:.6g},"
            f" n_valid={speciality_summary['speciality_index_n_valid']}/"
            f"{speciality_summary['speciality_index_n_total']}]"
        )
        print(
            "  speciality index S.imag: "
            f"median = {speciality_summary['speciality_index_imag_median']:.6g},"
            f" trimmed_mean = {speciality_summary['speciality_index_imag_trimmed_mean']:.6g},"
            f" mean = {speciality_summary['speciality_index_imag_mean']:.6g},"
            f" std = {speciality_summary['speciality_index_imag_std']:.6g},"
            f" trimmed_std = {speciality_summary['speciality_index_imag_trimmed_std']:.6g};"
            f" trimmed_outliers = {speciality_summary['speciality_index_real_trimmed_outlier_count']}"
        )
        print(
            "  rho constant |rho|:       "
            f"median = {rho_summary['rho_constant_abs_median']:.6g},"
            f" trimmed_mean = {rho_summary['rho_constant_abs_trimmed_mean']:.6g},"
            f" mean = {rho_summary['rho_constant_abs_mean']:.6g},"
            f" std = {rho_summary['rho_constant_abs_std']:.6g},"
            f" CoV = {rho_summary['rho_constant_abs_cov']:.6g}"
            f"  [target {rho_summary['rho_constant_target_abs']:.6g},"
            f" n_valid={rho_summary['rho_constant_n_valid']}/"
            f"{rho_summary['rho_constant_n_total']}]"
        )
        print(
            "  rho constant signed:      "
            f"median = {rho_summary['rho_constant_signed_median']:.6g},"
            f" trimmed_mean = {rho_summary['rho_constant_signed_trimmed_mean']:.6g},"
            f" mean = {rho_summary['rho_constant_signed_mean']:.6g},"
            f" std = {rho_summary['rho_constant_signed_std']:.6g}"
        )
        print(
            f"  K*r^6:                    mean = {K_r6_mean:.6g},  std = {K_r6_std:.6g}"
            f"  [target {K_r6_target:.6g}]"
        )
        print(f"  mean |g_ij|:             {mean_abs_metric:.6g}")
        print(f"  mean |R_ij|:             {mean_abs_ricci:.6g}")
        print(f"  mean |R_ij| / mean |g_ij|: {ricci_metric_abs_ratio:.6g}")
        print(
            f"  mean ||R||_F / ||g||_F:  {ricci_frobenius_relative_mean:.6g}"
        )
        print("  SO(3) Killing residuals ||L_ξ g||_F / ||g||_F (mean ± std):")
        for gen in ("z", "x", "y"):
            ks = killing_stats[f"killing_{gen}"]
            print(f"    L_{gen}: abs={ks['mean_abs']:.3e} ± {ks['std_abs']:.3e},"
                  f"  rel={ks['mean_rel']:.3e} ± {ks['std_rel']:.3e}")
        print("----------------------------------\n")

        # ---- Build JSON payload ----
        def _to_serialisable(v):
            if isinstance(v, list):
                return [float(x) for x in v]
            return float(v) if v is not None else None

        loss_data = {
            "test_losses": {
                **{k: _to_serialisable(v) for k, v in constituents.items()},
                "total_loss_normalised": total_val,
                "speciality_index_target": _speciality_index_target,
                "speciality_index_target_label": _speciality_index_target_label,
                **speciality_summary,
                **rho_summary,
                "K_r6_mean": K_r6_mean,
                "K_r6_std": K_r6_std,
                "K_r6_target": K_r6_target,
                "mean_abs_metric": mean_abs_metric,
                "mean_abs_ricci": mean_abs_ricci,
                "ricci_metric_abs_ratio": ricci_metric_abs_ratio,
                "ricci_frobenius_relative_mean": ricci_frobenius_relative_mean,
                **{f"killing_{g}_{s}": killing_stats[f"killing_{g}"][s]
                   for g in ("z", "x", "y")
                   for s in ("mean_abs", "std_abs", "mean_rel", "std_rel")},
            },
        }

        if train_losses is not None:
            train_floats = [float(l) for l in train_losses]
            loss_data["train_losses"] = {
                "history": train_floats,
                "final": train_floats[-1] if train_floats else None,
                "min": min(train_floats) if train_floats else None,
                "mean": float(np.mean(train_floats)) if train_floats else None,
            }

        out_path = Path(self.model_parent) / "losses.json"
        with open(out_path, "w") as f:
            json.dump(loss_data, f, indent=2)
        print(f"Losses saved to {out_path}")

        if train_losses is not None:
            self.plot_loss_curve(train_floats)

        return loss_data

    def plot_loss_curve(self, train_losses: list) -> plt.Figure:
        """
        Plot the training loss history and save to plots/loss_curve.pdf.

        Parameters
        ----------
        train_losses : list of float
            Per-epoch training losses returned by ``network.train()``.
        """
        epochs = np.arange(1, len(train_losses) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: linear scale
        axes[0].plot(epochs, train_losses, linewidth=1.0, color="steelblue")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training loss")
        axes[0].set_title("Training loss (linear)")
        axes[0].grid(True, alpha=0.4)

        # Right: log scale — makes convergence rate visible
        positive = [l for l in train_losses if l > 0]
        if positive:
            axes[1].semilogy(epochs, train_losses, linewidth=1.0, color="steelblue")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Training loss (log scale)")
            axes[1].set_title("Training loss (log scale)")
            axes[1].grid(True, which="both", alpha=0.4)
        else:
            axes[1].set_visible(False)

        run_name = getattr(self.config.metadata, "run_name", "") or ""
        if run_name:
            fig.suptitle(f"Run: {run_name}", fontsize=11)

        plt.tight_layout()
        output_path = Path(self.model_parent) / "plots" / "loss_curve.pdf"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Loss curve saved to {output_path}")
        return fig

    def run_all(self, train_losses=None) -> None:
        """Run all visualisation steps: compute, plot, and save."""
        self.compute_quantities()
        self.plot_points()
        self.plot_all_pairs()
        self.plot_analytic_metrics()
        self.plot_metric_determinant()
        self.plot_analytic_metric_determinant()
        self.plot_kretschmann_scalar()
        self.plot_analytic_kretschmann_scalar()
        self.plot_kretschmann_r6()
        self.plot_speciality_index()
        self.evaluate_and_save_losses(train_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise Schwarzschild models in a directory."
    )
    parser.add_argument(
        "--model_parent",
        type=str,
        required=True,
        help="Path to directory containing trained Schwarzschild models.",
    )
    args = parser.parse_args()

    model_parent = Path(args.model_parent)
    if not model_parent.exists():
        raise FileNotFoundError(f"Directory not found: {model_parent}")

    print(f"\nProcessing model: {model_parent}")

    try:
        # Load model and visualiser
        visualiser = SchwarzschildVisualiser(model_parent)

        # Run computations
        visualiser.compute_quantities()

        visualiser.plot_points()
        visualiser.plot_all_pairs()
        visualiser.plot_analytic_metrics()
        visualiser.plot_metric_determinant()
        visualiser.plot_analytic_metric_determinant()
        visualiser.plot_kretschmann_scalar()
        visualiser.plot_analytic_kretschmann_scalar()
        visualiser.plot_kretschmann_r6()
        visualiser.plot_speciality_index()
        visualiser.evaluate_and_save_losses()

        print(f"Finished processing {model_parent}")
    except Exception as e:
        print(f"Error while processing {model_parent}: {e}")
