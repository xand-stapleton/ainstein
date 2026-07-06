from __future__ import annotations

"""Sampling Schemes for the Schwarzschild Penrose diagram"""

# Import libraries
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


def PenroseSample(
    num_pts,
    patch_width=1.0,
    density_power=1.0,
):
    # Sample a 2d disc
    radii_sample = np.random.beta(1 / density_power, density_power, size=num_pts)
    radii_sample *= patch_width
    angle_sample = np.random.uniform(high=2 * np.pi, size=num_pts)
    disc_sample = np.stack(
        (radii_sample * np.cos(angle_sample), radii_sample * np.sin(angle_sample)),
        axis=1,
    )

    hexagonal_sample = disc_to_penrose(
        disc_sample, delta_scale=1.0, eps=1e-12, inverse=False
    )

    return hexagonal_sample


def PenroseRegionMixtureSample(
    num_pts,
    patch_width=1.0,
    density_power=1.0,
    exterior_fraction=0.25,
    interior_fraction=0.55,
    horizon_fraction=0.15,
    singularity_fraction=0.05,
    horizon_width=0.06,
    singularity_width=0.10,
    interior_only=False,
):
    """Sample the Penrose diagram from an explicit region mixture.

    The ordinary ``PenroseSample`` controls radial density but does not directly
    decide how much of each Penrose region is seen.  This sampler keeps that
    base density and adds rejection sampling for exterior, interior, horizon,
    and singularity-side interior buckets.
    """
    if interior_only:
        exterior_fraction = 0.0
        horizon_fraction = 0.0
        if interior_fraction + singularity_fraction <= 0.0:
            interior_fraction = 1.0

    counts = _region_counts(
        num_pts,
        {
            "exterior": exterior_fraction,
            "interior": interior_fraction,
            "horizon": horizon_fraction,
            "singularity": singularity_fraction,
        },
    )
    chunks = []
    for region, n_region in counts.items():
        if n_region <= 0:
            continue
        if interior_only and region in ("interior", "singularity"):
            future_count = n_region // 2
            past_count = n_region - future_count
            if future_count > 0:
                chunks.append(
                    _sample_penrose_region(
                        future_count,
                        region,
                        patch_width,
                        density_power,
                        horizon_width,
                        singularity_width,
                        interior_side="future",
                    )
                )
            if past_count > 0:
                chunks.append(
                    _sample_penrose_region(
                        past_count,
                        region,
                        patch_width,
                        density_power,
                        horizon_width,
                        singularity_width,
                        interior_side="past",
                    )
                )
            continue

        chunks.append(
            _sample_penrose_region(
                n_region,
                region,
                patch_width,
                density_power,
                horizon_width,
                singularity_width,
            )
        )
    if not chunks:
        return PenroseSample(num_pts, patch_width, density_power)

    sample = np.concatenate(chunks, axis=0)
    np.random.shuffle(sample)
    return sample


def _region_counts(num_pts, fractions):
    clean = {name: max(0.0, float(value)) for name, value in fractions.items()}
    total = sum(clean.values())
    if total <= 0.0:
        clean = {"exterior": 1.0, "interior": 0.0, "horizon": 0.0, "singularity": 0.0}
        total = 1.0

    exact = {name: num_pts * value / total for name, value in clean.items()}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    remainder = num_pts - sum(counts.values())
    if remainder > 0:
        by_residual = sorted(
            exact,
            key=lambda name: exact[name] - counts[name],
            reverse=True,
        )
        for name in by_residual[:remainder]:
            counts[name] += 1
    return counts


def _sample_penrose_region(
    num_pts,
    region,
    patch_width,
    density_power,
    horizon_width,
    singularity_width,
    interior_side=None,
):
    accepted = []
    remaining = num_pts
    attempts = 0
    while remaining > 0 and attempts < 30:
        candidate_count = max(2048, int(np.ceil(remaining * 12)))
        candidates = PenroseSample(candidate_count, patch_width, density_power)
        mask = _penrose_region_mask(
            candidates,
            region,
            horizon_width=horizon_width,
            singularity_width=singularity_width,
            patch_width=patch_width,
            interior_side=interior_side,
        )
        selected = candidates[mask]
        if selected.size > 0:
            take = selected[:remaining]
            accepted.append(take)
            remaining -= take.shape[0]
        attempts += 1

    if remaining > 0:
        scope = f"{interior_side} " if interior_side is not None else ""
        raise RuntimeError(
            f"Could not rejection-sample {remaining} remaining {scope}{region} "
            "Penrose points after 30 attempts. Relax the region fractions, "
            "increase the region width, or increase density_power."
        )

    return np.concatenate(accepted, axis=0)


def _penrose_region_mask(
    samples,
    region,
    horizon_width,
    singularity_width,
    patch_width=1.0,
    interior_side=None,
):
    T = samples[:, 0]
    X = samples[:, 1]
    horizon_distance = np.minimum(np.abs(T - X), np.abs(T + X)) / np.sqrt(2.0)
    horizon = horizon_distance <= horizon_width
    if interior_side == "future":
        interior_base = T > np.abs(X)
    elif interior_side == "past":
        interior_base = T < -np.abs(X)
    else:
        interior_base = np.abs(T) > np.abs(X)
    exterior_base = np.abs(X) > np.abs(T)
    sampled_singularity_edge = np.pi / 4.0 * patch_width
    singularity = (
        interior_base
        & (np.abs(T) >= (sampled_singularity_edge - singularity_width))
        & (np.abs(X) <= (sampled_singularity_edge - 0.5 * singularity_width))
    )

    match region:
        case "exterior":
            return exterior_base & ~horizon
        case "interior":
            return interior_base & ~horizon & ~singularity
        case "horizon":
            return horizon
        case "singularity":
            return singularity
        case _:
            raise ValueError(f"Unknown Penrose region '{region}'.")


# Function to draw the Penrose diagram outline
def draw_penrose(ax):
    """
    Draws the Penrose diagram structure on an existing Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    """
    # Constants
    pi = np.pi

    # Function for drawing the singularity line on the Penrose diagrams
    def singularity_line(x0, y0, x1, y1, waves=10, amp=0.02, pts=200):
        t = np.linspace(0, 1, pts)
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy)
        px, py = -dy / L, dx / L
        s = amp * np.sin(2 * np.pi * waves * t)
        return x + px * s, y + py * s

    # Singularities (top and bottom)
    top_wx, top_wy = singularity_line(-pi / 4, pi / 4, pi / 4, pi / 4)
    bot_wx, bot_wy = singularity_line(-pi / 4, -pi / 4, pi / 4, -pi / 4)
    ax.plot(top_wx, top_wy, "k")
    ax.plot(bot_wx, bot_wy, "k")

    # Upper lightlike boundaries
    ax.plot([-pi / 2, -pi / 4], [0, pi / 4], "k")  # left upper
    ax.plot([pi / 4, pi / 2], [pi / 4, 0], "k")  # right upper

    # Lower lightlike boundaries
    ax.plot([-pi / 2, -pi / 4], [0, -pi / 4], "k")  # left lower
    ax.plot([pi / 4, pi / 2], [-pi / 4, 0], "k")  # right lower

    # Event horizons
    ax.plot([-pi / 4, pi / 4], [-pi / 4, pi / 4], "k")
    ax.plot([-pi / 4, pi / 4], [pi / 4, -pi / 4], "k")

    # Horizontal and vertical axes
    ax.plot([-pi / 2, pi / 2], [0, 0], "k")
    ax.plot([0, 0], [-pi / 4, pi / 4], "k")
    ax.set_xlabel("X")
    ax.set_ylabel("T")

    # Ticks and labels
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

    if not hasattr(ax, "zaxis"):
        ax.set_aspect("equal")

    return


def disc_to_penrose(tx, delta_scale=1.0, eps=1e-12, inverse=False):
    tx = np.asarray(tx, dtype=float)

    if not inverse:
        # --- forward (circle -> euclidean offset hex) ---
        out = np.zeros_like(tx)
        r = np.linalg.norm(tx, axis=-1, keepdims=True)
        r = np.clip(r, 0.0, 1.0)
        delta = delta_scale * np.pi / 4 * (1.0 - r)

        u = np.zeros_like(tx)
        mask = r[..., 0] > 0
        u[mask] = tx[mask] / r[mask]
        ut = u[..., 0]
        ux = u[..., 1]

        numA = np.pi / 4.0 - delta[..., 0]
        numBC = np.pi / 2.0 - delta[..., 0] * np.sqrt(2)

        A = np.where(np.abs(ut) > eps, numA / np.abs(ut), np.inf)
        B = np.where(np.abs(ut + ux) > eps, numBC / np.abs(ut + ux), np.inf)
        C = np.where(np.abs(ux - ut) > eps, numBC / np.abs(ux - ut), np.inf)

        m = np.minimum.reduce([A, B, C])
        out[mask] = (m[mask])[..., None] * u[mask]
        return out

    else:
        # --- inverse (euclidean offset hex -> circle) ---
        n = np.linalg.norm(tx, axis=-1, keepdims=True)
        T = tx[..., 0]
        X = tx[..., 1]

        delta1 = np.pi / 4.0 - np.abs(T)
        delta2 = (np.pi / 2.0 - np.abs(T + X)) / np.sqrt(2.0)
        delta3 = (np.pi / 2.0 - np.abs(X - T)) / np.sqrt(2.0)
        delta = np.minimum.reduce([delta1, delta2, delta3])[..., None]

        r = np.clip(1.0 - delta / (np.pi / 4 * delta_scale), 0.0, 1.0)
        v = np.zeros_like(tx)
        mask = n[..., 0] > eps
        v[mask] = tx[mask] / n[mask]
        return r * v


###############################################################################
def disc_to_penrose_tf(tx, delta_scale=1.0, eps=1e-12):
    """
    TF-compatible inverse disc_to_penrose (euclidean-offset-hex -> circle).
    Equivalent to disc_to_penrose(tx, inverse=True) but uses TF ops so it
    can be used inside @tf.function / GradientTape contexts.
    """
    n = tf.norm(tx, axis=-1, keepdims=True)  # [batch, 1]
    T = tx[..., 0]  # [batch]
    X = tx[..., 1]  # [batch]

    sqrt2 = tf.cast(tf.sqrt(2.0), tx.dtype)
    pi = tf.cast(np.pi, tx.dtype)

    delta1 = pi / 4.0 - tf.abs(T)
    delta2 = (pi / 2.0 - tf.abs(T + X)) / sqrt2
    delta3 = (pi / 2.0 - tf.abs(X - T)) / sqrt2

    delta = tf.minimum(tf.minimum(delta1, delta2), delta3)[..., tf.newaxis]  # [batch,1]

    r = tf.clip_by_value(
        tf.cast(1.0, tx.dtype) - delta / (pi / 4.0 * delta_scale),
        tf.cast(0.0, tx.dtype),
        tf.cast(1.0, tx.dtype),
    )

    # Safe normalisation: avoid div-by-zero for points at the origin
    v = tx / tf.maximum(n, tf.cast(eps, tx.dtype))

    return r * v


###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Sampling hyperparameters
    num_samples = int(1e4)
    patch_width = 0.95
    scaling_power = 1.0

    # Test the sampling scheme
    test_sample = PenroseSample(
        num_samples, patch_width=patch_width, density_power=scaling_power
    )
    # Draw the Penrose diagram
    fig, ax = plt.subplots()
    ax.grid()
    draw_penrose(ax)
    fig.tight_layout()

    # Add the sampled points
    ax.scatter(test_sample[:, 1], test_sample[:, 0], alpha=0.1)

    plt.show()
