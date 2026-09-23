"""Figures and the summary for an inference run (pure matplotlib/numpy; no simulator needed).

Plot conventions: one y-axis per panel (small multiples instead of dual axes), iteration order in a
single-hue sequential ramp, at most three categorical series per panel, legends only when there are
two or more series, recessive grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from scene_physics import geometry  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
TRUTH = "#e34948"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel == "iteration":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def plot_scores(trace: dict, path: Path) -> Path:
    """Small multiples over iterations: visual score, displacement and rotation (best world)."""
    ch = trace["channels"]  # (T, N, 4)
    it = np.arange(len(ch)) - 1  # -1 = initial proposals
    energy = trace["energy"]
    best_so_far = np.maximum.accumulate(energy.max(axis=1))
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), layout="constrained")
    ax = axes[0]
    ax.plot(it, ch[:, :, 0].max(axis=1), color=SERIES[0], lw=2, label="max over worlds")
    ax.plot(it, ch[:, :, 0].mean(axis=1), color=SERIES[1], lw=2, label="mean over worlds")
    _style(ax, "Visual score (0 = perfect match)", "iteration", "log-likelihood delta")
    ax.legend(frameon=False, labelcolor=INK_2)
    ax = axes[1]
    ax.plot(it, ch[:, :, 1].min(axis=1) * 1000, color=SERIES[0], lw=2)
    _style(ax, "Least total displacement (mm)", "iteration")
    ax = axes[2]
    ax.plot(it, np.degrees(ch[:, :, 2].min(axis=1)), color=SERIES[0], lw=2)
    _style(ax, "Least total rotation (deg)", "iteration")
    ax = axes[3]
    ax.plot(it, best_so_far, color=SERIES[0], lw=2)
    _style(ax, "Best energy so far", "iteration")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_positions(name: str, xy: np.ndarray, truth: np.ndarray, best: np.ndarray, prior_means: np.ndarray,
                   bounds, path: Path) -> Path:
    """XY of every proposal (colour = iteration), with truth, best and the prior candidates."""
    t, n = xy.shape[:2]
    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, ec=INK_2, lw=1, ls="--"))
    sc = ax.scatter(xy[..., 0].ravel(), xy[..., 1].ravel(), c=np.repeat(np.arange(t) - 1, n), cmap=BLUES, s=10,
                    linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("iteration", color=INK_2)
    ax.scatter(prior_means[:, 0], prior_means[:, 1], marker="x", s=60, color=INK_2, lw=1.5, label="prior candidates")
    ax.scatter([best[0]], [best[1]], marker="o", s=90, facecolor="none", edgecolor=SERIES[1], lw=2, label="best hypothesis")
    ax.scatter([truth[0]], [truth[1]], marker="*", s=180, color=TRUTH, edgecolor="white", lw=1, label="truth")
    ax.set_aspect("equal")
    ax.set_xlim(x_min - 0.02, x_max + 0.02)
    ax.set_ylim(y_min - 0.02, y_max + 0.02)
    _style(ax, f"Proposals for {name}", "x (m)", "y (m)")
    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=INK_2)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_schedule(pos_std: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.2), layout="constrained")
    ax.plot(np.arange(len(pos_std)) - 1, np.asarray(pos_std) * 100, color=SERIES[0], lw=2)
    _style(ax, "Proposal position std (cm)", "iteration")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_posterior_on_image(image: np.ndarray, camera, particles: np.ndarray, weights: np.ndarray,
                            truth: np.ndarray, path: Path) -> Path:
    """Final particles projected into the stimulus: where the model thinks the object is."""
    h, w = image.shape[:2]
    cam = camera.scaled(w / camera.width) if w != camera.width else camera
    uv, _ = cam.project(particles[:, :3])
    tuv, _ = cam.project(truth[None, :3])
    fig, ax = plt.subplots(figsize=(8, 8 * h / w), layout="constrained")
    ax.imshow(image)
    size = 20 + 380 * (weights / max(weights.max(), 1e-12))
    ax.scatter(uv[:, 0], uv[:, 1], s=size, facecolor=SERIES[0] + "99", edgecolor="white", lw=0.8, label="final particles")
    ax.scatter(tuv[:, 0], tuv[:, 1], marker="*", s=260, color=TRUTH, edgecolor="white", lw=1.2, label="truth")
    ax.set_axis_off()
    ax.legend(frameon=True, loc="lower left", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def summarize(sampler, scene, truth_channels=None, timing: dict | None = None) -> dict:
    best = sampler.best()
    particles = sampler.final_particles()
    out = {"scene": scene.name, "best_iteration": sampler.best_iteration, "best_energy": sampler.best_energy,
           "objects": {}}
    for name in sampler.sampled:
        tp = scene[name].pose
        bp = best[name]
        poses, w = particles[name]
        mean_xy = (poses[:, :2] * w[:, None]).sum(axis=0)
        spread = np.sqrt((((poses[:, :2] - mean_xy) ** 2).sum(axis=1) * w).sum())
        out["objects"][name] = {
            "truth": tp.tolist(),
            "best": bp.tolist(),
            "best_position_error_m": float(geometry.position_error(bp, tp)),
            "best_xy_error_m": float(np.linalg.norm(bp[:2] - tp[:2])),
            "best_rotation_error_deg": float(np.degrees(geometry.rotation_angle(bp[3:], tp[3:]))),
            "posterior_mean_xy": mean_xy.tolist(),
            "posterior_mean_xy_error_m": float(np.linalg.norm(mean_xy - tp[:2])),
            "posterior_xy_spread_m": float(spread),
        }
    if truth_channels is not None:
        out["truth_check"] = {
            "visual": float(truth_channels.visual[0]),
            "displacement_m": float(truth_channels.displacement[0]),
            "rotation_deg": float(np.degrees(truth_channels.rotation[0])),
            "penetration_m3": float(truth_channels.penetration[0]),
        }
    if timing:
        out["timing"] = timing
    return out


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2))
    return path
