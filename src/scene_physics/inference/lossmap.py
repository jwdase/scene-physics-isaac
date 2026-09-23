"""Brute-force likelihood surface over the hidden object's XY prior box (``sp lossmap``).

Port of data-pipeline's ``loss_map`` / ``plot_loss_map``: the hidden object is placed at every point of
a grid (``--resolution`` metres) at its prior height and orientation, every other object stays at its
observed pose, and all channels are evaluated in batches of ``num_worlds`` parallel worlds.

Saved to ``results/<exp>/<scene>/lossmap-<timestamp>/``:

    lossmap.npz   xs, ys, channels (ny, nx, 4) with columns [visual, -penetration, -displacement, -rotation]
                  (every column oriented so higher = more likely, as the old non_rank_physics), truth_xy
    lossmap_channels.png, lossmap_combined.png
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from scene_physics import paths

CHANNEL_LABELS = ("visual", "-penetration", "-displacement", "-rotation")


def compute(scene, config, *, object_name: str | None = None, resolution: float = 0.01, log=print) -> dict:
    from scene_physics.isaac.simulator import ParallelWorlds
    from scene_physics.likelihood.scene_likelihood import SceneLikelihood

    hidden = [o for o in scene.dynamic_objects if o.role == "hidden"]
    if object_name:
        obj = scene[object_name]
    elif scene.target is not None and scene.target.role == "hidden":
        obj = scene.target
    else:
        obj = hidden[0] if hidden else None
    if obj is None or obj.prior is None:
        raise ValueError("lossmap needs a hidden object with a prior")
    x_min, x_max, y_min, y_max = obj.prior.bounds
    xs = np.arange(x_min, x_max + 1e-9, resolution)
    ys = np.arange(y_min, y_max + 1e-9, resolution)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    n = int(config.sampler.num_worlds)
    worlds = ParallelWorlds(scene, n, config.physics)
    lik = SceneLikelihood(scene, physics=worlds, cfg=config.likelihood, num_worlds=n, body_names=worlds.names)
    b = worlds.body_index(obj.name)
    template = obj.prior.means[0]
    out = np.zeros((len(cells), 4))
    for start in range(0, len(cells), n):
        chunk = cells[start:start + n]
        poses = np.repeat(worlds.truth[None], n, axis=0)
        poses[: len(chunk), b, :] = template
        poses[: len(chunk), b, :2] = chunk
        ch, _ = lik.evaluate(poses)
        k = len(chunk)
        out[start:start + k] = np.stack([ch.visual[:k], -ch.penetration[:k], -ch.displacement[:k], -ch.rotation[:k]], axis=1)
        log(f"lossmap {min(start + n, len(cells))}/{len(cells)} cells")
    worlds.stop()
    return {"xs": xs, "ys": ys, "channels": out.reshape(len(ys), len(xs), 4), "truth_xy": obj.pose[:2],
            "object": obj.name}


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max())
    return e / e.sum()


def plot(result: dict, out_dir: Path, lambdas=(1.0, 1.0, 1.0, 1.0)) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scene_physics.inference.report import BLUES, INK_2, TRUTH, _style

    xs, ys, c, truth = result["xs"], result["ys"], result["channels"], result["truth_xy"]
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    panels = [(CHANNEL_LABELS[k], _softmax(c[..., k])) for k in range(4)]
    panels.append(("combined softmax(sum lambda * channel)", _softmax((c * np.asarray(lambdas)).sum(axis=-1))))
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), layout="constrained")
    for ax, (label, prob) in zip(axes, panels):
        ax.imshow(prob, origin="lower", extent=extent, cmap=BLUES, interpolation="nearest")
        ax.scatter([truth[0]], [truth[1]], marker="*", s=160, color=TRUTH, edgecolor="white", label="truth")
        if np.ptp(prob) > 0:
            i, j = np.unravel_index(np.argmax(prob), prob.shape)
            ax.scatter([xs[j]], [ys[i]], marker="^", s=90, color="#eb6834", edgecolor="white", label="argmax")
        _style(ax, label, "x (m)", "y (m)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    path = out_dir / "lossmap_channels.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]


def main(args) -> None:
    from scene_physics.inference.runner import config_for_run
    from scene_physics.scene import load_scene

    path, cfg = config_for_run(args)
    scene = load_scene(path)
    exp = scene.metadata.get("experiment") or cfg.experiment or "default"
    out = Path(args.out) if args.out else paths.results_dir() / exp / scene.name / (
        "lossmap-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    result = compute(scene, cfg, resolution=args.resolution, log=lambda m: print(f"[sp] {m}", flush=True))
    np.savez_compressed(out / "lossmap.npz", xs=result["xs"], ys=result["ys"], channels=result["channels"],
                        truth_xy=result["truth_xy"], object=result["object"], channel_labels=np.array(CHANNEL_LABELS))
    plot(result, out)
    print(f"[sp] loss map {result['channels'].shape[:2]} -> {out}", flush=True)
