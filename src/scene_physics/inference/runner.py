"""Run the model on one scene (``sp infer``).

Outputs, in ``results/<experiment>/<scene>/<timestamp>[-tag]/``:

    summary.json          best hypothesis, errors vs truth, posterior mean/spread, truth check, timing
    trace.npz             every proposal, channel, weight and energy, per iteration (see sampler.Trace)
    config.json           the full configuration, scene path, git commit, world count
    sampling.txt          iteration,object,likelihood  (the old log format)
    reconstruction.usd    the scene with each sampled object at its best pose (render it with sp render)
    scores.png, schedule.png, positions_<object>.png, posterior_<object>.png

Multi-world inference is meant for the remote server; on a small GPU use ``--num-worlds 1`` for smoke tests.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from scene_physics import paths
from scene_physics.config import Config


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(paths.root()), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(paths.root()), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except Exception:
        return None


def results_dir_for(scene, experiment: str, tag: str | None = None) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return paths.results_dir() / experiment / scene.name / (f"{stamp}-{tag}" if tag else stamp)


def run_inference(scene_path: str | Path, config: Config, *, out_dir: str | Path | None = None,
                  tag: str | None = None) -> Path:
    """Build N parallel worlds + the likelihood, run the sampler, write the results directory."""
    from scene_physics.isaac import app

    app.launch()  # no-op if the CLI already started Kit
    from scene_physics.inference import report
    from scene_physics.inference.sampler import GibbsImportanceSampler
    from scene_physics.isaac.simulator import ParallelWorlds
    from scene_physics.likelihood.scene_likelihood import SceneLikelihood
    from scene_physics.scene import load_scene, usd_io

    scene_path = paths.resolve_scene(scene_path)
    scene = load_scene(scene_path)
    experiment = scene.metadata.get("experiment") or config.experiment or "default"
    out = Path(out_dir) if out_dir else results_dir_for(scene, experiment, tag)
    out.mkdir(parents=True, exist_ok=True)
    n = int(config.sampler.num_worlds)
    log_file = open(out / "log.txt", "w")

    def log(msg: str) -> None:
        print(f"[sp] {msg}", flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"scene {scene_path}")
    log(f"{n} worlds, {config.sampler.iterations} iterations, proposer {config.sampler.proposer}/{config.sampler.schedule}, "
        f"combine {config.likelihood.combine}, rollout {config.likelihood.rollout_seconds} s, device {config.physics.device}")
    if n == 1:
        log("warning: with one world the sampler only keeps its elite (world 0) and cannot explore; "
            "this is a smoke test, use --num-worlds >= 2 for inference")
    t0 = time.perf_counter()
    worlds = ParallelWorlds(scene, n, config.physics)
    lik = SceneLikelihood(scene, physics=worlds, cfg=config.likelihood, num_worlds=n, body_names=worlds.names)
    setup_s = time.perf_counter() - t0
    log(f"set up worlds + likelihood in {setup_s:.1f} s")

    # truth check: how the ground truth itself scores under the model (visual 0, near-zero motion if settled)
    truth_ch, _ = lik.evaluate(np.repeat(worlds.truth[None], n, axis=0))
    log(f"truth check: visual {truth_ch.visual[0]:.3f}, displacement {truth_ch.displacement[0] * 1000:.2f} mm, "
        f"rotation {np.degrees(truth_ch.rotation[0]):.2f} deg")

    sampler = GibbsImportanceSampler(scene, lik, config.sampler, config.likelihood, log=log)
    t1 = time.perf_counter()
    trace = sampler.run()
    run_s = time.perf_counter() - t1

    # ---- outputs
    arrays = trace.arrays()
    np.savez_compressed(out / "trace.npz", **arrays)
    with open(out / "sampling.txt", "w") as f:
        f.write("Iteration,Object,Likelihood\n")
        for i, (k, ch) in enumerate(zip(arrays["chosen"], arrays["channels"])):
            f.write(f"{i - 1},{'initial' if k < 0 else trace.sampled[k]},{ch[:, 0].max()}\n")
    timing = {"setup_s": setup_s, "sampling_s": run_s, "per_iteration_s": run_s / max(len(trace.chosen), 1),
              **{f"likelihood_{k}_s": v for k, v in lik.timing.as_dict().items() if k != "calls"},
              "likelihood_calls": lik.timing.calls}
    summary = report.summarize(sampler, scene, truth_ch, timing)
    summary.update({"num_worlds": n, "iterations": config.sampler.iterations, "scene_path": str(scene_path)})
    report.write_json(out / "summary.json", summary)
    report.write_json(out / "config.json", {**config.to_dict(), "scene_path": str(scene_path), "num_worlds": n,
                                            "git_commit": _git_commit(), "created": _dt.datetime.now().isoformat()})

    report.plot_scores(arrays, out / "scores.png")
    report.plot_schedule(arrays["pos_std"], out / "schedule.png")
    best = sampler.best()
    stim = scene_path.parent / "render" / "rgb.png"
    image = None
    if stim.exists():
        from PIL import Image

        image = np.asarray(Image.open(stim).convert("RGB"))
    for s, name in enumerate(trace.sampled):
        obj = scene[name]
        report.plot_positions(name, arrays["poses"][:, :, s, :2], obj.pose, best[name], obj.prior.means,
                              obj.prior.bounds, out / f"positions_{name}.png")
        if image is not None:
            poses, w = sampler.final_particles()[name]
            report.plot_posterior_on_image(image, scene.camera, poses, w, obj.pose, out / f"posterior_{name}.png")

    # the model's answer as a scene
    from pxr import Usd

    stage = Usd.Stage.Open(str(scene_path))
    prims = {p.GetName(): p for p in usd_io.object_prims(stage)}
    for name, pose in best.items():
        usd_io.set_pose(prims[name], pose)
    usd_io.set_attr(stage.GetPrimAtPath("/World"), "sp:scene:status", "reconstruction")
    stage.GetRootLayer().Export(str(out / "reconstruction.usd"))

    for name, rec in summary["objects"].items():
        log(f"{name}: best xy error {rec['best_xy_error_m'] * 100:.1f} cm, posterior mean xy error "
            f"{rec['posterior_mean_xy_error_m'] * 100:.1f} cm (spread {rec['posterior_xy_spread_m'] * 100:.1f} cm)")
    log(f"done in {run_s:.1f} s ({timing['per_iteration_s']:.2f} s/iteration) -> {out}")
    log_file.close()
    worlds.stop()
    return out


def config_for_run(args) -> tuple[Path, Config]:
    """Resolve the scene and its experiment config, then apply CLI / environment overrides."""
    from scene_physics.config import load_config, load_experiment
    from scene_physics.scene.usd_io import read_scene_metadata, open_stage

    path = paths.resolve_scene(args.scene)
    meta = read_scene_metadata(open_stage(path))
    exp = getattr(args, "exp", None) or meta.get("experiment")
    extra = [args.config] if getattr(args, "config", None) else []
    exp_file = paths.experiments_dir() / f"{str(exp).replace('/', '-')}.toml" if exp else None
    cfg = load_experiment(str(exp).replace("/", "-"), *extra) if exp_file and exp_file.exists() else load_config(*extra)
    # the old scripts read NUM_WORLDS / NUM_EPOCHS from the environment; keep honouring them
    num_worlds = getattr(args, "num_worlds", None) or os.environ.get("NUM_WORLDS")
    iterations = getattr(args, "iterations", None) or os.environ.get("NUM_EPOCHS")
    if num_worlds:
        cfg.sampler.num_worlds = int(num_worlds)
    if iterations:
        cfg.sampler.iterations = int(iterations)
    if getattr(args, "seed", None) is not None:
        cfg.sampler.seed = int(args.seed)
    return path, cfg


def main(args) -> None:
    """One Isaac Sim session for every scene given (``sp infer data/data-set/scene0*``)."""
    import argparse

    scenes = args.scene if isinstance(args.scene, list) else [args.scene]
    if args.out and len(scenes) > 1:
        raise ValueError("--out only works with a single scene")
    failed = []
    for spec in scenes:
        one = argparse.Namespace(**{**vars(args), "scene": spec})
        try:
            path, cfg = config_for_run(one)
            run_inference(path, cfg, out_dir=args.out, tag=args.tag)
        except Exception as exc:
            if len(scenes) == 1:
                raise
            failed.append(spec)
            print(f"[sp] FAILED {spec}: {exc}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(scenes)} scenes failed: {', '.join(failed)}")
