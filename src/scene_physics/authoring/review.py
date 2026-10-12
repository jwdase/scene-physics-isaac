"""Review finished scenes of an experiment (``sp review <exp>``): every check a hidden-object trial needs,
measured on the finished files, plus a GIF of each scene simulated with and without its target.

Per scene:

* **physics**: an inference-length rollout (fresh start) from the stored truth -- nothing should move -- and
  the same rollout with the target taken away: what the target holds up moves (its physical evidence);
* **occlusion**: target pixels in the path-traced segmentation of the stimulus, and in the ray caster with a
  2 and a 4 pixel margin;
* **occlusion only** (rule 4 of ``docs/COMPOSING_SCENES.md``): the occlusion-only posterior must prefer a decoy;
* **validity** (roles, priors, ids: ``sp validate``);
* optionally (``--leak``) the stimulus rendered again with the target invisible (``render/rgb_without_target.png``):
  where the hidden target still changes the image (its shadow), against the path-tracing noise of a repeated render.

Writes ``data/<exp>/review.json``, ``review.md`` (a table) and ``review_stimuli.png`` (all stimuli), and
``render/counterfactual.gif`` in every scene folder. Requires Isaac Sim with rendering (the CLI launches it).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from scene_physics import geometry, paths
from scene_physics.config import Config

FAR = np.array([20.0, 20.0, 5.0])


def _print(msg: str) -> None:
    print(msg, flush=True)


def _physics(path: Path, target: str, config: Config) -> dict:
    """Truth rollout and rollout without the target (both from a fresh start of the scene file)."""
    import omni.usd

    from scene_physics.isaac.simulator import SceneSimulator

    sim = SceneSimulator(path, config.physics)
    steps = int(round(config.likelihood.rollout_seconds / sim.dt))
    truth = sim.authored.copy()
    ti = sim.names.index(target)

    def rollout(start):
        sim.reset(start)
        sim.step(steps)
        end = sim.poses()
        return geometry.position_error(end, start), np.degrees(geometry.rotation_angle(end[:, 3:], start[:, 3:]))

    d0, r0 = rollout(truth)
    parked = truth.copy()
    parked[ti, :3] = FAR
    d1, r1 = rollout(parked)
    sim.stop()
    omni.usd.get_context().new_stage()
    others = [i for i in range(len(sim.names)) if i != ti]
    ev = {sim.names[i]: [max(0.0, float(d1[i] - d0[i])) * 1e3, max(0.0, float(r1[i] - r0[i]))] for i in others}
    worst = max(ev, key=lambda n: ev[n][0]) if ev else ""
    return {"truth_max_mm": float(d0.max() * 1e3), "truth_max_deg": float(r0.max()), "evidence": ev,
            "evidence_object": worst, "evidence_mm": ev[worst][0] if worst else 0.0,
            "evidence_deg": ev[worst][1] if worst else 0.0}


def _segmentation_pixels(scene_dir: Path, target: str) -> int | None:
    from PIL import Image

    seg_path = scene_dir / "render" / "segmentation.png"
    if not seg_path.exists():
        return None
    im = Image.open(seg_path)
    labels = json.loads(im.info.get("labels", "{}"))
    return int((np.asarray(im) == labels.get(target, -1)).sum())


def counterfactual_gif(path: Path, target: str, config: Config, out: Path, *, seconds: float = 1.5,
                       fps: int = 24) -> Path:
    """The scene simulated from its truth (left) and without its target (right), from the scene camera."""
    from scene_physics.authoring import workflow
    from scene_physics.isaac.simulator import SceneSimulator
    from scene_physics.render import images
    from scene_physics.scene import load_scene

    scene = load_scene(path)
    stage, renderer = workflow._render_setup(path, scene, config, scale=0.5)
    sim = SceneSimulator(path, config.physics, open_stage=False)
    truth = sim.authored.copy()
    ti = sim.names.index(target)
    every = max(1, int(round(1.0 / (fps * sim.dt))))
    panels = []
    for remove in (False, True):
        start = truth.copy()
        if remove:
            start[ti, :3] = FAR
        sim.reset(start)
        sim.step(1, update_fabric=True)
        frames = []
        for _ in range(int(seconds * fps)):
            frames.append(renderer.capture("fast").rgb)
            sim.step(every, update_fabric=True)
        panels.append(frames)
    images.save_gif(out, [np.concatenate([a, b], axis=1) for a, b in zip(*panels)], fps=fps)
    renderer.destroy()
    sim.stop()
    return out


def _render_without_target(path: Path, target: str | None, config: Config) -> np.ndarray:
    """The stimulus rendered again (``target`` None: unchanged, to measure the path-tracing noise)."""
    from pxr import UsdGeom

    from scene_physics.authoring import workflow
    from scene_physics.scene import load_scene
    from scene_physics.scene import schema as S
    from scene_physics.scene.model import safe_prim_name

    scene = load_scene(path)
    stage, renderer = workflow._render_setup(path, scene, config, scale=config.render.scale)
    if target is not None:
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{S.OBJECTS}/{safe_prim_name(target)}")).MakeInvisible()
    renderer.capture("final")  # textures keep streaming in after a stage opens: the first frame is not comparable
    frame = renderer.capture("final")
    renderer.destroy()
    return frame.rgb


def _local_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Largest 9x9 local mean of the per-pixel colour difference (noise averages out, shadows do not)."""
    from scipy import ndimage

    d = np.abs(a.astype(float) - b.astype(float)).max(axis=2)
    return float(ndimage.uniform_filter(d, size=9).max())


def review(exp: str, scenes: list[str] | None, config: Config, *, gif: bool = True, leak: bool = False,
           log=None) -> dict:
    from PIL import Image

    from scene_physics.authoring import analysis, compose
    from scene_physics.authoring.occlusion import OcclusionMeter
    from scene_physics.scene import load_scene

    log = log or _print
    root = paths.data_dir() / exp
    names = scenes or sorted(p.name for p in root.iterdir() if (p / "scene.usd").exists())
    out: dict = {}
    noise = None
    for name in names:
        t0 = time.time()
        d = root / name
        path = d / "scene.usd"
        scene = load_scene(path)
        target = scene.target.name if scene.target else None
        if target is None:
            out[name] = {"problems": ["no target"]}
            log(f"[review] {name}: no target")
            continue
        rec: dict = {"title": scene.metadata.get("title", ""), "relationship": scene.metadata.get("relationship", ""),
                     "target": target, "problems": analysis.validate(scene)}
        if leak:
            # rendered first, on a freshly opened stage (a simulation of the same stage would show in it), and both
            # images in the same session: renderer state carried over from earlier scenes affects both alike
            with_target = _render_without_target(path, None, config)
            if noise is None:  # path-tracing noise floor: the same scene rendered twice
                noise = _local_difference(with_target, _render_without_target(path, None, config))
            without = _render_without_target(path, target, config)
            Image.fromarray(without).save(d / "render" / "rgb_without_target.png")
            rec["leak"] = _local_difference(with_target, without)
            rec["leak_noise"] = noise
        rec.update(_physics(path, target, config))
        rec["rtx_target_px"] = _segmentation_pixels(d, target)
        meter = OcclusionMeter(scene, scene.camera.scaled(config.render.scale))
        for k in (2, 4):
            rec[f"margin{k}"] = meter.measure(target, margin_px=k).margin
        post = compose.occlusion_posterior(scene, target)
        rec["occlusion_only"] = compose.posterior_summary(post)
        rec["decoy_ok"] = compose.decoy_ok(rec["occlusion_only"])
        if gif:
            counterfactual_gif(path, target, config, d / "render" / "counterfactual.gif")
        rec["ok"] = bool(not rec["problems"] and rec["truth_max_mm"] <= 1.0 and rec["evidence_mm"] >= 10.0
                         and rec["margin2"] == 0 and (rec["rtx_target_px"] or 0) == 0 and rec["decoy_ok"])
        out[name] = rec
        log(f"[review] {name}: {'OK' if rec['ok'] else 'NOT OK'}; truth moves {rec['truth_max_mm']:.3f} mm; without "
            f"the target {rec['evidence_object']} moves {rec['evidence_mm']:.0f} mm / {rec['evidence_deg']:.0f} deg; "
            f"target {rec['rtx_target_px']} px (RTX), margin2 {rec['margin2']}, margin4 {rec['margin4']}; "
            f"{post.line()}" + (f"; leak {rec['leak']:.1f} (noise {noise:.1f})" if "leak" in rec else "")
            + (f"; problems {rec['problems']}" if rec["problems"] else "") + f" ({time.time() - t0:.0f} s)")
    previous = json.loads((root / "review.json").read_text()) if (root / "review.json").exists() else {}
    previous.update(out)
    (root / "review.json").write_text(json.dumps(previous, indent=1, default=float))
    _write_markdown(root, previous)
    _contact_sheet(root, sorted(previous))
    log(f"[review] wrote {root / 'review.md'}, {root / 'review.json'}, {root / 'review_stimuli.png'}")
    return out


def _write_markdown(root: Path, recs: dict) -> None:
    rows = ["| scene | title | target | RTX px | margin 2/4 px | truth rollout | without the target | occlusion only "
            "(largest / target's / near) | leak | OK |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(recs):
        r = recs[name]
        if "truth_max_mm" not in r:
            rows.append(f"| {name} | | | | | | | | | {', '.join(r.get('problems', []))} |")
            continue
        o = r["occlusion_only"]
        groups = o.get("groups") or {}
        largest = f"{o['decoy']} {groups.get(o['decoy'], 0):.0%}" if groups else "-"
        own = f"{groups.get(o['truth_group'], 0):.0%}"
        leak = f"{r['leak']:.0f} (noise {r['leak_noise']:.0f})" if "leak" in r else "-"
        rows.append(f"| {name} | {r['title']} | {r['target']} | {r['rtx_target_px']} | {r['margin2']}/{r['margin4']} | "
                    f"{r['truth_max_mm']:.2f} mm | {r['evidence_object']} {r['evidence_mm'] / 10:.1f} cm / "
                    f"{r['evidence_deg']:.0f}° | {largest} / {own} / {o['near']:.0%} | {leak} | "
                    f"{'yes' if r['ok'] else 'NO'} |")
    text = ["# Review", "", "Made by `sp review`. Leak: largest local colour difference (0-255) when the stimulus is "
            "rendered again with the target invisible; up to about twice the noise is path-tracing noise, more is the "
            "target's shadow.", "", *rows, ""]
    (root / "review.md").write_text("\n".join(text))


def _contact_sheet(root: Path, names: list[str], cols: int = 2) -> None:
    from PIL import Image, ImageDraw

    tiles = []
    for n in names:
        p = root / n / "render" / "rgb.png"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB").resize((640, 480))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 640, 18], fill=(0, 0, 0))
        title = ""
        rj = root / "review.json"
        if rj.exists():
            title = json.loads(rj.read_text()).get(n, {}).get("title", "")
        d.text((6, 3), f"{n}  {title}", fill=(255, 255, 0))
        tiles.append(im)
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (640 * cols, 480 * rows), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 640, (i // cols) * 480))
    sheet.save(root / "review_stimuli.png")
