"""The manual-scene workflow: import -> check -> (edit, repeat) -> finalize.

* :func:`import_scene`  - Blender export or legacy scene folder -> ``authoring/<exp>/<name>.usd``
* :func:`check_scene`   - simulate under the current physics, render from the scene camera, report
                          motion and occlusion; writes ``<name>.check/`` next to the scene
* :func:`finalize_scene`- settle, measure visibility, assign roles / target / priors, render the
                          stimulus + segmentation, validate, write ``data/<exp>/<name>/``
* :func:`render_scene`  - (re)render any scene file
* :func:`open_in_editor`- Isaac Sim editor on the scene (PhysX, scene camera, studio look)
* :func:`migrate_legacy`- legacy dataset folder -> finished scene without moving any object

Functions that need Isaac Sim launch it on first use (see :mod:`scene_physics.isaac.app`).
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scene_physics import paths
from scene_physics.config import Config
from scene_physics.render import images
from scene_physics.scene import priors as prior_rules
from scene_physics.scene.model import Scene

# NOTE: modules that import warp or pxr (scene_physics.authoring.analysis, scene.usd_io, isaac.*) are
# imported inside the functions, after Isaac Sim has started: Kit brings its own warp and OpenUSD and
# must load them first.


def _log(msg: str) -> None:
    print(f"[sp] {msg}", flush=True)


def _launch(headless: bool = True, experience: str | None = None):
    from scene_physics.isaac import app

    return app.launch(headless=headless, experience=experience)


def authoring_path(experiment: str, name: str) -> Path:
    return paths.authoring_dir() / experiment / f"{name}.usd"


def check_dir(scene_path: Path) -> Path:
    return scene_path.with_name(scene_path.stem + ".check")


# ============================================================================================ import


def import_scene(source: str | Path, config: Config, *, name: str | None = None, out: str | Path | None = None,
                 static: list[str] | None = None, tags: dict[str, list[str]] | None = None,
                 overwrite: bool = False) -> Path:
    """Convert a Blender USD export (or a legacy data-pipeline scene folder) into an authoring scene."""
    from scene_physics.scene import importers, usd_io

    source = Path(source)
    if source.is_dir():
        scene = importers.from_legacy_scene(source, camera=config.camera, name=name)
    else:
        scene = importers.from_usd_export(source, name=name or source.stem, camera=config.camera,
                                          static_names=static or [], tags=tags)
    if config.experiment:
        scene.metadata["experiment"] = config.experiment
    scene.metadata["status"] = "authoring"
    scene.assign_segmentation_ids()
    out = Path(out) if out else authoring_path(config.experiment or "default", scene.name)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (use --overwrite to replace it)")
    usd_io.write_scene(scene, out, config.physics)
    _log(f"imported {source} -> {out}")
    print(scene.summary())
    for w in scene.metadata.get("import_warnings", []):
        _log(f"warning: {w}")
    return out


# ============================================================================================ check


def _render_setup(scene_path: Path, scene: Scene, config: Config, *, scale: float):
    import omni.usd

    from scene_physics.isaac import app, environment, rtx

    from isaacsim.core.simulation_manager import SimulationManager

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(scene_path)):
        raise RuntimeError(f"Isaac Sim could not open {scene_path}")
    app.update(2)
    # choose the physics device before anything can start the timeline (PhysX picks CPU otherwise)
    SimulationManager.set_device(config.physics.device)
    stage = ctx.get_stage()
    environment.apply_studio(stage, table_paths=environment.table_paths(stage))
    labels = {o.name: int(o.segmentation_id) for o in scene if o.segmentation_id is not None}
    renderer = rtx.StimulusRenderer(labels, scene.camera, config.render, scale=scale)
    return stage, renderer


def check_scene(scene_path: str | Path, config: Config, *, gui: bool = False, gif: bool = True,
                out_dir: str | Path | None = None) -> Path:
    """Simulate + render + occlusion report. Nothing in the scene file changes."""
    from scene_physics.scene import load_scene

    scene_path = paths.resolve_scene(scene_path, prefer="authoring")
    _launch(headless=not gui)
    from scene_physics.authoring import analysis
    from scene_physics.isaac import app
    from scene_physics.isaac.simulator import SceneSimulator

    scene = load_scene(scene_path)
    if any(o.segmentation_id is None for o in scene):
        scene.assign_segmentation_ids()
    out_dir = Path(out_dir) if out_dir else check_dir(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    stage, renderer = _render_setup(scene_path, scene, config, scale=1.0)
    stray = analysis.stray_bodies(stage)
    if gui:
        _set_viewport_camera()
    before = renderer.capture("preview")

    sim = SceneSimulator(scene_path, config.physics, open_stage=False)
    frames: list[np.ndarray] = []

    def on_step(_sim, _steps):
        if gui:
            app.update(1)
        if gif:
            frames.append(renderer.capture("fast").rgb)

    report = sim.settle(config.settle, on_step=on_step if (gif or gui) else None, update_fabric=True)
    after = renderer.capture("preview")

    # occlusion, measured on the settled arrangement
    settled = dict(zip(report.names, report.settled))
    for name, pose in settled.items():
        scene[name].pose = pose
    vis = analysis.measure_visibility(scene)
    decision = analysis.assign_roles(scene, vis, config.authoring)

    # images
    labels = scene.segmentation_labels()
    hidden = [o for o in scene.with_role("hidden")]
    outlines = [(vis.amodal[o.name], (255, 40, 40)) for o in hidden if o.name in vis.amodal]
    qa = images.overlay(after.rgb, after.segmentation, images.qa_palette(labels.values()), alpha=0.45, outlines=outlines)
    images.save_png(out_dir / "before.png", before.rgb)
    images.save_png(out_dir / "after.png", after.rgb)
    images.save_png(out_dir / "overlay.png", qa)
    images.write_segmentation(out_dir / "segmentation.png", after.segmentation, labels)
    panel = images.labelled_panel(
        [before.rgb, after.rgb, qa],
        ["authored (before physics)", "after physics settles", "segmentation; hidden objects outlined in red"],
        cols=3, tile_width=640,
    )
    images.save_png(out_dir / "check.png", panel)
    if frames:
        images.save_gif(out_dir / "settle.gif", frames, fps=max(1, int(round(1.0 / (config.settle.check_every * config.physics.dt)))))

    moved = report.moved(config.settle)
    lines = [
        f"scene: {scene_path}",
        f"physics: PhysX {config.physics.solver}, {1 / config.physics.dt:.0f} Hz, {config.physics.dynamic_approximation} colliders, "
        f"friction {config.physics.dynamic_friction}, device {config.physics.device}",
        "",
        "1) physics (authored poses -> rest)",
        report.table(),
        "",
    ]
    if moved:
        lines.append(f"   {len(moved)} object(s) moved more than {config.settle.moved_threshold_m * 1000:.0f} mm or "
                     f"{config.settle.moved_threshold_deg:.0f} deg: {', '.join(m.name for m in moved)}")
        lines.append("   finalize will store the settled poses as ground truth (use --no-settle to keep the authored ones)")
    else:
        lines.append("   the authored arrangement is stable under the model's physics")
    lines += ["", f"2) occlusion from camera {scene.camera.name} (hidden below {config.authoring.hidden_threshold:.0%} visible)",
              analysis.visibility_table(scene), ""]
    if decision.target:
        lines.append(f"   target: {decision.target}")
    lines += [f"   warning: {w}" for w in decision.warnings]
    if stray:
        lines.append(f"   warning: rigid bodies outside /World/Objects are ignored by the model: {', '.join(stray)} "
                     "(move them under /World/Objects in the Stage panel)")
    lines += ["", f"images: {out_dir}/check.png (before | after | segmentation), settle.gif", f"wall time {time.time() - t0:.1f} s"]
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text + "\n")
    print(text)
    renderer.destroy()
    sim.stop()
    if gui:
        _log("close the Isaac Sim window to exit")
        while app.get().is_running():
            app.update(1)
    return out_dir


def _set_viewport_camera(camera_path: str = "/World/Camera") -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport

        vp = get_active_viewport()
        if vp is not None:
            vp.camera_path = camera_path
    except Exception as exc:  # the viewport is optional (headless)
        _log(f"could not set the viewport camera: {exc}")


# ============================================================================================ finalize


def _write_final_scene(src_path: Path, scene: Scene, config: Config, out_path: Path) -> Path:
    """Apply the model's poses/metadata + the current physics to a fresh copy of the authoring stage,
    flatten it (self-contained, library references resolved) and write ``out_path``."""
    from pxr import Usd

    from scene_physics.scene import usd_io

    stage = Usd.Stage.Open(str(src_path))
    usd_io.update_scene_file(stage, scene)
    usd_io.apply_physics(stage, config.physics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    stage.Flatten().Export(str(out_path))
    return out_path


def render_scene(scene_path: str | Path, config: Config, *, out_dir: str | Path | None = None,
                 quality: str = "final") -> Path:
    """Render stimulus + segmentation (+ overlay and target silhouette) for a scene file."""
    from scene_physics.scene import load_scene

    scene_path = paths.resolve_scene(scene_path)
    _launch()
    from scene_physics.authoring import analysis

    scene = load_scene(scene_path)
    if any(o.segmentation_id is None for o in scene):
        scene.assign_segmentation_ids()
    out_dir = Path(out_dir) if out_dir else scene_path.parent / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    _, renderer = _render_setup(scene_path, scene, config, scale=config.render.scale)
    t0 = time.time()
    frame = renderer.capture(quality)
    labels = scene.segmentation_labels()
    images.save_png(out_dir / "rgb.png", frame.rgb)
    images.write_segmentation(out_dir / "segmentation.png", frame.segmentation, labels)
    vis = analysis.measure_visibility(scene)
    hidden = scene.with_role("hidden")
    outlines = [(vis.amodal[o.name], (255, 40, 40)) for o in hidden if o.name in vis.amodal]
    images.save_png(out_dir / "overlay.png",
                    images.overlay(frame.rgb, frame.segmentation, images.qa_palette(labels.values()), alpha=0.45,
                                   outlines=outlines))
    target = scene.target
    if target is not None and target.name in vis.amodal:
        images.save_png(out_dir / "target.png", images.target_image(vis.amodal[target.name], target.color, frame.rgb.shape[:2]))
    renderer.destroy()
    _log(f"rendered {scene_path.name} ({frame.rgb.shape[1]}x{frame.rgb.shape[0]}, {quality}) in {time.time() - t0:.1f} s -> {out_dir}")
    return out_dir


def finalize_scene(scene_path: str | Path, config: Config, *, name: str | None = None, experiment: str | None = None,
                   settle: bool = True, target: str | None = None, render: bool = True, regenerate_priors: bool = True,
                   strict: bool = False, overwrite: bool = False) -> Path:
    """Produce the finished scene ``data/<experiment>/<name>/``."""
    scene_path = paths.resolve_scene(scene_path, prefer="authoring")
    _launch()  # Kit must start before anything imports pxr / warp
    from scene_physics.authoring import analysis
    from scene_physics.isaac.simulator import SceneSimulator
    from scene_physics.scene import load_scene
    from scene_physics.scene.usd_io import physics_config_dict

    scene = load_scene(scene_path)
    experiment = experiment or config.experiment or scene.metadata.get("experiment") or "default"
    name = name or scene.name
    out_dir = paths.scene_dir(experiment, name)
    out_path = out_dir / "scene.usd"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (use --overwrite)")

    # 1) physics: measure (and by default apply) the settle
    sim = SceneSimulator(scene_path, config.physics)
    report = sim.settle(config.settle)
    sim.stop()
    _log("physics:\n" + report.table())
    moved = report.moved(config.settle)
    if moved and strict:
        raise RuntimeError(f"objects moved while settling: {', '.join(m.name for m in moved)} (run sp check; "
                           "drop --strict to accept the settled poses)")
    settled = dict(zip(report.names, report.settled))
    motions = {m.name: m for m in report.motions}
    for obj in scene.dynamic_objects:
        m = motions.get(obj.name)
        if m is None:
            continue
        obj.settle_displacement, obj.settle_rotation_deg, obj.settle_residual = m.displacement, m.rotation_deg, m.residual
        if settle:
            obj.pose = settled[obj.name]
    if not report.at_rest:
        _log("warning: the scene did not come to rest within settle.max_seconds")

    # 2) visibility -> roles, target; priors; ids
    scene.assign_segmentation_ids()
    vis = analysis.measure_visibility(scene)
    decision = analysis.assign_roles(scene, vis, config.authoring, target=target)
    for w in decision.warnings:
        _log(f"warning: {w}")
    if regenerate_priors or any(o.dynamic and o.prior is None for o in scene):
        prior_rules.assign_priors(scene, config.authoring, overwrite=regenerate_priors)
    problems = analysis.validate(scene)
    if problems:
        msg = "; ".join(problems)
        if strict:
            raise RuntimeError(f"scene is not valid: {msg}")
        _log(f"warning: {msg}")

    # 3) write the finished scene file
    scene.metadata.update({
        "experiment": experiment,
        "status": "finalized",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "settled": bool(settle),
        "physics_config": physics_config_dict(config.physics),
        "source": scene.metadata.get("source") or str(scene_path),
        "authoring_file": str(scene_path),
    })
    scene.name = name
    _write_final_scene(scene_path, scene, config, out_path)
    _log(f"wrote {out_path}")
    print(analysis.visibility_table(scene))

    # 4) stimulus + segmentation
    if render:
        render_scene(out_path, config, out_dir=out_dir / "render")
    return out_dir


# ============================================================================================ migrate


def migrate_legacy(legacy_dir: str | Path, config: Config, *, name: str | None = None, render: bool = True,
                   overwrite: bool = False) -> Path:
    """Legacy data-pipeline scene -> finished scene, keeping every object where it was.

    The settle is simulated only to *measure* stability (stored per object); truth poses, legacy roles
    and priors are kept.
    """
    _launch()  # Kit must start before anything imports pxr / warp
    from scene_physics.authoring import analysis
    from scene_physics.isaac.simulator import SceneSimulator
    from scene_physics.scene import importers
    from scene_physics.scene.usd_io import physics_config_dict, write_scene

    legacy_dir = Path(legacy_dir)
    scene = importers.from_legacy_scene(legacy_dir, camera=config.camera, name=name)
    experiment = config.experiment or "default"
    out_dir = paths.scene_dir(experiment, scene.name)
    out_path = out_dir / "scene.usd"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (use --overwrite)")
    scene.assign_segmentation_ids()
    # simulate a temporary copy: Kit keeps the file it simulated open, so it cannot be rewritten
    tmp = out_dir / f".{scene.name}.migrating.usd"
    write_scene(scene, tmp, config.physics)
    sim = SceneSimulator(tmp, config.physics)
    report = sim.settle(config.settle)
    sim.stop()
    import omni.usd

    omni.usd.get_context().new_stage()
    tmp.unlink(missing_ok=True)
    for m in report.motions:
        o = scene[m.name]
        o.settle_displacement, o.settle_rotation_deg, o.settle_residual = m.displacement, m.rotation_deg, m.residual
    vis = analysis.measure_visibility(scene)
    decision = analysis.assign_roles(scene, vis, config.authoring)
    missing = [o for o in scene.dynamic_objects if o.prior is None]
    if missing:
        prior_rules.assign_priors(scene, config.authoring, overwrite=False)
    scene.metadata.update({
        "experiment": experiment,
        "status": "finalized",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "settled": False,
        "physics_config": physics_config_dict(config.physics),
    })
    write_scene(scene, out_path, config.physics)
    _log(f"migrated {legacy_dir} -> {out_path}")
    _log("physics (measured only, poses kept):\n" + report.table())
    for w in decision.warnings + scene.metadata.get("import_warnings", []):
        _log(f"warning: {w}")
    if render:
        render_scene(out_path, config, out_dir=out_dir / "render")
    return out_dir


# ============================================================================================ editor


def open_in_editor(scene_path: str | Path, config: Config, *, headless: bool = False, max_frames: int | None = None) -> None:
    """Open the scene in the Isaac Sim editor with the studio look and the scene camera.

    PhysX is forced (the full experience would otherwise switch to Newton). Edit, then save with
    Ctrl+S; the look lives in the session layer and is not saved into the scene. ``headless`` and
    ``max_frames`` exist for tests.
    """
    scene_path = paths.resolve_scene(scene_path, prefer="authoring")
    from scene_physics.isaac import app

    app.launch(headless=headless, experience="isaacsim.exp.full.kit", quiet=True,
               extra_args=["--/exts/isaacsim.core.simulation_manager/default_engine=physx",
                           "--/exts/isaacsim.physics.newton/auto_switch_on_startup=false",
                           "--/isaac/startup/ros_bridge_extension="])
    import omni.usd

    from scene_physics.isaac import environment
    from scene_physics.scene import usd_io

    ctx = omni.usd.get_context()
    ctx.open_stage(str(scene_path))
    app.update(5)
    stage = ctx.get_stage()
    usd_io.apply_physics(stage, config.physics)
    environment.apply_studio(stage, table_paths=environment.table_paths(stage))
    _set_viewport_camera()
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.set_device(config.physics.device)
    _log(f"editing {scene_path}. Press Play to watch the physics; Ctrl+S saves the scene (not the look).")
    frames = 0
    while app.get().is_running() and (max_frames is None or frames < max_frames):
        app.update(1)
        frames += 1
