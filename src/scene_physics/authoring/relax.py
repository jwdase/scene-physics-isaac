"""Relaxation: remove interpenetration gently and bring a scene to rest without explosions.

Authored arrangements (Blender exports, hand edits, small perturbations of a scene) usually start with
objects a few millimetres inside each other or the table, or a few millimetres in the air. Simulated
directly under full gravity, every overlap becomes a depenetration push at the first step and a floating
object is dropped; stacked or leaning objects then slide off their supports. Relaxation instead:

1. **Separate** under almost no gravity (``low_gravity``): PhysX pushes overlapping bodies apart no
   faster than ``separate_speed`` (its max depenetration velocity), with heavy damping and hard speed
   caps, so the push is a slow slide. Bodies that do not overlap stay where they are.
2. **Ramp** gravity from ``low_gravity`` to the model's gravity over ``ramp_seconds`` (still damped and
   capped), so floating objects are lowered onto their supports instead of dropped, then **hold** at
   full gravity for ``hold_seconds``.
3. **Settle** under exactly the model's physics (``[physics]``, the inference settings) until at rest.
4. **Verify**: from the settled poses, a rollout as long as an inference rollout must leave every object
   where it is (``verify_tolerance``). Optionally, small random perturbations of all objects must not
   make anything fall (``perturb_trials``): a stable rest, not a balance on a knife edge.

Any object that moves more than ``max_displacement`` from where it started is reported (the explosion
guard). Requires Isaac Sim (call through :func:`relax_scene`, which launches it).
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from scene_physics import geometry
from scene_physics.config import Config, PhysicsConfig, SettleConfig
from scene_physics.scene.model import Scene

_TMP_COUNTER = itertools.count()


@dataclass
class RelaxConfig:
    separate_seconds: float = 1.5  # phase 1: low gravity, overlaps pushed apart
    low_gravity: float = 0.3  # m/s^2 during phase 1 (~3% of g)
    ramp_seconds: float = 2.0  # phase 2: gravity ramps low -> full
    hold_seconds: float = 0.5  # full gravity, still damped
    separate_speed: float = 0.05  # m/s, PhysX max depenetration velocity while relaxing
    max_linear_speed: float = 0.1  # m/s, hard cap while relaxing
    max_angular_speed: float = 45.0  # deg/s, hard cap while relaxing
    linear_damping: float = 4.0  # 1/s while relaxing
    angular_damping: float = 4.0
    chunk_steps: int = 6  # gravity is updated (and motion sampled) every this many steps
    max_displacement: float = 0.03  # m; larger motion is reported as an explosion
    max_rotation_deg: float = 30.0
    verify_seconds: float = 2.5  # an inference rollout
    verify_tolerance: float = 0.001  # m an object may move in the verify rollout
    verify_tolerance_deg: float = 0.5
    perturb_trials: int = 0  # robustness: rollouts from randomly perturbed rest poses
    perturb_position: float = 0.001  # m (x, y; z is only lifted, never pushed into a support)
    perturb_rotation_deg: float = 0.5
    perturb_tolerance: float = 0.005  # m an object may end up from its rest pose after a perturbation
    seed: int = 0


@dataclass
class ObjectRelax:
    name: str
    relax_displacement: float  # m, authored -> after phases 1-2
    relax_rotation_deg: float
    settle_displacement: float  # m, after phases 1-2 -> settled under the model's physics
    settle_rotation_deg: float
    total_displacement: float  # m, authored -> final
    total_rotation_deg: float
    verify_displacement: float  # m moved in the verify rollout
    verify_rotation_deg: float
    perturb_displacement: float = 0.0  # worst over the perturbation trials
    max_speed: float = 0.0  # m/s, largest speed seen while relaxing (from pose samples)


@dataclass
class RelaxReport:
    names: list[str]
    authored: np.ndarray  # (B, 7)
    relaxed: np.ndarray  # after phases 1-2
    final: np.ndarray  # settled under the model's physics
    objects: list[ObjectRelax]
    at_rest: bool
    settle_seconds: float
    wall_seconds: float
    exploded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.at_rest and not self.exploded and self.verified and self.robust

    @property
    def verified(self) -> bool:
        return not [w for w in self.warnings if w.startswith("verify:")]

    @property
    def robust(self) -> bool:
        return not [w for w in self.warnings if w.startswith("perturb:")]

    def final_poses(self) -> dict[str, np.ndarray]:
        return dict(zip(self.names, self.final))

    def table(self) -> str:
        lines = [f"  {'object':44s} {'relax mm/deg':>14s} {'settle mm/deg':>14s} {'total mm/deg':>14s} "
                 f"{'verify mm':>9s} {'perturb mm':>10s} {'vmax m/s':>8s}"]
        for o in self.objects:
            lines.append(
                f"  {o.name:44s} {o.relax_displacement * 1e3:7.2f}/{o.relax_rotation_deg:5.1f} "
                f"{o.settle_displacement * 1e3:7.2f}/{o.settle_rotation_deg:5.1f} "
                f"{o.total_displacement * 1e3:7.2f}/{o.total_rotation_deg:5.1f} {o.verify_displacement * 1e3:9.3f} "
                f"{o.perturb_displacement * 1e3:10.2f} {o.max_speed:8.3f}")
        status = "OK" if self.ok else "NOT OK"
        lines.append(f"  settled in {self.settle_seconds:.2f} s, wall {self.wall_seconds:.1f} s: {status}")
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def relax_physics(physics: PhysicsConfig, cfg: RelaxConfig) -> PhysicsConfig:
    """The model's physics with the relaxation's gentle depenetration, damping and speed caps."""
    return replace(physics, max_depenetration_velocity=cfg.separate_speed, linear_damping=cfg.linear_damping,
                   angular_damping=cfg.angular_damping, max_linear_velocity=cfg.max_linear_speed,
                   max_angular_velocity=cfg.max_angular_speed)


def _motion(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return geometry.position_error(b, a), np.degrees(geometry.rotation_angle(b[:, 3:], a[:, 3:]))


def _tmp_path(directory: Path, name: str, tag: str) -> Path:
    # Kit keeps every file it opened in its layer registry: never reuse a path within one process
    return directory / f".{name}.{tag}.{os.getpid()}.{next(_TMP_COUNTER)}.usd"


def _simulate_file(path: Path, physics: PhysicsConfig):
    from scene_physics.isaac.simulator import SceneSimulator

    return SceneSimulator(path, physics)


def _close(sim, path: Path) -> None:
    import omni.usd

    sim.stop()
    omni.usd.get_context().new_stage()
    path.unlink(missing_ok=True)


def _set_gravity(g: float) -> None:
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.get_physics_simulation_view().set_gravity([0.0, 0.0, -float(g)])


def relax(scene: Scene, config: Config, cfg: RelaxConfig | None = None, *, workdir: str | Path | None = None,
          log=print) -> RelaxReport:
    """Relax ``scene`` (its current poses) under ``config.physics``; the scene object is not changed.

    Isaac Sim must be running. ``workdir`` holds the temporary scene files (default: next to the scene).
    """
    from scene_physics.scene.usd_io import write_scene

    cfg = cfg or RelaxConfig()
    t0 = time.time()
    physics = config.physics
    workdir = Path(workdir) if workdir else (scene.path.parent if scene.path else Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)
    names = [o.name for o in scene.dynamic_objects]
    authored = scene.truth_poses(names)

    # ---- phases 1-2: low gravity, gentle separation, gravity ramp
    rphys = relax_physics(physics, cfg)
    tmp = _tmp_path(workdir, scene.name, "relax")
    own_path = scene.path
    write_scene(scene, tmp, rphys)
    scene.path = own_path  # write_scene points the scene at the file it wrote
    sim = _simulate_file(tmp, rphys)
    order = [names.index(n) for n in sim.names]  # simulator order -> our order
    sim.reset(authored[order])
    dt = physics.dt
    max_speed = np.zeros(len(names))
    previous = sim.poses()
    exploded: set[str] = set()

    def run(seconds: float, g_from: float, g_to: float) -> None:
        nonlocal previous
        n = int(np.ceil(seconds / dt))
        done = 0
        while done < n:
            k = min(cfg.chunk_steps, n - done)
            frac = (done + k) / max(n, 1)
            _set_gravity(g_from + (g_to - g_from) * frac)
            sim.step(k)
            done += k
            current = sim.poses()
            speed = geometry.position_error(current, previous) / (k * dt)
            for i, j in enumerate(order):
                max_speed[j] = max(max_speed[j], speed[i])
            previous = current

    run(cfg.separate_seconds, cfg.low_gravity, cfg.low_gravity)
    run(cfg.ramp_seconds, cfg.low_gravity, physics.gravity)
    run(cfg.hold_seconds, physics.gravity, physics.gravity)
    relaxed = np.empty_like(authored)
    relaxed[order] = sim.poses()
    _close(sim, tmp)

    # ---- phase 3: settle under the model's physics
    relaxed_scene = _with_poses(scene, dict(zip(names, relaxed)))
    tmp = _tmp_path(workdir, scene.name, "settle")
    write_scene(relaxed_scene, tmp, physics)
    sim = _simulate_file(tmp, physics)
    order = [names.index(n) for n in sim.names]
    report = sim.settle(config.settle, stability=False)
    final = np.empty_like(authored)
    final[order] = report.settled

    # ---- phase 4: verify (a full inference rollout from the final poses), then perturbations
    sim.reset(report.settled)
    sim.step(int(round(cfg.verify_seconds / dt)))
    after = np.empty_like(authored)
    after[order] = sim.poses()
    v_disp, v_rot = _motion(final, after)
    p_disp = np.zeros(len(names))
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.perturb_trials):
        start = final.copy()
        start[:, :2] += rng.uniform(-cfg.perturb_position, cfg.perturb_position, size=(len(names), 2))
        start[:, 2] += rng.uniform(0.0, cfg.perturb_position, size=len(names))
        start[:, 3:] = geometry.perturb_rotations(start[:, 3:], np.radians(cfg.perturb_rotation_deg) / np.sqrt(3), rng)
        sim.reset(start[[names.index(n) for n in sim.names]])
        sim.step(int(round(cfg.verify_seconds / dt)))
        end = np.empty_like(authored)
        end[order] = sim.poses()
        p_disp = np.maximum(p_disp, geometry.position_error(end, final))
    _close(sim, tmp)

    r_disp, r_rot = _motion(authored, relaxed)
    s_disp, s_rot = _motion(relaxed, final)
    t_disp, t_rot = _motion(authored, final)
    objects = [ObjectRelax(n, *map(float, vals)) for n, *vals in
               zip(names, r_disp, r_rot, s_disp, s_rot, t_disp, t_rot, v_disp, v_rot, p_disp, max_speed)]
    warnings = []
    for o in objects:
        if o.total_displacement > cfg.max_displacement or o.total_rotation_deg > cfg.max_rotation_deg:
            exploded.add(o.name)
            warnings.append(f"{o.name} moved {o.total_displacement * 1e3:.1f} mm / {o.total_rotation_deg:.1f} deg "
                            f"(limit {cfg.max_displacement * 1e3:.0f} mm / {cfg.max_rotation_deg:.0f} deg)")
        if o.verify_displacement > cfg.verify_tolerance or o.verify_rotation_deg > cfg.verify_tolerance_deg:
            warnings.append(f"verify: {o.name} still moves {o.verify_displacement * 1e3:.2f} mm / "
                            f"{o.verify_rotation_deg:.2f} deg in a {cfg.verify_seconds:g} s rollout from the result")
        if cfg.perturb_trials and o.perturb_displacement > cfg.perturb_tolerance:
            warnings.append(f"perturb: {o.name} moves {o.perturb_displacement * 1e3:.1f} mm after a small push "
                            f"(rest is not robust)")
    out = RelaxReport(names, authored, relaxed, final, objects, report.at_rest, report.seconds, time.time() - t0,
                      sorted(exploded), warnings)
    if not report.at_rest:
        out.warnings.append("the scene did not come to rest within settle.max_seconds")
    return out


def _with_poses(scene: Scene, poses: dict[str, np.ndarray]) -> Scene:
    import copy

    out = copy.copy(scene)
    out.objects = {}
    for name, obj in scene.objects.items():
        o = copy.copy(obj)
        if name in poses:
            o.pose = geometry.canonical_pose(np.asarray(poses[name], dtype=float))
        out.objects[name] = o
    return out


def relax_scene(scene_path: str | Path, config: Config, cfg: RelaxConfig | None = None, *,
                out: str | Path | None = None, in_place: bool = False) -> RelaxReport:
    """``sp relax``: relax a scene file and write the relaxed + settled poses (``out`` or in place)."""
    from scene_physics import paths
    from scene_physics.isaac import app

    scene_path = paths.resolve_scene(scene_path, prefer="authoring")
    app.launch()
    from pxr import Usd

    from scene_physics.scene import load_scene, usd_io

    scene = load_scene(scene_path)
    report = relax(scene, config, cfg, workdir=scene_path.parent)
    print(report.table(), flush=True)
    target = Path(out) if out else (scene_path if in_place else None)
    if target is not None:
        final = report.final_poses()
        for obj in scene.dynamic_objects:
            obj.pose = final[obj.name]
            m = next(o for o in report.objects if o.name == obj.name)
            obj.settle_displacement, obj.settle_rotation_deg = m.total_displacement, m.total_rotation_deg
            obj.settle_residual = m.verify_displacement
        if target.resolve() == scene_path.resolve():
            stage = Usd.Stage.Open(str(scene_path))
            usd_io.update_scene_file(stage, scene)
            stage.GetRootLayer().Save()
        else:
            usd_io.write_scene(scene, target, config.physics)
        print(f"[sp] wrote relaxed poses -> {target}", flush=True)
    return report
