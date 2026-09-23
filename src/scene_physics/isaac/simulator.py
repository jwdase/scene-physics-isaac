"""PhysX simulation of scenes: one scene (authoring) or N parallel worlds (inference).

Both classes apply the same :class:`~scene_physics.config.PhysicsConfig` and move bodies through the
PhysX tensor API (``omni.physics.tensors``), so authoring and inference see identical physics.

* :class:`SceneSimulator` opens a scene file as the Kit stage (so the viewport, lights and camera are the
  scene's own) and simulates it in place. Used by ``sp check`` / ``sp finalize`` / ``sp open``.
* :class:`ParallelWorlds` builds a fresh stage with the scene's static objects once (shared by all
  worlds) and its dynamic objects cloned into ``num_worlds`` co-located worlds. PhysX env IDs keep the
  worlds from interacting; with enhanced determinism a world's result does not depend on the others.

Poses are ``(..., 7)`` arrays ``[x, y, z, qx, qy, qz, qw]`` in the world frame (PhysX tensor order).

Requires Isaac Sim to be running (:func:`scene_physics.isaac.app.launch`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
import warp as wp
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics

from scene_physics import geometry
from scene_physics.config import PhysicsConfig, SettleConfig
from scene_physics.isaac import app as isaac_app
from scene_physics.scene import schema as S
from scene_physics.scene import usd_io
from scene_physics.scene.model import Scene, SceneObject, safe_prim_name


def _start_physics(physics: PhysicsConfig) -> None:
    """Play the timeline so PhysX parses the stage, create the tensor views, then pause.

    The timeline is paused afterwards so rendering (``app.update``) never advances physics; all
    stepping goes through :meth:`SimulationManager.step`.
    """
    SimulationManager.set_device(physics.device)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    isaac_app.update(1)
    SimulationManager.initialize_physics()
    timeline.pause()
    isaac_app.update(1)
    if SimulationManager.get_physics_simulation_view() is None:
        raise RuntimeError("PhysX did not initialise (no simulation view); check the stage for errors")


def _stop_physics() -> None:
    omni.timeline.get_timeline_interface().stop()
    isaac_app.update(1)


class _BodyView:
    """Thin wrapper over a PhysX ``RigidBodyView`` with numpy/warp helpers."""

    def __init__(self, pattern, device: str):
        view = SimulationManager.get_physics_simulation_view()
        self.view = view.create_rigid_body_view(pattern)
        if self.view.count == 0:
            raise RuntimeError(f"no rigid bodies matched {pattern}")
        self.device = device if device.startswith("cuda") else "cpu"
        self.count = self.view.count
        self.paths = list(self.view.prim_paths)
        self.all_idx = wp.array(np.arange(self.count, dtype=np.uint32), device=self.device)
        self._zero_vel = wp.zeros((self.count, 6), dtype=wp.float32, device=self.device)

    def poses_wp(self) -> wp.array:
        """Current transforms as a warp array ``(count, 7)`` (valid until the next call)."""
        return self.view.get_transforms()

    def poses(self) -> np.ndarray:
        return self.view.get_transforms().numpy().reshape(self.count, 7).astype(np.float64)

    def velocities(self) -> np.ndarray:
        return self.view.get_velocities().numpy().reshape(self.count, 6).astype(np.float64)

    def set_poses(self, poses, zero_velocity: bool = True) -> None:
        if isinstance(poses, wp.array):
            data = poses
        else:
            data = wp.array(np.ascontiguousarray(np.asarray(poses, dtype=np.float32).reshape(self.count, 7)),
                            device=self.device)
        self.view.set_transforms(data, self.all_idx)
        if zero_velocity:
            self.view.set_velocities(self._zero_vel, self.all_idx)


def step(n: int, *, update_fabric: bool = False) -> None:
    if n > 0:
        SimulationManager.step(steps=int(n), update_fabric=update_fabric)


# =========================================================================================== one scene


@dataclass
class ObjectMotion:
    name: str
    displacement: float  # metres
    rotation_deg: float
    residual: float = 0.0  # metres moved during the extra stability window

    def moved(self, cfg: SettleConfig) -> bool:
        return self.displacement > cfg.moved_threshold_m or self.rotation_deg > cfg.moved_threshold_deg


@dataclass
class SettleReport:
    names: list[str]
    start: np.ndarray  # (B, 7)
    settled: np.ndarray  # (B, 7)
    seconds: float
    at_rest: bool
    motions: list[ObjectMotion] = field(default_factory=list)
    wall_seconds: float = 0.0

    def moved(self, cfg: SettleConfig) -> list[ObjectMotion]:
        return [m for m in self.motions if m.moved(cfg)]

    def still_moving(self, cfg: SettleConfig) -> list[ObjectMotion]:
        """Objects that kept moving during the stability window after the scene was called at rest."""
        return [m for m in self.motions if m.residual > cfg.moved_threshold_m]

    def table(self) -> str:
        lines = [f"  {'object':44s} {'moved (mm)':>10s} {'rotated (deg)':>13s} {'residual (mm)':>13s}"]
        for m in self.motions:
            lines.append(f"  {m.name:44s} {m.displacement * 1000:10.2f} {m.rotation_deg:13.2f} {m.residual * 1000:13.3f}")
        status = "at rest" if self.at_rest else "STILL MOVING"
        lines.append(f"  simulated {self.seconds:.2f} s ({status}), wall time {self.wall_seconds:.1f} s")
        return "\n".join(lines)


class SceneSimulator:
    """Simulate one scene file in place (the Kit stage *is* the scene)."""

    def __init__(self, scene_path: str | Path, physics: PhysicsConfig, *, open_stage: bool = True):
        self.physics = physics
        self.path = Path(scene_path)
        ctx = omni.usd.get_context()
        if open_stage:
            if not ctx.open_stage(str(self.path)):
                raise RuntimeError(f"Isaac Sim could not open {self.path}")
            isaac_app.update(2)
        self.stage: Usd.Stage = ctx.get_stage()
        usd_io.apply_physics(self.stage, physics)
        self.body_prims = [p for p in usd_io.object_prims(self.stage) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not self.body_prims:
            raise ValueError(f"{self.path}: no dynamic objects to simulate")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        authored = {str(p.GetPath()): usd_io.decompose(usd_io.world_matrix(p, cache))[0] for p in self.body_prims}
        _start_physics(physics)
        self.bodies = _BodyView([str(p.GetPath()) for p in self.body_prims], physics.device)
        self.names = [p.rsplit("/", 1)[-1] for p in self.bodies.paths]
        self.authored = np.stack([authored[p] for p in self.bodies.paths])
        self.reset()

    @property
    def dt(self) -> float:
        return self.physics.dt

    def reset(self, poses: np.ndarray | None = None) -> None:
        """Put every body at ``poses`` (default: the authored poses) with zero velocity."""
        self.bodies.set_poses(self.authored if poses is None else poses)

    def poses(self) -> np.ndarray:
        return self.bodies.poses()

    def pose_dict(self) -> dict[str, np.ndarray]:
        return dict(zip(self.names, self.poses()))

    def step(self, n: int, *, update_fabric: bool = False) -> None:
        step(n, update_fabric=update_fabric)

    @staticmethod
    def _speeds(before: np.ndarray, after: np.ndarray, seconds: float) -> tuple[float, float]:
        """Largest linear (m/s) and angular (rad/s) speed implied by two pose snapshots.

        Measured from poses rather than PhysX velocities: bodies in resting contact report small
        non-zero velocities even when their poses do not change at all.
        """
        lin = geometry.position_error(after, before).max() / seconds
        ang = geometry.rotation_angle(after[:, 3:], before[:, 3:]).max() / seconds
        return float(lin), float(ang)

    def settle(self, cfg: SettleConfig, *, on_step=None, stability: bool = True, update_fabric: bool | None = None
               ) -> SettleReport:
        """Simulate from the authored poses until everything is at rest (or ``cfg.max_seconds``).

        ``on_step(sim, steps_done)`` is called every ``cfg.check_every`` steps (used to record frames).
        ``update_fabric`` mirrors physics into Fabric so renders see it (default: when ``on_step`` is set).
        """
        t0 = time.time()
        fabric = (on_step is not None) if update_fabric is None else update_fabric
        self.reset()
        start = self.poses()
        steps, max_steps = 0, int(np.ceil(cfg.max_seconds / self.dt))
        min_steps = int(np.ceil(cfg.min_seconds / self.dt))
        at_rest = False
        previous = start
        quiet = 0
        while steps < max_steps:
            self.step(cfg.check_every, update_fabric=fabric)
            steps += cfg.check_every
            if on_step is not None:
                on_step(self, steps)
            current = self.poses()
            lin, ang = self._speeds(previous, current, cfg.check_every * self.dt)
            previous = current
            quiet = quiet + 1 if (lin < cfg.rest_linear_velocity and ang < cfg.rest_angular_velocity) else 0
            if steps >= min_steps and quiet >= max(1, cfg.rest_checks):
                at_rest = True
                break
        settled = self.poses()
        residual = np.zeros(len(self.names))
        if stability and cfg.stability_seconds > 0:
            n = int(np.ceil(cfg.stability_seconds / self.dt))
            done = 0
            while done < n:
                k = min(cfg.check_every, n - done)
                self.step(k, update_fabric=fabric)
                done += k
                if on_step is not None:
                    on_step(self, steps + done)
            residual = geometry.position_error(self.poses(), settled)
            at_rest = at_rest and not bool((residual > cfg.moved_threshold_m).any())
        disp = geometry.position_error(settled, start)
        rot = np.degrees(geometry.rotation_angle(settled[:, 3:], start[:, 3:]))
        motions = [ObjectMotion(n, float(d), float(r), float(res)) for n, d, r, res in zip(self.names, disp, rot, residual)]
        return SettleReport(self.names, start, settled, steps * self.dt, at_rest, motions, time.time() - t0)

    def write_poses_to_usd(self, poses: dict[str, np.ndarray] | None = None) -> None:
        """Author poses onto the stage's object prims (physics runs in Fabric, not USD)."""
        poses = poses or self.pose_dict()
        for prim in self.body_prims:
            if prim.GetName() in poses:
                usd_io.set_pose(prim, poses[prim.GetName()])

    def stop(self) -> None:
        _stop_physics()


# ======================================================================================= N worlds


class ParallelWorlds:
    """``num_worlds`` copies of a scene's dynamic objects on one GPU (for inference rollouts)."""

    ENV_ROOT = "/World/envs"

    def __init__(self, scene: Scene, num_worlds: int, physics: PhysicsConfig, *, layout: str = "colocated",
                 spacing: float = 4.0):
        from isaacsim.core.cloner import Cloner

        if num_worlds < 1:
            raise ValueError("num_worlds must be >= 1")
        self.scene = scene
        self.num_worlds = num_worlds
        self.physics = physics
        self.layout = layout
        self.dynamic: list[SceneObject] = scene.dynamic_objects
        if not self.dynamic:
            raise ValueError(f"scene {scene.name} has no dynamic objects")

        ctx = omni.usd.get_context()
        ctx.new_stage()
        isaac_app.update(1)
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, S.WORLD).GetPrim())
        usd_io.author_physics_scene(stage, physics)
        usd_io.author_physics_material(stage, physics)
        if physics.ground_plane:
            usd_io.author_ground(stage)

        origins = np.zeros((num_worlds, 3))
        if layout == "grid":
            side = int(np.ceil(np.sqrt(num_worlds)))
            origins[:, 0] = (np.arange(num_worlds) % side) * spacing
            origins[:, 1] = (np.arange(num_worlds) // side) * spacing
        elif layout != "colocated":
            raise ValueError(f"layout must be 'colocated' or 'grid', not {layout!r}")
        self.origins = origins

        # static objects: shared by all worlds (colocated) or one copy per world (grid)
        static_root = "/World/Static" if layout == "colocated" else f"{self.ENV_ROOT}/env_0/Static"
        UsdGeom.Scope.Define(stage, self.ENV_ROOT)
        UsdGeom.Xform.Define(stage, f"{self.ENV_ROOT}/env_0")
        UsdGeom.Scope.Define(stage, static_root)
        for obj in scene.static_objects:
            usd_io.author_object(stage, obj, physics, parent=static_root, looks=False)
        UsdGeom.Scope.Define(stage, f"{self.ENV_ROOT}/env_0/Bodies")
        for obj in self.dynamic:
            usd_io.author_object(stage, obj, physics, parent=f"{self.ENV_ROOT}/env_0/Bodies", looks=False)

        cloner = Cloner(stage=stage)
        cloner.define_base_env(self.ENV_ROOT)
        env_paths = cloner.generate_paths(f"{self.ENV_ROOT}/env", num_worlds)
        if layout == "colocated":
            cloner.clone(f"{self.ENV_ROOT}/env_0", env_paths, replicate_physics=True, enable_env_ids=True)
        else:
            cloner.clone(f"{self.ENV_ROOT}/env_0", env_paths, positions=origins, replicate_physics=True)
            cloner.filter_collisions(S.PHYSICS_SCENE, "/World/collisions", env_paths,
                                     global_paths=[S.GROUND] if physics.ground_plane else [])
        isaac_app.update(1)
        _start_physics(physics)
        if layout == "grid":
            SimulationManager.get_physics_simulation_view().set_subspace_roots(f"{self.ENV_ROOT}/*")
        self.bodies = _BodyView(f"{self.ENV_ROOT}/env_*/Bodies/*", physics.device)

        # view index for (world, body) -- must be world-major in scene.dynamic order
        names = [safe_prim_name(o.name) for o in self.dynamic]
        expected = [f"{self.ENV_ROOT}/env_{w}/Bodies/{n}" for w in range(num_worlds) for n in names]
        if self.bodies.paths != expected:
            order = {p: i for i, p in enumerate(self.bodies.paths)}
            missing = [p for p in expected if p not in order]
            if missing:
                raise RuntimeError(f"rigid body view is missing {missing[:3]}...")
            self._perm = np.array([order[p] for p in expected])
        else:
            self._perm = None
        self.names = [o.name for o in self.dynamic]
        self.num_bodies = len(self.dynamic)
        self.truth = scene.truth_poses(self.names)  # (B, 7)

    # ------------------------------------------------------------------ poses
    @property
    def dt(self) -> float:
        return self.physics.dt

    @property
    def device(self) -> str:
        return self.bodies.device

    def body_index(self, name: str) -> int:
        return self.names.index(name)

    def set_poses(self, poses: np.ndarray) -> None:
        """``poses``: ``(num_worlds, num_bodies, 7)`` world-frame poses; velocities are zeroed."""
        poses = np.asarray(poses, dtype=np.float64).reshape(self.num_worlds, self.num_bodies, 7)
        # grid layout: the view's subspace roots make these env-local, i.e. the scene's own frame
        flat = poses.reshape(-1, 7).copy()
        if self._perm is not None:
            out = np.empty_like(flat)
            out[self._perm] = flat
            flat = out
        self.bodies.set_poses(flat)

    def poses(self) -> np.ndarray:
        flat = self.bodies.poses()
        if self._perm is not None:
            flat = flat[self._perm]
        return flat.reshape(self.num_worlds, self.num_bodies, 7)

    def poses_wp(self) -> wp.array:
        """Transforms in view order as a warp array ``(num_worlds * num_bodies, 7)`` (no host copy)."""
        if self._perm is not None:
            raise RuntimeError("view order differs from world-major order; use poses()")
        return self.bodies.poses_wp()

    def step(self, n: int) -> None:
        step(n)

    def park(self) -> None:
        """Move every body far above the scene, apart from each other, and step once.

        PhysX keeps contact pairs between steps; after a teleport, contacts cached from the previous
        rollout slightly change the first steps of the next one. Parking drops all contacts, which
        makes identical inputs give bit-identical rollouts (residual history effects are ~1e-6 m).
        """
        park = np.zeros((self.num_worlds, self.num_bodies, 7))
        park[..., 6] = 1.0
        park[..., 0] = np.arange(self.num_bodies)[None, :] * 3.0
        park[..., 2] = 50.0
        self.set_poses(park)
        self.step(1)

    def rollout(self, poses: np.ndarray, seconds: float, *, park: bool = True) -> np.ndarray:
        """Start every world at ``poses``, simulate ``seconds``, return the final poses."""
        if park:
            self.park()
        self.set_poses(poses)
        self.step(int(round(seconds / self.dt)))
        return self.poses()

    def stop(self) -> None:
        _stop_physics()
