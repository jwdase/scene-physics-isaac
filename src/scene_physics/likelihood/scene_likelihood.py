"""The model's likelihood for a batch of scene hypotheses (one per world).

    lik = SceneLikelihood(scene, physics=worlds, cfg=config.likelihood, num_worlds=N, body_names=worlds.names)
    channels = lik.evaluate(poses)          # poses: (N, B, 7) for every dynamic body, world frame
    w = composite.weights(channels, cfg)    # resampling weights

The observation is the ray-cast point cloud of the ground-truth arrangement from the scene camera (the
same renderer as the hypotheses, so a perfect hypothesis scores exactly 0 on the visual channel).
``physics`` is anything with ``rollout(poses, seconds) -> final_poses`` (a
:class:`~scene_physics.isaac.simulator.ParallelWorlds` in practice).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from scene_physics.camera import CameraSpec
from scene_physics.config import LikelihoodConfig
from scene_physics.likelihood.composite import Channels
from scene_physics.likelihood.physics_terms import PenetrationEstimator, total_displacement, total_rotation
from scene_physics.likelihood.pointcloud import PointCloudLikelihood
from scene_physics.render.raycast import Raycaster
from scene_physics.scene.model import Mesh, Scene


@dataclass
class Timing:
    render: float = 0.0
    visual: float = 0.0
    physics: float = 0.0
    penetration: float = 0.0
    calls: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("render", "visual", "physics", "penetration", "calls")}


class SceneLikelihood:
    def __init__(self, scene: Scene, *, physics, cfg: LikelihoodConfig, num_worlds: int, body_names: list[str],
                 device: str | None = None, camera: CameraSpec | None = None):
        self.cfg = cfg
        self.physics = physics
        self.num_worlds = num_worlds
        self.body_names = list(body_names)
        cam = camera or scene.camera
        if cfg.render_scale != 1.0:
            cam = cam.scaled(cfg.render_scale)
        self.camera = cam
        self.raycaster = Raycaster.from_scene(scene, num_worlds=num_worlds, camera=cam, device=device,
                                              dynamic_names=self.body_names)
        self.device = self.raycaster.device
        # observation: the true arrangement, rendered once
        truth = scene.truth_poses(self.body_names)
        obs_rc = Raycaster.from_scene(scene, num_worlds=1, camera=cam, device=self.device, dynamic_names=self.body_names)
        self.observed = obs_rc.render(truth[None]).points_wp().reshape((cam.height * cam.width,))
        self.observed_points = self.observed.numpy().reshape(cam.height, cam.width, 3)
        self.pointcloud = PointCloudLikelihood(self.observed, cam.height, cam.width, variance=cfg.variance,
                                               outlier_prob=cfg.outlier_prob, outlier_volume=cfg.outlier_volume,
                                               filter_size=cfg.filter_size, device=self.device)
        self.penetration = None
        if cfg.lambda_penetration != 0.0:
            dyn = [scene[n].mesh for n in self.body_names]
            statics = [Mesh(o.world_points, o.mesh.triangles) for o in scene.static_objects]
            self.penetration = PenetrationEstimator(dyn, statics, num_worlds=num_worlds,
                                                    samples=cfg.penetration_samples, device=self.device)
        self.timing = Timing()

    def visual(self, poses: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.raycaster.render(poses)
        pts = out.points_wp()
        t1 = time.perf_counter()
        v = self.pointcloud.delta(pts)
        self.timing.render += t1 - t0
        self.timing.visual += time.perf_counter() - t1
        return v

    def evaluate(self, poses: np.ndarray) -> tuple[Channels, np.ndarray]:
        """Channels for every world, plus the final poses of the rollout ``(N, B, 7)``."""
        poses = np.asarray(poses, dtype=np.float64).reshape(self.num_worlds, len(self.body_names), 7)
        visual = self.visual(poses)
        t0 = time.perf_counter()
        pen = self.penetration(poses) if self.penetration is not None else np.zeros(self.num_worlds)
        t1 = time.perf_counter()
        final = self.physics.rollout(poses, self.cfg.rollout_seconds)
        t2 = time.perf_counter()
        self.timing.penetration += t1 - t0
        self.timing.physics += t2 - t1
        self.timing.calls += 1
        ch = Channels(visual=visual, displacement=total_displacement(poses, final),
                      rotation=total_rotation(poses, final), penetration=pen)
        return ch, final
