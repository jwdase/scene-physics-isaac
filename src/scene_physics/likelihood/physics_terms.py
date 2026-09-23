"""Physical-plausibility terms: how much a hypothesis moves under physics, and how much it overlaps.

* :func:`total_displacement` / :func:`total_rotation`: summed over all dynamic bodies between the
  proposal and the end of the PhysX rollout (the old ``_total_displacement`` / ``_total_rotation``).
* :class:`PenetrationEstimator`: the sampled interpenetration volume from the old design note, which
  was never implemented there (the term was a stub returning zeros). Each object gets K points spread
  through its interior once; per world, the fraction of an object's points that lie inside another
  object (or a static collider) times its volume estimates the overlap volume. Counts are integers, so
  the result is deterministic.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from scene_physics import geometry
from scene_physics.scene.model import Mesh


def total_displacement(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    """``(N, B, 7)`` x2 -> ``(N,)`` sum over bodies of the L2 position change."""
    return np.linalg.norm(final[..., :3] - initial[..., :3], axis=-1).sum(axis=-1)


def total_rotation(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    """``(N, B, 7)`` x2 -> ``(N,)`` sum over bodies of the rotation angle (radians)."""
    return geometry.rotation_angle(initial[..., 3:], final[..., 3:]).sum(axis=-1)


@wp.kernel(enable_backward=False)
def _inside_counts(
    points: wp.array2d(dtype=wp.vec3f),  # (B, K) interior points of each dynamic body, body frame
    poses: wp.array(dtype=wp.transformf),  # (N*B,)
    dyn_meshes: wp.array(dtype=wp.uint64),  # (B,)
    static_meshes: wp.array(dtype=wp.uint64),  # (S,) world frame
    num_bodies: int,
    num_static: int,
    max_dist: float,
    counts: wp.array3d(dtype=wp.int32),  # (N, B, K) -> 1 if the point is inside anything else
):
    w, a, k = wp.tid()
    xa = poses[w * num_bodies + a]
    pw = wp.transform_point(xa, points[a, k])
    inside = int(0)
    for b in range(num_bodies):
        if b == a:
            continue
        xb = poses[w * num_bodies + b]
        pb = wp.transform_point(wp.transform_inverse(xb), pw)
        q = wp.mesh_query_point_sign_winding_number(dyn_meshes[b], pb, max_dist, 2.0, 0.5)
        if q.result and q.sign < 0.0:
            inside = 1
    for s in range(num_static):
        q = wp.mesh_query_point_sign_winding_number(static_meshes[s], pw, max_dist, 2.0, 0.5)
        if q.result and q.sign < 0.0:
            inside = 1
    counts[w, a, k] = inside


def _wp_mesh(mesh: Mesh, device: str) -> wp.Mesh:
    # winding-number inside tests need the extra BVH data
    return wp.Mesh(points=wp.array(mesh.points, dtype=wp.vec3f, device=device),
                   indices=wp.array(mesh.triangles.reshape(-1), dtype=wp.int32, device=device),
                   support_winding_number=True)


def _interior_points(mesh: Mesh, k: int, rng: np.random.Generator, device: str) -> tuple[np.ndarray, float]:
    """``k`` points uniformly inside ``mesh`` (winding-number test) and its volume estimate."""
    wm = _wp_mesh(mesh, device)
    lo, hi = mesh.aabb
    box_vol = float(np.prod(np.maximum(hi - lo, 1e-9)))
    kept, tried = [], 0
    while sum(len(x) for x in kept) < k and tried < 200 * k:
        cand = rng.uniform(lo, hi, size=(4 * k, 3)).astype(np.float32)
        inside = _inside_mesh(wm, cand, device)
        kept.append(cand[inside])
        tried += len(cand)
    pts = np.concatenate(kept) if kept else np.zeros((0, 3), np.float32)
    frac = sum(len(x) for x in kept) / max(tried, 1)
    if len(pts) == 0:  # open / degenerate mesh: fall back to its vertices
        pts = mesh.points[rng.choice(len(mesh.points), size=k)]
        frac = 0.0
    idx = rng.choice(len(pts), size=k, replace=len(pts) < k)
    return pts[idx].astype(np.float32), box_vol * frac


@wp.kernel(enable_backward=False)
def _inside_one(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 1.0e6, 2.0, 0.5)
    out[i] = 1 if (q.result and q.sign < 0.0) else 0


def _inside_mesh(wm: wp.Mesh, pts: np.ndarray, device: str) -> np.ndarray:
    p = wp.array(pts, dtype=wp.vec3f, device=device)
    out = wp.zeros(len(pts), dtype=wp.int32, device=device)
    wp.launch(_inside_one, dim=len(pts), inputs=[wm.id, p], outputs=[out], device=device)
    return out.numpy().astype(bool)


class PenetrationEstimator:
    """Sampled overlap volume per world (cubic metres)."""

    def __init__(self, dynamic_meshes: list[Mesh], static_meshes_world: list[Mesh], *, num_worlds: int,
                 samples: int = 512, seed: int = 0, device: str | None = None):
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        rng = np.random.default_rng(seed)
        self.num_worlds = num_worlds
        self.num_bodies = len(dynamic_meshes)
        pts, vols = zip(*[_interior_points(m, samples, rng, self.device) for m in dynamic_meshes])
        self.volumes = np.asarray(vols, dtype=np.float64)
        self.samples = samples
        self.points = wp.array(np.stack(pts), dtype=wp.vec3f, device=self.device)
        self._dyn = [_wp_mesh(m, self.device) for m in dynamic_meshes]
        self._static = [_wp_mesh(m, self.device) for m in static_meshes_world]
        self.dyn_ids = wp.array([m.id for m in self._dyn], dtype=wp.uint64, device=self.device)
        self.static_ids = wp.array([m.id for m in self._static] or [0], dtype=wp.uint64, device=self.device)
        self.num_static = len(self._static)

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        """``poses`` ``(N, B, 7)`` -> ``(N,)`` overlap volume (sum over bodies of the volume inside others)."""
        flat = np.ascontiguousarray(np.asarray(poses, dtype=np.float32).reshape(-1, 7))
        tf = wp.array(flat, dtype=wp.transformf, device=self.device)
        counts = wp.zeros((self.num_worlds, self.num_bodies, self.samples), dtype=wp.int32, device=self.device)
        wp.launch(_inside_counts, dim=(self.num_worlds, self.num_bodies, self.samples),
                  inputs=[self.points, tf, self.dyn_ids, self.static_ids, self.num_bodies, self.num_static, 1.0e6],
                  outputs=[counts], device=self.device)
        frac = counts.numpy().mean(axis=2)  # (N, B)
        return (frac * self.volumes[None, :]).sum(axis=1)
