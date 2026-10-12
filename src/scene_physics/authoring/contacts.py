"""Which objects touch, and which rests on which: the scene's physical relationships.

A hidden-object trial is often built around a relationship (a bowl resting on the hidden block, a loaf
leaning on it), so edits to a scene (relaxation, small moves that improve occlusion) must keep it.
For every pair (dynamic A, any B) the signed distance of points spread over A's surface to B's surface is
computed (winding-number sign: negative = inside B); the pair is in contact when the smallest one is below
``tol``. A contact is a support (``A on B``) when the touching points are on A's underside and B's surface
faces up there; otherwise it is a side contact (``touch``).

Pure (Warp); no simulator needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from scene_physics import geometry
from scene_physics.likelihood.physics_terms import _wp_mesh
from scene_physics.scene.model import Mesh, Scene


@wp.kernel(enable_backward=False)
def _signed_distance(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3f), max_d: float,
                     out_d: wp.array(dtype=wp.float32), out_n: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], max_d, 2.0, 0.5)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        out_d[i] = q.sign * wp.length(pts[i] - cp)
        out_n[i] = wp.mesh_eval_face_normal(mesh, q.face)
    else:
        out_d[i] = max_d
        out_n[i] = wp.vec3f(0.0, 0.0, 0.0)


def surface_samples(mesh: Mesh, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` area-weighted random points on the mesh surface, plus its vertices."""
    v, t = mesh.points.astype(np.float64), mesh.triangles
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        return v
    idx = rng.choice(len(t), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
    return np.concatenate([pts, v])


@dataclass(frozen=True)
class Contact:
    a: str
    b: str
    separation: float  # metres; negative = interpenetrating
    relation: str  # "on" (a rests on b), "under" (b rests on a), "touch"

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.a, self.b)))

    def describe(self) -> str:
        if self.relation == "on":
            return f"{self.a} on {self.b}"
        if self.relation == "under":
            return f"{self.b} on {self.a}"
        return f"{self.a} touches {self.b}"

    def supports(self) -> tuple[str, str] | None:
        """(upper, lower) for a support contact."""
        if self.relation == "on":
            return self.a, self.b
        if self.relation == "under":
            return self.b, self.a
        return None


class ContactAnalyzer:
    def __init__(self, scene: Scene, *, samples: int = 6000, seed: int = 0, device: str | None = None):
        self.scene = scene
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        rng = np.random.default_rng(seed)
        self.meshes = {o.name: _wp_mesh(o.mesh, self.device) for o in scene}
        self.samples = {o.name: surface_samples(o.mesh, samples, rng) for o in scene}

    def _pose(self, name: str, poses: dict[str, np.ndarray] | None) -> np.ndarray:
        if poses and name in poses:
            return np.asarray(poses[name], dtype=float)
        return self.scene[name].pose

    def distances(self, a: str, b: str, poses=None, max_d: float = 0.05):
        """Signed distance of a's surface samples to b, their world positions and b's surface normal there."""
        pa, pb = self._pose(a, poses), self._pose(b, poses)
        world = geometry.transform_points(pa, self.samples[a])
        local = geometry.transform_points(geometry.inverse(pb), world).astype(np.float32)
        pts = wp.array(local, dtype=wp.vec3f, device=self.device)
        d = wp.zeros(len(local), dtype=wp.float32, device=self.device)
        n = wp.zeros(len(local), dtype=wp.vec3f, device=self.device)
        wp.launch(_signed_distance, dim=len(local), inputs=[self.meshes[b].id, pts, max_d], outputs=[d, n],
                  device=self.device)
        return d.numpy(), world, Rotation.from_quat(pb[3:]).apply(n.numpy())

    def separation(self, a: str, b: str, poses: dict[str, np.ndarray] | None = None) -> float:
        """Smallest signed distance between the surfaces of ``a`` and ``b`` (negative = overlap)."""
        d, _, _ = self.distances(a, b, poses)
        sep = float(d.min())
        if self.scene[b].dynamic or not self.scene[a].dynamic:
            d2, _, _ = self.distances(b, a, poses)
            sep = min(sep, float(d2.min()))
        return sep

    def separations(self, names: list[str], poses: dict[str, np.ndarray] | None = None,
                    others: list[str] | None = None) -> dict[tuple[str, str], float]:
        """``{(a, b): separation}`` for every a in ``names`` and every other object b (or ``others``)."""
        others = others if others is not None else [o.name for o in self.scene]
        return {(a, b): self.separation(a, b, poses) for a in names for b in others if b != a and
                not (b in names and b < a)}

    @staticmethod
    def _rests_on(d, world, normals, tol) -> bool:
        close = d < tol
        if not close.any():
            return False
        z = world[:, 2]
        lo, hi = z.min(), z.max()
        return bool(world[close, 2].mean() < lo + 0.25 * (hi - lo) and normals[close, 2].mean() > 0.5)

    def contacts(self, poses: dict[str, np.ndarray] | None = None, tol: float = 0.002) -> list[Contact]:
        out = []
        dyn = [o.name for o in self.scene.dynamic_objects]
        static = [o.name for o in self.scene.static_objects]
        for i, a in enumerate(dyn):
            for b in dyn[i + 1:] + static:
                dab, wab, nab = self.distances(a, b, poses)
                sep = float(dab.min())
                if b in dyn:
                    dba, wba, nba = self.distances(b, a, poses)
                    sep = min(sep, float(dba.min()))
                if sep >= tol:
                    continue
                rel = "touch"
                if self._rests_on(dab, wab, nab, tol):
                    rel = "on"
                elif b in dyn and self._rests_on(dba, wba, nba, tol):
                    rel = "under"
                out.append(Contact(a, b, sep, rel))
        return out


def relations(contacts: list[Contact]) -> set[tuple[str, str, str]]:
    """Contacts as comparable facts: ``(upper, "on", lower)`` or ``(a, "touch", b)`` (sorted pair)."""
    out = set()
    for c in contacts:
        s = c.supports()
        if s is not None:
            out.add((s[0], "on", s[1]))
        else:
            a, b = c.pair
            out.add((a, "touch", b))
    return out


def changed(before: dict, after: dict, *, touching: float = 0.002, apart: float = 0.004) -> tuple[list, list]:
    """(lost, gained) contacts between two separation maps, with hysteresis so that gaps near the contact
    tolerance do not flip: lost = was touching (< ``touching``) and is now apart (> ``apart``); gained =
    was apart and is now touching."""
    lost = sorted(k for k, s in before.items() if s < touching and after.get(k, np.inf) > apart)
    gained = sorted(k for k, s in before.items() if s > apart and after.get(k, np.inf) < touching)
    return lost, gained


def compare(before: set, after: set, *, involving: set[str] | None = None) -> tuple[set, set]:
    """(lost, gained) relations; with ``involving``, only relations that mention one of those objects.

    A support that became a side contact (or the other way round) between the same two objects counts as
    kept: the objects still touch, and which surface carries the weight is a matter of degrees.
    """
    def keep(r):
        return involving is None or r[0] in involving or r[2] in involving

    def pairs(rs):
        return {tuple(sorted((r[0], r[2]))) for r in rs}

    b, a = {r for r in before if keep(r)}, {r for r in after if keep(r)}
    lost = {r for r in b - a if tuple(sorted((r[0], r[2]))) not in pairs(a)}
    gained = {r for r in a - b if tuple(sorted((r[0], r[2]))) not in pairs(b)}
    return lost, gained
