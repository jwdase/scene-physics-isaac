"""Find the smallest edit of a scene that hides its target: rigid moves of the target (together with the
objects resting on it, so the relationship the trial is built on is kept) and, if that is not enough,
small moves of the objects that occlude it.

Candidates are tried cheapest first (cost = translation / ``unit_m`` + rotation / ``unit_deg``, occluder
moves weighted ``occluder_weight`` times more) and accepted when

* the target is hidden with ``margin_px`` pixels to spare at the stimulus resolution
  (:class:`~scene_physics.authoring.occlusion.OcclusionMeter`),
* no moved object overlaps another object by more than ``max_penetration`` (penetration that was already
  there before the move is not counted),
* every moved object keeps exactly its contacts with unmoved objects (no relationship lost or gained).

The edit is geometric; physics is checked afterwards by relaxing the edited scene (:mod:`.relax`),
which can shift objects by a millimetre or two, so the search asks for a little more margin than the
final acceptance test.

Pure (Warp ray caster and contact queries); no simulator needed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from scene_physics.authoring.contacts import ContactAnalyzer, relations
from scene_physics.authoring.occlusion import OcclusionMeter
from scene_physics.camera import CameraSpec
from scene_physics.scene.model import Scene


@dataclass
class HideConfig:
    margin_px: int = 4  # at the meter's camera resolution (use the stimulus resolution)
    max_shift: float = 0.04  # m, target group translation
    shift_step: float = 0.005
    max_yaw_deg: float = 15.0
    yaw_step_deg: float = 5.0
    occluder_max_shift: float = 0.02  # m
    occluder_step: float = 0.005
    occluder_max_yaw_deg: float = 10.0
    occluder_weight: float = 3.0
    unit_m: float = 0.01
    unit_deg: float = 10.0
    max_penetration: float = 0.0005  # m of new overlap tolerated (sampling noise)
    contact_tol: float = 0.002
    max_candidates: int = 20000
    prefilter: bool = True  # reject candidates at half resolution first (4x cheaper)
    table_margin: float = 0.02  # moved objects keep their centre this far inside the table top


# Search stages for sp rework: small moves first; larger target moves (and turns) only if those fail.
STAGES = (
    HideConfig(),
    HideConfig(max_shift=0.10, shift_step=0.01, max_yaw_deg=45.0, yaw_step_deg=15.0, occluder_max_shift=0.0,
               occluder_max_yaw_deg=0.0, unit_m=0.02, unit_deg=15.0),
    HideConfig(max_shift=0.16, shift_step=0.02, max_yaw_deg=90.0, yaw_step_deg=15.0, occluder_max_shift=0.03,
               occluder_step=0.01, occluder_max_yaw_deg=10.0, unit_m=0.02, unit_deg=15.0),
)


@dataclass
class HideResult:
    found: bool
    moves: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # name -> (dx, dy, dyaw_deg)
    poses: dict[str, np.ndarray] = field(default_factory=dict)  # new poses of the moved objects
    cost: float = 0.0
    tried: int = 0
    margin: int = -1
    visible: int = -1
    reason: str = ""

    def describe(self) -> str:
        if not self.found:
            return f"no hiding edit found after {self.tried} candidates ({self.reason})"
        parts = [f"{n}: {dx * 1e3:+.0f}/{dy * 1e3:+.0f} mm, {dyaw:+.0f} deg" for n, (dx, dy, dyaw) in self.moves.items()]
        return f"hidden after {self.tried} candidates (cost {self.cost:.2f}): " + "; ".join(parts)


def support_group(scene: Scene, rels: set, root: str) -> list[str]:
    """``root`` and everything resting on it, transitively (dynamic objects only)."""
    group, frontier = {root}, [root]
    while frontier:
        lower = frontier.pop()
        for upper, kind, below in rels:
            if kind == "on" and below == lower and upper not in group and scene[upper].dynamic:
                group.add(upper)
                frontier.append(upper)
    return sorted(group)


def move_poses(scene: Scene, names: list[str], dx: float, dy: float, dyaw_deg: float,
               pivot: np.ndarray, base: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Rigidly move ``names`` by a yaw about the vertical axis through ``pivot`` and a translation."""
    rot = Rotation.from_euler("z", dyaw_deg, degrees=True)
    out = {}
    for n in names:
        p = np.array(base[n] if base and n in base else scene[n].pose, dtype=float)
        pos = rot.apply(p[:3] - pivot) + pivot + np.array([dx, dy, 0.0])
        q = (rot * Rotation.from_quat(p[3:])).as_quat()
        out[n] = np.concatenate([pos, q])
    return out


def _grid(max_v: float, step: float) -> np.ndarray:
    n = int(round(max_v / step))
    return np.arange(-n, n + 1) * step


class Hider:
    def __init__(self, scene: Scene, target: str, camera: CameraSpec, cfg: HideConfig | None = None,
                 occluders: list[str] | None = None, device: str | None = None):
        self.scene, self.target, self.cfg = scene, target, cfg or HideConfig()
        self.meter = OcclusionMeter(scene, camera, device=device)
        self.coarse = None
        if self.cfg.prefilter and camera.width >= 4 and camera.height >= 4:
            self.coarse = OcclusionMeter(scene, camera.scaled(0.5), device=device)
        self.table = scene.support_bounds(self.cfg.table_margin)
        self.contacts = ContactAnalyzer(scene, samples=4000, device=device)
        self.rels = relations(self.contacts.contacts(tol=self.cfg.contact_tol))
        self.group = support_group(scene, self.rels, target)
        self.occluders = [o for o in (occluders if occluders is not None else self.find_occluders())
                          if o not in self.group and scene[o].dynamic]
        self._base_pen: dict[tuple[str, str], float] = {}

    def find_occluders(self) -> list[str]:
        """Dynamic objects in front of the target's silhouette (largest coverage first)."""
        m = self.meter
        k = m.names.index(self.target)
        p = m.poses()
        ids = m.rc.render(p).ids.numpy()[0]
        sil = m.rc.render(p, solo=k).ids.numpy()[0] == k
        ids = ids[sil]
        counts = {m.names[i]: int((ids == i).sum()) for i in np.unique(ids) if i >= 0 and i != k}
        return [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if n in m.dynamic]

    # ------------------------------------------------------------------ checks
    def _relations_of(self, names: list[str], poses: dict[str, np.ndarray]) -> set:
        rels = relations(self.contacts.contacts(poses, tol=self.cfg.contact_tol))
        return {r for r in rels if r[0] in names or r[2] in names}

    def _penetration(self, a: str, b: str, poses) -> float:
        d, _, _ = self.contacts.distances(a, b, poses)
        pen = max(0.0, -float(d.min()))
        if self.scene[b].dynamic:
            d2, _, _ = self.contacts.distances(b, a, poses)
            pen = max(pen, -float(d2.min()))
        return pen

    def _overlaps(self, moved: list[str], poses) -> bool:
        others = [o.name for o in self.scene if o.name not in moved]
        for a in moved:
            for b in others:
                key = (a, b)
                if key not in self._base_pen:
                    self._base_pen[key] = self._penetration(a, b, None)
                if self._penetration(a, b, poses) > self._base_pen[key] + self.cfg.max_penetration:
                    return True
        return False

    # ------------------------------------------------------------------ search
    def candidates(self):
        c = self.cfg
        shifts = _grid(c.max_shift, c.shift_step)
        yaws = _grid(c.max_yaw_deg, c.yaw_step_deg) if c.yaw_step_deg > 0 else np.zeros(1)
        t_moves = [(dx, dy, yw) for dx in shifts for dy in shifts for yw in yaws]
        o_shifts = _grid(c.occluder_max_shift, c.occluder_step)
        o_yaws = _grid(c.occluder_max_yaw_deg, c.yaw_step_deg) if c.yaw_step_deg > 0 else np.zeros(1)
        o_moves = [(dx, dy, yw) for dx in o_shifts for dy in o_shifts for yw in o_yaws]

        def cost(m):
            return np.hypot(m[0], m[1]) / c.unit_m + abs(m[2]) / c.unit_deg

        out = [(cost(m), m, None, None) for m in t_moves]
        for occ in self.occluders[:1]:  # the main occluder only
            for om in o_moves:
                if om == (0.0, 0.0, 0.0):
                    continue
                oc = c.occluder_weight * cost(om)
                for m in t_moves:
                    tc = cost(m)
                    if tc <= 2.0:  # combined with small target moves only
                        out.append((tc + oc, m, occ, om))
        out.sort(key=lambda x: x[0])
        return out[: c.max_candidates]

    def apply(self, move, occ, omove) -> tuple[dict[str, np.ndarray], list[str]]:
        pivot = self.scene[self.target].world_points.mean(axis=0)
        pivot[2] = 0.0
        poses = move_poses(self.scene, self.group, *move, pivot)
        moved = list(self.group)
        if occ is not None:
            opiv = self.scene[occ].world_points.mean(axis=0)
            opiv[2] = 0.0
            poses.update(move_poses(self.scene, [occ], *omove, opiv))
            moved.append(occ)
        return poses, moved

    def search(self) -> HideResult:
        c = self.cfg
        before = {}
        tried = 0
        best_margin = None
        x0, x1, y0, y1 = self.table
        for cost, move, occ, omove in self.candidates():
            tried += 1
            poses, moved = self.apply(move, occ, omove)
            if any(not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for p in poses.values()):
                continue
            if self.coarse is not None:
                # a pixel at half resolution covers four: the hidden region shrinks by at most one coarse pixel
                oc = self.coarse.measure(self.target, margin_px=max(0, c.margin_px // 2 - 1), poses=poses)
                if not oc.hidden:
                    continue
            o = self.meter.measure(self.target, margin_px=c.margin_px, poses=poses)
            if best_margin is None or o.margin < best_margin:
                best_margin = o.margin
            if not o.hidden:
                continue
            if self._overlaps(moved, poses):
                continue
            key = tuple(moved)
            if key not in before:
                before[key] = self._relations_of(moved, {})
            after = self._relations_of(moved, poses)
            if after != before[key]:
                continue
            moves = {n: tuple(float(v) for v in move) for n in self.group}
            if occ is not None:
                moves[occ] = tuple(float(v) for v in omove)
            return HideResult(True, moves, poses, float(cost), tried, o.margin, o.visible)
        return HideResult(False, tried=tried, margin=best_margin if best_margin is not None else -1,
                          reason="no candidate both hides the target and keeps its contacts")


def hide(scene: Scene, target: str, camera: CameraSpec, cfg: HideConfig | None = None, **kw) -> HideResult:
    return Hider(scene, target, camera, cfg, **kw).search()
