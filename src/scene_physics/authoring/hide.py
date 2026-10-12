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

from scene_physics.authoring.contacts import ContactAnalyzer, changed, relations
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
    max_candidates: int = 300000  # effectively all: the search stops at the first that passes
    prefilter: bool = True  # reject candidates at half resolution first (4x cheaper)
    table_margin: float = 0.02  # moved objects keep their centre this far inside the table top
    min_design_coverage: float = 0.5  # the design's occluders must cover at least this much of the target


# Search stages for sp rework: small moves first; then moderate occluder moves; larger target moves (and
# turns) only if those fail.
STAGES = (
    HideConfig(),
    HideConfig(max_shift=0.02, shift_step=0.01, max_yaw_deg=10.0, yaw_step_deg=5.0, occluder_max_shift=0.04,
               occluder_step=0.01, occluder_max_yaw_deg=15.0),
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
    edit: tuple | None = None  # (target move, members, occluder, occluder move): `exclude` it if it fails

    def describe(self) -> str:
        if not self.found:
            return f"no hiding edit found after {self.tried} candidates ({self.reason})"
        parts = [f"{n}: {dx * 1e3:+.0f}/{dy * 1e3:+.0f} mm, {dyaw:+.0f} deg" for n, (dx, dy, dyaw) in self.moves.items()]
        return f"hidden after {self.tried} candidates (cost {self.cost:.2f}): " + "; ".join(parts)


def connected_group(scene: Scene, rels: set, root: str) -> list[str]:
    """``root`` and every dynamic object physically connected to it: resting on it, leaning on it, or
    supporting it, transitively. Moving the group rigidly keeps every contact inside it."""
    group, frontier = {root}, [root]
    while frontier:
        cur = frontier.pop()
        for a, _, b in rels:
            if cur not in (a, b):
                continue
            other = b if a == cur else a
            if other not in group and scene[other].dynamic:
                group.add(other)
                frontier.append(other)
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
                 occluders: list[str] | None = None, device: str | None = None,
                 intent: dict[tuple[str, str], float] | None = None, design_occluders: list[str] | None = None,
                 visibility_floor: dict[str, float] | None = None):
        """``intent``: separations ``{(target, other): metres}`` that the target's contacts must match
        (the source design); without it the current contacts are kept. ``design_occluders``: the objects
        the target hides behind in the design; together they must keep covering most of it
        (``min_design_coverage``), so the target is not simply moved behind something else.
        ``visibility_floor``: ``{object: fraction}`` each listed object must stay at least this visible
        (an edit must not bury the objects the design shows, above all the physical cues)."""
        self.scene, self.target, self.cfg = scene, target, cfg or HideConfig()
        self.intent = intent or {}
        self.design_occluders = list(design_occluders or [])
        self.visibility_floor = dict(visibility_floor or {})
        self.meter = OcclusionMeter(scene, camera, device=device)
        self.coarse = None
        if self.cfg.prefilter and camera.width >= 4 and camera.height >= 4:
            self.coarse = OcclusionMeter(scene, camera.scaled(0.5), device=device)
        self.table = scene.support_bounds(self.cfg.table_margin)
        self.contacts = ContactAnalyzer(scene, samples=4000, device=device)
        self.rels = relations(self.contacts.contacts(tol=self.cfg.contact_tol))
        group_rels = self.rels
        if self.intent:
            # the target's own contacts as designed: an object that only touches it by accident (after
            # settling) is not moved along, so the search can separate them again
            designed = {b if a == target else a for (a, b), sep in self.intent.items() if sep < 2 * self.cfg.contact_tol}
            group_rels = {r for r in self.rels if target not in (r[0], r[2]) or ({r[0], r[2]} - {target}) <= designed}
        self.group = connected_group(scene, group_rels, target)
        self.occluders = [o for o in (occluders if occluders is not None else self.find_occluders())
                          if o not in self.group and scene[o].dynamic]
        # moving visible objects along with the hidden target changes the stimulus: weigh such group moves
        # like occluder moves
        visible_members = []
        for n in self.group:
            if n == target:
                continue
            o = self.meter.measure(n, margin_px=0)
            if o.silhouette and o.visible / o.silhouette >= 0.1:
                visible_members.append(n)
        self.group_weight = self.cfg.occluder_weight if visible_members else 1.0
        self._base_sep: dict[tuple, dict] = {}

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
    def _separations(self, moved: list[str], poses) -> dict:
        others = [o.name for o in self.scene if o.name not in moved]
        return self.contacts.separations(moved, poses, others=others)

    def intent_ok(self, seps: dict) -> bool:
        """The target touches exactly what the design says it touches (``intent``, with hysteresis)."""
        tol = self.cfg.contact_tol
        for (a, b), want in self.intent.items():
            s = seps.get((a, b), seps.get((b, a)))
            if s is None:
                continue
            if want < tol and s > 2 * tol:
                return False
            if want > 2 * tol and s < tol:
                return False
        return True

    def _target_seps(self, poses) -> dict:
        """Separations of the target from the objects named in ``intent`` (moved or not)."""
        others = {b if a == self.target else a for a, b in self.intent}
        return {(self.target, o): self.contacts.separation(self.target, o, poses) for o in sorted(others)
                if o != self.target}

    def contacts_kept(self, moved: list[str], poses) -> bool:
        """No new overlap beyond ``max_penetration``; the target's contacts as in ``intent`` (when given);
        every other contact of a moved object neither lost nor gained."""
        key = tuple(moved)
        if key not in self._base_sep:
            self._base_sep[key] = self._separations(moved, None)
        before, after = self._base_sep[key], self._separations(moved, poses)
        for k, s in after.items():
            if -s > max(0.0, -before[k]) + self.cfg.max_penetration:
                return False
        if self.intent:
            if not self.intent_ok(self._target_seps(poses)):
                return False
            governed = set(self.intent) | {(b, a) for a, b in self.intent}
            before = {k: v for k, v in before.items() if k not in governed}
        lost, gained = changed(before, after, touching=self.cfg.contact_tol, apart=2 * self.cfg.contact_tol)
        return not lost and not gained

    def others_visible(self, poses) -> bool:
        for name, floor in self.visibility_floor.items():
            o = self.meter.measure(name, margin_px=0, poses=poses)
            if o.silhouette and o.visible / o.silhouette < floor:
                return False
        return True

    def behind_design_occluders(self, occ) -> bool:
        if not self.design_occluders:
            return True
        return sum(occ.coverage.get(n, 0.0) for n in self.design_occluders) >= self.cfg.min_design_coverage

    def _support_shift(self, old_xy: np.ndarray, new_xy: np.ndarray) -> float:
        """Height change of the support surface between two points (non-zero on a tilted table)."""
        sup = self.scene.support
        if sup is None:
            return 0.0
        rot = Rotation.from_quat(sup.pose[3:])
        n = rot.apply([0.0, 0.0, 1.0])
        if abs(n[2]) < 0.5:
            return 0.0
        d = np.asarray(new_xy, dtype=float) - np.asarray(old_xy, dtype=float)
        return float(-(n[0] * d[0] + n[1] * d[1]) / n[2])

    # ------------------------------------------------------------------ search
    def candidates(self):
        """``(cost, move, members, occluder, occluder_move)`` cheapest first. ``members`` move rigidly with
        the target: the target alone (what rests on it stays and re-settles; physics checks the support
        afterwards) or its whole connected group (weighted by ``group_weight`` when it holds visible
        objects). Translations are capped by their length, not per axis."""
        c = self.cfg
        shifts = _grid(c.max_shift, c.shift_step)
        yaws = _grid(c.max_yaw_deg, c.yaw_step_deg) if c.yaw_step_deg > 0 else np.zeros(1)
        t_moves = [(dx, dy, yw) for dx in shifts for dy in shifts for yw in yaws if np.hypot(dx, dy) <= c.max_shift + 1e-9]
        o_shifts = _grid(c.occluder_max_shift, c.occluder_step)
        o_yaws = _grid(c.occluder_max_yaw_deg, c.yaw_step_deg) if c.yaw_step_deg > 0 else np.zeros(1)
        o_moves = [(dx, dy, yw) for dx in o_shifts for dy in o_shifts for yw in o_yaws
                   if np.hypot(dx, dy) <= c.occluder_max_shift + 1e-9 and (dx, dy, yw) != (0.0, 0.0, 0.0)]

        def cost(m):
            return np.hypot(m[0], m[1]) / c.unit_m + abs(m[2]) / c.unit_deg

        variants = [((self.target,), 1.0)]
        if len(self.group) > 1:
            variants.append((tuple(self.group), self.group_weight))
        out = []
        for members, weight in variants:
            out += [(weight * cost(m), m, members, None, None) for m in t_moves]
            for occ in self.occluders[:1]:  # the main occluder only
                for om in o_moves:
                    oc = c.occluder_weight * cost(om)
                    out += [(weight * cost(m) + oc, m, members, occ, om) for m in t_moves if cost(m) <= 2.0]
        out.sort(key=lambda x: x[0])
        return out[: c.max_candidates]

    def apply(self, move, members, occ, omove) -> tuple[dict[str, np.ndarray], list[str]]:
        pivot = self.scene[self.target].world_points.mean(axis=0)
        pivot[2] = 0.0
        poses = move_poses(self.scene, list(members), *move, pivot)
        dz = self._support_shift(pivot[:2], pivot[:2] + np.asarray(move[:2]))
        for n in members:
            poses[n][2] += dz
        moved = list(members)
        if occ is not None:
            opiv = self.scene[occ].world_points.mean(axis=0)
            opiv[2] = 0.0
            poses.update(move_poses(self.scene, [occ], *omove, opiv))
            poses[occ][2] += self._support_shift(opiv[:2], opiv[:2] + np.asarray(omove[:2]))
            moved.append(occ)
        return poses, moved

    @staticmethod
    def _near(move, members, occ, omove, failed) -> bool:
        """Close to an edit that already failed under physics (same objects, within 1 cm / 10 deg)."""
        for fm, fmembers, focc, fom in failed:
            if fm is None:  # a whole kind of edit failed (e.g. moving the target without what rests on it)
                if fmembers == members:
                    return True
                continue
            if fmembers != members or occ != focc:
                continue
            if np.hypot(move[0] - fm[0], move[1] - fm[1]) <= 0.0101 and abs(move[2] - fm[2]) <= 10.01:
                if occ is None or (np.hypot(omove[0] - fom[0], omove[1] - fom[1]) <= 0.0101 and abs(omove[2] - fom[2]) <= 10.01):
                    return True
        return False

    def search(self, exclude: list | None = None) -> HideResult:
        """``exclude``: edits ``(move, occluder, occluder_move)`` that failed under physics; candidates
        near them are skipped."""
        c = self.cfg
        exclude = exclude or []
        tried = 0
        best_margin = None
        x0, x1, y0, y1 = self.table
        for cost, move, members, occ, omove in self.candidates():
            if exclude and self._near(move, members, occ, omove, exclude):
                continue
            tried += 1
            poses, moved = self.apply(move, members, occ, omove)
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
            if not self.behind_design_occluders(o):
                continue
            if not self.others_visible(poses):
                continue
            if not self.contacts_kept(moved, poses):
                continue
            moves = {n: tuple(float(v) for v in move) for n in members}
            if occ is not None:
                moves[occ] = tuple(float(v) for v in omove)
            res = HideResult(True, moves, poses, float(cost), tried, o.margin, o.visible)
            res.edit = (tuple(float(v) for v in move), tuple(members), occ,
                        None if omove is None else tuple(float(v) for v in omove))
            return res
        return HideResult(False, tried=tried, margin=best_margin if best_margin is not None else -1,
                          reason="no candidate both hides the target and keeps its contacts")


def hide(scene: Scene, target: str, camera: CameraSpec, cfg: HideConfig | None = None, **kw) -> HideResult:
    return Hider(scene, target, camera, cfg, **kw).search()
