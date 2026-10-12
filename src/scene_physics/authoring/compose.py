"""Compose new scenes from a hand-written spec and assess them as hidden-object trials (``sp compose``).

Scenes are designed one at a time (by a person or an agent): propose an arrangement in a spec, compose it,
relax it, look at the report and the pictures, edit the spec, repeat. A spec (TOML) lists library objects
and how each is placed, in order::

    name = "scene001"
    title = "Propped phone"
    relationship = "lean back: the phone (occluder) leans on the hidden can"
    idea = "A phone stands propped up; only something behind it can hold it at that angle."
    target = "can"
    touching = [["phone", "can"]]        # contacts the design needs (checked after relaxation)
    apart = []                           # pairs that must not touch

    [[object]]
    name = "can"
    asset = "b03_cocacola_can_cage"
    at = [0.05, 0.12]                    # footprint centre on the table (m)
    yaw = 30                             # deg about z, after `rotate`

    [[object]]
    name = "phone"
    asset = "f10_apple_iphone_4_v2"
    rotate = [["y", 90], ["z", 90], ["x", -20]]   # world-axis rotations of the resting asset, in order (deg)
    at = [0.05, 0.07]
    slide = [{ dir = [0, 1], until = "can" }]     # then move along dir until it touches `until`

Placement: rotate the resting asset (upright as it rests in the library, footprint centred, bottom at
z = 0), put its footprint centre at ``at``, drop it from above until it touches whatever is below (the
table or objects placed before it), then apply each ``slide``. ``z`` (height of the lowest point) replaces
the drop and ``pose`` gives the whole pose. Contacts are found by casting rays from surface samples of the
moving object against the other meshes (and back), so objects end ``gap`` from touching; relaxation
(:mod:`.relax`) then brings the arrangement to rest under the model's physics.

:func:`assess` (Isaac Sim) relaxes a composed scene and reports what a hidden-object trial needs: the
target hidden with a margin, the other objects visible and in frame, a verified (and robust) rest state,
the design's contacts, and the physical evidence of the target -- what moves when it is taken away --
with pictures from the scene camera and from the side.
"""

from __future__ import annotations

import copy
import json
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scene_physics import geometry
from scene_physics.config import Config
from scene_physics.scene.model import Mesh, Scene, SceneObject, safe_prim_name

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
FAR = np.array([20.0, 20.0, 5.0])  # where a removed object is parked


def _print(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================================== spec


@dataclass
class ObjectSpec:
    name: str
    asset: str
    at: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    rotate: list = field(default_factory=list)  # [[axis, deg], ...] world axes (x|y|z|vector), in order
    z: float | None = None  # lowest point (m, world); None = drop
    on: str = "table"  # what the drop must land on (an object name, "table" or "any")
    slide: list = field(default_factory=list)  # [{dir = [dx, dy(, dz)], until = name | "any"}, ...]
    offset: list | None = None  # final small translation [dx, dy, dz] (m)
    pose: list | None = None  # full pose [x, y, z, qx, qy, qz, qw]
    scale: list | None = None  # one factor or three (resting frame; z is height)
    color: list | None = None  # linear RGB override
    tags: list = field(default_factory=list)


@dataclass
class SceneSpec:
    name: str
    target: str
    objects: list[ObjectSpec]
    title: str = ""
    relationship: str = ""
    idea: str = ""
    touching: list = field(default_factory=list)
    apart: list = field(default_factory=list)
    gap: float = 0.0005
    path: Path | None = None

    @staticmethod
    def load(path: str | Path) -> "SceneSpec":
        path = Path(path)
        with open(path, "rb") as fh:
            d = tomllib.load(fh)
        objs = [ObjectSpec(**o) for o in d.pop("object", [])]
        spec = SceneSpec(objects=objs, path=path, **d)
        names = [o.name for o in objs]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate object names")
        if spec.target not in names:
            raise ValueError(f"{path}: target {spec.target!r} is not one of the objects {names}")
        return spec


# ========================================================================================= placement


def rotation_of(o: ObjectSpec) -> Rotation:
    """``rotate`` entries are ``[axis, deg]`` with axis ``"x" | "y" | "z"`` or a vector (world frame); a
    positive angle about ``d x z`` lifts the side of the object facing horizontal direction ``d``."""
    r = Rotation.identity()
    for axis, deg in o.rotate:
        a = np.asarray(AXES[axis.lower()] if isinstance(axis, str) else axis, dtype=float)
        r = Rotation.from_rotvec(np.radians(float(deg)) * a / np.linalg.norm(a)) * r
    return Rotation.from_euler("z", float(o.yaw), degrees=True) * r


class Sweeper:
    """How far an object can move along a direction before it touches another (ray casting both ways)."""

    def __init__(self, device: str = "cpu", samples: int = 4000, seed: int = 0):
        import warp as wp

        from scene_physics.authoring.contacts import surface_samples

        self.wp = wp
        self.device = device
        self._samples = surface_samples
        self.rng = np.random.default_rng(seed)
        self.n = samples
        self.meshes: dict[str, object] = {}
        self.points: dict[str, np.ndarray] = {}
        self._kernel = _ray_kernel()

    def add(self, name: str, mesh: Mesh) -> None:
        wp = self.wp
        self.meshes[name] = wp.Mesh(points=wp.array(mesh.points, dtype=wp.vec3f, device=self.device),
                                    indices=wp.array(mesh.triangles.reshape(-1), dtype=wp.int32, device=self.device))
        self.points[name] = self._samples(mesh, self.n, self.rng)

    def _cast(self, mesh_name: str, origins: np.ndarray, d: np.ndarray, max_t: float) -> float:
        wp = self.wp
        o = wp.array(np.ascontiguousarray(origins, dtype=np.float32), dtype=wp.vec3f, device=self.device)
        out = wp.empty(len(origins), dtype=wp.float32, device=self.device)
        wp.launch(self._kernel, dim=len(origins), inputs=[self.meshes[mesh_name].id, o, wp.vec3f(*map(float, d)),
                                                          float(max_t)], outputs=[out], device=self.device)
        return float(out.numpy().min())

    def travel(self, name: str, pose: np.ndarray, others: dict[str, np.ndarray], d, max_t: float = 2.0
               ) -> tuple[float, str | None]:
        """Largest distance ``name`` (at ``pose``) can move along unit ``d`` before touching one of
        ``others`` (name -> pose), and which one it touches."""
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        best, hit = max_t, None
        world_a = geometry.transform_points(pose, self.points[name])
        ra = Rotation.from_quat(pose[3:])
        for other, opose in others.items():
            inv = geometry.inverse(opose)
            rb = Rotation.from_quat(opose[3:])
            t1 = self._cast(other, geometry.transform_points(inv, world_a), rb.inv().apply(d), best)
            world_b = geometry.transform_points(opose, self.points[other])
            t2 = self._cast(name, geometry.transform_points(geometry.inverse(pose), world_b), ra.inv().apply(-d), best)
            t = min(t1, t2)
            if t < best:
                best, hit = t, other
        return best, hit


def _ray_kernel():
    import warp as wp

    @wp.kernel(enable_backward=False)
    def ray_hits(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3f), d: wp.vec3f, max_t: float,
                 out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        q = wp.mesh_query_ray(mesh, origins[i], d, max_t)
        t = max_t
        if q.result:
            t = q.t
        out[i] = t

    return ray_hits


def compose(spec: SceneSpec, config: Config, *, device: str = "cpu", log=None) -> Scene:
    """The scene the spec describes (table, camera, objects placed in order)."""
    from scene_physics.scene import library, looks

    log = log or _print
    table_mesh, tinfo = library.read_asset("dining_room_table")
    table = SceneObject("dining_room_table", geometry.IDENTITY_POSE, table_mesh, dynamic=False, asset=tinfo["asset"],
                        color=tinfo["color"], material=looks.look_for(tinfo["asset"]))
    if config.camera is None:
        raise ValueError("the experiment config has no [camera]")
    scene = Scene(spec.name, {table.name: table}, config.camera,
                  metadata={"experiment": config.experiment, "status": "authoring",
                            "source": f"sp compose {spec.path}" if spec.path else "sp compose",
                            "title": spec.title, "relationship": spec.relationship, "idea": spec.idea})
    sweep = Sweeper(device)
    sweep.add(table.name, table.mesh)
    placed = {table.name: table.pose}
    for o in spec.objects:
        mesh, info = library.read_asset(o.asset)
        if o.scale is not None:
            f = np.broadcast_to(np.asarray(o.scale, dtype=np.float64).reshape(-1), (3,))
            mesh = Mesh(mesh.points * f, mesh.triangles)
        look = dict(looks.look_for(info["asset"]))
        color = info["color"]
        if o.color is not None:
            color = tuple(float(c) for c in o.color)
            look["base_color"] = color
        name = safe_prim_name(o.name)
        tags = list(o.tags) + (["target"] if o.name == spec.target and "target" not in o.tags else [])
        sweep.add(name, mesh)
        if o.pose is not None:
            pose = geometry.canonical_pose(np.asarray(o.pose, dtype=float))
        else:
            rot = rotation_of(o)
            pts = rot.apply(mesh.points.astype(np.float64))
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            top = max(float(geometry.transform_points(p, sweep.points[n])[:, 2].max()) for n, p in placed.items())
            z0 = (o.z if o.z is not None else top + 0.05) - lo[2]
            pose = geometry.make_pose([o.at[0] - (lo[0] + hi[0]) / 2, o.at[1] - (lo[1] + hi[1]) / 2, z0], rot.as_quat())
            if o.z is None:
                t, hit = sweep.travel(name, pose, placed, [0.0, 0.0, -1.0], max_t=5.0)
                if hit is None:
                    raise ValueError(f"{o.name}: nothing below {o.at}")
                expected = table.name if o.on == "table" else safe_prim_name(o.on)
                if o.on != "any" and hit != expected:
                    raise ValueError(f"{o.name}: dropped at {o.at} it lands on {hit}, not on {o.on} "
                                     "(move it, or set `on` in the spec)")
                pose[2] -= max(t - spec.gap, 0.0)
                log(f"[compose] {o.name}: dropped onto {hit}")
            for s in o.slide:
                d = np.zeros(3)
                d[: len(s["dir"])] = s["dir"]
                until = s.get("until", "any")
                targets = placed if until == "any" else {safe_prim_name(until): placed[safe_prim_name(until)]}
                t, hit = sweep.travel(name, pose, targets, d, max_t=float(s.get("max", 0.5)))
                if hit is None:
                    raise ValueError(f"{o.name}: sliding along {s['dir']} touches nothing within {s.get('max', 0.5)} m")
                # stop at the first contact with anything, not only with `until`
                t_any, hit_any = sweep.travel(name, pose, placed, d, max_t=t + 1e-6)
                if hit_any is not None and hit_any != hit and t_any < t - 1e-4:
                    log(f"[compose] warning: {o.name} touches {hit_any} before {hit}")
                    t, hit = t_any, hit_any
                pose[:3] += (t - spec.gap) * d / np.linalg.norm(d)
                log(f"[compose] {o.name}: slid {t * 100:.1f} cm to {hit}")
        if o.offset is not None:
            pose[: len(o.offset)] += np.asarray(o.offset, dtype=float)
        scene.objects[name] = SceneObject(name, pose, mesh, dynamic=info["dynamic"], asset=info["asset"], color=color,
                                          material=look, tags=tags)
        placed[name] = scene.objects[name].pose
    scene.assign_segmentation_ids()
    return scene


# ======================================================================== occlusion-only posterior


@dataclass
class OcclusionPosterior:
    """Where an observer that only knows what is hidden would put the target.

    Hypotheses are the inference's own (``proposer = "xy"``): the target moved in XY over a grid on the table,
    at its true height and orientation, everything else at its true pose. Each hypothesis is weighted by the
    hidden object's prior (``observed_objects``: a Gaussian behind every visible object) and kept if the target
    would be hidden from the camera and does not overlap another object. If most of that mass is near the true
    position -- or behind the object that hides the target -- occlusion alone finds it and the scene does not
    test physical reasoning; a good scene has a bigger hidden region elsewhere (a decoy).
    """

    spacing: float
    radius: float
    xy: np.ndarray  # (K, 2) hidden, non-overlapping hypotheses
    weight: np.ndarray  # (K,) normalised prior mass
    occluder: list  # main occluder of each hypothesis
    truth_xy: np.ndarray
    truth_group: str
    near: float  # mass within `radius` of the true position
    groups: dict  # main occluder -> mass

    @property
    def decoy(self) -> str:
        return max(self.groups, key=self.groups.get) if self.groups else ""

    @property
    def aligned(self) -> bool:
        """The largest hidden region (by prior mass) is the one the target is in."""
        return self.decoy == self.truth_group

    def line(self) -> str:
        top = ", ".join(f"{n} {m:.0%}" for n, m in sorted(self.groups.items(), key=lambda kv: -kv[1])[:4])
        return (f"{self.near:.0%} of the occlusion-only mass within {self.radius * 100:.0f} cm of the target; "
                f"by occluder: {top} (target behind {self.truth_group})")


def occlusion_posterior(scene: Scene, target: str, *, spacing: float = 0.015, std: float = 0.1, radius: float = 0.04,
                        overlap: float = 0.003, device: str | None = None) -> OcclusionPosterior:
    from scene_physics.authoring.contacts import ContactAnalyzer
    from scene_physics.authoring.occlusion import OcclusionMeter

    meter = OcclusionMeter(scene, scene.camera, device=device)  # the model's resolution
    ca = ContactAnalyzer(scene, samples=1500, device=device)
    others = [o.name for o in scene if o.name != target]
    centers = np.array([o.pose[:2] for o in scene.dynamic_objects if o.name != target])
    truth = scene[target].pose.copy()

    def main_occluder(occ) -> str:
        return max(occ.coverage, key=occ.coverage.get) if occ.coverage else ""

    truth_group = main_occluder(meter.measure(target, margin_px=0))
    x0, x1, y0, y1 = scene.support_bounds()
    xy, w, occl = [], [], []
    for x in np.arange(x0 + spacing / 2, x1, spacing):
        for y in np.arange(y0 + spacing / 2, y1, spacing):
            prior = float(np.exp(-((centers - (x, y)) ** 2).sum(1) / (2 * std ** 2)).mean()) if len(centers) else 1.0
            if prior < 1e-3:
                continue
            pose = truth.copy()
            pose[:2] = (x, y)
            occ = meter.measure(target, margin_px=0, poses={target: pose})
            if occ.visible > 0 or not occ.in_frame:
                continue
            if any(ca.separation(target, o, {target: pose}) < -overlap for o in others):
                continue
            xy.append((x, y))
            w.append(prior)
            occl.append(main_occluder(occ))
    xy, w = np.array(xy).reshape(-1, 2), np.array(w)
    w = w / w.sum() if w.sum() > 0 else w
    near = float(w[np.linalg.norm(xy - truth[:2], axis=1) <= radius].sum()) if len(w) else 0.0
    groups: dict[str, float] = {}
    for g, m in zip(occl, w):
        groups[g] = groups.get(g, 0.0) + float(m)
    return OcclusionPosterior(spacing, radius, xy, w, occl, truth[:2], truth_group, near, groups)


def posterior_map(scene: Scene, post: OcclusionPosterior, size: int = 480) -> np.ndarray:
    """Top view of the table: occlusion-only mass (blue), object footprints (grey), the target (red)."""
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    x0, x1, y0, y1 = scene.support_bounds()
    sx = size / (x1 - x0)

    def px(p):  # table xy -> image (x right, y up)
        return (float((p[0] - x0) * sx), float(size - (p[1] - y0) * sx))

    img = Image.new("RGB", (size, size), (250, 248, 240))
    d = ImageDraw.Draw(img)
    if len(post.weight):
        top = post.weight.max()
        r = post.spacing * sx / 2
        for (x, y), m in zip(post.xy, post.weight):
            t = float(m / top) ** 0.5
            c = (int(250 - 200 * t), int(248 - 150 * t), int(240 - 20 * t))
            cx, cy = px((x, y))
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=c)
    for o in scene.dynamic_objects:
        pts = o.world_points[:, :2]
        try:
            hull = pts[ConvexHull(pts).vertices]
        except Exception:  # noqa: BLE001 - degenerate footprint
            continue
        poly = [px(p) for p in hull]
        col = (220, 30, 30) if o.name == post_target(scene, post) else (90, 90, 90)
        d.polygon(poly, outline=col, width=2)
        cx, cy = px(o.pose[:2])
        d.text((cx + 4, cy - 6), o.name[:14], fill=col)
    d.text((4, 4), "occlusion-only mass (blue); camera below", fill=(0, 0, 0))
    return np.asarray(img)


def decoy_ok(d: dict) -> bool:
    """Rule 4: another hidden region (a decoy) holds at least 1.5 times the occlusion-only mass of the target's
    own region, and at most a quarter of the mass lies within ``radius`` of the target. ``d`` is the summary
    stored in reports (``near``, ``groups``, ``truth_group``, ``decoy``, ``aligned``)."""
    if not d or not d.get("groups") or d["aligned"]:
        return False
    groups = d["groups"]
    return d["near"] <= 0.25 and groups[d["decoy"]] >= 1.5 * groups.get(d["truth_group"], 0.0)


def posterior_summary(post: OcclusionPosterior) -> dict:
    return {"near": post.near, "groups": post.groups, "truth_group": post.truth_group, "decoy": post.decoy,
            "aligned": post.aligned, "line": post.line()}


def post_target(scene: Scene, post: OcclusionPosterior) -> str:
    for o in scene.dynamic_objects:
        if np.allclose(o.pose[:2], post.truth_xy):
            return o.name
    return ""


# ========================================================================================= assessment


@dataclass
class Assessment:
    name: str
    target: str
    relax_ok: bool
    verified: bool
    robust: bool
    warnings: list[str]
    hidden: bool  # target: no visible pixel with the margin at the stimulus resolution, in frame
    target_visible_px: int
    target_margin_px: int
    coverage: dict  # occluder -> fraction of the target's silhouette it covers
    visibility: dict  # object -> visible fraction (model resolution)
    out_of_frame: list[str]
    relations: list[str]  # contacts after relaxation ("a on b", "a touches b")
    missing: list  # design contacts that are not there
    unwanted: list  # pairs that should not touch but do
    evidence: dict  # object -> [mm, deg] moved in a rollout without the target (minus with it)
    evidence_object: str
    evidence_mm: float
    perturb_mm: dict
    motion_mm: dict  # composed -> relaxed, per object
    rotation_deg: dict
    masses: dict = field(default_factory=dict)
    occlusion_only: dict = field(default_factory=dict)  # OcclusionPosterior summary (decoy check)
    wall_seconds: float = 0.0

    @property
    def decoy_ok(self) -> bool:
        """Occlusion alone would not find the target (rule 4, :func:`decoy_ok`)."""
        return decoy_ok(self.occlusion_only)

    @property
    def observed_ok(self) -> bool:
        return all(v >= 0.10 for n, v in self.visibility.items() if n != self.target)

    @property
    def ok(self) -> bool:
        return (self.relax_ok and self.robust and self.hidden and not self.out_of_frame and not self.missing
                and not self.unwanted and self.evidence_mm >= 10.0 and self.observed_ok and self.decoy_ok)

    def lines(self) -> list[str]:
        flag = lambda b: "ok" if b else "NO"  # noqa: E731
        lines = [
            f"{self.name}: {'PASS' if self.ok else 'not yet'}",
            f"  rest state      : {flag(self.relax_ok)} (verified {flag(self.verified)}, robust {flag(self.robust)})",
            f"  target hidden   : {flag(self.hidden)} ({self.target_visible_px} px visible, {self.target_margin_px} px "
            f"inside the 2 px margin); covered by " + ", ".join(f"{n} {f:.0%}" for n, f in
                                                              sorted(self.coverage.items(), key=lambda kv: -kv[1])),
            "  others visible  : " + ", ".join(f"{n} {v:.0%}" for n, v in self.visibility.items() if n != self.target)
            + ("" if self.observed_ok else "  (some below 10%)"),
            f"  in frame        : {flag(not self.out_of_frame)} {self.out_of_frame or ''}",
            f"  contacts        : {'; '.join(self.relations)}",
            f"  design contacts : {flag(not self.missing and not self.unwanted)}"
            + (f" missing {self.missing}" if self.missing else "") + (f" unwanted {self.unwanted}" if self.unwanted else ""),
            f"  evidence        : without the target {self.evidence_object} moves {self.evidence_mm:.0f} mm  ("
            + ", ".join(f"{n} {v[0]:.0f} mm/{v[1]:.0f} deg" for n, v in self.evidence.items()) + ")",
            "  relax motion    : " + ", ".join(f"{n} {d:.1f} mm/{self.rotation_deg[n]:.1f} deg"
                                              for n, d in self.motion_mm.items()),
            "  perturbation    : " + ", ".join(f"{n} {d:.1f} mm" for n, d in self.perturb_mm.items()),
            "  masses          : " + ", ".join(f"{n} {m * 1000:.0f} g" for n, m in self.masses.items()),
            f"  occlusion only  : {flag(self.decoy_ok)} {self.occlusion_only.get('line', '')}",
        ]
        lines += [f"  warning: {w}" for w in self.warnings]
        return lines


def _pairs(spec_pairs, scene: Scene) -> list[tuple[str, str]]:
    return [(safe_prim_name(a), safe_prim_name(b)) for a, b in spec_pairs]


def _masses(scene: Scene, config: Config, workdir: Path) -> dict[str, float]:
    """Masses PhysX gives the dynamic objects (from their colliders and the density)."""
    from scene_physics.authoring.relax import _close, _simulate_file, _tmp_path
    from scene_physics.scene.usd_io import write_scene

    tmp = _tmp_path(workdir, scene.name, "mass")
    own = scene.path
    write_scene(scene, tmp, config.physics)
    scene.path = own
    sim = _simulate_file(tmp, config.physics)
    sim.step(1)
    try:
        m = sim.bodies.view.get_masses().numpy().reshape(-1)
        out = {n: float(v) for n, v in zip(sim.names, m)}
    except Exception:  # noqa: BLE001 - informative only
        out = {}
    _close(sim, tmp)
    return out


def assess(scene: Scene, spec: SceneSpec, config: Config, *, workdir: Path, perturb: int = 6,
           log=None) -> tuple[Scene, Assessment, dict]:
    """Relax ``scene`` and assess it. Returns the relaxed scene, the assessment and extra poses for pictures
    (``"without_target"``: where everything is after a rollout without the target). Isaac Sim must run."""
    from scene_physics.authoring import analysis, contacts, occlusion, relax

    log = log or _print
    t0 = time.time()
    target = safe_prim_name(spec.target)
    workdir.mkdir(parents=True, exist_ok=True)
    masses = _masses(scene, config, workdir)
    rcfg = relax.RelaxConfig(perturb_trials=perturb, max_displacement=0.05, max_rotation_deg=45.0)
    rep = relax.relax(scene, config, rcfg, workdir=workdir, log=log)
    poses = rep.final_poses()
    out = relax._with_poses(scene, poses)
    out.path = scene.path

    # occlusion (stimulus resolution, margin) and the others' visibility (model resolution)
    meter = occlusion.OcclusionMeter(out, out.camera.scaled(config.render.scale))
    occ = meter.measure(target, margin_px=2)
    vis = analysis.measure_visibility(out)
    frame = []
    for o in out.dynamic_objects:
        if not meter._roi(o.name, o.pose, 0)[4]:
            frame.append(o.name)

    # contacts
    ca = contacts.ContactAnalyzer(out)
    rels = sorted(c.describe() for c in ca.contacts())
    # PhysX collides convex hulls that lie up to a few millimetres outside the visual meshes (thin walls,
    # rims), so a resting contact can show as a few millimetres of mesh separation
    missing = [p for p in _pairs(spec.touching, out) if ca.separation(*p) > 0.006]
    unwanted = [p for p in _pairs(spec.apart, out) if ca.separation(*p) < 0.001]

    # physical evidence: a rollout without the target vs. with it (both from a fresh start)
    seconds = config.likelihood.rollout_seconds
    truth_end = relax.fresh_rollout(out, poses, config, seconds, workdir)
    parked = dict(poses)
    parked[target] = geometry.make_pose(FAR, [0, 0, 0, 1])
    cf_end = relax.fresh_rollout(out, parked, config, seconds, workdir)
    evidence = {}
    for n in poses:
        if n == target:
            continue
        d1 = float(geometry.position_error(cf_end[n], poses[n]))
        d0 = float(geometry.position_error(truth_end[n], poses[n]))
        r1 = float(np.degrees(geometry.rotation_angle(cf_end[n][3:], poses[n][3:])))
        r0 = float(np.degrees(geometry.rotation_angle(truth_end[n][3:], poses[n][3:])))
        evidence[n] = [max(0.0, d1 - d0) * 1e3, max(0.0, r1 - r0)]
    worst = max(evidence, key=lambda n: evidence[n][0]) if evidence else ""
    post = occlusion_posterior(out, target)
    res = Assessment(
        name=spec.name, target=target, relax_ok=rep.at_rest and rep.verified, verified=rep.verified,
        robust=rep.robust, warnings=rep.warnings, hidden=occ.hidden, target_visible_px=occ.visible,
        target_margin_px=occ.margin, coverage={k: float(v) for k, v in occ.coverage.items()},
        visibility={n: float(v) for n, v in vis.fractions.items() if out[n].dynamic}, out_of_frame=frame,
        relations=rels, missing=missing, unwanted=unwanted, evidence=evidence, evidence_object=worst,
        evidence_mm=evidence[worst][0] if worst else 0.0,
        perturb_mm={o.name: o.perturb_displacement * 1e3 for o in rep.objects},
        motion_mm={o.name: o.total_displacement * 1e3 for o in rep.objects},
        rotation_deg={o.name: o.total_rotation_deg for o in rep.objects},
        masses=masses, wall_seconds=time.time() - t0,
        occlusion_only=posterior_summary(post))
    return out, res, {"without_target": {n: p for n, p in cf_end.items() if n != target}, "posterior": post}


# ============================================================================================ pictures


def _shaded(scene: Scene, poses: dict[str, np.ndarray], camera, target: str | None, device: str | None = None,
            palette=None) -> np.ndarray:
    """Ray-cast picture: objects in their colours with simple shading, the target's outline in red."""
    from scipy import ndimage

    from scene_physics.render import raycast

    rc = raycast.Raycaster.from_scene(scene, num_worlds=1, camera=camera, device=device, floor=True)
    names = rc.object_names
    dyn = names[: rc.num_dynamic]
    pose_arr = np.stack([poses.get(n, geometry.make_pose(FAR, [0, 0, 0, 1])) for n in dyn])[None]
    out = rc.render(pose_arr)
    ids = out.id_image()[0]
    P = out.points()[0]
    gx, gy = np.gradient(P, axis=1), np.gradient(P, axis=0)
    nrm = np.cross(gx, gy)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-12
    eye = np.asarray(camera.eye, dtype=float)
    v = eye[None, None] - P
    v /= np.linalg.norm(v, axis=2, keepdims=True) + 1e-12
    nrm = np.where((nrm * v).sum(2, keepdims=True) < 0, -nrm, nrm)
    light = np.array([0.35, -0.55, 0.75])
    light /= np.linalg.norm(light)
    shade = np.nan_to_num(np.clip((nrm * light).sum(2), 0, 1) * 0.7 + 0.3, nan=1.0)
    img = np.full(ids.shape + (3,), 235.0)
    for i, n in enumerate(names):
        m = ids == i
        if not m.any():
            continue
        if n == "__floor__":
            col = np.array([0.35, 0.35, 0.35])
        elif i >= rc.num_dynamic:
            col = np.array([0.80, 0.68, 0.53])
        else:
            col = np.asarray(scene[n].material.get("base_color", scene[n].color), dtype=float)
            col = np.clip(col, 0, 1) ** (1 / 2.2)  # linear -> display
        img[m] = shade[m, None] * col[None] * 255
    img = img.astype(np.uint8)
    if target is not None and target in dyn:
        k = names.index(target)
        sil = rc.render(pose_arr, solo=k).id_image()[0] == k
        edge = sil & ~ndimage.binary_erosion(sil, iterations=2)
        img[edge] = [255, 0, 0]
        img[ids == k] = [255, 0, 255]  # visible target pixels
    return img


def pictures(scene: Scene, poses: dict[str, np.ndarray], target: str, extra: dict, out_png: Path,
             device: str | None = None) -> Path:
    """Panel: scene camera | side | top, and the scene camera after a rollout without the target."""
    from PIL import Image, ImageDraw

    from scene_physics.authoring.occlusion import orbit_camera

    cam = scene.camera
    look = np.mean([poses[n][:3] for n in poses], axis=0)
    look[2] = 0.78
    side = orbit_camera(70.0, 12.0, 0.9, look_at=tuple(look), fov_degree=40.0, reference=cam, name="side")
    top = orbit_camera(20.0, 65.0, 0.9, look_at=tuple(look), fov_degree=40.0, reference=cam, name="top")
    panels = [("scene camera (target outlined red, visible target magenta)", _shaded(scene, poses, cam, target, device)),
              ("side (from the right)", _shaded(scene, poses, side, target, device)),
              ("from above", _shaded(scene, poses, top, target, device))]
    if extra.get("without_target"):
        panels.append(("2.5 s without the target", _shaded(scene, extra["without_target"], cam, None, device)))
    if extra.get("posterior") is not None:
        post = extra["posterior"]
        panels.append((f"occlusion only: {post.near:.0%} near the target, most behind {post.decoy}",
                       posterior_map(scene, post, size=360)))
    tiles = []
    for title, img in panels:
        im = Image.fromarray(img)
        if im.size != (480, 360):
            if im.size[0] == im.size[1]:  # the square map: pad to the tile size
                pad = Image.new("RGB", (480, 360), (255, 255, 255))
                pad.paste(im.resize((360, 360)), (60, 0))
                im = pad
            else:
                im = im.resize((480, 360))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 480, 16], fill=(0, 0, 0))
        d.text((4, 2), title, fill=(255, 255, 0))
        tiles.append(im)
    rows = (len(tiles) + 1) // 2
    sheet = Image.new("RGB", (960, 360 * rows), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * 480, (i // 2) * 360))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def rtx_preview(scene: Scene, config: Config, out_png: Path, workdir: Path, quality: str = "preview") -> Path:
    """Path-traced picture of ``scene`` (its current poses) from the scene camera."""
    import omni.usd

    from scene_physics.authoring import workflow
    from scene_physics.authoring.relax import _tmp_path
    from scene_physics.render import images
    from scene_physics.scene.usd_io import write_scene

    tmp = _tmp_path(workdir, scene.name, "rtx")
    own = scene.path
    write_scene(scene, tmp, config.physics)
    scene.path = own
    stage, renderer = workflow._render_setup(tmp, scene, config, scale=1.0)
    frame = renderer.capture(quality)
    images.save_png(out_png, frame.rgb)
    renderer.destroy()
    omni.usd.get_context().new_stage()
    tmp.unlink(missing_ok=True)
    return out_png


# ============================================================================== preview (no simulator)


@dataclass
class MarginReport:
    visible: int  # target pixels in front of everything
    margin: int  # pixels of the target's silhouette grown by margin_px that nothing covers
    where: str  # where those pixels are on the target's outline ("right edge 9, top 2")
    crop: np.ndarray | None = None  # zoomed picture: occluders coloured, outline red, offending pixels magenta


def margin_pixels(scene: Scene, target: str, camera, *, margin_px: int = 2, device: str | None = "cpu") -> MarginReport:
    """Like :meth:`OcclusionMeter.measure`, but says *where* the target shows (or comes within ``margin_px`` of
    showing), which tells which way to move it or its occluder."""
    from scipy import ndimage

    from scene_physics.authoring.occlusion import transparent_objects
    from scene_physics.render import raycast

    rc = raycast.Raycaster.from_scene(scene, num_worlds=1, camera=camera, device=device, floor=False)
    names = rc.object_names
    poses = scene.truth_poses(names[: rc.num_dynamic])[None]
    k = names.index(target)
    others = rc.render(poses, hidden=[k, *[names.index(n) for n in transparent_objects(scene) if n in names]])
    h, w = camera.height, camera.width
    d_o = np.nan_to_num(others.distance.numpy().reshape(h, w), nan=np.inf)
    d_t = np.nan_to_num(rc.render(poses, solo=k).distance.numpy().reshape(h, w), nan=np.inf)
    sil = np.isfinite(d_t)
    if not sil.any():
        return MarginReport(0, 0, "outside the image")
    d_min = ndimage.minimum_filter(d_t, size=2 * margin_px + 1, mode="constant", cval=np.inf) if margin_px else d_t
    bad = np.isfinite(d_min) & ~(d_o < d_min)
    visible = int((sil & ~(d_o < d_t)).sum())
    ys, xs = np.nonzero(sil)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sides: dict[str, int] = {}
    for y, x in zip(*np.nonzero(bad)):
        fx, fy = (x - x0) / max(x1 - x0, 1), (y - y0) / max(y1 - y0, 1)
        side = ("left" if fx < 0.25 else "right" if fx > 0.75 else "") + (" top" if fy < 0.25 else " bottom" if fy > 0.75 else "")
        side = side.strip() or "middle"
        sides[side] = sides.get(side, 0) + 1
    where = ", ".join(f"{k_} {v}" for k_, v in sorted(sides.items(), key=lambda kv: -kv[1])) or "-"
    ids = others.ids.numpy().reshape(h, w)
    img = np.full((h, w, 3), 230, np.uint8)
    pal = np.array([[200, 120, 40], [60, 160, 90], [90, 160, 220], [200, 100, 170], [170, 170, 80], [100, 200, 200]])
    for i in range(len(names)):
        img[ids == i] = pal[i % len(pal)]
    img[sil & ~ndimage.binary_erosion(sil)] = [255, 0, 0]
    img[bad] = [255, 0, 255]
    pad = 12
    crop = img[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    crop = np.repeat(np.repeat(crop, 3, axis=0), 3, axis=1)
    return MarginReport(visible, int(bad.sum()), where, crop)


def preview(spec_path: str | Path, config: Config, *, out_dir: str | Path | None = None, device: str = "cpu",
            log=None) -> dict:
    """``sp preview``: compose a spec and check what can be checked without physics, in seconds on the CPU:
    the target's occlusion with the margin (and where it fails), the other objects' visibility and framing,
    interpenetration at the composed poses, and the occlusion-only posterior (rule 4). Rest state, robustness
    and the physical evidence need ``sp compose``. Pictures: ``<out_dir>/preview.png`` and ``margin.png``."""
    from PIL import Image

    from scene_physics.authoring import analysis, workflow
    from scene_physics.authoring.occlusion import OcclusionMeter

    log = log or _print
    spec = SceneSpec.load(spec_path)
    notes: list[str] = []
    scene = compose(spec, config, device=device, log=notes.append)
    target = safe_prim_name(spec.target)
    if out_dir is None:
        auth = workflow.authoring_path(config.experiment or "default", spec.name)
        out_dir = auth.with_name(auth.stem + ".compose")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = scene.camera.scaled(config.render.scale)
    m2 = margin_pixels(scene, target, cam, margin_px=2, device=device)
    m4 = margin_pixels(scene, target, cam, margin_px=4, device=device)
    vis = analysis.measure_visibility(scene, device=device)
    meter = OcclusionMeter(scene, cam, device=device)
    frame = [o.name for o in scene.dynamic_objects if not meter._roi(o.name, o.pose, 0)[4]]
    over = analysis.overlaps(scene, device=device)
    post = occlusion_posterior(scene, target, device=device)
    summary = posterior_summary(post)
    observed = {n: float(v) for n, v in vis.fractions.items() if scene[n].dynamic and n != target}
    poses = {o.name: o.pose for o in scene.dynamic_objects}
    pictures(scene, poses, target, {"posterior": post}, out_dir / "preview.png", device=device)
    if m2.crop is not None:
        Image.fromarray(m2.crop).save(out_dir / "margin.png")
    flag = lambda b: "ok" if b else "NO"  # noqa: E731
    hidden = m2.visible == 0 and m2.margin == 0 and not frame
    lines = [
        f"{spec.name} preview (composed poses, no physics: rest, robustness and evidence need `sp compose`)",
        f"  target hidden   : {flag(hidden)} ({m2.visible} px visible, {m2.margin} px inside the 2 px margin"
        + (f": {m2.where}" if m2.margin else "") + f"); {m4.margin} px inside a 4 px margin"
        + (f" ({m4.where})" if m4.margin else "") + " - relaxation moves things by millimetres, keep some spare",
        "  others visible  : " + ", ".join(f"{n} {v:.0%}" for n, v in observed.items())
        + ("" if all(v >= 0.10 for v in observed.values()) else "  (some below 10%)"),
        f"  in frame        : {flag(not frame)} {frame or ''}",
        f"  overlaps        : {flag(not over)} " + ("; ".join(analysis.overlap_lines(over)) if over else ""),
        f"  occlusion only  : {flag(decoy_ok(summary))} {post.line()}",
        "  placement       : " + "; ".join(n.replace("[compose] ", "") for n in notes),
    ]
    text = "\n".join(lines)
    (out_dir / "preview.txt").write_text(text + "\n")
    log(text)
    log(f"[preview] pictures: {out_dir}/preview.png" + (f", {out_dir}/margin.png" if m2.crop is not None else ""))
    return {"name": spec.name, "hidden": hidden, "margin2": m2.margin, "margin4": m4.margin, "where": m2.where,
            "observed": observed, "out_of_frame": frame, "overlaps": analysis.overlap_lines(over),
            "occlusion_only": summary, "decoy_ok": decoy_ok(summary)}


# ============================================================================================== driver


def run(spec_path: str | Path, config: Config, *, perturb: int = 6, rtx: bool = True, finalize: bool = False,
        overwrite: bool = False, log=None) -> Assessment:
    """``sp compose``: compose the spec, relax and assess it, write the authoring scene (relaxed poses) and the
    report + pictures next to it (``authoring/<exp>/<name>.compose/``); with ``finalize``, a passing scene is
    finalized into ``data/<exp>/<name>/`` with its stimulus."""
    from scene_physics import paths
    from scene_physics.authoring import workflow
    from scene_physics.scene.usd_io import write_scene

    log = log or _print
    spec = SceneSpec.load(spec_path)
    exp = config.experiment
    if not exp:
        raise ValueError("sp compose needs an experiment (--exp)")
    auth = workflow.authoring_path(exp, spec.name)
    out_dir = auth.with_name(auth.stem + ".compose")
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = compose(spec, config, log=log)
    scene.path = auth
    composed = {o.name: o.pose.copy() for o in scene.dynamic_objects}
    relaxed, res, extra = assess(scene, spec, config, workdir=out_dir / ".tmp", perturb=perturb, log=log)
    target = safe_prim_name(spec.target)
    pictures(relaxed, {o.name: o.pose for o in relaxed.dynamic_objects}, target, extra, out_dir / "views.png")
    pictures(scene, composed, target, {}, out_dir / "composed.png")
    if rtx:
        rtx_preview(relaxed, config, out_dir / "rtx.png", out_dir / ".tmp")
    text = "\n".join(res.lines())
    (out_dir / "report.txt").write_text(text + "\n")
    (out_dir / "report.json").write_text(json.dumps({"spec": str(spec.path), **res.__dict__}, indent=1, default=float))
    log(text)
    log(f"[compose] pictures: {out_dir}/views.png" + (f", {out_dir}/rtx.png" if rtx else ""))

    # the authoring scene: relaxed poses, roles pinned (the target hidden, everything else observed)
    final = copy.deepcopy(relaxed)
    for o in final.dynamic_objects:
        o.role = "hidden" if o.name == target else "observed"
        o.role_source = "authored"
        o.settle_displacement = res.motion_mm.get(o.name, 0.0) / 1e3
        o.settle_rotation_deg = res.rotation_deg.get(o.name, 0.0)
    final.metadata["compose"] = json.dumps({"spec": str(spec.path), "ok": res.ok, "evidence_mm": res.evidence_mm,
                                            "evidence_object": res.evidence_object})
    write_scene(final, auth, config.physics)
    log(f"[compose] wrote {auth}")
    if finalize:
        if not res.ok and not overwrite:
            raise RuntimeError(f"{spec.name} does not pass the checks; not finalized")
        dest = paths.scene_dir(exp, spec.name)
        workflow.finalize_scene(auth, config, name=spec.name, experiment=exp, target=target, render=rtx,
                                settle=False, overwrite=True)
        log(f"[compose] finalized -> {dest}")
    return res
