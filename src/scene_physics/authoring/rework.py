"""Rework a finished scene for a new experiment (``sp rework``): the experiment's camera, a verified rest
state, and the target hidden with a margin, changing the original arrangement as little as possible.

For each scene, starting from the source scene's poses (plus any hand edits from an edits file):

1. **relax** (:mod:`.relax`): overlaps pushed apart under low gravity, settled under the model's physics,
   verified by an inference-length rollout and small perturbations;
2. **measure** the target's occlusion at the stimulus resolution with a margin (:mod:`.occlusion`) and
   compare the scene's relationships with the source design (:mod:`.contacts`): every contact between the
   target and another object must still be there, and no new one may appear;
3. if the target shows, **hide** it (:mod:`.hide`): the cheapest rigid move of the target together with
   what rests on it (and, if needed, a small move of its main occluder) that hides it with a larger margin
   and keeps every contact, then back to 1.

A scene that passes is written to ``authoring/<exp>/<name>.usd`` and finalized (``data/<exp>/<name>/``,
stimulus rendered). Scenes the loop cannot fix are reported with the reason; they need a hand edit in
the edits file (see :func:`load_edits`).
"""

from __future__ import annotations

import copy
import json
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from scene_physics import geometry, paths
from scene_physics.authoring.hide import STAGES, HideConfig, move_poses
from scene_physics.authoring.relax import RelaxConfig
from scene_physics.config import Config
from scene_physics.scene.model import Scene


@dataclass
class ReworkConfig:
    margin_px: int = 2  # acceptance: target hidden with this many stimulus pixels to spare
    max_rounds: int = 5
    relax: RelaxConfig = field(default_factory=lambda: RelaxConfig(perturb_trials=4))
    hide_stages: tuple[HideConfig, ...] = STAGES  # tried in order until one finds an edit
    contact_tol: float = 0.002
    require_robust: bool = False  # fail scenes whose rest does not survive the perturbation test
    design_occluder_min: float = 0.05  # an object covering this much of the target in the source is an occluder
    min_design_coverage: float = 0.5  # the design's occluders must cover at least this much in the result


@dataclass
class Round:
    relax_ok: bool
    relax_warnings: list[str]
    visible: int
    margin: int
    lost: list
    gained: list
    worst_motion_mm: float
    behind_design_occluders: float = 0.0
    hide: str = ""


@dataclass
class ReworkResult:
    name: str
    target: str
    ok: bool
    reason: str = ""
    rounds: list[Round] = field(default_factory=list)
    displacement_mm: dict[str, float] = field(default_factory=dict)  # source pose -> final, per object
    rotation_deg: dict[str, float] = field(default_factory=dict)
    edits: dict = field(default_factory=dict)
    design_occluders: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        status = "OK" if self.ok else f"FAILED ({self.reason})"
        moved = ", ".join(f"{n} {d:.1f} mm/{self.rotation_deg[n]:.1f} deg" for n, d in
                          sorted(self.displacement_mm.items(), key=lambda kv: -kv[1]) if d > 1.0 or self.rotation_deg[n] > 1.0)
        return f"{self.name}: {status}; {len(self.rounds)} round(s); moved: {moved or 'nothing > 1 mm'}"


def load_edits(path: str | Path | None) -> dict:
    """Hand edits per scene, applied to the source poses before the first relaxation::

        [scene017]
        note = "loaf moved behind the bowl"
        move = { b03_loafbread_008_008 = [0.02, 0.03, 10.0] }   # dx, dy (m), yaw (deg) about the object's centre
        pose = { coffeemug_008 = [x, y, z, qx, qy, qz, qw] }     # absolute pose
        camera = { azimuth = 5.0 }                               # per-scene camera override (orbit terms)
    """
    if path is None or not Path(path).exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def apply_edits(scene: Scene, edit: dict) -> None:
    """``move``: ``name = [dx, dy, yaw_deg]`` or ``[dx, dy, yaw_deg, dz]``, rigid, about the object's centre
    (``with = { name = [other, ...] }`` moves those objects along); ``pose``: absolute poses."""
    groups = edit.get("with") or {}
    for name, move in (edit.get("move") or {}).items():
        dx, dy, yaw = (float(v) for v in move[:3])
        dz = float(move[3]) if len(move) > 3 else 0.0
        names = [name, *groups.get(name, [])]
        piv = scene[name].world_points.mean(axis=0)
        piv[2] = 0.0
        for n, pose in move_poses(scene, names, dx, dy, yaw, piv).items():
            pose[2] += dz
            scene[n].pose = pose
    for name, pose in (edit.get("pose") or {}).items():
        scene[name].pose = geometry.canonical_pose(np.asarray(pose, dtype=float))


def rework_scene(source: Scene, config: Config, cfg: ReworkConfig | None = None, *, edit: dict | None = None,
                 workdir: str | Path | None = None, log=print) -> tuple[Scene, ReworkResult]:
    """Rework one scene in memory (Isaac Sim must be running). Returns the new scene and the log."""
    from scene_physics.authoring import contacts, occlusion, relax
    from scene_physics.authoring.hide import Hider

    cfg = cfg or ReworkConfig()
    edit = edit or {}
    t0 = time.time()
    scene = copy.deepcopy(source)
    target = source.target.name
    if config.camera is not None:
        scene.camera = config.camera
    if edit.get("camera"):
        from scene_physics.authoring.occlusion import orbit_camera, orbit_of

        az, el, d = orbit_of(scene.camera)
        c = edit["camera"]
        scene.camera = orbit_camera(c.get("azimuth", az), c.get("elevation", el), c.get("distance", d),
                                    look_at=scene.camera.target, reference=scene.camera, name=scene.camera.name)
    apply_edits(scene, edit)
    result = ReworkResult(scene.name, target, False, edits=edit)
    others = [o.name for o in source.dynamic_objects if o.name != target]
    intent = contacts.ContactAnalyzer(source).separations([target], others=others)  # the design's contacts
    stimulus_cam = scene.camera.scaled(config.render.scale)
    # the design's occluders: what covers the target in the source scene, seen from the source camera
    src_occ = occlusion.OcclusionMeter(source, source.camera.scaled(config.render.scale)).measure(target, margin_px=0)
    design_occluders = [n for n, f in src_occ.coverage.items() if f >= cfg.design_occluder_min and n in others]
    result.design_occluders = design_occluders
    for k in range(cfg.max_rounds):
        rep = relax.relax(scene, config, cfg.relax, workdir=workdir, log=log)
        for name, pose in rep.final_poses().items():
            scene[name].pose = pose
        occ = occlusion.OcclusionMeter(scene, stimulus_cam).measure(target, margin_px=cfg.margin_px)
        behind = sum(occ.coverage.get(n, 0.0) for n in design_occluders)
        now = contacts.ContactAnalyzer(scene).separations([target], others=others)
        lost, gained = contacts.changed(intent, now, touching=cfg.contact_tol, apart=2 * cfg.contact_tol)
        worst = max(o.total_displacement for o in rep.objects) * 1e3
        rnd = Round(rep.ok, rep.warnings, occ.visible, occ.margin, sorted(lost), sorted(gained), worst, behind)
        result.rounds.append(rnd)
        log(f"[rework] {scene.name} round {k + 1}: relax {'ok' if rep.ok else 'NOT ok'}, target visible {occ.visible} px, "
            f"margin {occ.margin} px, behind design occluders {behind:.0%}, lost {sorted(lost)}, gained {sorted(gained)}")
        hard = [w for w in rep.warnings if not w.startswith("perturb:")]
        if hard and all(w.startswith("verify:") for w in hard) and k < cfg.max_rounds - 1:
            continue  # still creeping: relax again from where it got to
        if hard or (cfg.require_robust and not rep.robust):
            result.reason = "relaxation: " + "; ".join(hard or rep.warnings)
            break
        if occ.hidden and not lost and not gained and (not design_occluders or behind >= cfg.min_design_coverage):
            result.ok = True
            result.warnings = [w for w in rep.warnings if w.startswith("perturb:")]
            break
        # hide the target and/or restore the design's contacts with the smallest edit
        for stage in cfg.hide_stages:
            h = Hider(scene, target, stimulus_cam, stage, intent=intent, design_occluders=design_occluders).search()
            if h.found:
                break
        rnd.hide = h.describe()
        log(f"[rework] {scene.name}: {h.describe()}")
        if not h.found:
            problems = []
            if not occ.hidden:
                problems.append(f"target visible ({occ.visible} px, margin {occ.margin})")
            if lost or gained:
                problems.append(f"contacts changed (lost {sorted(lost)}, gained {sorted(gained)})")
            if design_occluders and behind < cfg.min_design_coverage:
                problems.append(f"only {behind:.0%} behind {design_occluders}")
            result.reason = " and ".join(problems) + f"; {h.describe()}"
            break
        for name, pose in h.poses.items():
            scene[name].pose = pose
    else:
        result.reason = f"not solved in {cfg.max_rounds} rounds"
    for o in scene.dynamic_objects:
        src = source[o.name].pose
        result.displacement_mm[o.name] = float(geometry.position_error(o.pose, src) * 1e3)
        result.rotation_deg[o.name] = float(np.degrees(geometry.rotation_angle(o.pose[3:], src[3:])))
    result.wall_seconds = time.time() - t0
    return scene, result


def rework(sources: list[str], config: Config, cfg: ReworkConfig | None = None, *, edits_path: str | Path | None = None,
           render: bool = True, overwrite: bool = False, only_failed_report: bool = False) -> list[ReworkResult]:
    """``sp rework``: rework source scenes into ``config.experiment`` (finalized under data/)."""
    from scene_physics.isaac import app

    app.launch()
    from scene_physics.authoring import workflow
    from scene_physics.scene import load_scene
    from scene_physics.scene.usd_io import write_scene

    exp = config.experiment
    if not exp:
        raise ValueError("sp rework needs an experiment (--exp) for the new scenes")
    edits = load_edits(edits_path or paths.experiments_dir() / exp / "edits.toml")
    out_dir = paths.data_dir() / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "rework_log.json"
    history = json.loads(log_path.read_text()) if log_path.exists() else {}
    results = []
    for spec in sources:
        src = load_scene(spec)
        final_dir = paths.scene_dir(exp, src.name)
        if (final_dir / "scene.usd").exists() and not overwrite:
            print(f"[rework] {src.name}: {final_dir} exists (use --overwrite)", flush=True)
            continue
        scene, res = rework_scene(src, config, cfg, edit=edits.get(src.name), workdir=paths.authoring_dir() / exp / ".tmp")
        print(f"[rework] {res.summary()}", flush=True)
        results.append(res)
        history[src.name] = res.to_dict()
        log_path.write_text(json.dumps(history, indent=1, default=float))
        if not res.ok:
            continue
        for o in scene.dynamic_objects:
            o.role_source = "measured"
        scene.metadata.update({"experiment": exp, "status": "authoring", "source": str(src.path),
                               "rework": json.dumps({"from": str(src.path), "edits": res.edits,
                                                      "displacement_mm": res.displacement_mm,
                                                      "rotation_deg": res.rotation_deg})})
        for key in ("legacy_format", "import_warnings"):
            scene.metadata.pop(key, None)
        auth = workflow.authoring_path(exp, src.name)
        auth = auth.with_name(f"{auth.stem}.{time.strftime('%H%M%S')}.usd") if auth.exists() else auth
        write_scene(scene, auth, config.physics)
        workflow.finalize_scene(auth, config, name=src.name, experiment=exp, target=res.target, render=render,
                                overwrite=True)
    return results
