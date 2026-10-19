"""Pictures for the human study (``sp study-renders``): what participants see around each stimulus.

The study website shows, for every trial, the target alone on the table, then the empty table, then the stimulus;
after a practice trial it shows where the target was. ``sp study-renders`` makes those pictures for finished scenes,
through the scene camera and with the studio look of the stimulus, and writes them next to it in ``render/``::

    target_card.png        the target alone on the table, in red, at the centre of the table: resting on the face it
                           rests on in the scene and turned to show its broadest side. Every other object is
                           invisible, so its position, tilt and what it leans on in the scene are not shown.
    empty_table.png        the table with every object invisible
    reveal.png             (--reveal) where the target is: the stimulus with the objects that hide the target faded
                           to a ghost, and the target, in red, behind them
    reveal_unoccluded.png  (--reveal) the render behind the ghost: the scene with the target in red and the objects
                           that hide it invisible
    study.json             what was rendered: the red, the card pose and how well the card is centred, the objects
                           the reveal fades

Every change is render-time only (the stage's session layer): ``scene.usd`` is never modified.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scene_physics import geometry
from scene_physics.config import Config

RED = (0.70, 0.015, 0.015)  # linear RGB of the target in every study picture
CARD_YAW_OFFSET_DEG = 25.0  # the card turns the target this far past its broadest side (a three-quarter view)
FADE_MIN_COVERAGE = 0.02  # the reveal fades the objects that hide at least this much of the target's outline
GHOST = 0.25  # opacity the faded objects keep in the reveal


def _print(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================================ geometry (pure numpy)


def _yaw(degree: float) -> np.ndarray:
    t = np.radians(degree)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def card_rotation(rotation: np.ndarray, points: np.ndarray, right: np.ndarray,
                  yaw_offset_deg: float = CARD_YAW_OFFSET_DEG) -> np.ndarray:
    """Body -> world rotation of the target on the card.

    The body axis closest to world up in the scene stays up, snapped exactly upright (the object rests on the face it
    rests on in the scene), and the object is turned about z to its widest silhouette (widest along the camera's
    horizontal ``right`` axis; for a box, its diagonal across the view), plus ``yaw_offset_deg``. The tilt and
    heading it has in the scene are dropped.
    """
    k = int(np.argmax(np.abs(rotation[2, :])))
    sign = 1.0 if rotation[2, k] >= 0 else -1.0
    i, j = [a for a in range(3) if a != k]
    r = np.zeros((3, 3))
    r[:, k] = (0.0, 0.0, sign)
    r[:, i] = (1.0, 0.0, 0.0)
    r[:, j] = (0.0, 1.0, 0.0)
    if np.linalg.det(r) < 0:
        r[:, j] *= -1.0
    horizontal = np.array([right[0], right[1], 0.0])
    horizontal /= np.linalg.norm(horizontal)
    upright = np.asarray(points, dtype=float) @ r.T
    widths = [np.ptp(upright @ _yaw(d).T @ horizontal) for d in range(0, 180, 2)]
    return _yaw(2.0 * int(np.argmax(widths)) + yaw_offset_deg) @ r


def surface_height(points: np.ndarray, triangles: np.ndarray, xy: np.ndarray) -> float:
    """Highest point of a triangle mesh (world ``points``) on the vertical lines through ``xy`` ``(N, 2)``;
    ``-inf`` if no line hits it."""
    tri = np.asarray(points, dtype=float)[np.asarray(triangles)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e0, e1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    den = e0[:, 0] * e1[:, 1] - e1[:, 0] * e0[:, 1]
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    best = -np.inf
    for p in np.atleast_2d(np.asarray(xy, dtype=float)):
        d = p - a[:, :2]
        u = (d[:, 0] * e1[:, 1] - e1[:, 0] * d[:, 1]) / den
        v = (e0[:, 0] * d[:, 1] - d[:, 0] * e0[:, 1]) / den
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        if hit.any():
            z = a[hit, 2] + u[hit] * (b[hit, 2] - a[hit, 2]) + v[hit] * (c[hit, 2] - a[hit, 2])
            best = max(best, float(z.max()))
    return best


def card_pose(target, table, camera, yaw_offset_deg: float = CARD_YAW_OFFSET_DEG) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the target on the card and the point it stands on: rotated by :func:`card_rotation`, its
    footprint (bounding box) centred on the centre of the table top, resting on the highest point of the table
    under that footprint (a few tables are tilted by up to 1.7 degrees)."""
    rotation = Rotation.from_quat(target.pose[3:]).as_matrix()
    rot = card_rotation(rotation, target.mesh.points, camera.rotation[:, 0], yaw_offset_deg)
    v = target.mesh.points.astype(float) @ rot.T
    lo, hi = v.min(axis=0), v.max(axis=0)
    tlo, thi = table.world_aabb()
    centre = (np.asarray(tlo[:2], dtype=float) + np.asarray(thi[:2], dtype=float)) / 2.0
    half = (hi[:2] - lo[:2]) / 2.0
    corners = np.array([[0, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    top = surface_height(table.world_points, table.mesh.triangles, centre + corners * half)
    if not np.isfinite(top):
        raise ValueError(f"no table surface under the centre of {table.name}")
    position = np.array([centre[0] - (lo[0] + hi[0]) / 2.0, centre[1] - (lo[1] + hi[1]) / 2.0, top - lo[2]])
    return geometry.make_pose(position, Rotation.from_matrix(rot).as_quat()), np.array([*centre, top])


def coverage(segmentation: np.ndarray, labels: dict[str, int], silhouette: np.ndarray) -> dict[str, float]:
    """Fraction of the target's full (amodal) silhouette that each object covers in the stimulus."""
    n = int(silhouette.sum())
    if n == 0:
        return {}
    counts = np.bincount(np.asarray(segmentation)[silhouette].ravel().astype(np.int64))
    names = {int(v): k for k, v in labels.items() if k != "background"}
    return {names[i]: float(c) / n for i, c in enumerate(counts) if c and i in names}


def ghost_composite(stimulus: np.ndarray, unoccluded: np.ndarray, mask: np.ndarray, ghost: float = GHOST) -> np.ndarray:
    """The stimulus with the pixels in ``mask`` (the objects being faded) mixed with ``unoccluded``: those objects
    keep ``ghost`` of their opacity, and what is behind them shows through."""
    a = (1.0 - ghost) * np.asarray(mask, dtype=np.float32)[..., None]
    out = np.asarray(stimulus, dtype=np.float32) * (1.0 - a) + np.asarray(unoccluded, dtype=np.float32) * a
    return np.round(out).clip(0, 255).astype(np.uint8)


def detail(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean horizontal brightness change between neighbouring pixels inside ``mask``: the table's wood grain. A
    frame captured before the path tracer has settled after a change comes back smoothed, with far less of it."""
    g = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    both = mask[:, 1:] & mask[:, :-1]
    return float(np.abs(np.diff(g, axis=1))[both].mean()) if both.any() else 0.0


def card_check(segmentation: np.ndarray, target_id: int, camera, stand_point: np.ndarray) -> dict:
    """Where the target appears on the card: its pixel bounding box, and how far the box's centre is from the
    projection of the centre of the table top (``offset_px``, horizontal)."""
    ys, xs = np.nonzero(np.asarray(segmentation) == target_id)
    px, _ = camera.project(np.asarray(stand_point, dtype=float)[None, :])
    out = {"table_centre_px": [round(float(px[0, 0]), 1), round(float(px[0, 1]), 1)], "target_px": int(len(xs))}
    if len(xs) == 0:
        return out
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    h, w = np.asarray(segmentation).shape
    out.update({
        "bbox": box,
        "bbox_centre_px": [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0],
        "offset_px": round((box[0] + box[2]) / 2.0 - float(px[0, 0]), 1),
        "in_frame": bool(box[0] > 0 and box[1] > 0 and box[2] < w and box[3] < h),
    })
    return out


# ============================================================================================ rendering (Isaac Sim)


def render_study(scene_path: str | Path, config: Config, *, reveal: bool = False, color=RED, quality: str = "final",
                 log=None) -> dict:
    """Render the study pictures of one finished scene into its ``render/`` folder (see the module notes).

    Every picture is checked against the stimulus before it is written: the right objects show in its
    segmentation (a frame RTX failed to render repeats the previous one), and its table has the same wood grain as
    the stimulus's (a render made with streamed, low-resolution textures is smoothed, see ``_render_setup``).
    ``record["problems"]`` lists what failed; nothing is written then.
    """
    import omni.usd
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    from scene_physics.authoring import workflow
    from scene_physics.isaac import app
    from scene_physics.isaac.environment import LOOK_ROOT
    from scene_physics.render import images
    from scene_physics.scene import load_scene, usd_io
    from scene_physics.scene import schema as S
    from scene_physics.scene.model import safe_prim_name

    log = log or _print
    t0 = time.time()
    scene_path = Path(scene_path)
    scene = load_scene(scene_path)
    target, table = scene.target, scene.support
    if target is None or table is None:
        raise ValueError(f"{scene_path}: the study pictures need a target and a table")
    if any(o.segmentation_id is None for o in scene):
        scene.assign_segmentation_ids()
    out = scene_path.parent / "render"
    for name in ("rgb.png", "segmentation.png") + (("target.png",) if reveal else ()):
        if not (out / name).is_file():
            raise FileNotFoundError(f"{out / name} is missing: render the stimulus first (sp render)")
    stimulus = np.asarray(Image.open(out / "rgb.png").convert("RGB"))
    seg, labels = images.read_segmentation(out / "segmentation.png")
    statics = {int(o.segmentation_id) for o in scene.static_objects}
    table_px = np.isin(seg, list(statics))
    grain = detail(stimulus, table_px)
    cam = scene.camera.scaled(config.render.scale)
    stage, renderer = workflow._render_setup(scene_path, scene, config, scale=config.render.scale)  # full-res textures
    if (renderer.height, renderer.width) != stimulus.shape[:2]:
        raise ValueError(f"{out / 'rgb.png'} is {stimulus.shape[1]}x{stimulus.shape[0]}, the renders "
                         f"{renderer.width}x{renderer.height}")
    prims = {o.name: stage.GetPrimAtPath(f"{S.OBJECTS}/{safe_prim_name(o.name)}") for o in scene}
    session = stage.GetSessionLayer()
    problems: list[str] = []

    def show_only(names: set[str]) -> None:
        with Usd.EditContext(stage, session):
            for o in scene.dynamic_objects:
                imageable = UsdGeom.Imageable(prims[o.name])
                imageable.MakeVisible() if o.name in names else imageable.MakeInvisible()

    def capture(what: str, shown: set[str]):
        """A frame of the current state, checked. The path tracer needs a frame to settle after a change: the
        first frame is thrown away."""
        renderer.capture(quality)
        frame = renderer.capture(quality)
        ids = {int(i) for i in np.unique(frame.segmentation)} - {0} - statics
        wanted = {int(scene[n].segmentation_id) for n in shown}
        if ids != wanted:
            names = {int(o.segmentation_id): o.name for o in scene}
            problems.append(f"{what}: shows {sorted(names.get(i, str(i)) for i in ids)}, should show {sorted(shown)}")
        # the table must look the same in every picture and in the stimulus (participants see one after the other)
        ratio = detail(frame.rgb, table_px & np.isin(frame.segmentation, list(statics))) / max(grain, 1e-6)
        if ratio < 0.8:
            problems.append(f"{what}: the table is smoothed ({ratio:.2f} of the stimulus's wood grain)")
        elif ratio > 1.25:
            problems.append(f"{what}: the stimulus's table is smoothed ({1 / ratio:.2f} of this picture's wood "
                            "grain): render the stimulus again (sp render)")
        return frame, round(ratio, 3)

    # the target in red, wherever it appears (the look's scope lives in the session layer too)
    with Usd.EditContext(stage, session):
        path = f"{LOOK_ROOT}/Materials/StudyTarget"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        for mesh in usd_io.mesh_prims(prims[target.name]):
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material, UsdShade.Tokens.strongerThanDescendants)

    record: dict = {"scene": str(scene_path), "target": target.name, "color_linear": [float(c) for c in color],
                    "quality": quality, "samples": config.render.samples,
                    "created": _dt.datetime.now().isoformat(timespec="seconds")}
    grains: dict[str, float] = {}

    # 1) the empty table
    show_only(set())
    empty, grains["empty_table"] = capture("empty table", set())

    # 2) the reveal, with the target where it is: the objects that hide it are taken away, then faded back in as a
    #    ghost over the stimulus
    behind = None
    if reveal:
        silhouette = np.asarray(Image.open(out / "target.png"))
        silhouette = silhouette[..., 3] > 0 if silhouette.ndim == 3 else silhouette > 0
        cover = coverage(seg, labels, silhouette)
        faded = sorted(n for n, f in cover.items() if f >= FADE_MIN_COVERAGE and n != target.name and n in scene
                       and scene[n].dynamic)
        shown = {o.name for o in scene.dynamic_objects} - set(faded)
        show_only(shown)
        behind, grains["reveal_unoccluded"] = capture("reveal", shown)
        record["reveal"] = {"faded": faded, "ghost": GHOST, "min_coverage": FADE_MIN_COVERAGE,
                            "coverage": {n: round(f, 4) for n, f in sorted(cover.items(), key=lambda kv: -kv[1])}}

    # 3) the card: the target alone, moved to the centre of the table
    pose, stand = card_pose(target, table, cam)
    with Usd.EditContext(stage, session):
        usd_io.set_pose(prims[target.name], pose)
    show_only({target.name})
    card, grains["target_card"] = capture("card", {target.name})
    check = card_check(card.segmentation, int(target.segmentation_id), cam, stand)
    record["card"] = {"pose": [round(float(v), 6) for v in pose], "stands_on": [round(float(v), 4) for v in stand],
                      "yaw_offset_deg": CARD_YAW_OFFSET_DEG, **check}
    record["table_grain_vs_stimulus"] = grains
    renderer.destroy()
    omni.usd.get_context().new_stage()  # let RTX release this scene before the next one opens
    app.update(4)

    if not check.get("target_px"):
        problems.append("the target does not show on the card")
    elif not check["in_frame"]:
        problems.append("the target on the card touches the image border")
    elif abs(check["offset_px"]) > 4.0:
        problems.append(f"the target on the card is {check['offset_px']:+.1f} px off the table centre")
    record["problems"] = problems
    if not problems:  # only pictures that passed every check replace earlier ones
        images.save_png(out / "empty_table.png", empty.rgb)
        images.save_png(out / "target_card.png", card.rgb)
        if behind is not None:
            mask = np.isin(seg, [labels[n] for n in record["reveal"]["faded"]])
            images.save_png(out / "reveal_unoccluded.png", behind.rgb)
            images.save_png(out / "reveal.png", ghost_composite(stimulus, behind.rgb, mask))
        (out / "study.json").write_text(json.dumps(record, indent=1))
    faded_note = f", reveal fades {', '.join(record['reveal']['faded']) or 'nothing'}" if reveal else ""
    log(f"[study] {scene_path.parent.name}: card {check.get('target_px', 0)} px, {check.get('offset_px', 'n/a')} px "
        f"from the table centre, table grain {min(grains.values()):.2f} of the stimulus{faded_note} "
        f"({time.time() - t0:.0f} s)" + (f"; NOT WRITTEN: {'; '.join(problems)}" if problems else ""))
    return record
