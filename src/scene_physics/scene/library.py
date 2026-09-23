"""Object library for authoring scenes without Blender.

Assets are extracted from existing scenes (``sp library extract``), so every asset has exactly the
size and look it has in the experiments (the raw OBJ library is Y-up and some items are ~1 m at native
scale). Each asset is one small USD file, ``assets/objects/<asset>.usd``, holding the mesh in its
*resting* frame: upright as it stood in the scene, footprint centred on the origin, lowest point at
z = 0. With the physics schemas and material applied it can also be dragged straight into Isaac Sim.

    sp library extract                       # from every scene under data/
    sp new data-set/scene060                 # table + camera
    sp add data-set/scene060 jug04 --at 0.1 -0.2
    sp add data-set/scene060 square_wood_block --at 0.1 0.0 --tag target
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from scene_physics import geometry, paths
from scene_physics.camera import CameraSpec
from scene_physics.config import PhysicsConfig
from scene_physics.scene import looks, usd_io
from scene_physics.scene import schema as S
from scene_physics.scene.model import Mesh, Scene, SceneObject, base_asset_name, safe_prim_name


def library_dir() -> Path:
    return paths.assets_dir() / "objects"


def asset_path(asset: str) -> Path:
    return library_dir() / f"{safe_prim_name(asset)}.usd"


def footprint_yaw(xy: np.ndarray) -> float:
    """Yaw that turns a footprint into its smallest axis-aligned rectangle, long side along x."""
    from scipy.spatial import ConvexHull

    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:  # degenerate footprint (a line or a point)
        return 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    best = (np.inf, 0.0)
    for angle in np.arctan2(edges[:, 1], edges[:, 0]):
        c, s = np.cos(-angle), np.sin(-angle)
        r = hull @ np.array([[c, s], [-s, c]])
        ext = r.max(axis=0) - r.min(axis=0)
        area = ext[0] * ext[1]
        if area < best[0] - 1e-12:
            best = (area, angle if ext[0] >= ext[1] else angle + np.pi / 2)
    return float(best[1])


def resting_mesh(obj: SceneObject) -> Mesh:
    """The object's mesh as it rests in the scene (same tilt against gravity), turned about z so its
    footprint is axis-aligned with the long side along x, then re-centred: footprint centre at the
    origin, bottom at z = 0."""
    world = obj.world_points.astype(np.float64)
    yaw = footprint_yaw(world[:, :2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    turned = world @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    lo, hi = turned.min(axis=0), turned.max(axis=0)
    offset = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return Mesh(turned - offset, obj.mesh.triangles)


def write_asset(asset: str, mesh: Mesh, *, dynamic: bool, color, look: dict, physics: PhysicsConfig | None = None,
                source: str = "", instances: int = 1, overwrite: bool = False) -> Path:
    path = asset_path(asset)
    if path.exists() and not overwrite:
        return path
    physics = physics or PhysicsConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    name = safe_prim_name(asset)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    m = usd_io.author_mesh(stage, f"/{name}/{S.MESH_NAME}", mesh, color)
    mat = usd_io.author_preview_material(stage, f"/{name}", look, color)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    usd_io.author_object_physics(root.GetPrim(), dynamic, physics, material_path=None)
    usd_io.set_attr(root.GetPrim(), S.ASSET, asset)
    usd_io.set_attr(root.GetPrim(), S.COLOR, Gf.Vec3f(*[float(c) for c in color]), "Color3f")
    usd_io.set_attr(root.GetPrim(), "sp:library:source", source)
    usd_io.set_attr(root.GetPrim(), "sp:library:dynamic", bool(dynamic))
    usd_io.set_attr(root.GetPrim(), "sp:library:instances", int(instances), "Int")
    stage.GetRootLayer().Save()
    return path


def read_asset(asset: str) -> tuple[Mesh, dict]:
    path = asset_path(asset)
    if not path.exists():
        known = ", ".join(list_assets()) or "none (run `sp library extract`)"
        raise FileNotFoundError(f"no library asset {asset!r} ({path}); known: {known}")
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh = usd_io.read_object_mesh(root, np.eye(4), cache)
    color = usd_io.get_attr(root, S.COLOR)
    info = {
        "asset": usd_io.get_attr(root, S.ASSET, asset),
        "color": tuple(float(c) for c in color) if color is not None else looks.color_for(asset),
        "dynamic": bool(usd_io.get_attr(root, "sp:library:dynamic", True)),
        "source": usd_io.get_attr(root, "sp:library:source", ""),
        "instances": int(usd_io.get_attr(root, "sp:library:instances", 1)),
    }
    return mesh, info


def list_assets() -> list[str]:
    return sorted(p.stem for p in library_dir().glob("*.usd"))


def _instability(obj: SceneObject, moved_m: float = 0.005, moved_deg: float = 2.0) -> float:
    """How far an object moved when its scene was simulated, in units of the settle thresholds (0 = unknown)."""
    d = (obj.settle_displacement or 0.0) / moved_m
    r = (obj.settle_rotation_deg or 0.0) / moved_deg
    return max(d, r)


def _same_size(a: np.ndarray, b: np.ndarray, rel: float = 0.02) -> bool:
    return bool(np.all(np.abs(a - b) <= rel * np.maximum(a, b)))


def _representative(instances: list[tuple[Scene, SceneObject]], max_angle_deg: float = 5.0):
    """The instance whose resting pose the asset takes: from the most common way the object rests
    (instances grouped by which way is up in the object's own frame; groups with an instance that
    stayed put come first), the flattest instance that stayed put, else the one that moved least."""
    cos_max = np.cos(np.radians(max_angle_deg))
    groups: list[list[tuple[np.ndarray, Scene, SceneObject]]] = []
    for scene, obj in instances:
        up = geometry.pose_to_matrix(obj.pose)[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # world up, object frame
        for g in groups:
            if float(np.dot(g[0][0], up)) >= cos_max:
                g.append((up, scene, obj))
                break
        else:
            groups.append([(up, scene, obj)])
    groups.sort(key=lambda g: (not any(_instability(o) <= 1.0 for _, _, o in g), -len(g)))
    stable = [t for t in groups[0] if _instability(t[2]) <= 1.0]
    if stable:  # the flattest of them (a slight lean against a neighbour raises the resting height)
        _, scene, obj = min(stable, key=lambda t: float(np.ptp(t[2].world_points[:, 2])))
    else:
        _, scene, obj = min(groups[0], key=lambda t: _instability(t[2]))
    return scene, obj


def extract_from_scenes(scenes: list[Scene], *, overwrite: bool = False, physics: PhysicsConfig | None = None,
                        log=print) -> list[Path]:
    """One asset per mesh *and size*: the experiments scale the same mesh differently from scene to
    scene (blocks of several proportions, 10 and 20 cm coasters), so every size variant is kept. The
    most common variant gets the base name, the others ``<name>_v2``, ``_v3``, ... in order of use.
    Each asset rests the way the object most often rests in the experiments, taken from an instance
    that stayed put when its scene was simulated (``sp:settle:*``)."""
    variants: dict[str, list[dict]] = {}  # base name -> [{size, instances: [(scene, obj)]}]
    for scene in scenes:
        for obj in scene:
            base = obj.asset or base_asset_name(obj.name)
            size = obj.mesh.extents.astype(np.float64)  # object frame, scale baked in
            groups = variants.setdefault(base, [])
            for g in groups:
                if _same_size(g["size"], size):
                    g["instances"].append((scene, obj))
                    break
            else:
                groups.append({"size": size, "instances": [(scene, obj)]})
    written = []
    for base, groups in variants.items():
        groups.sort(key=lambda g: -len(g["instances"]))  # stable: ties keep first-seen order
        for k, g in enumerate(groups):
            asset = base if k == 0 else f"{base}_v{k + 1}"
            scene, obj = _representative(g["instances"])
            score = _instability(obj)
            existed = asset_path(asset).exists()
            mesh = resting_mesh(obj)
            path = write_asset(asset, mesh, dynamic=obj.dynamic, color=obj.color,
                               look=obj.material or looks.look_for(base), physics=physics,
                               source=f"{scene.name}/{obj.name}", instances=len(g["instances"]), overwrite=overwrite)
            ext = mesh.extents
            state = "kept" if existed and not overwrite else "wrote"
            log(f"{state} {path.name:52s} {'static ' if not obj.dynamic else 'dynamic'} "
                f"{ext[0] * 100:6.1f} x {ext[1] * 100:6.1f} x {ext[2] * 100:6.1f} cm  used {len(g['instances']):2d}x "
                f"(from {scene.name}/{obj.name})")
            if score > 1.0 and not (existed and not overwrite):
                log(f"warning: {asset}: every instance moved when simulated (least: {obj.settle_displacement * 1000:.0f} mm, "
                    f"{obj.settle_rotation_deg:.0f} deg in {scene.name}); it may not rest the way it is stored")
            written.append(path)
    return written


# ------------------------------------------------------------------------------------------ authoring


def new_scene(path: str | Path, name: str, camera: CameraSpec, *, table: str = "dining_room_table",
              experiment: str = "", physics: PhysicsConfig | None = None) -> Path:
    """An authoring scene with just the support table (bottom at z=0, centred) and the camera."""
    mesh, info = read_asset(table)
    obj = SceneObject(safe_prim_name(table), geometry.IDENTITY_POSE, mesh, dynamic=False, asset=info["asset"],
                      color=info["color"], material=looks.look_for(info["asset"]))
    scene = Scene(name, {obj.name: obj}, camera, metadata={"experiment": experiment, "status": "authoring",
                                                           "source": "sp new"})
    return usd_io.write_scene(scene, path, physics)


def _unique_name(existing: set[str], asset: str) -> str:
    base = safe_prim_name(asset)
    k = 1
    while f"{base}_{k:03d}" in existing:
        k += 1
    return f"{base}_{k:03d}"


def overlapping(scene: Scene, obj: SceneObject) -> list[str]:
    """Other dynamic objects whose world bounding box intersects ``obj``'s (likely interpenetration)."""
    lo, hi = obj.world_aabb()
    out = []
    for other in scene.dynamic_objects:
        if other.name == obj.name:
            continue
        olo, ohi = other.world_aabb()
        if np.all(lo < ohi) and np.all(olo < hi):
            out.append(other.name)
    return out


def add_object(scene_path: str | Path, asset: str, *, at: tuple[float, float] | None = None,
               yaw_deg: float | None = None, name: str | None = None, tags: list[str] | None = None,
               drop: float = 0.001, rng: np.random.Generator | None = None, scale=None,
               physics: PhysicsConfig | None = None, log=print) -> str:
    """Add a library asset to a scene file, standing on the support surface. Returns the object name.

    ``scale``: one factor or three (x, y, z in the asset's resting frame; z is height).
    """
    from scene_physics.scene.usd_io import read_scene

    rng = rng or np.random.default_rng()
    scene_path = Path(scene_path)
    scene = read_scene(scene_path)
    mesh, info = read_asset(asset)
    if scale is not None:
        factors = np.broadcast_to(np.asarray(scale, dtype=np.float64).reshape(-1), (3,))
        if np.any(factors <= 0):
            raise ValueError(f"scale must be positive, got {scale}")
        mesh = Mesh(mesh.points * factors, mesh.triangles)  # bottom stays at z = 0, footprint centred
    x_min, x_max, y_min, y_max = scene.support_bounds(margin=0.05)
    if at is None:
        at = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
    if not (x_min - 0.05 <= at[0] <= x_max + 0.05 and y_min - 0.05 <= at[1] <= y_max + 0.05):
        raise ValueError(f"position {at} is off the table (x in [{x_min:.2f}, {x_max:.2f}], y in [{y_min:.2f}, {y_max:.2f}])")
    yaw = np.radians(yaw_deg) if yaw_deg is not None else float(rng.uniform(0, 2 * np.pi))
    z = scene.support_top() + drop  # resting frame: the mesh bottom is at z = 0
    pose = geometry.make_pose([at[0], at[1], z], geometry.yaw_quat(yaw))
    obj_name = safe_prim_name(name) if name else _unique_name(set(scene.objects), asset)
    if obj_name in scene.objects:
        raise ValueError(f"{obj_name} already exists in {scene.name}")
    obj = SceneObject(obj_name, pose, mesh, dynamic=info["dynamic"], asset=info["asset"], color=info["color"],
                      material=looks.look_for(info["asset"]), tags=list(tags or []))
    if tags and any(t in ("observed", "hidden") for t in tags):
        raise ValueError("roles are measured; use --tag for annotations like target / occluder")
    hits = overlapping(scene, obj)
    if hits:
        log(f"[sp] warning: {obj_name} overlaps the bounding box of {', '.join(hits)}; "
            "sp check will show whether they interpenetrate")
    stage = Usd.Stage.Open(str(scene_path))
    usd_io.author_object(stage, obj, physics or PhysicsConfig())
    stage.GetRootLayer().Save()
    return obj_name
