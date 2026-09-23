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


def resting_mesh(obj: SceneObject) -> Mesh:
    """The object's mesh as it stands in the scene, re-centred: XY footprint centre at the origin, bottom at z=0."""
    world = obj.world_points.astype(np.float64)
    lo, hi = world.min(axis=0), world.max(axis=0)
    offset = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return Mesh(world - offset, obj.mesh.triangles)


def write_asset(asset: str, mesh: Mesh, *, dynamic: bool, color, look: dict, physics: PhysicsConfig | None = None,
                source: str = "", overwrite: bool = False) -> Path:
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
    }
    return mesh, info


def list_assets() -> list[str]:
    return sorted(p.stem for p in library_dir().glob("*.usd"))


def extract_from_scenes(scenes: list[Scene], *, overwrite: bool = False, physics: PhysicsConfig | None = None,
                        log=print) -> list[Path]:
    """One asset per base name, taken from the first scene that contains it."""
    written, seen = [], set()
    for scene in scenes:
        for obj in scene:
            asset = obj.asset or base_asset_name(obj.name)
            if asset in seen:
                continue
            seen.add(asset)
            existed = asset_path(asset).exists()
            path = write_asset(asset, resting_mesh(obj), dynamic=obj.dynamic, color=obj.color,
                               look=obj.material or looks.look_for(asset), physics=physics,
                               source=f"{scene.name}/{obj.name}", overwrite=overwrite)
            ext = obj.mesh.extents
            state = "kept" if existed and not overwrite else "wrote"
            log(f"{state} {path.name:48s} {'static ' if not obj.dynamic else 'dynamic'} "
                f"{ext[0] * 100:6.1f} x {ext[1] * 100:6.1f} x {ext[2] * 100:6.1f} cm  (from {scene.name}/{obj.name})")
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


def add_object(scene_path: str | Path, asset: str, *, at: tuple[float, float] | None = None,
               yaw_deg: float | None = None, name: str | None = None, tags: list[str] | None = None,
               drop: float = 0.001, rng: np.random.Generator | None = None,
               physics: PhysicsConfig | None = None) -> str:
    """Add a library asset to a scene file, standing on the support surface. Returns the object name."""
    from scene_physics.scene.usd_io import read_scene

    rng = rng or np.random.default_rng()
    scene_path = Path(scene_path)
    scene = read_scene(scene_path)
    mesh, info = read_asset(asset)
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
    stage = Usd.Stage.Open(str(scene_path))
    usd_io.author_object(stage, obj, physics or PhysicsConfig())
    stage.GetRootLayer().Save()
    return obj_name
