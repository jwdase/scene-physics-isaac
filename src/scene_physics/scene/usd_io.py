"""Read and write ``scene.usd``.

Works with plain ``pxr`` (the standalone OpenUSD in the venv) as well as inside Isaac Sim, so scene
files can be created, inspected and validated without booting Kit. PhysX-specific attributes are
authored as ordinary typed attributes plus their applied-schema tokens, which is exactly what
``PhysxSchema.*API.Apply`` writes.

Layout written by :func:`write_scene` (see ``docs/DESIGN.md`` for the attribute reference)::

    /World                       Xform, sp:scene:* metadata, defaultPrim
      /PhysicsScene              UsdPhysics.Scene + PhysxSceneAPI
      /PhysicsMaterial           shared contact material
      /Ground                    invisible static collider, top face at z = 0
      /Objects/<name>            Xform whose transform is the object's ground-truth pose
        /mesh                    geometry + collider
        /Looks/Material          UsdPreviewSurface
      /Camera                    scene camera
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from scene_physics import geometry
from scene_physics.camera import CameraSpec
from scene_physics.config import PhysicsConfig
from scene_physics.scene import looks
from scene_physics.scene import schema as S
from scene_physics.scene.model import Mesh, Prior, Scene, SceneObject, Visibility, safe_prim_name

# ============================================================================ low-level helpers


def _vtype(name: str):
    return getattr(Sdf.ValueTypeNames, name)


def set_attr(prim: Usd.Prim, name: str, value, type_name: str | None = None) -> None:
    """Create (if needed) and set an attribute, inferring the USD type from the Python value."""
    if value is None:
        if prim.HasAttribute(name):
            prim.RemoveProperty(name)
        return
    if type_name is None:
        if isinstance(value, bool):
            type_name = "Bool"
        elif isinstance(value, (int, np.integer)):
            type_name = "Int"
        elif isinstance(value, (float, np.floating)):
            type_name = "Double"
        elif isinstance(value, str):
            type_name = "String"
        elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            type_name = "TokenArray"
        else:
            raise TypeError(f"cannot infer a USD type for {name}={value!r}")
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, _vtype(type_name), custom=True)
    attr.Set(value)


def get_attr(prim: Usd.Prim, name: str, default=None):
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    value = attr.Get()
    return default if value is None else value


def applied_api_names(prim: Usd.Prim) -> list[str]:
    """API schema names in the prim's ``apiSchemas`` metadata, including ones whose schema plugin is
    not loaded (outside Kit, ``GetAppliedSchemas`` hides the PhysX APIs)."""
    listop = prim.GetMetadata("apiSchemas")
    if listop is None:
        return []
    return [str(t) for t in listop.GetAddedOrExplicitItems()]


def apply_api(prim: Usd.Prim, api_name: str) -> None:
    if api_name not in applied_api_names(prim):
        prim.AddAppliedSchema(api_name)


def set_physx_attr(prim: Usd.Prim, name: str, value) -> None:
    """Author a PhysX schema attribute (and apply its API) without needing PhysxSchema."""
    for prefix, api in S.PHYSX_API_FOR_PREFIX.items():
        if name.startswith(prefix):
            apply_api(prim, api)
            break
    set_attr(prim, name, value, S.PHYSX_ATTR_TYPES[name])


def gf_quat(q_xyzw) -> Gf.Quatf:
    x, y, z, w = (float(v) for v in q_xyzw)
    return Gf.Quatf(w, x, y, z)


def matrix_to_gf(m: np.ndarray) -> Gf.Matrix4d:
    """numpy column-vector matrix -> Gf (row-vector convention, i.e. transposed)."""
    return Gf.Matrix4d(*[float(v) for v in np.asarray(m, dtype=float).T.reshape(-1)])


def gf_to_matrix(m: Gf.Matrix4d) -> np.ndarray:
    return np.array(m, dtype=float).T


def set_pose(prim: Usd.Prim, pose: np.ndarray) -> None:
    """Make ``prim``'s world pose equal ``pose`` using ``translate`` + ``orient`` (+ existing scale).

    Any other xform-op stack (e.g. Blender's ``rotateXYZ``) is replaced; an authored scale is kept.
    """
    pose = np.asarray(pose, dtype=float)
    parent_world = world_matrix(prim.GetParent())
    if not np.allclose(parent_world, np.eye(4)):
        local = np.linalg.inv(parent_world) @ geometry.pose_to_matrix(pose)
        pose = geometry.matrix_to_pose(local)
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    names = [op.GetOpName() for op in ops]
    if names[:2] == ["xformOp:translate", "xformOp:orient"] and names[2:] in ([], ["xformOp:scale"]):
        translate, orient = ops[0], ops[1]
    else:
        scale = Gf.Transform(xf.GetLocalTransformation()).GetScale()
        xf.ClearXformOpOrder()
        for op in ops:
            prim.RemoveProperty(op.GetAttr().GetName())
        translate = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        orient = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        if not Gf.IsClose(scale, Gf.Vec3d(1.0, 1.0, 1.0), 1e-9):
            xf.AddScaleOp().Set(Gf.Vec3f(scale))
    if translate.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        translate.Set(Gf.Vec3d(*pose[:3]))
    else:
        translate.Set(Gf.Vec3f(*pose[:3]))
    if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        x, y, z, w = pose[3:]
        orient.Set(Gf.Quatd(w, x, y, z))
    else:
        orient.Set(gf_quat(pose[3:]))


def world_matrix(prim: Usd.Prim, cache: UsdGeom.XformCache | None = None) -> np.ndarray:
    cache = cache or UsdGeom.XformCache(Usd.TimeCode.Default())
    return gf_to_matrix(cache.GetLocalToWorldTransform(prim))


def decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid+scale 4x4 -> (pose (7,), scale (3,)). Assumes no shear."""
    lin = m[:3, :3]
    scale = np.linalg.norm(lin, axis=0)
    rot = lin / np.where(scale == 0, 1.0, scale)
    if np.linalg.det(rot) < 0:  # mirrored: fold the reflection into the scale
        scale[0] = -scale[0]
        rot[:, 0] = -rot[:, 0]
    pose_m = np.eye(4)
    pose_m[:3, :3] = rot
    pose_m[:3, 3] = m[:3, 3]
    return geometry.matrix_to_pose(pose_m), scale


# ============================================================================ reading


def open_stage(path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage {path}")
    return stage


def object_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    root = stage.GetPrimAtPath(S.OBJECTS)
    if not root:
        return []
    return [p for p in root.GetChildren() if p.IsA(UsdGeom.Xformable)]


def read_object_mesh(prim: Usd.Prim, obj_world: np.ndarray, cache: UsdGeom.XformCache) -> Mesh:
    """Merge every Mesh under ``prim`` into one mesh in the object's (unscaled) body frame."""
    inv = np.linalg.inv(obj_world)
    parts = []
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(p).ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        counts = m.GetFaceVertexCountsAttr().Get()
        indices = m.GetFaceVertexIndicesAttr().Get()
        if pts.size == 0 or counts is None:
            continue
        rel = inv @ world_matrix(p, cache)
        pts = pts @ rel[:3, :3].T + rel[:3, 3]
        parts.append(Mesh.from_polygons(pts, counts, indices))
    if not parts:
        raise ValueError(f"{prim.GetPath()}: no mesh geometry under the object")
    return parts[0] if len(parts) == 1 else Mesh.merge(parts)


def _read_prior(prim: Usd.Prim) -> Prior | None:
    positions = get_attr(prim, S.PRIOR_POSITIONS)
    if positions is None:
        return None
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    quats = get_attr(prim, S.PRIOR_ORIENTATIONS)
    if quats is None:
        q = np.tile([0.0, 0.0, 0.0, 1.0], (len(positions), 1))
    else:
        q = np.array([[qq.GetImaginary()[0], qq.GetImaginary()[1], qq.GetImaginary()[2], qq.GetReal()] for qq in quats])
    weights = get_attr(prim, S.PRIOR_WEIGHTS)
    bounds = get_attr(prim, S.PRIOR_BOUNDS, (-0.5, 0.5, -0.5, 0.5))
    return Prior(
        means=np.concatenate([positions, q], axis=1),
        pos_std=float(get_attr(prim, S.PRIOR_POS_STD, 0.1)),
        rot_std=float(get_attr(prim, S.PRIOR_ROT_STD, 0.1)),
        bounds=tuple(float(b) for b in bounds),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        rule=str(get_attr(prim, S.PRIOR_RULE, "manual")),
    )


_KNOWN_OBJECT_ATTRS = {
    S.ROLE, S.ROLE_SOURCE, S.TAGS, S.ASSET, S.COLOR, S.SEGMENTATION_ID, S.VIS_FRACTION, S.VIS_VISIBLE,
    S.VIS_SILHOUETTE, S.PRIOR_POSITIONS, S.PRIOR_ORIENTATIONS, S.PRIOR_WEIGHTS, S.PRIOR_POS_STD,
    S.PRIOR_ROT_STD, S.PRIOR_BOUNDS, S.PRIOR_RULE, S.SETTLE_DISPLACEMENT, S.SETTLE_ROTATION,
    S.SETTLE_RESIDUAL,
}


def _python_value(v):
    if isinstance(v, (Vt.TokenArray, Vt.StringArray)):
        return list(v)
    if hasattr(v, "__len__") and not isinstance(v, str):
        try:
            return [float(x) for x in v]
        except TypeError:
            return str(v)
    return v


def read_object(prim: Usd.Prim, cache: UsdGeom.XformCache) -> SceneObject:
    m = world_matrix(prim, cache)
    pose, scale = decompose(m)
    unscaled = geometry.pose_to_matrix(pose)
    mesh = read_object_mesh(prim, unscaled, cache)
    dynamic = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    if dynamic and not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() in (None, True):
        dynamic = False
    mass = None
    if prim.HasAPI(UsdPhysics.MassAPI):
        mv = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        mass = float(mv) if mv and mv > 0 else None
    asset = get_attr(prim, S.ASSET, "")
    color = get_attr(prim, S.COLOR)
    vis = None
    if get_attr(prim, S.VIS_FRACTION) is not None:
        vis = Visibility(
            fraction=float(get_attr(prim, S.VIS_FRACTION)),
            visible_pixels=int(get_attr(prim, S.VIS_VISIBLE, 0)),
            silhouette_pixels=int(get_attr(prim, S.VIS_SILHOUETTE, 0)),
        )
    extra = {}
    for attr in prim.GetAttributes():
        n = attr.GetName()
        if n.startswith("sp:") and n not in _KNOWN_OBJECT_ATTRS and attr.HasAuthoredValue():
            extra[n] = _python_value(attr.Get())
    obj = SceneObject(
        name=prim.GetName(),
        pose=pose,
        mesh=mesh,
        dynamic=dynamic,
        role=get_attr(prim, S.ROLE),
        role_source=str(get_attr(prim, S.ROLE_SOURCE, "measured")),
        tags=[str(t) for t in (get_attr(prim, S.TAGS) or [])],
        asset=str(asset),
        mass=mass,
        segmentation_id=get_attr(prim, S.SEGMENTATION_ID),
        visibility=vis,
        prior=_read_prior(prim),
        settle_displacement=get_attr(prim, S.SETTLE_DISPLACEMENT),
        settle_rotation_deg=get_attr(prim, S.SETTLE_ROTATION),
        settle_residual=get_attr(prim, S.SETTLE_RESIDUAL),
        extra=extra,
    )
    obj.color = tuple(float(c) for c in color) if color is not None else looks.color_for(obj.asset)
    obj.material = looks.look_for(obj.asset)
    if np.any(np.abs(np.abs(scale) - 1.0) > 1e-6):
        obj.extra["sp:readScale"] = [float(s) for s in scale]
    return obj


def read_camera(stage: Usd.Stage) -> CameraSpec | None:
    prim = stage.GetPrimAtPath(S.CAMERA)
    if not prim or not prim.IsA(UsdGeom.Camera):
        return None
    cam = UsdGeom.Camera(prim)
    m = world_matrix(prim)
    eye = m[:3, 3]
    rot = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    forward, up_axis = -rot[:, 2], rot[:, 1]
    width = int(get_attr(prim, S.CAMERA_WIDTH, 640))
    height = int(get_attr(prim, S.CAMERA_HEIGHT, 480))
    focal = cam.GetFocalLengthAttr().Get()
    vap = cam.GetVerticalApertureAttr().Get()
    fov = float(np.degrees(2.0 * np.arctan(vap / (2.0 * focal))))
    stored_eye = get_attr(prim, S.CAMERA_EYE)
    stored_target = get_attr(prim, S.CAMERA_TARGET)
    stored_up = get_attr(prim, S.CAMERA_UP)
    target, up = eye + forward, np.array([0.0, 0.0, 1.0])
    if stored_eye is not None and stored_target is not None:
        s_eye, s_target = np.array(stored_eye, dtype=float), np.array(stored_target, dtype=float)
        s_fwd = (s_target - s_eye) / np.linalg.norm(s_target - s_eye)
        if np.linalg.norm(s_eye - eye) < 1e-5 and np.dot(s_fwd, forward) > 1 - 1e-8:
            eye, target = s_eye, s_target  # exact stored look-at (the camera was not moved)
            up = np.array(stored_up if stored_up is not None else (0, 0, 1), dtype=float)
        else:
            target = eye + forward * np.linalg.norm(s_target - s_eye)
    if np.dot(up, up_axis) <= 0:
        up = up_axis
    fov_stored = get_attr(prim, S.CAMERA_FOV)
    if fov_stored is not None and abs(float(fov_stored) - fov) < 1e-4:
        fov = float(fov_stored)
    return CameraSpec(
        width=width,
        height=height,
        fov_degree=fov,
        eye=tuple(float(v) for v in eye),
        target=tuple(float(v) for v in target),
        up=tuple(float(v) for v in up),
        max_depth=float(get_attr(prim, S.CAMERA_MAX_DEPTH, 4.0)),
        name=str(get_attr(prim, S.CAMERA_NAME, "camera")),
    )


_SCENE_KEYS = {
    S.SCENE_NAME: "name", S.SCENE_EXPERIMENT: "experiment", S.SCENE_CONDITION: "condition",
    S.SCENE_CUE: "cue", S.SCENE_DESCRIPTION: "description", S.SCENE_SOURCE: "source",
    S.SCENE_STATUS: "status", S.SCENE_CREATED: "created", S.SCENE_SETTLED: "settled",
    S.SCENE_FORMAT: "format_version", S.SCENE_PHYSICS_CONFIG: "physics_config",
}


def read_scene_metadata(stage: Usd.Stage) -> dict[str, Any]:
    world = stage.GetPrimAtPath(S.WORLD)
    meta: dict[str, Any] = {}
    if not world:
        return meta
    for attr in world.GetAttributes():
        n = attr.GetName()
        if not n.startswith(S.SCENE_PREFIX) or not attr.HasAuthoredValue():
            continue
        key = _SCENE_KEYS.get(n, n[len(S.SCENE_PREFIX):])
        value = _python_value(attr.Get())
        if key == "physics_config" and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        meta[key] = value
    return meta


def read_scene(source) -> Scene:
    """Load a scene from a path or an open ``Usd.Stage``."""
    stage = source if isinstance(source, Usd.Stage) else open_stage(source)
    path = None if isinstance(source, Usd.Stage) else Path(source)
    if path is None and stage.GetRootLayer().realPath:
        path = Path(stage.GetRootLayer().realPath)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    objects = {}
    for prim in object_prims(stage):
        obj = read_object(prim, cache)
        objects[obj.name] = obj
    if not objects:
        raise ValueError(f"{path or stage}: no objects under {S.OBJECTS} (is this a scene-physics scene?)")
    meta = read_scene_metadata(stage)
    camera = read_camera(stage)
    if camera is None:
        from scene_physics.camera import DEFAULT_CAMERA

        camera = DEFAULT_CAMERA
        meta.setdefault("warnings", []).append("no /World/Camera; using the default camera")
    name = meta.pop("name", None) or (path.parent.name if path and path.stem == "scene" else (path.stem if path else "scene"))
    return Scene(name=name, objects=objects, camera=camera, metadata=meta, path=path)


# ============================================================================ writing


def author_physics_scene(stage: Usd.Stage, physics: PhysicsConfig) -> Usd.Prim:
    scene = UsdPhysics.Scene.Define(stage, S.PHYSICS_SCENE)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(float(physics.gravity))
    prim = scene.GetPrim()
    set_physx_attr(prim, "physxScene:solverType", physics.solver)
    gpu = physics.device.startswith("cuda")
    set_physx_attr(prim, "physxScene:enableGPUDynamics", gpu)
    set_physx_attr(prim, "physxScene:broadphaseType", "GPU" if gpu else "MBP")
    set_physx_attr(prim, "physxScene:timeStepsPerSecond", int(round(1.0 / physics.dt)))
    set_physx_attr(prim, "physxScene:enableEnhancedDeterminism", bool(physics.enhanced_determinism))
    set_physx_attr(prim, "physxScene:enableStabilization", bool(physics.stabilization))
    set_physx_attr(prim, "physxScene:enableCCD", bool(physics.ccd))
    set_physx_attr(prim, "physxScene:disableSleeping", bool(physics.disable_sleeping))
    return prim


def author_physics_material(stage: Usd.Stage, physics: PhysicsConfig, path: str = "/World/PhysicsMaterial") -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(physics.static_friction))
    api.CreateDynamicFrictionAttr(float(physics.dynamic_friction))
    api.CreateRestitutionAttr(float(physics.restitution))
    api.GetStaticFrictionAttr().Set(float(physics.static_friction))
    api.GetDynamicFrictionAttr().Set(float(physics.dynamic_friction))
    api.GetRestitutionAttr().Set(float(physics.restitution))
    set_physx_attr(mat.GetPrim(), "physxMaterial:frictionCombineMode", "average")
    set_physx_attr(mat.GetPrim(), "physxMaterial:restitutionCombineMode", "average")
    return mat


def author_ground(stage: Usd.Stage, size: float = 40.0) -> Usd.Prim:
    """Invisible static slab whose top face is z = 0 (renders use the look layer's floor)."""
    xf = UsdGeom.Xform.Define(stage, S.GROUND)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    box = UsdGeom.Cube.Define(stage, f"{S.GROUND}/collider")
    box.CreateSizeAttr(1.0)
    UsdGeom.Xformable(box).ClearXformOpOrder()
    UsdGeom.Xformable(box).AddScaleOp().Set(Gf.Vec3f(size, size, 0.1))
    box.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    return xf.GetPrim()


def mesh_prims(obj_prim: Usd.Prim) -> list[Usd.Prim]:
    return [p for p in Usd.PrimRange(obj_prim) if p.IsA(UsdGeom.Mesh)]


def author_object_physics(prim: Usd.Prim, dynamic: bool, physics: PhysicsConfig, mass: float | None = None,
                          material_path: str | None = "/World/PhysicsMaterial") -> None:
    """(Re)author the physics schemas of one object prim and its meshes from ``physics``."""
    meshes = mesh_prims(prim)
    material = UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path)) if material_path else None
    for mp in meshes:
        UsdPhysics.CollisionAPI.Apply(mp)
        mc = UsdPhysics.MeshCollisionAPI.Apply(mp)
        approx = physics.dynamic_approximation if dynamic else physics.static_approximation
        mc.CreateApproximationAttr().Set(approx)
        if dynamic and approx == "convexDecomposition":
            set_physx_attr(mp, "physxConvexDecompositionCollision:maxConvexHulls", int(physics.max_convex_hulls))
            set_physx_attr(mp, "physxConvexDecompositionCollision:hullVertexLimit", int(physics.hull_vertex_limit))
        elif dynamic and approx == "convexHull":
            set_physx_attr(mp, "physxConvexHullCollision:hullVertexLimit", int(physics.hull_vertex_limit))
        elif dynamic and approx == "sdf":
            set_physx_attr(mp, "physxSDFMeshCollision:sdfResolution", int(physics.sdf_resolution))
        if physics.contact_offset is not None:
            set_physx_attr(mp, "physxCollision:contactOffset", float(physics.contact_offset))
        if physics.rest_offset is not None:
            set_physx_attr(mp, "physxCollision:restOffset", float(physics.rest_offset))
        if material is not None and material.GetPrim():
            UsdShade.MaterialBindingAPI.Apply(mp).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
    if not dynamic:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        return
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    mapi = UsdPhysics.MassAPI.Apply(prim)
    if mass is not None and mass > 0:
        mapi.CreateMassAttr().Set(float(mass))
        if mapi.GetDensityAttr().HasAuthoredValue():
            mapi.GetDensityAttr().Clear()
    else:
        mapi.CreateDensityAttr().Set(float(physics.density))
        if mapi.GetMassAttr().HasAuthoredValue():
            mapi.GetMassAttr().Clear()
    set_physx_attr(prim, "physxRigidBody:maxDepenetrationVelocity", float(physics.max_depenetration_velocity))
    set_physx_attr(prim, "physxRigidBody:solverPositionIterationCount", int(physics.position_iterations))
    set_physx_attr(prim, "physxRigidBody:solverVelocityIterationCount", int(physics.velocity_iterations))
    set_physx_attr(prim, "physxRigidBody:linearDamping", float(physics.linear_damping))
    set_physx_attr(prim, "physxRigidBody:angularDamping", float(physics.angular_damping))
    set_physx_attr(prim, "physxRigidBody:enableCCD", bool(physics.ccd))
    if physics.disable_sleeping:
        set_physx_attr(prim, "physxRigidBody:sleepThreshold", 0.0)


def apply_physics(stage: Usd.Stage, physics: PhysicsConfig) -> None:
    """Make an open scene stage use ``physics`` (scene settings, material, every object's colliders)."""
    author_physics_scene(stage, physics)
    author_physics_material(stage, physics)
    if physics.ground_plane:
        author_ground(stage)
    for prim in object_prims(stage):
        dynamic = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        mass = None
        if prim.HasAPI(UsdPhysics.MassAPI):
            mv = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            mass = float(mv) if mv and mv > 0 else None
        author_object_physics(prim, dynamic, physics, mass)


def author_preview_material(stage: Usd.Stage, obj_path: str, look: dict, color) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, f"{obj_path}/{S.LOOKS_NAME}/Material")
    shader = UsdShade.Shader.Define(stage, f"{obj_path}/{S.LOOKS_NAME}/Material/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(look.get("roughness", 0.45)))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(look.get("metallic", 0.0)))
    if look.get("opacity", 1.0) < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(look["opacity"]))
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def author_mesh(stage: Usd.Stage, path: str, mesh: Mesh, color=None) -> UsdGeom.Mesh:
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mesh.points, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.triangles), 3, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.triangles.reshape(-1), dtype=np.int32)))
    lo, hi = mesh.aabb
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    if color is not None:
        m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in color])]))
    return m


def author_object_metadata(prim: Usd.Prim, obj: SceneObject) -> None:
    set_attr(prim, S.ROLE, obj.role, "Token")
    set_attr(prim, S.ROLE_SOURCE, obj.role_source if obj.role else None, "Token")
    set_attr(prim, S.TAGS, list(obj.tags) if obj.tags else None, "TokenArray")
    set_attr(prim, S.ASSET, obj.asset)
    set_attr(prim, S.COLOR, Gf.Vec3f(*[float(c) for c in obj.color]), "Color3f")
    set_attr(prim, S.SEGMENTATION_ID, None if obj.segmentation_id is None else int(obj.segmentation_id), "Int")
    if obj.visibility is not None:
        set_attr(prim, S.VIS_FRACTION, float(obj.visibility.fraction), "Double")
        set_attr(prim, S.VIS_VISIBLE, int(obj.visibility.visible_pixels), "Int")
        set_attr(prim, S.VIS_SILHOUETTE, int(obj.visibility.silhouette_pixels), "Int")
    if obj.prior is not None:
        pr = obj.prior
        set_attr(prim, S.PRIOR_POSITIONS, Vt.Vec3dArray([Gf.Vec3d(*m[:3]) for m in pr.means]), "Double3Array")
        set_attr(prim, S.PRIOR_ORIENTATIONS,
                 Vt.QuatdArray([Gf.Quatd(float(m[6]), float(m[3]), float(m[4]), float(m[5])) for m in pr.means]),
                 "QuatdArray")
        set_attr(prim, S.PRIOR_WEIGHTS, Vt.DoubleArray([float(w) for w in pr.weights]), "DoubleArray")
        set_attr(prim, S.PRIOR_POS_STD, float(pr.pos_std), "Double")
        set_attr(prim, S.PRIOR_ROT_STD, float(pr.rot_std), "Double")
        set_attr(prim, S.PRIOR_BOUNDS, Gf.Vec4d(*pr.bounds), "Double4")
        set_attr(prim, S.PRIOR_RULE, pr.rule)
    else:
        for n in (S.PRIOR_POSITIONS, S.PRIOR_ORIENTATIONS, S.PRIOR_WEIGHTS, S.PRIOR_POS_STD, S.PRIOR_ROT_STD,
                  S.PRIOR_BOUNDS, S.PRIOR_RULE):
            set_attr(prim, n, None)
    set_attr(prim, S.SETTLE_DISPLACEMENT, None if obj.settle_displacement is None else float(obj.settle_displacement), "Double")
    set_attr(prim, S.SETTLE_ROTATION, None if obj.settle_rotation_deg is None else float(obj.settle_rotation_deg), "Double")
    set_attr(prim, S.SETTLE_RESIDUAL, None if obj.settle_residual is None else float(obj.settle_residual), "Double")
    for name, value in obj.extra.items():
        if name == "sp:readScale":
            continue
        if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
            set_attr(prim, name, Vt.DoubleArray([float(v) for v in value]), "DoubleArray")
        elif isinstance(value, (dict, list)):
            set_attr(prim, name, json.dumps(value))
        else:
            set_attr(prim, name, value)


def author_camera(stage: Usd.Stage, camera: CameraSpec) -> Usd.Prim:
    cam = UsdGeom.Camera.Define(stage, S.CAMERA)
    attrs = camera.usd_attributes()
    cam.CreateFocalLengthAttr().Set(attrs["focalLength"])
    cam.CreateVerticalApertureAttr().Set(attrs["verticalAperture"])
    cam.CreateHorizontalApertureAttr().Set(attrs["horizontalAperture"])
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(*attrs["clippingRange"]))
    cam.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(matrix_to_gf(camera.camera_to_world))
    prim = cam.GetPrim()
    set_attr(prim, S.CAMERA_NAME, camera.name)
    set_attr(prim, S.CAMERA_WIDTH, int(camera.width), "Int")
    set_attr(prim, S.CAMERA_HEIGHT, int(camera.height), "Int")
    set_attr(prim, S.CAMERA_FOV, float(camera.fov_degree), "Double")
    set_attr(prim, S.CAMERA_EYE, Gf.Vec3d(*camera.eye), "Double3")
    set_attr(prim, S.CAMERA_TARGET, Gf.Vec3d(*camera.target), "Double3")
    set_attr(prim, S.CAMERA_UP, Gf.Vec3d(*camera.up), "Double3")
    set_attr(prim, S.CAMERA_MAX_DEPTH, float(camera.max_depth), "Double")
    return prim


def author_scene_metadata(stage: Usd.Stage, name: str, metadata: dict[str, Any]) -> None:
    world = stage.GetPrimAtPath(S.WORLD)
    reverse = {v: k for k, v in _SCENE_KEYS.items()}
    set_attr(world, S.SCENE_FORMAT, int(S.FORMAT_VERSION), "Int")
    set_attr(world, S.SCENE_NAME, name)
    for key, value in metadata.items():
        if key in ("format_version", "name", "warnings"):
            continue
        attr = reverse.get(key, S.SCENE_PREFIX + key)
        if key == "physics_config" and not isinstance(value, str):
            value = json.dumps(value)
        if value is None:
            set_attr(world, attr, None)
        elif isinstance(value, (dict, list)) and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            set_attr(world, attr, json.dumps(value))
        else:
            set_attr(world, attr, value)


def new_stage(path: str | Path | None = None) -> Usd.Stage:
    if path is None:
        stage = Usd.Stage.CreateInMemory()
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, S.WORLD)
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, S.OBJECTS)
    return stage


def author_object(stage: Usd.Stage, obj: SceneObject, physics: PhysicsConfig) -> Usd.Prim:
    name = safe_prim_name(obj.name)
    path = f"{S.OBJECTS}/{name}"
    xf = UsdGeom.Xform.Define(stage, path)
    prim = xf.GetPrim()
    set_pose(prim, obj.pose)
    mesh = author_mesh(stage, f"{path}/{S.MESH_NAME}", obj.mesh, obj.color)
    look = obj.material or looks.look_for(obj.asset)
    mat = author_preview_material(stage, path, look, obj.color)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    author_object_physics(prim, obj.dynamic, physics, obj.mass)
    author_object_metadata(prim, obj)
    return prim


def write_scene(scene: Scene, path: str | Path, physics: PhysicsConfig | None = None) -> Path:
    """Write ``scene`` as a canonical, self-contained scene file (overwrites ``path``)."""
    physics = physics or PhysicsConfig()
    path = Path(path)
    stage = new_stage(path)
    author_scene_metadata(stage, scene.name, scene.metadata)
    author_physics_scene(stage, physics)
    author_physics_material(stage, physics)
    if physics.ground_plane:
        author_ground(stage)
    for obj in scene:
        author_object(stage, obj, physics)
    author_camera(stage, scene.camera)
    stage.GetRootLayer().Save()
    scene.path = path
    return path


def update_scene_file(stage: Usd.Stage, scene: Scene, *, poses: bool = True) -> None:
    """Write the model's poses and ``sp:*`` metadata onto an existing scene stage (in place)."""
    prims = {p.GetName(): p for p in object_prims(stage)}
    for obj in scene:
        prim = prims.get(safe_prim_name(obj.name)) or prims.get(obj.name)
        if prim is None:
            raise KeyError(f"object {obj.name} not found under {S.OBJECTS}")
        if poses:
            set_pose(prim, obj.pose)
        author_object_metadata(prim, obj)
    author_scene_metadata(stage, scene.name, scene.metadata)
    author_camera(stage, scene.camera)


def physics_config_dict(physics: PhysicsConfig) -> dict:
    return asdict(physics)
