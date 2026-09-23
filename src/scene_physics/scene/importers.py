"""Bring outside scenes into the canonical format.

* :func:`from_usd_export`: a hand-authored scene exported from Blender (or any USD with one Mesh per
  object, e.g. ``data-pipeline/stage/sceneNNN.usdc``). Replaces data-pipeline's ``gen_physics`` +
  ``gen_truth``.
* :func:`from_legacy_scene`: a finished scene folder from data-pipeline / Scene-Physics
  (``<scene>/data/<scene>_physics.usdc`` + ``_truth/_makeup/_priors.json`` + ``results/target.json``).
  Roles, annotation tags, priors and the recorded visibility are carried over.

Both return a :class:`~scene_physics.scene.model.Scene`; write it with
:func:`scene_physics.scene.usd_io.write_scene`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from scene_physics.camera import CameraSpec
from scene_physics.scene import looks
from scene_physics.scene import schema as S
from scene_physics.scene.model import (
    Mesh,
    Prior,
    Scene,
    SceneObject,
    Visibility,
    base_asset_name,
    safe_prim_name,
)
from scene_physics.scene.usd_io import decompose, world_matrix

STATIC_PREFIXES = ("dining_room_table",)


def _is_static_name(name: str, static_names: set[str]) -> bool:
    return name in static_names or name.startswith(STATIC_PREFIXES) or base_asset_name(name) in static_names


def _mesh_object(prim: Usd.Prim, cache: UsdGeom.XformCache) -> tuple[np.ndarray, Mesh]:
    """World pose (T+R) of a Mesh prim and its geometry in that frame, with any scale baked in."""
    m = world_matrix(prim, cache)
    pose, scale = decompose(m)
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) * scale
    geo = Mesh.from_polygons(pts, mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
    return pose, geo


def from_usd_export(
    path: str | Path,
    *,
    name: str | None = None,
    camera: CameraSpec | None = None,
    static_names: list[str] | tuple[str, ...] = (),
    tags: dict[str, list[str]] | None = None,
    metadata: dict | None = None,
) -> Scene:
    """Convert a USD export with one Mesh prim per object.

    ``static_names``: objects (full or base names) that are fixed colliders; anything starting with
    ``dining_room_table`` is static automatically. ``tags``: ``{object_name: [tag, ...]}``.
    """
    path = Path(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(path)
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"{path}: stage up axis is {UsdGeom.GetStageUpAxis(stage)}; export Z-up from Blender")
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(mpu - 1.0) > 1e-9:
        raise ValueError(f"{path}: metersPerUnit is {mpu}; scenes must be authored in metres")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    statics = set(static_names)
    tags = tags or {}
    # A physics USD (legacy `_physics.usdc`) marks dynamic bodies with RigidBodyAPI; trust that when
    # present. A raw export has no physics, so fall back to the static-name rule.
    has_physics = any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    objects: dict[str, SceneObject] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.inherited:
            continue
        obj_name = safe_prim_name(prim.GetName())
        if obj_name in objects:
            raise ValueError(f"{path}: two meshes are named {obj_name}; object names must be unique")
        pose, mesh = _mesh_object(prim, cache)
        asset = base_asset_name(obj_name)
        if has_physics:
            dynamic = prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetParent().HasAPI(UsdPhysics.RigidBodyAPI)
        else:
            dynamic = not _is_static_name(obj_name, statics)
        objects[obj_name] = SceneObject(
            name=obj_name,
            pose=pose,
            mesh=mesh,
            dynamic=dynamic,
            asset=asset,
            color=looks.color_for(asset),
            material=looks.look_for(asset),
            tags=list(tags.get(obj_name, [])),
        )
    if not objects:
        raise ValueError(f"{path}: no Mesh prims found")
    from scene_physics.camera import DEFAULT_CAMERA

    meta = {"source": str(path.resolve()), "status": "authoring"}
    meta.update(metadata or {})
    return Scene(name=name or path.stem, objects=objects, camera=camera or DEFAULT_CAMERA, metadata=meta)


# ---------------------------------------------------------------------------------------- legacy

_LEGACY_ROLE_KEYS = ("static", "observed", "hidden")
_LEGACY_TAG_KEYS = ("target", "occluder", "decoy", "probe", "reference")
_LEGACY_SCENE_KEYS = ("condition", "cue")


def _match_object(ref: str, names: list[str]) -> str | None:
    """Resolve a makeup reference (full name or base name like ``jug04``) to an object name."""
    if ref in names:
        return ref
    ref_safe = safe_prim_name(ref)
    if ref_safe in names:
        return ref_safe
    hits = [n for n in names if base_asset_name(n) == base_asset_name(ref) or n.startswith(ref + "_")]
    if "_ref" not in ref:
        hits = [n for n in hits if "_ref" not in n] or hits
    return hits[0] if len(hits) == 1 else (sorted(hits, key=len)[0] if hits else None)


def _find(data: Path, scene: str, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        p = data / f"{scene}{suffix}"
        if p.exists():
            return p
    return None


def from_legacy_scene(scene_dir: str | Path, *, camera: CameraSpec | None = None, name: str | None = None) -> Scene:
    """Read an old ``<scene>/data`` + ``<scene>/results`` folder."""
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    data = scene_dir / "data"
    usd = _find(data, scene_name, ("_physics.usdc",))
    if usd is None:
        raise FileNotFoundError(f"{scene_dir}: no data/{scene_name}_physics.usdc")
    stage = Usd.Stage.Open(str(usd))
    has_physics = any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    raw = _find(data, scene_name, (".usdc",))
    if not has_physics and raw is not None:
        usd = raw  # a raw Blender export was dropped in the _physics slot; use the export itself
    scene = from_usd_export(usd, name=name or scene_name, camera=camera)
    names = list(scene.objects)

    makeup_path = _find(data, scene_name, ("_makeup.json", f"_{scene_name}_makeup.json", "_scene_makeup.json"))
    priors_path = _find(data, scene_name, ("_modified_priors.json", "_priors.json", f"_{scene_name}_priors.json",
                                           "_scene_priors.json"))
    truth_path = _find(data, scene_name, ("_truth.json",))
    warnings: list[str] = []

    if makeup_path is not None:
        mk = json.loads(makeup_path.read_text())
        for role in _LEGACY_ROLE_KEYS:
            for ref in mk.get(role, []) or []:
                n = _match_object(ref, names)
                if n is None:
                    warnings.append(f"makeup {role} {ref!r} matches no object")
                    continue
                obj = scene.objects[n]
                if role == "static":
                    obj.dynamic, obj.role = False, "static"
                else:
                    obj.role, obj.role_source = role, "legacy"
        for key in _LEGACY_TAG_KEYS:
            ref = mk.get(key)
            if not ref:
                continue
            n = _match_object(ref, names)
            if n is None:
                warnings.append(f"makeup {key} {ref!r} matches no object")
                continue
            if key not in scene.objects[n].tags:
                scene.objects[n].tags.append(key)
        for key in _LEGACY_SCENE_KEYS:
            if mk.get(key) is not None:
                scene.metadata[key] = mk[key]
        extra = {k: v for k, v in mk.items() if k not in (*_LEGACY_ROLE_KEYS, *_LEGACY_TAG_KEYS, *_LEGACY_SCENE_KEYS)}
        if extra:
            scene.metadata["legacy_makeup_extra"] = extra

    if priors_path is not None:
        for ref, entry in json.loads(priors_path.read_text()).items():
            n = _match_object(ref, names)
            if n is None:
                warnings.append(f"prior for {ref!r} matches no object")
                continue
            scene.objects[n].prior = Prior.from_legacy_dict(entry)
        scene.metadata["legacy_priors_file"] = priors_path.name

    if truth_path is not None:
        truth = json.loads(truth_path.read_text())
        worst = 0.0
        for ref, pose in truth.items():
            n = _match_object(ref, names)
            if n is not None:
                worst = max(worst, float(np.linalg.norm(np.asarray(pose[:3]) - scene.objects[n].pose[:3])))
        if worst > 1e-3:
            warnings.append(f"truth.json differs from the USD poses by up to {worst * 1000:.1f} mm (USD kept)")

    target_json = scene_dir / "results" / "target.json"
    if target_json.exists():
        tj = json.loads(target_json.read_text())
        for rec in tj.get("objects", []):
            n = _match_object(rec["name"], names)
            if n is not None:
                scene.objects[n].visibility = Visibility(
                    fraction=float(rec["visible_fraction"]),
                    visible_pixels=int(rec["visible_px"]),
                    silhouette_pixels=int(rec["silhouette_px"]),
                )
        if tj.get("target"):
            n = _match_object(tj["target"], names)
            if n is not None and "target" not in scene.objects[n].tags and not scene.with_tag("target"):
                scene.objects[n].tags.append("target")

    scene.metadata["source"] = str(scene_dir.resolve())
    scene.metadata["legacy_format"] = True
    if warnings:
        scene.metadata["import_warnings"] = warnings
    return scene
