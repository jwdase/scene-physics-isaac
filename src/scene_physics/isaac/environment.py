"""The render-time "studio" look: lights, floor, backdrop wall and the wood table texture.

The look is authored into the stage's **session layer**, so it shows up in renders and in the editor but
is never saved into ``scene.usd``: scene files stay pure content (geometry, physics, truth, metadata)
and the whole dataset can be re-rendered with a different look without touching any scene.

It reproduces the Blender stimulus setup: interior HDRI for ambient light and reflections, a
key / fill / rim rig of soft area lights, a tiled floor at ``z = 0``, a warm-grey backdrop wall behind the
table, and a wood texture on the table (box-projected UVs; the meshes have none).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from scene_physics import paths
from scene_physics.scene import schema as S
from scene_physics.scene import usd_io

LOOK_ROOT = "/World/Look"


def _textures() -> dict[str, Path | None]:
    tex = paths.assets_dir() / "textures"
    hdri = paths.assets_dir() / "hdri"

    def first(*cands):
        return next((c for c in cands if c.exists()), None)

    return {
        "wood": first(tex / "Wood063" / "Wood063_4K-JPG_Color.jpg"),
        "wood_rough": first(tex / "Wood063" / "Wood063_4K-JPG_Roughness.jpg"),
        "tiles": first(tex / "Tiles108" / "Tiles108_4K-JPG_Color.jpg"),
        "tiles_rough": first(tex / "Tiles108" / "Tiles108_4K-JPG_Roughness.jpg"),
        "hdri": first(hdri / "lythwood_room_4k.hdr", hdri / "studio_small_08_2k.hdr"),
    }


def _textured_material(stage: Usd.Stage, path: str, color_tex: Path | None, rough_tex: Path | None,
                       fallback_color, roughness: float) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*fallback_color))
    if color_tex is not None:
        reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/diffuseTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(color_tex))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        if rough_tex is not None:
            rt = UsdShade.Shader.Define(stage, f"{path}/roughnessTexture")
            rt.CreateIdAttr("UsdUVTexture")
            rt.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(rough_tex))
            rt.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            rt.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            rt.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            rt.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            rt.CreateOutput("r", Sdf.ValueTypeNames.Float)
            shader.GetInput("roughness").ConnectToSource(rt.ConnectableAPI(), "r")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def box_projected_uvs(points: np.ndarray, face_indices: np.ndarray, repeat_m: float) -> np.ndarray:
    """Face-varying UVs by projecting each triangle onto the plane of its dominant normal axis."""
    tri = points[face_indices.reshape(-1, 3)]  # (T, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    axis = np.abs(n).argmax(axis=1)
    keep = np.array([[1, 2], [0, 2], [0, 1]])[axis]  # (T, 2): coordinates spanning the projection plane
    uv = np.take_along_axis(tri, keep[:, None, :].repeat(3, axis=1), axis=2)  # (T, 3, 2)
    return (uv.reshape(-1, 2) / repeat_m).astype(np.float32)


def _plane(stage: Usd.Stage, path: str, corners: np.ndarray, uv_repeat_m: float) -> UsdGeom.Mesh:
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in corners]))
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    extent_u = np.linalg.norm(corners[1] - corners[0]) / uv_repeat_m
    extent_v = np.linalg.norm(corners[3] - corners[0]) / uv_repeat_m
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(extent_u, 0), Gf.Vec2f(extent_u, extent_v), Gf.Vec2f(0, extent_v)]))
    return m


def _rect_light(stage, path, position, target, intensity, size, color=(1.0, 1.0, 1.0)):
    light = UsdLux.RectLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateWidthAttr(float(size))
    light.CreateHeightAttr(float(size))
    light.CreateColorAttr(Gf.Vec3f(*color))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(light)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    return light


def apply_studio(stage: Usd.Stage, *, table_paths: list[str] | None = None, intensity_scale: float = 1.0) -> None:
    """Author the studio look into the session layer of ``stage``."""
    tex = _textures()
    session = stage.GetSessionLayer()
    with Usd.EditContext(stage, session):
        if stage.GetPrimAtPath(LOOK_ROOT):
            stage.RemovePrim(LOOK_ROOT)
        UsdGeom.Xform.Define(stage, LOOK_ROOT)
        # ---- lights
        dome = UsdLux.DomeLight.Define(stage, f"{LOOK_ROOT}/Dome")
        dome.CreateIntensityAttr(700.0 * intensity_scale)
        if tex["hdri"] is not None:
            dome.CreateTextureFileAttr(str(tex["hdri"]))
            dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        centre = (0.0, 0.0, 0.8)
        _rect_light(stage, f"{LOOK_ROOT}/Key", (-1.3, -1.5, 2.1), centre, 9000 * intensity_scale, 2.0, (1.0, 0.98, 0.95))
        _rect_light(stage, f"{LOOK_ROOT}/Fill", (1.7, -1.1, 1.2), centre, 2500 * intensity_scale, 2.6, (0.96, 0.98, 1.0))
        _rect_light(stage, f"{LOOK_ROOT}/Rim", (0.6, 1.7, 1.9), centre, 5000 * intensity_scale, 1.6)
        # ---- floor (z = 0) and backdrop wall
        looks = f"{LOOK_ROOT}/Materials"
        UsdGeom.Scope.Define(stage, looks)
        floor_mat = _textured_material(stage, f"{looks}/Floor", tex["tiles"], tex["tiles_rough"], (0.55, 0.55, 0.55), 0.45)
        s = 30.0
        floor = _plane(stage, f"{LOOK_ROOT}/Floor", np.array([[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]]), 2.2)
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_mat)
        wall_mat = _textured_material(stage, f"{looks}/Wall", None, None, (0.62, 0.60, 0.57), 0.9)
        wy, ws = 2.2, 24.0
        wall = _plane(stage, f"{LOOK_ROOT}/Wall", np.array([[-ws / 2, wy, 0.0], [ws / 2, wy, 0.0], [ws / 2, wy, ws], [-ws / 2, wy, ws]]), 2.0)
        UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(wall_mat)
        # ---- wood table
        if table_paths:
            wood = _textured_material(stage, f"{looks}/Wood", tex["wood"], tex["wood_rough"], (0.52, 0.33, 0.19), 0.5)
            for obj_path in table_paths:
                obj = stage.GetPrimAtPath(obj_path)
                if not obj:
                    continue
                for mp in usd_io.mesh_prims(obj):
                    mesh = UsdGeom.Mesh(mp)
                    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
                    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
                    if not np.all(counts == 3):
                        continue
                    uv = box_projected_uvs(pts, idx, repeat_m=0.9)
                    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                                  UsdGeom.Tokens.faceVarying)
                    st.Set(Vt.Vec2fArray.FromNumpy(uv))
                    UsdShade.MaterialBindingAPI.Apply(mp).Bind(wood, UsdShade.Tokens.strongerThanDescendants)


def remove_look(stage: Usd.Stage) -> None:
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if stage.GetPrimAtPath(LOOK_ROOT):
            stage.RemovePrim(LOOK_ROOT)


def table_paths(stage: Usd.Stage) -> list[str]:
    """Static objects (no rigid body) under /World/Objects: they get the wood look."""
    from pxr import UsdPhysics

    return [str(p.GetPath()) for p in usd_io.object_prims(stage) if not p.HasAPI(UsdPhysics.RigidBodyAPI)]
