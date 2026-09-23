"""Scene model <-> scene.usd round trips (plain OpenUSD, no Isaac Sim)."""

from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdPhysics

from scene_physics.camera import DEFAULT_CAMERA
from scene_physics.config import AuthoringConfig, PhysicsConfig
from scene_physics.scene import Mesh, Prior, Scene, SceneObject, Visibility, base_asset_name, load_scene
from scene_physics.scene import importers, priors, usd_io
from scene_physics.scene import schema as S

LEGACY = Path("/home/jdase/Desktop/Coding/data-pipeline")


def box_mesh(hx, hy, hz):
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


def toy_scene() -> Scene:
    table = SceneObject("dining_room_table_001", [0, 0, 0.35, 0, 0, 0, 1], box_mesh(0.5, 0.5, 0.35), dynamic=False)
    block = SceneObject("square_wood_block_013", [0.1, 0.2, 0.715, 0, 0, 0.38268343, 0.92387953],
                        box_mesh(0.04, 0.04, 0.015), role="hidden", tags=["target"])
    jug = SceneObject("jug04_006", [0.1, 0.05, 0.8, 0, 0, 0, 1], box_mesh(0.05, 0.05, 0.1), role="observed",
                      tags=["occluder"], visibility=Visibility(0.97, 970, 1000))
    scene = Scene("toy", {o.name: o for o in (table, block, jug)}, DEFAULT_CAMERA,
                  metadata={"experiment": "unit", "condition": "control", "cue": "on_top"})
    priors.assign_priors(scene, AuthoringConfig())
    scene.assign_segmentation_ids()
    return scene


def test_base_asset_name():
    assert base_asset_name("jug04_006") == "jug04"
    assert base_asset_name("f10_apple_iphone_4_003_001") == "f10_apple_iphone_4"
    assert base_asset_name("b04_wineglass_ref_003") == "b04_wineglass"
    assert base_asset_name("_9v_battery") == "9v_battery"
    assert base_asset_name("square_wood_block") == "square_wood_block"


def test_write_read_roundtrip(tmp_path):
    scene = toy_scene()
    path = usd_io.write_scene(scene, tmp_path / "scene.usd")
    back = load_scene(path)
    assert set(back.objects) == set(scene.objects)
    assert back.camera == scene.camera
    assert back.metadata["condition"] == "control" and back.metadata["cue"] == "on_top"
    for name, obj in scene.objects.items():
        b = back[name]
        assert np.allclose(b.pose, obj.pose, atol=1e-6), name
        assert b.dynamic == obj.dynamic and b.role == obj.role and b.tags == obj.tags
        assert np.allclose(b.mesh.points, obj.mesh.points) and np.array_equal(b.mesh.triangles, obj.mesh.triangles)
        assert b.segmentation_id == obj.segmentation_id
        if obj.prior is not None:
            assert np.allclose(b.prior.means, obj.prior.means, atol=1e-7)
            assert b.prior.bounds == pytest.approx(obj.prior.bounds)
            assert b.prior.pos_std == obj.prior.pos_std and b.prior.rule == obj.prior.rule
    assert back["jug04_006"].visibility.fraction == pytest.approx(0.97)
    assert back.target.name == "square_wood_block_013"


def test_physics_schemas_are_authored(tmp_path):
    path = usd_io.write_scene(toy_scene(), tmp_path / "scene.usd", PhysicsConfig())
    stage = Usd.Stage.Open(str(path))
    table = stage.GetPrimAtPath(f"{S.OBJECTS}/dining_room_table_001")
    block = stage.GetPrimAtPath(f"{S.OBJECTS}/square_wood_block_013")
    assert not table.HasAPI(UsdPhysics.RigidBodyAPI)
    assert block.HasAPI(UsdPhysics.RigidBodyAPI) and block.HasAPI(UsdPhysics.MassAPI)
    assert "PhysxRigidBodyAPI" in usd_io.applied_api_names(block)
    assert block.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get() == pytest.approx(1.0)
    mesh = stage.GetPrimAtPath(f"{S.OBJECTS}/square_wood_block_013/mesh")
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexDecomposition"
    tmesh = stage.GetPrimAtPath(f"{S.OBJECTS}/dining_room_table_001/mesh")
    assert UsdPhysics.MeshCollisionAPI(tmesh).GetApproximationAttr().Get() == "none"
    scene_prim = stage.GetPrimAtPath(S.PHYSICS_SCENE)
    assert scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get() == 120
    assert scene_prim.GetAttribute("physxScene:enableEnhancedDeterminism").Get() is True


def test_update_in_place_moves_pose(tmp_path):
    scene = toy_scene()
    path = usd_io.write_scene(scene, tmp_path / "scene.usd")
    stage = Usd.Stage.Open(str(path))
    scene["jug04_006"].pose = np.array([0.2, 0.1, 0.81, 0, 0, 0, 1.0])
    scene["jug04_006"].tags = ["occluder", "probe"]
    usd_io.update_scene_file(stage, scene)
    stage.GetRootLayer().Save()
    back = load_scene(path)
    assert np.allclose(back["jug04_006"].pose[:3], [0.2, 0.1, 0.81])
    assert back["jug04_006"].tags == ["occluder", "probe"]


def test_hidden_prior_rule_uses_observed_xy_and_own_pose():
    scene = toy_scene()
    p = scene["square_wood_block_013"].prior
    assert len(p.means) == 1  # one observed object -> one candidate
    assert np.allclose(p.means[0, :2], scene["jug04_006"].pose[:2])
    assert p.means[0, 2] == pytest.approx(scene["square_wood_block_013"].pose[2])
    assert np.allclose(p.means[0, 3:], scene["square_wood_block_013"].pose[3:])
    assert p.bounds == pytest.approx(scene.support_bounds())


def test_legacy_prior_dict_roundtrip():
    d = {"position": [[0.1, 0.2, 0.7, 0, 0, 0, 1], [0.3, 0.1, 0.7, 0, 0, 0, 1]], "pos_std": 0.2, "rot_std": 0.1,
         "x_min": -0.5, "x_max": 0.5, "y_min": -0.4, "y_max": 0.4}
    p = Prior.from_legacy_dict(d)
    assert p.means.shape == (2, 7) and p.weights.sum() == pytest.approx(1.0)
    assert Prior.from_legacy_dict(p.to_legacy_dict()).bounds == p.bounds


@pytest.mark.skipif(not (LEGACY / "stage/scene001.usdc").exists(), reason="legacy data not present")
def test_import_blender_export(tmp_path):
    scene = importers.from_usd_export(LEGACY / "stage/scene001.usdc", name="scene001")
    assert [o.name for o in scene.static_objects] == ["dining_room_table_018"]
    assert len(scene.dynamic_objects) == 6
    jug = scene["jug04_006"]
    assert np.allclose(jug.pose[:3], [-0.0918, -0.1275, 0.7352], atol=1e-3)
    assert scene.support_top() == pytest.approx(0.7351, abs=1e-3)
    path = usd_io.write_scene(scene, tmp_path / "scene.usd")
    back = load_scene(path)
    assert np.allclose(back["jug04_006"].pose, jug.pose, atol=1e-6)


@pytest.mark.skipif(not (LEGACY / "data/data-set/scene017").exists(), reason="legacy data not present")
def test_import_legacy_scene_keeps_annotations():
    scene = importers.from_legacy_scene(LEGACY / "data/data-set/scene017")
    assert scene["coffeemug"].role == "hidden" and "target" in scene["coffeemug"].tags
    assert "occluder" in scene["jug04"].tags and "decoy" in scene["jug01"].tags
    assert scene.metadata["condition"] == "control"
    assert len(scene["coffeemug"].prior.means) == 1  # a single 7-vector mean in this file
    assert scene["coffeemug"].visibility.fraction == 0.0
    assert not scene["dining_room_table"].dynamic


@pytest.mark.skipif(not (LEGACY / "data/data-set-2/scene004").exists(), reason="legacy data not present")
def test_import_legacy_mixture_prior():
    scene = importers.from_legacy_scene(LEGACY / "data/data-set-2/scene004")
    target = scene.target
    assert target.name == "square_wood_block_007_001"
    assert target.prior.means.shape == (4, 7)
    assert not scene["dining_room_table_007_001"].dynamic
