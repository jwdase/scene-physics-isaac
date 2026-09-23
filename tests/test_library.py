"""Library extract -> sp new -> sp add, without Isaac Sim."""

import numpy as np
import pytest

from scene_physics.camera import DEFAULT_CAMERA
from scene_physics.scene import Mesh, Scene, SceneObject, load_scene


def box(hx, hy, hz):
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENE_PHYSICS_ROOT", str(tmp_path))
    return tmp_path


def test_extract_new_add(root):
    from scene_physics.scene import library

    table = SceneObject("dining_room_table_018", [0.3, 0.1, 0.36, 0, 0, 0.38268343, 0.92387953],
                        box(0.5, 0.4, 0.36), dynamic=False)
    # a tilted, yawed block: the asset should keep it standing as it stood in the scene
    block = SceneObject("square_wood_block_013", [0.2, 0.2, 0.735, 0, 0, 0.38268343, 0.92387953], box(0.04, 0.04, 0.015))
    src = Scene("src", {o.name: o for o in (table, block)}, DEFAULT_CAMERA)
    library.extract_from_scenes([src], log=lambda m: None)
    assert library.list_assets() == ["dining_room_table", "square_wood_block"]
    mesh, info = library.read_asset("square_wood_block")
    assert mesh.aabb[0][2] == pytest.approx(0.0, abs=1e-6) and info["dynamic"]
    assert np.allclose((mesh.aabb[0][:2] + mesh.aabb[1][:2]) / 2, 0.0, atol=1e-6)

    path = root / "authoring" / "exp" / "s1.usd"
    library.new_scene(path, "s1", DEFAULT_CAMERA, experiment="exp")
    name = library.add_object(path, "square_wood_block", at=(0.05, -0.1), yaw_deg=30, tags=["target"])
    scene = load_scene(path)
    t = scene.support_top()
    assert t == pytest.approx(0.72, abs=1e-6)  # table asset re-centred: bottom on the floor, top at 0.72
    b = scene[name]
    lo, hi = b.world_aabb()
    assert lo[2] == pytest.approx(t + 0.001, abs=1e-5)  # resting (1 mm drop) on the table
    assert np.allclose(b.pose[:2], [0.05, -0.1]) and b.tags == ["target"]
    with pytest.raises(ValueError):
        library.add_object(path, "square_wood_block", at=(3.0, 0.0))  # off the table
