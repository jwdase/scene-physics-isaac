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
    assert np.allclose(mesh.extents, [0.08, 0.08, 0.03], atol=1e-6)  # the 45 deg yaw is taken out
    table_mesh, _ = library.read_asset("dining_room_table")
    assert np.allclose(table_mesh.extents, [1.0, 0.8, 0.72], atol=1e-6)  # long side along x

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
    warnings = []
    library.add_object(path, "square_wood_block", at=(0.06, -0.1), yaw_deg=0, log=warnings.append)
    assert len(warnings) == 1 and name in warnings[0]  # dropped into the first block
    library.add_object(path, "square_wood_block", at=(0.3, 0.1), yaw_deg=0, log=warnings.append)
    assert len(warnings) == 1  # clear of both
    tall = library.add_object(path, "square_wood_block", at=(-0.3, 0.1), yaw_deg=0, scale=[1, 1, 2], log=warnings.append)
    lo, hi = load_scene(path)[tall].world_aabb()
    assert hi[2] - lo[2] == pytest.approx(0.06, abs=1e-5) and lo[2] == pytest.approx(t + 0.001, abs=1e-5)


def test_size_variants_become_separate_assets(root):
    from scene_physics.scene import library

    table = SceneObject("dining_room_table", [0, 0, 0.36, 0, 0, 0, 1], box(0.5, 0.4, 0.36), dynamic=False)
    a = SceneObject("square_wood_block_001", [0.1, 0, 0.735, 0, 0, 0, 1], box(0.04, 0.04, 0.015))
    b = SceneObject("square_wood_block_002", [0.3, 0, 0.735, 0, 0, 0, 1], box(0.04, 0.04, 0.015))
    c = SceneObject("square_wood_block_003", [-0.1, 0, 0.735, 0, 0, 0, 1], box(0.05, 0.03, 0.015))  # rescaled
    c.settle_displacement, c.settle_rotation_deg = 0.0, 0.0
    s1 = Scene("s1", {o.name: o for o in (table, c)}, DEFAULT_CAMERA)
    s2 = Scene("s2", {o.name: o for o in (table, a, b)}, DEFAULT_CAMERA)
    library.extract_from_scenes([s1, s2], log=lambda m: None)
    assert library.list_assets() == ["dining_room_table", "square_wood_block", "square_wood_block_v2"]
    main, info = library.read_asset("square_wood_block")  # the most used size keeps the base name
    assert np.allclose(main.extents, [0.08, 0.08, 0.03]) and info["instances"] == 2
    v2, info2 = library.read_asset("square_wood_block_v2")
    assert np.allclose(v2.extents, [0.10, 0.06, 0.03]) and info2["instances"] == 1


def test_asset_rests_the_common_way(root):
    from scene_physics.scene import library

    table = SceneObject("dining_room_table", [0, 0, 0.36, 0, 0, 0, 1], box(0.5, 0.4, 0.36), dynamic=False)
    flat = [SceneObject(f"square_wood_block_00{i}", [0.1 * i, 0, 0.735, 0, 0, np.sin(0.3 * i), np.cos(0.3 * i)],
                        box(0.04, 0.04, 0.015)) for i in range(1, 3)]
    # one instance leans (tilted 30 deg about x) and never moved: still not how the block usually rests
    leaning = SceneObject("square_wood_block_009", [0.3, 0.2, 0.75, np.sin(np.radians(15)), 0, 0, np.cos(np.radians(15))],
                          box(0.04, 0.04, 0.015))
    leaning.settle_displacement, leaning.settle_rotation_deg = 0.0, 0.0
    for o in flat:
        o.settle_displacement, o.settle_rotation_deg = 0.001, 0.5
    s = Scene("s", {o.name: o for o in (table, *flat, leaning)}, DEFAULT_CAMERA)
    library.extract_from_scenes([s], log=lambda m: None)
    mesh, _ = library.read_asset("square_wood_block")
    assert np.allclose(mesh.extents, [0.08, 0.08, 0.03], atol=1e-6)  # flat, not leaning
