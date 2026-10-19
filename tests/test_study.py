"""Study pictures: the target card's pose, the table surface under it, the reveal's ghost (no simulator)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scene_physics.authoring import study
from scene_physics.camera import CameraSpec
from scene_physics.scene import Mesh, SceneObject

CAM = CameraSpec(width=640, height=480, fov_degree=34.0, eye=(0.0, -1.66, 1.15), target=(0.0, 0.0, 0.8))


def box(hx, hy, hz):
    """Box mesh centred on its body origin."""
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


def obj(name, mesh, position=(0.0, 0.0, 0.0), rotation=None, dynamic=True):
    quat = Rotation.identity().as_quat() if rotation is None else rotation.as_quat()
    return SceneObject(name=name, pose=np.array([*position, *quat]), mesh=mesh, dynamic=dynamic)


def table(top=0.7, tilt_deg=0.0):
    rot = Rotation.from_euler("x", tilt_deg, degrees=True)
    return obj("table", box(0.5, 0.5, 0.35), (0.02, -0.01, top - 0.35), rot, dynamic=False)


def test_card_keeps_the_resting_face_and_turns_the_broad_side_to_the_camera():
    points = box(0.05, 0.02, 0.01).points  # 10 x 4 x 2 cm, lying flat
    # in the scene it is tilted 12 degrees and turned 70 degrees: the card drops both
    scene_rot = (Rotation.from_euler("z", 70, degrees=True) * Rotation.from_euler("x", 12, degrees=True)).as_matrix()
    rot = study.card_rotation(scene_rot, points, CAM.rotation[:, 0], yaw_offset_deg=0.0)
    assert rot @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)  # still lies flat, upright
    world = points @ rot.T
    # turned to its widest silhouette across the view: for a box, its diagonal (10.77 cm)
    assert np.ptp(world[:, 0]) == pytest.approx(np.hypot(0.10, 0.04), abs=2e-4)
    # an object resting on its side keeps resting on that side
    on_side = Rotation.from_euler("x", 88, degrees=True).as_matrix()  # body y points (almost) up
    rot = study.card_rotation(on_side, points, CAM.rotation[:, 0])
    assert abs((rot @ np.array([0.0, 1.0, 0.0]))[2]) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_surface_height_follows_the_mesh_under_each_probe():
    t = table(top=0.7)
    assert study.surface_height(t.world_points, t.mesh.triangles, [[0.0, 0.0]]) == pytest.approx(0.7, abs=1e-6)
    assert study.surface_height(t.world_points, t.mesh.triangles, [[3.0, 3.0]]) == -np.inf
    tilted = table(top=0.7, tilt_deg=2.0)  # the side towards +y is higher
    low = study.surface_height(tilted.world_points, tilted.mesh.triangles, [[0.0, -0.05]])
    high = study.surface_height(tilted.world_points, tilted.mesh.triangles, [[0.0, -0.05], [0.0, 0.05]])
    assert high - low == pytest.approx(0.1 * np.tan(np.radians(2.0)), abs=1e-4)


def test_card_pose_centres_the_footprint_on_the_table_and_rests_on_it():
    t = table(top=0.7)
    target = obj("block", box(0.04, 0.03, 0.015), (0.3, 0.2, 0.715), Rotation.from_euler("z", 33, degrees=True))
    pose, stand = study.card_pose(target, t, CAM)
    world = target.mesh.transformed(pose)
    lo, hi = world.min(axis=0), world.max(axis=0)
    assert (lo[:2] + hi[:2]) / 2 == pytest.approx([0.02, -0.01], abs=1e-6)  # the table's centre, not the origin
    assert lo[2] == pytest.approx(0.7, abs=1e-6)
    assert stand == pytest.approx([0.02, -0.01, 0.7], abs=1e-6)


def test_coverage_and_the_ghost():
    seg = np.zeros((4, 6), dtype=np.uint8)
    seg[:, :3] = 5  # the jug
    seg[:, 3] = 7  # the loaf
    silhouette = np.zeros_like(seg, dtype=bool)
    silhouette[1:3, 1:5] = True  # 8 px: 4 behind the jug, 2 behind the loaf, 2 behind nothing
    cover = study.coverage(seg, {"background": 0, "jug": 5, "loaf": 7}, silhouette)
    assert cover == {"jug": 0.5, "loaf": 0.25}

    stimulus = np.full((4, 6, 3), 200, dtype=np.uint8)
    behind = np.zeros_like(stimulus)
    mixed = study.ghost_composite(stimulus, behind, seg == 5, ghost=0.25)
    assert (mixed[:, :3] == 50).all() and (mixed[:, 3:] == 200).all()


def test_card_check_measures_the_offset_from_the_table_centre():
    stand = np.array([0.0, 0.0, 0.7])
    (cx, cy), = CAM.project(stand[None])[0]
    seg = np.zeros((CAM.height, CAM.width), dtype=np.uint8)
    x0 = int(round(cx)) - 20
    seg[int(cy) - 30:int(cy), x0:x0 + 46] = 3  # 46 px wide, its centre 3 px right of the table centre
    check = study.card_check(seg, 3, CAM, stand)
    assert check["offset_px"] == pytest.approx(x0 + 23 - cx, abs=1e-6)
    assert check["in_frame"] and check["target_px"] == 30 * 46
    assert study.card_check(seg, 9, CAM, stand)["target_px"] == 0
