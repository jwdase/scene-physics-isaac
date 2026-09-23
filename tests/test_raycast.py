"""Warp ray caster on CPU (no Isaac Sim)."""

import numpy as np
import pytest
import warp as wp

from scene_physics.camera import CameraSpec
from scene_physics.render.raycast import Raycaster
from scene_physics.scene import Mesh, Scene, SceneObject

wp.config.quiet = True
CAM = CameraSpec(width=64, height=48, fov_degree=40.0, eye=(0.0, -2.0, 1.0), target=(0.0, 0.0, 1.0), max_depth=10.0)


def box(h):
    pts = np.array([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


@pytest.fixture(scope="module")
def rc():
    return Raycaster(CAM, [box(0.1), box(0.05)], num_worlds=2, device="cpu")


def poses(front_y=-0.5, world1_front_y=None):
    p = np.zeros((2, 2, 7))
    p[..., 6] = 1.0
    p[:, 0, :3] = [0, 0, 1]
    p[0, 1, :3] = [0, front_y, 1]
    p[1, 1, :3] = [0, front_y if world1_front_y is None else world1_front_y, 1]
    return p


def centre(img):
    return img[CAM.height // 2, CAM.width // 2]


def test_occlusion_and_depth(rc):
    out = rc.render(poses())
    ids = out.id_image()
    assert centre(ids[0]) == 1  # the small box is in front
    depth = out.depth()[0]
    assert centre(depth) == pytest.approx(2.0 - 0.5 - 0.05, abs=1e-3)
    assert ids[0][0, 0] == -1 and np.isnan(out.points()[0][0, 0]).all()


def test_worlds_are_independent(rc):
    out = rc.render(poses(world1_front_y=3.0))  # world 1: small box moved behind the camera's view
    ids = out.id_image()
    assert centre(ids[0]) == 1 and centre(ids[1]) == 0
    assert centre(out.depth()[1]) == pytest.approx(2.0 - 0.1, abs=1e-3)


def test_solo_is_amodal(rc):
    out = rc.render(poses(), solo=0)
    ids = out.id_image()[0]
    assert centre(ids) == 0
    assert set(np.unique(ids)) <= {-1, 0}


def test_points_project_back_to_their_pixels(rc):
    pts = rc.render(poses()).points()[0]
    js, is_ = np.nonzero(np.isfinite(pts[..., 0]))
    uv, _ = CAM.project(pts[js, is_])
    assert np.allclose(uv[:, 0], is_ + 0.5, atol=1e-3) and np.allclose(uv[:, 1], js + 0.5, atol=1e-3)


def test_accepts_warp_pose_buffer(rc):
    p = poses()
    flat = wp.array(p.reshape(-1, 7).astype(np.float32), dtype=wp.float32, device="cpu")
    a = rc.render(p).id_image()
    b = rc.render(flat).id_image()
    assert np.array_equal(a, b)


def test_scene_visibility_counts():
    from scene_physics.render.raycast import visibility

    back = SceneObject("back", [0, 0, 1, 0, 0, 0, 1], box(0.1))
    front = SceneObject("front", [0, -0.5, 1, 0, 0, 0, 1], box(0.05))
    scene = Scene("t", {"back": back, "front": front}, CAM)
    vis = visibility(scene, device="cpu")
    assert vis["front"]["fraction"] == pytest.approx(1.0)
    assert 0.0 < vis["back"]["fraction"] < 1.0
    assert vis["back"]["silhouette"] > vis["back"]["visible"]


def test_depth_png_roundtrip(tmp_path):
    from scene_physics.render import images

    d = np.array([[1.2346, np.inf], [np.nan, 0.5]])
    back = images.read_depth(images.write_depth(tmp_path / "d.png", d))
    assert back[0, 0] == pytest.approx(1.235) and back[1, 1] == pytest.approx(0.5)
    assert np.isnan(back[0, 1]) and np.isnan(back[1, 0])
