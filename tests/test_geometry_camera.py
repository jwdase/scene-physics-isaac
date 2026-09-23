import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scene_physics import geometry
from scene_physics.camera import DEFAULT_CAMERA, CameraSpec


def random_pose(rng):
    return np.concatenate([rng.normal(size=3), Rotation.random(random_state=rng.integers(1 << 30)).as_quat()])


def test_quat_order_roundtrip():
    q = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(geometry.quat_wxyz_to_xyzw(geometry.quat_xyzw_to_wxyz(q)), q)
    assert np.allclose(geometry.quat_xyzw_to_wxyz(q), [0.9, 0.1, 0.2, 0.3])


def test_pose_matrix_roundtrip_and_compose():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = random_pose(rng), random_pose(rng)
        assert np.allclose(geometry.matrix_to_pose(geometry.pose_to_matrix(a)), geometry.canonical_pose(a), atol=1e-9)
        ab = geometry.compose(a, b)
        assert np.allclose(geometry.pose_to_matrix(ab), geometry.pose_to_matrix(a) @ geometry.pose_to_matrix(b), atol=1e-9)
        ident = geometry.compose(a, geometry.inverse(a))
        assert np.allclose(ident[:3], 0, atol=1e-9) and geometry.rotation_angle(ident[3:], [0, 0, 0, 1]) < 1e-6


def test_rotation_angle_sign_invariant():
    q = Rotation.from_rotvec([0, 0, 0.3]).as_quat()
    assert geometry.rotation_angle(q, -q) == pytest.approx(0.0, abs=1e-7)
    assert geometry.rotation_angle(q, [0, 0, 0, 1]) == pytest.approx(0.3, abs=1e-7)


def test_camera_rays_and_projection_agree():
    cam = DEFAULT_CAMERA
    rays = cam.ray_directions()
    assert rays.shape == (cam.height, cam.width, 3)
    # a point along the ray through pixel (i, j) projects back to that pixel's centre
    for (i, j) in [(0, 0), (cam.width - 1, cam.height - 1), (100, 400), (320, 240)]:
        p = np.asarray(cam.eye) + 1.7 * rays[j, i]
        uv, depth = cam.project(p[None])
        assert np.allclose(uv[0], [i + 0.5, j + 0.5], atol=1e-6)
        assert depth[0] > 0


def test_camera_centre_ray_points_at_target():
    cam = DEFAULT_CAMERA
    fwd = np.asarray(cam.target) - np.asarray(cam.eye)
    fwd /= np.linalg.norm(fwd)
    uv, _ = cam.project(np.asarray(cam.target)[None])
    assert np.allclose(uv[0], [cam.width / 2, cam.height / 2], atol=1e-6)
    # vertical FOV: the top-centre edge ray is fov/2 from the optical axis
    top = np.array([cam.width / 2, 0.0])
    x = (top[0] - cam.width / 2) / cam.focal_px
    y = -(top[1] - cam.height / 2) / cam.focal_px
    assert np.degrees(np.arctan2(y, 1.0)) == pytest.approx(cam.fov_degree / 2)


def test_usd_camera_attributes_reproduce_fov():
    cam = CameraSpec(width=1280, height=720, fov_degree=50.0)
    a = cam.usd_attributes()
    fov = np.degrees(2 * np.arctan(a["verticalAperture"] / (2 * a["focalLength"])))
    assert fov == pytest.approx(50.0)
    assert a["horizontalAperture"] / a["verticalAperture"] == pytest.approx(1280 / 720)


def test_camera_dict_roundtrip_accepts_legacy_keys():
    legacy = {"id": "p01_f03", "position": {"index": 1}, "fov_degree": 34, "width": 640, "height": 480,
              "eye": [0, -1.66, 1.15], "target": [0, 0, 0.8], "up": [0, 0, 1]}
    cam = CameraSpec.from_dict(legacy)
    assert cam.name == "p01_f03" and cam.extra["position"] == {"index": 1}
    again = CameraSpec.from_dict(cam.to_dict())
    assert again == cam
