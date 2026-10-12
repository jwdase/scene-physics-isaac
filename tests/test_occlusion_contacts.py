"""Occlusion with a safety margin, orbit cameras, contact relations, speed-cap authoring (no simulator)."""

import numpy as np
import pytest
import warp as wp
from pxr import Usd

from scene_physics.authoring import contacts, occlusion
from scene_physics.camera import CameraSpec
from scene_physics.config import PhysicsConfig
from scene_physics.scene import Mesh, Scene, SceneObject, usd_io

wp.config.quiet = True
CAM = CameraSpec(width=160, height=120, fov_degree=40.0, eye=(0.0, -2.0, 1.0), target=(0.0, 0.0, 1.0), max_depth=10.0)


def box(hx, hy=None, hz=None):
    hy = hx if hy is None else hy
    hz = hx if hz is None else hz
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


def occlusion_scene(target_x=0.0, glass=False):
    table = SceneObject("table", [0, 0, 0.45, 0, 0, 0, 1], box(0.6, 0.6, 0.45), dynamic=False)
    wall = SceneObject("glass1" if glass else "wall", [0, -0.5, 1.1, 0, 0, 0, 1], box(0.15, 0.02, 0.2))
    target = SceneObject("block", [target_x, 0.3, 1.0, 0, 0, 0, 1], box(0.04))
    return Scene("occ", {o.name: o for o in (table, wall, target)}, CAM)


def test_hidden_with_margin_and_revealed_when_moved():
    hidden = occlusion.occlusion(occlusion_scene(0.0), "block", margin_px=2, scale=1.0)
    assert hidden.visible == 0 and hidden.margin == 0 and hidden.hidden and hidden.silhouette > 0
    peek = occlusion.occlusion(occlusion_scene(0.42), "block", margin_px=2, scale=1.0)
    assert peek.visible > 0 and not peek.hidden


def test_margin_catches_a_barely_hidden_target():
    meter = occlusion.OcclusionMeter(occlusion_scene(0.0), CAM)
    xs = np.linspace(0.0, 0.5, 101)
    res = [meter.measure("block", margin_px=3, poses={"block": np.array([x, 0.3, 1.0, 0, 0, 0, 1.0])}) for x in xs]
    first_visible = next(i for i, r in enumerate(res) if r.visible > 0)
    first_unsafe = next(i for i, r in enumerate(res) if r.margin > 0)
    assert first_unsafe < first_visible  # the margin flags the target before any pixel shows


def test_transparent_objects_do_not_occlude():
    assert occlusion.occlusion(occlusion_scene(0.0, glass=True), "block", scale=1.0).visible > 0


def test_orbit_camera_roundtrip_and_dolly_zoom():
    ref = CameraSpec(fov_degree=34.0, eye=(0.0, -1.7, 0.8), target=(0.0, 0.0, 0.8))
    cam = occlusion.orbit_camera(10.0, 15.0, 1.2, look_at=(0.0, 0.0, 0.8), reference=ref)
    az, el, d = occlusion.orbit_of(cam)
    assert (az, el, d) == pytest.approx((10.0, 15.0, 1.2))
    # dolly zoom: a point at the look-at plane keeps its image size
    assert np.tan(np.radians(cam.fov_degree / 2)) * 1.2 == pytest.approx(np.tan(np.radians(17.0)) * 1.7)


def stack_scene():
    table = SceneObject("table", [0, 0, 0.35, 0, 0, 0, 1], box(0.5, 0.5, 0.35), dynamic=False)
    a = SceneObject("block_a", [0.0, 0.0, 0.72, 0, 0, 0, 1], box(0.05, 0.05, 0.02))
    b = SceneObject("block_b", [0.0, 0.0, 0.76, 0, 0, 0, 1], box(0.03, 0.03, 0.02))
    c = SceneObject("block_c", [0.25, 0.1, 0.80, 0, 0, 0, 1], box(0.03, 0.03, 0.03))
    d = SceneObject("block_d", [0.081, 0.0, 0.72, 0, 0, 0, 1], box(0.03, 0.03, 0.02))  # beside a, touching it
    return Scene("stack", {o.name: o for o in (table, a, b, c, d)}, CAM)


def test_contact_relations():
    s = stack_scene()
    rel = contacts.relations(contacts.ContactAnalyzer(s, samples=3000).contacts())
    assert ("block_b", "on", "block_a") in rel
    assert ("block_a", "on", "table") in rel and ("block_d", "on", "table") in rel
    assert ("block_a", "touch", "block_d") in rel
    assert not any("block_c" in (r[0], r[2]) for r in rel)  # floating
    moved = {"block_b": np.array([0.3, -0.3, 0.76, 0, 0, 0, 1.0])}  # lifted off the stack, in the air
    after = contacts.relations(contacts.ContactAnalyzer(s, samples=3000).contacts(moved))
    lost, gained = contacts.compare(rel, after, involving={"block_a"})
    assert lost == {("block_b", "on", "block_a")} and not gained


def test_speed_caps_are_authored_only_when_set(tmp_path):
    s = stack_scene()
    plain = usd_io.write_scene(s, tmp_path / "plain.usd", PhysicsConfig())
    capped = usd_io.write_scene(s, tmp_path / "capped.usd", PhysicsConfig(max_linear_velocity=0.1, max_angular_velocity=45.0))
    stages = [Usd.Stage.Open(str(plain)), Usd.Stage.Open(str(capped))]
    p, c = (st.GetPrimAtPath("/World/Objects/block_a") for st in stages)
    assert not p.HasAttribute("physxRigidBody:maxLinearVelocity")
    assert c.GetAttribute("physxRigidBody:maxLinearVelocity").Get() == pytest.approx(0.1)
    assert c.GetAttribute("physxRigidBody:maxAngularVelocity").Get() == pytest.approx(45.0)


def test_hide_moves_the_target_group_behind_the_occluder():
    from scene_physics.authoring import hide

    table = SceneObject("table", [0, 0, 0.45, 0, 0, 0, 1], box(0.6, 0.6, 0.45), dynamic=False)
    wall = SceneObject("wall", [0, -0.3, 0.98, 0, 0, 0, 1], box(0.12, 0.02, 0.08))
    block = SceneObject("block", [0.13, 0.2, 0.92, 0, 0, 0, 1], box(0.03, 0.03, 0.02))  # peeks out at the side
    lid = SceneObject("lid", [0.13, 0.2, 0.95, 0, 0, 0, 1], box(0.02, 0.02, 0.01))  # rests on the block
    s = Scene("hide", {o.name: o for o in (table, wall, block, lid)}, CAM)
    assert not occlusion.occlusion(s, "block", margin_px=1, scale=1.0).hidden
    cfg = hide.HideConfig(margin_px=1, max_shift=0.06, shift_step=0.01, yaw_step_deg=0.0)
    res = hide.hide(s, "block", CAM, cfg, device="cpu")
    assert res.found, res.describe()
    assert set(res.moves) == {"block", "lid"}  # the lid moves with the block it rests on
    dx, dy, _ = res.moves["block"]
    assert dx < 0  # towards the middle of the wall
    for n, p in res.poses.items():
        s[n].pose = p
    assert occlusion.occlusion(s, "block", margin_px=1, scale=1.0).hidden
    assert ("lid", "on", "block") in contacts.relations(contacts.ContactAnalyzer(s, samples=2000).contacts())
