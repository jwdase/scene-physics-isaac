"""Composing scenes from specs (placement by ray casting) and the occlusion-only posterior (no simulator)."""

import numpy as np
import pytest
import warp as wp

from scene_physics.authoring import compose
from scene_physics.camera import CameraSpec
from scene_physics.config import Config
from scene_physics.scene import Mesh, Scene, SceneObject, library

wp.config.quiet = True
CAM = CameraSpec(width=160, height=120, fov_degree=40.0, eye=(0.0, -2.0, 1.2), target=(0.0, 0.0, 0.8), max_depth=10.0)


def box(hx, hy, hz, resting=True):
    """Box mesh; ``resting``: bottom at z = 0, footprint centred (the library's resting frame)."""
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    if resting:
        pts[:, 2] += hz
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


ASSETS = {
    "dining_room_table": box(0.5, 0.5, 0.35),  # top at z = 0.7
    "cube": box(0.04, 0.04, 0.04),
    "slab": box(0.05, 0.005, 0.05),
}


@pytest.fixture
def fake_library(monkeypatch):
    def read_asset(asset):
        return ASSETS[asset], {"asset": asset, "color": (0.5, 0.5, 0.5), "dynamic": asset != "dining_room_table"}

    monkeypatch.setattr(library, "read_asset", read_asset)


def config():
    cfg = Config()
    cfg.camera = CAM
    cfg.experiment = "test"
    return cfg


def test_rotation_axis_convention():
    # a positive angle about d x z lifts the side of the object that faces horizontal direction d
    for d in ([1.0, 0.0], [0.0, 1.0], [0.6, 0.8]):
        axis = [d[1], -d[0], 0.0]
        r = compose.rotation_of(compose.ObjectSpec("o", "cube", rotate=[[axis, 10.0]]))
        assert r.apply([*d, 0.0])[2] > 0.1
    # named axes and yaw (applied last)
    r = compose.rotation_of(compose.ObjectSpec("o", "cube", rotate=[["x", 90.0]], yaw=90.0))
    assert r.apply([0.0, 1.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert r.apply([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_drop_stack_and_slide_until_contact(fake_library):
    spec = compose.SceneSpec(name="t", target="a", gap=0.0005, objects=[
        compose.ObjectSpec("a", "cube", at=(0.0, 0.0)),
        compose.ObjectSpec("b", "cube", at=(-0.01, 0.0), on="a"),  # lands on a (overhanging to the left)
        compose.ObjectSpec("c", "slab", at=(0.25, 0.0), slide=[{"dir": [-1, 0], "until": "a"}]),
        compose.ObjectSpec("d", "slab", at=(-0.2, 0.0), rotate=[["y", 20.0]]),  # tilted: lowest point on the table
    ])
    scene = compose.compose(spec, config(), log=lambda m: None)
    lo = {n: scene[n].world_aabb()[0] for n in "abcd"}
    hi = {n: scene[n].world_aabb()[1] for n in "abcd"}
    assert lo["a"][2] == pytest.approx(0.7005, abs=1e-4)
    assert lo["b"][2] == pytest.approx(hi["a"][2] + 0.0005, abs=2e-4)
    assert lo["c"][0] == pytest.approx(hi["a"][0] + 0.0005, abs=1e-3)  # slid until it touches a
    assert lo["d"][2] == pytest.approx(0.7005, abs=1e-4)
    assert scene.target.name == "a" and "target" in scene["a"].tags
    # an object that would land on something other than its `on` is an error, not a silent stack
    spec.objects[1].on = "table"
    with pytest.raises(ValueError, match="lands on a"):
        compose.compose(spec, config(), log=lambda m: None)


def test_spec_file_round_trip(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text('name = "s"\ntarget = "a"\ntouching = [["a", "b"]]\n\n[[object]]\nname = "a"\nasset = "cube"\n'
                    'at = [0.1, 0.2]\n\n[[object]]\nname = "b"\nasset = "cube"\nrotate = [["x", 90], [[1, 0, 0], 5]]\n')
    spec = compose.SceneSpec.load(path)
    assert [o.name for o in spec.objects] == ["a", "b"] and spec.touching == [["a", "b"]]
    path.write_text('name = "s"\ntarget = "z"\n\n[[object]]\nname = "a"\nasset = "cube"\n')
    with pytest.raises(ValueError):
        compose.SceneSpec.load(path)


def test_occlusion_posterior_finds_the_decoy():
    """A target tightly hidden behind a small wall; a big wall elsewhere hides a much larger area."""
    table = SceneObject("table", [0, 0, 0.0, 0, 0, 0, 1], box(0.5, 0.5, 0.35), dynamic=False)
    small = SceneObject("small", [0.15, -0.1, 0.7, 0, 0, 0, 1], box(0.035, 0.01, 0.06))
    big = SceneObject("big", [-0.2, -0.1, 0.7, 0, 0, 0, 1], box(0.15, 0.01, 0.15))
    target = SceneObject("target", [0.15, -0.06, 0.7, 0, 0, 0, 1], box(0.025, 0.025, 0.04))
    scene = Scene("decoy", {o.name: o for o in (table, small, big, target)}, CAM)
    post = compose.occlusion_posterior(scene, "target", spacing=0.02, device="cpu")
    assert post.truth_group == "small"
    assert post.decoy == "big" and not post.aligned
    assert post.groups["big"] > 1.5 * post.groups["small"]
    assert post.near < 0.25
    assert len(post.xy) and np.isclose(post.weight.sum(), 1.0)


def test_margin_pixels_say_where_the_target_peeks_out():
    table = SceneObject("table", [0, 0, 0.0, 0, 0, 0, 1], box(0.5, 0.5, 0.35), dynamic=False)
    wall = SceneObject("wall", [0.0, -0.1, 0.7, 0, 0, 0, 1], box(0.05, 0.01, 0.08))
    hidden = SceneObject("target", [0.0, -0.05, 0.7, 0, 0, 0, 1], box(0.02, 0.02, 0.04))
    scene = Scene("m", {o.name: o for o in (table, wall, hidden)}, CAM)
    ok = compose.margin_pixels(scene, "target", CAM.scaled(2.0), margin_px=2)
    assert ok.visible == 0 and ok.margin == 0 and ok.crop is not None
    scene["target"].pose = np.array([0.04, -0.05, 0.7, 0, 0, 0, 1.0])  # slid right: its right edge shows
    peek = compose.margin_pixels(scene, "target", CAM.scaled(2.0), margin_px=2)
    assert peek.margin > 0 and peek.where.startswith("right")


def test_preview_checks_a_spec_without_physics(fake_library, tmp_path, monkeypatch):
    coarse = compose.occlusion_posterior
    monkeypatch.setattr(compose, "occlusion_posterior", lambda s, t, **kw: coarse(s, t, spacing=0.05, **kw))
    spec = tmp_path / "s.toml"
    spec.write_text('name = "s"\ntarget = "a"\n\n[[object]]\nname = "slab"\nasset = "slab"\nat = [0.0, -0.1]\n\n'
                    '[[object]]\nname = "a"\nasset = "cube"\nat = [0.0, -0.05]\n')
    res = compose.preview(spec, config(), out_dir=tmp_path / "out", log=lambda m: None)
    assert res["name"] == "s" and "occlusion_only" in res and not res["overlaps"]
    assert (tmp_path / "out" / "preview.png").exists() and (tmp_path / "out" / "preview.txt").exists()
