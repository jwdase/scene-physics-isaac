"""Visibility -> roles -> target, and validation (ray caster on CPU)."""

import numpy as np
import pytest
import warp as wp

from scene_physics.authoring import analysis
from scene_physics.camera import CameraSpec
from scene_physics.config import AuthoringConfig
from scene_physics.scene import Mesh, Scene, SceneObject, priors

wp.config.quiet = True
CAM = CameraSpec(width=96, height=72, fov_degree=40.0, eye=(0.0, -2.0, 1.0), target=(0.0, 0.0, 1.0), max_depth=10.0)


def box(h):
    pts = np.array([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


def scene():
    floor = SceneObject("table", [0, 0, 0.5, 0, 0, 0, 1], box(0.3), dynamic=False)
    front = SceneObject("occluder", [0, -0.6, 1.0, 0, 0, 0, 1], box(0.12))
    back = SceneObject("block", [0, 0.3, 1.0, 0, 0, 0, 1], box(0.05))  # fully behind the occluder
    side = SceneObject("mug", [0.5, 0.0, 1.0, 0, 0, 0, 1], box(0.05))
    return Scene("t", {o.name: o for o in (floor, front, back, side)}, CAM)


def test_roles_and_target_from_visibility():
    s = scene()
    rep = analysis.measure_visibility(s, device="cpu")
    dec = analysis.assign_roles(s, rep, AuthoringConfig())
    assert s["block"].role == "hidden" and s["block"].visibility.fraction == 0.0
    assert s["occluder"].role == "observed" and s["mug"].role == "observed"
    assert s["table"].role == "static"
    assert dec.target == "block" and "target" in s["block"].tags


def test_authored_role_kept_with_warning():
    s = scene()
    s["mug"].role, s["mug"].role_source = "hidden", "authored"
    dec = analysis.assign_roles(s, analysis.measure_visibility(s, device="cpu"), AuthoringConfig())
    assert s["mug"].role == "hidden"
    assert any("mug" in w for w in dec.warnings)


def test_validate_requires_priors_and_ids():
    s = scene()
    analysis.assign_roles(s, analysis.measure_visibility(s, device="cpu"), AuthoringConfig())
    assert any("prior" in p for p in analysis.validate(s))
    priors.assign_priors(s, AuthoringConfig())
    s.assign_segmentation_ids()
    assert analysis.validate(s) == []


def test_sp_roles_fixes_swapped_legacy_roles(tmp_path):
    import pytest

    from scene_physics import cli
    from scene_physics.scene import load_scene, save_scene

    s = scene()
    s["mug"].role, s["mug"].role_source, s["mug"].tags = "hidden", "legacy", ["target"]  # visible, marked hidden
    s["block"].role, s["block"].role_source = "observed", "legacy"  # fully hidden, marked observed
    priors.assign_priors(s, AuthoringConfig())
    s.assign_segmentation_ids()
    path = tmp_path / "t.usd"
    save_scene(s, path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["roles", str(path), "--measure", "--target", "block"])
    assert exc.value.code == 0
    fixed = load_scene(path)
    assert fixed["block"].role == "hidden" and fixed["mug"].role == "observed"
    assert fixed.target.name == "block" and "target" not in fixed["mug"].tags
    assert len(fixed["block"].prior.means) == 2  # regenerated: one candidate behind each observed object


def test_import_brings_a_finished_scene_back_for_editing(tmp_path):
    import pytest

    from scene_physics import cli
    from scene_physics.scene import load_scene, save_scene

    s = scene()
    analysis.assign_roles(s, analysis.measure_visibility(s, device="cpu"), AuthoringConfig())
    priors.assign_priors(s, AuthoringConfig())
    s.assign_segmentation_ids()
    s.metadata["status"] = "finalized"
    done = tmp_path / "data" / "exp" / "t" / "scene.usd"
    save_scene(s, done)
    out = tmp_path / "authoring" / "t.usd"
    with pytest.raises(SystemExit) as exc:
        cli.main(["import", str(done.parent), "--out", str(out)])
    assert exc.value.code == 0
    back = load_scene(out)
    assert back.metadata["status"] == "authoring" and back.name == "t"
    assert back.target.name == "block" and back["block"].prior is not None
    assert np.allclose(back["mug"].pose, s["mug"].pose)


def test_overlaps_finds_interpenetrating_pairs():
    s = scene()
    assert analysis.overlaps(s, device="cpu") == []
    s["mug"].pose = np.array([0.0, 0.3, 1.02, 0, 0, 0, 1])  # 10 cm cubes, 2 cm apart: 80% overlap
    found = analysis.overlaps(s, device="cpu")
    assert [(o.a, o.b) for o in found] in ([("mug", "block")], [("block", "mug")])
    assert found[0].fraction == pytest.approx(0.8, abs=0.05)
    assert found[0].volume == pytest.approx(0.8e-3, rel=0.1)
