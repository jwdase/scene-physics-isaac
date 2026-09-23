"""Visibility -> roles -> target, and validation (ray caster on CPU)."""

import numpy as np
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
