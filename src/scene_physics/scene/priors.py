"""Named rules that generate each object's pose prior.

Priors are written into ``scene.usd`` at finalize time (``sp:prior:*``) so a scene fully specifies the
inference problem, and they can be regenerated with a different rule (``sp priors``) or edited by hand.

Rules for **hidden** objects (the unknowns):

``observed_objects`` (default, the data-pipeline rule)
    One candidate behind every observed dynamic object: its XY, at the hidden object's resting height.
    "The hidden object is probably behind one of the things I can see."
``occluder``
    Like ``observed_objects`` but only objects tagged ``occluder`` (falls back to all observed objects).
    This is the Scene-Physics generator rule.
``truth``
    A single candidate at the true pose (debugging / oracle runs).

Observed objects always get a single candidate at their observed pose.

Orientation and height of hidden-object candidates (``prior_orientation`` / ``prior_z`` in the
authoring config): ``truth`` keeps the hidden object's own resting orientation and height, which fixes
the old data-pipeline behaviour of borrowing each *occluder's* orientation for the candidate.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scene_physics.config import AuthoringConfig
from scene_physics.scene.model import Prior, Scene, SceneObject

RULES = ("observed_objects", "occluder", "truth")


def resting_height(obj: SceneObject, quat_xyzw: np.ndarray, surface_z: float) -> float:
    """Body-origin z that puts the mesh's lowest point on ``surface_z`` for orientation ``quat``."""
    rotated = Rotation.from_quat(quat_xyzw).apply(obj.mesh.points.astype(float))
    return float(surface_z - rotated[:, 2].min())


def _candidate_objects(scene: Scene, hidden: SceneObject, rule: str) -> list[SceneObject]:
    observed = [o for o in scene.dynamic_objects if o.role == "observed" and o.name != hidden.name]
    if rule == "occluder":
        tagged = [o for o in observed if "occluder" in o.tags]
        return tagged or observed
    return observed


def hidden_prior(scene: Scene, obj: SceneObject, cfg: AuthoringConfig) -> Prior:
    bounds = scene.support_bounds()
    rule = cfg.prior_rule
    if rule not in RULES:
        raise ValueError(f"unknown prior rule {rule!r}; choose from {RULES}")
    if rule == "truth":
        return Prior(means=obj.pose[None], pos_std=cfg.prior_pos_std, rot_std=cfg.prior_rot_std,
                     bounds=bounds, rule="truth")
    quat = obj.pose[3:].copy() if cfg.prior_orientation == "truth" else np.array([0.0, 0.0, 0.0, 1.0])
    if cfg.prior_z == "truth":
        z = float(obj.pose[2])
    elif cfg.prior_z == "table_rest":
        z = resting_height(obj, quat, scene.support_top())
    else:
        raise ValueError(f"unknown prior_z {cfg.prior_z!r} (truth | table_rest)")
    cands = _candidate_objects(scene, obj, rule)
    if not cands:
        # nothing visible to hide behind: centre of the support surface
        x_min, x_max, y_min, y_max = bounds
        xy = [np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])]
    else:
        xy = [c.pose[:2] for c in cands]
    means = np.array([[p[0], p[1], z, *quat] for p in xy])
    return Prior(means=means, pos_std=cfg.prior_pos_std, rot_std=cfg.prior_rot_std, bounds=bounds,
                 rule=f"{rule}(z={cfg.prior_z},orientation={cfg.prior_orientation})")


def observed_prior(scene: Scene, obj: SceneObject, cfg: AuthoringConfig) -> Prior:
    return Prior(means=obj.pose[None], pos_std=cfg.prior_pos_std, rot_std=cfg.prior_rot_std,
                 bounds=scene.support_bounds(), rule="observed_pose")


def assign_priors(scene: Scene, cfg: AuthoringConfig, *, overwrite: bool = True) -> None:
    """Give every dynamic object a prior according to its role (static objects get none)."""
    for obj in scene:
        if not obj.dynamic:
            obj.prior = None
            continue
        if obj.prior is not None and not overwrite:
            continue
        if obj.role == "hidden":
            obj.prior = hidden_prior(scene, obj, cfg)
        else:
            obj.prior = observed_prior(scene, obj, cfg)
