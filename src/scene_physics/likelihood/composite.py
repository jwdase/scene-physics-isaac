"""Combining the likelihood channels into resampling weights.

Channels (``(N,)`` each, one value per world):

* ``visual``        point-cloud log-likelihood minus the perfect-match score (<= 0, higher is better)
* ``displacement``  metres moved by all dynamic bodies during the rollout (lower is better)
* ``rotation``      radians rotated by all dynamic bodies during the rollout (lower is better)
* ``penetration``   overlap volume at the proposal, cubic metres (lower is better)

``combine = "rank"`` reproduces the old composite exactly: each channel is ranked across worlds (average
ranks, 0-based), divided by its sum, weighted by ``lambda_*`` and summed. ``combine = "energy"`` keeps
magnitudes: ``log w = lambda_v*visual/visual_scale - lambda_d*displacement/displacement_scale - ...``
and softmax weights. Ranks compare worlds only *within* one batch, so the "best hypothesis so far" is
tracked with the energy score in either mode (:func:`energy`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from scene_physics.config import LikelihoodConfig

CHANNELS = ("visual", "displacement", "rotation", "penetration")


@dataclass
class Channels:
    visual: np.ndarray
    displacement: np.ndarray
    rotation: np.ndarray
    penetration: np.ndarray

    def stack(self) -> np.ndarray:
        """``(N, 4)`` in :data:`CHANNELS` order."""
        return np.stack([self.visual, self.displacement, self.rotation, self.penetration], axis=1)


def _rank(a: np.ndarray) -> np.ndarray:
    return rankdata(a, method="average") - 1.0


def _norm(r: np.ndarray) -> np.ndarray:
    s = r.sum()
    return r / s if s > 0 else np.zeros_like(r)


def rank_composite(ch: Channels, cfg: LikelihoodConfig) -> np.ndarray:
    """The old ``ParallelPhysicsLikelihood.physics`` composite (higher is better)."""
    return (
        cfg.lambda_visual * _norm(_rank(ch.visual))
        + cfg.lambda_penetration * _norm(_rank(-ch.penetration))
        + cfg.lambda_displacement * _norm(_rank(-ch.displacement))
        + cfg.lambda_rotation * _norm(_rank(-ch.rotation))
    )


def energy(ch: Channels, cfg: LikelihoodConfig) -> np.ndarray:
    """Absolute log-score, comparable across iterations (higher is better)."""
    return (
        cfg.lambda_visual * ch.visual / cfg.visual_scale
        - cfg.lambda_displacement * ch.displacement / cfg.displacement_scale
        - cfg.lambda_rotation * ch.rotation / cfg.rotation_scale
        - cfg.lambda_penetration * ch.penetration / cfg.penetration_scale
    )


def weights(ch: Channels, cfg: LikelihoodConfig) -> np.ndarray:
    """Non-negative resampling weights (not necessarily normalised)."""
    if cfg.combine == "rank":
        return rank_composite(ch, cfg)
    if cfg.combine == "energy":
        e = energy(ch, cfg)
        e = e - e.max()
        return np.exp(e)
    raise ValueError(f"likelihood.combine must be 'rank' or 'energy', not {cfg.combine!r}")
