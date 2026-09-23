"""Proposal kernels and step-size schedules (port of Scene-Physics ``sampling/proposals.py``).

Geometries:

* ``xy``   (``XYProposer``, the old default): only X/Y change; the orientation and height stay those of
           the prior candidate each particle came from.
* ``xyzq`` (``XYZQProposer``): position plus an axis-angle rotation perturbation.

Schedules: ``exp_decay`` multiplies both standard deviations by ``0.5 ** (num_halvings / iterations)``
after every proposal (the old ``ExpDecayProposal``); ``none`` keeps them fixed (``NoDecayProposal``).

The resampling step is split out (:func:`resample_indices`) so the sampler can resample whole particles
(every sampled object in a world) with the same indices; only the object being updated gets noise.
With one hidden object this is exactly the old behaviour.
"""

from __future__ import annotations

import numpy as np

from scene_physics import geometry
from scene_physics.scene.model import Prior


def resample_indices(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row 0 keeps the best world (elite); rows 1.. are drawn proportionally to ``weights``."""
    n = len(weights)
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = w.sum()
    p = w / total if total > 0 and np.isfinite(total) else np.full(n, 1.0 / n)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(np.argmax(weights))
    if n > 1:
        idx[1:] = rng.choice(n, size=n - 1, p=p, replace=True)
    return idx


class Proposer:
    def __init__(self, prior: Prior, num_worlds: int, iterations: int, rng: np.random.Generator, *,
                 geometry_: str = "xy", schedule: str = "exp_decay", num_halvings: float = 2.0):
        if geometry_ not in ("xy", "xyzq"):
            raise ValueError(f"proposer geometry must be 'xy' or 'xyzq', not {geometry_!r}")
        if schedule not in ("exp_decay", "none"):
            raise ValueError(f"schedule must be 'exp_decay' or 'none', not {schedule!r}")
        self.prior = prior
        self.num_worlds = num_worlds
        self.iterations = max(int(iterations), 1)
        self.rng = rng
        self.geometry = geometry_
        self.schedule = schedule
        self.pos_std = float(prior.pos_std)
        self.rot_std = 0.0 if geometry_ == "xy" else float(prior.rot_std)
        self.multiplier = 0.5 ** (num_halvings / self.iterations) if schedule == "exp_decay" else 1.0
        self.pos_std_history: list[float] = []
        self.rot_std_history: list[float] = []

    # ------------------------------------------------------------------ helpers
    def _clip(self, poses: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.prior.bounds
        poses[:, 0] = np.clip(poses[:, 0], x_min, x_max)
        poses[:, 1] = np.clip(poses[:, 1], y_min, y_max)
        return poses

    def _advance(self) -> None:
        self.pos_std_history.append(self.pos_std)
        self.rot_std_history.append(self.rot_std)
        self.pos_std *= self.multiplier
        self.rot_std *= self.multiplier

    def _noise(self, poses: np.ndarray, rows: slice) -> np.ndarray:
        n = poses[rows].shape[0]
        if n == 0:
            return poses
        if self.geometry == "xy":
            poses[rows, :2] += self.rng.normal(0.0, self.pos_std, size=(n, 2))
        else:
            poses[rows, :3] += self.rng.normal(0.0, self.pos_std, size=(n, 3))
            poses[rows, 3:] = geometry.perturb_rotations(poses[rows, 3:], self.rot_std, self.rng)
        return poses

    # ------------------------------------------------------------------ API
    def initial(self) -> np.ndarray:
        """``(num_worlds, 7)`` first proposals: prior candidates (by weight) plus noise."""
        idx = self.rng.choice(len(self.prior.means), size=self.num_worlds, p=self.prior.weights / self.prior.weights.sum())
        poses = self.prior.means[idx].copy()
        # the old XYProposer perturbed every row; XYZQProposer kept row 0 at the candidate
        self._noise(poses, slice(0, None) if self.geometry == "xy" else slice(1, None))
        self._advance()
        return self._clip(poses)

    def propose(self, positions: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Resampled ``positions[idx]`` with noise on every row except the elite (row 0)."""
        poses = np.asarray(positions, dtype=np.float64)[idx].copy()
        self._noise(poses, slice(1, None))
        self._advance()
        return self._clip(poses)
