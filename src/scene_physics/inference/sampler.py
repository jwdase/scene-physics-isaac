"""Gibbs-style importance sampling over the poses of the sampled objects (port of Scene-Physics
``sampling/importance.py``).

State: ``poses`` ``(N, B, 7)``, one hypothesis of the whole scene per world. Objects whose role is not
sampled (``sampler.sample_roles``, default only ``hidden``) stay at their observed pose in every world.

    initialise: each sampled object gets prior draws; score all worlds
    repeat:     pick a sampled object (random, as before, or a sweep)
                resample whole particles by the current weights (world 0 keeps the best)
                perturb only the picked object (Gaussian, shrinking schedule)
                score all worlds (visual + physics rollout [+ penetration]) -> new weights

Differences from the old sampler (see docs/MIGRATION.md): particles are resampled jointly (identical
with one hidden object); the best hypothesis is tracked with the absolute energy score, since rank
composites are only comparable within one batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from scene_physics.config import LikelihoodConfig, SamplerConfig
from scene_physics.inference.proposals import Proposer, resample_indices
from scene_physics.likelihood import composite
from scene_physics.likelihood.composite import Channels
from scene_physics.scene.model import Scene


@dataclass
class Trace:
    """Everything the run did, as arrays (saved to ``trace.npz``)."""

    sampled: list[str]
    body_names: list[str]
    chosen: list[int] = field(default_factory=list)  # index into `sampled` (-1 = initialisation)
    poses: list[np.ndarray] = field(default_factory=list)  # (N, S, 7) sampled-object poses per iteration
    channels: list[np.ndarray] = field(default_factory=list)  # (N, 4)
    weights: list[np.ndarray] = field(default_factory=list)  # (N,)
    energy: list[np.ndarray] = field(default_factory=list)  # (N,)
    pos_std: list[float] = field(default_factory=list)
    rot_std: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "sampled": np.array(self.sampled),
            "body_names": np.array(self.body_names),
            "chosen": np.array(self.chosen),
            "poses": np.stack(self.poses),
            "channels": np.stack(self.channels),
            "channel_names": np.array(composite.CHANNELS),
            "weights": np.stack(self.weights),
            "energy": np.stack(self.energy),
            "pos_std": np.array(self.pos_std),
            "rot_std": np.array(self.rot_std),
            "seconds": np.array(self.seconds),
        }


class GibbsImportanceSampler:
    def __init__(self, scene: Scene, likelihood, sampler_cfg: SamplerConfig, likelihood_cfg: LikelihoodConfig,
                 *, rng: np.random.Generator | None = None, log=print):
        self.scene = scene
        self.lik = likelihood
        self.cfg = sampler_cfg
        self.lcfg = likelihood_cfg
        self.log = log
        self.rng = rng or np.random.default_rng(sampler_cfg.seed)
        self.body_names = list(likelihood.body_names)
        self.num_worlds = likelihood.num_worlds
        self.sampled = [n for n in self.body_names if scene[n].role in sampler_cfg.sample_roles]
        if not self.sampled:
            raise ValueError(f"no object in {scene.name} has a role in {sampler_cfg.sample_roles}")
        missing = [n for n in self.sampled if scene[n].prior is None]
        if missing:
            raise ValueError(f"objects without a prior: {missing} (run `sp priors` or re-finalize the scene)")
        self.index = {n: self.body_names.index(n) for n in self.sampled}
        children = self.rng.spawn(len(self.sampled))
        self.proposers = {
            n: Proposer(scene[n].prior, self.num_worlds, sampler_cfg.iterations, child, geometry_=sampler_cfg.proposer,
                        schedule=sampler_cfg.schedule, num_halvings=sampler_cfg.num_halvings)
            for n, child in zip(self.sampled, children)
        }
        truth = scene.truth_poses(self.body_names)
        self.poses = np.repeat(truth[None], self.num_worlds, axis=0)
        self.trace = Trace(sampled=self.sampled, body_names=self.body_names)
        self.best_energy = -np.inf
        self.best_poses: np.ndarray | None = None  # (B, 7)
        self.best_iteration = -1
        self.weights: np.ndarray | None = None

    # ------------------------------------------------------------------ internals
    def _score(self, chosen: int, pos_std: float, rot_std: float, t0: float) -> Channels:
        ch, _final = self.lik.evaluate(self.poses)
        self.weights = composite.weights(ch, self.lcfg)
        e = composite.energy(ch, self.lcfg)
        k = int(np.argmax(e))
        if e[k] > self.best_energy:
            self.best_energy, self.best_poses, self.best_iteration = float(e[k]), self.poses[k].copy(), len(self.trace.chosen)
        s_idx = [self.index[n] for n in self.sampled]
        self.trace.chosen.append(chosen)
        self.trace.poses.append(self.poses[:, s_idx].copy())
        self.trace.channels.append(ch.stack())
        self.trace.weights.append(self.weights.copy())
        self.trace.energy.append(e)
        self.trace.pos_std.append(pos_std)
        self.trace.rot_std.append(rot_std)
        self.trace.seconds.append(time.perf_counter() - t0)
        return ch

    def _report(self, it: int, name: str, ch: Channels) -> None:
        self.log(f"iter {it:4d}  {name:36s}  max visual {ch.visual.max():12.2f}  "
                 f"min disp {ch.displacement.min() * 1000:8.2f} mm  min rot {np.degrees(ch.rotation.min()):7.2f} deg  "
                 f"best energy {self.best_energy:10.3f}")

    # ------------------------------------------------------------------ API
    def initialize(self) -> Channels:
        t0 = time.perf_counter()
        for n in self.sampled:
            self.poses[:, self.index[n]] = self.proposers[n].initial()
        p = self.proposers[self.sampled[0]]
        ch = self._score(-1, p.pos_std_history[-1], p.rot_std_history[-1], t0)
        self._report(-1, "(initial proposals)", ch)
        return ch

    def step(self, it: int) -> Channels:
        t0 = time.perf_counter()
        if self.cfg.gibbs_order == "sweep":
            k = it % len(self.sampled)
        else:
            k = int(self.rng.integers(len(self.sampled)))
        name = self.sampled[k]
        idx = resample_indices(self.weights, self.rng)
        prev = self.poses
        self.poses = prev[idx].copy()  # whole particles follow the resampling
        b = self.index[name]
        prop = self.proposers[name]
        std_used = (prop.pos_std, prop.rot_std)
        self.poses[:, b] = prop.propose(prev[:, b], idx)
        ch = self._score(k, std_used[0], std_used[1], t0)
        self._report(it, name, ch)
        return ch

    def run(self, iterations: int | None = None) -> Trace:
        iterations = self.cfg.iterations if iterations is None else iterations
        self.initialize()
        for it in range(iterations):
            self.step(it)
        return self.trace

    def best(self) -> dict[str, np.ndarray]:
        """Best hypothesis seen (by energy), for the sampled objects."""
        return {n: self.best_poses[self.index[n]].copy() for n in self.sampled}

    def final_particles(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``{object: (poses (N, 7), normalised weights (N,))}`` after the last iteration."""
        w = np.clip(self.weights, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
        return {n: (self.poses[:, self.index[n]].copy(), w) for n in self.sampled}
