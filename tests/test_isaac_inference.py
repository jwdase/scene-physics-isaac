"""End-to-end inference with Isaac Sim.

* ``-m isaac``: one world, two iterations (fine on a small GPU).
* ``-m multiworld``: several worlds; checks that a hypothesis scores the same whichever world slot it
  lands in and that repeated evaluation is stable. Run on the inference server, not on a small GPU.
"""

import numpy as np
import pytest

from scene_physics.camera import CameraSpec
from scene_physics.config import load_config
from scene_physics.scene import Mesh, Prior, Scene, SceneObject

CAM = CameraSpec(width=160, height=120, fov_degree=40.0, eye=(0.0, -1.6, 1.2), target=(0.0, 0.0, 0.75))


def box(hx, hy, hz):
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


def occluded_scene() -> Scene:
    table = SceneObject("table", [0, 0, 0.35, 0, 0, 0, 1], box(0.5, 0.5, 0.35), dynamic=False)
    wall = SceneObject("occluder", [0.0, -0.2, 0.8, 0, 0, 0, 1], box(0.15, 0.03, 0.1), role="observed")
    block = SceneObject("block", [0.05, 0.0, 0.72, 0, 0, 0, 1], box(0.04, 0.04, 0.02), role="hidden", tags=["target"])
    block.prior = Prior(means=np.array([[0.0, -0.2, 0.72, 0, 0, 0, 1]]), pos_std=0.1, bounds=(-0.45, 0.45, -0.45, 0.45))
    wall.prior = Prior(means=wall.pose[None])
    scene = Scene("occluded", {o.name: o for o in (table, wall, block)}, CAM)
    scene.assign_segmentation_ids()
    return scene


def _run(num_worlds, iterations, tmp_path):
    from scene_physics.inference.sampler import GibbsImportanceSampler
    from scene_physics.isaac.simulator import ParallelWorlds
    from scene_physics.likelihood.scene_likelihood import SceneLikelihood

    cfg = load_config()
    cfg.sampler.num_worlds, cfg.sampler.iterations = num_worlds, iterations
    cfg.likelihood.rollout_seconds = 0.5
    scene = occluded_scene()
    worlds = ParallelWorlds(scene, num_worlds, cfg.physics)
    lik = SceneLikelihood(scene, physics=worlds, cfg=cfg.likelihood, num_worlds=num_worlds, body_names=worlds.names)
    return scene, worlds, lik, cfg, GibbsImportanceSampler


@pytest.mark.isaac
def test_single_world_inference_runs(isaac, tmp_path):
    scene, worlds, lik, cfg, Sampler = _run(1, 2, tmp_path)
    truth_ch, _ = lik.evaluate(worlds.truth[None])
    assert truth_ch.visual[0] == pytest.approx(0.0, abs=1e-6)
    assert truth_ch.displacement[0] < 1e-3
    s = Sampler(scene, lik, cfg.sampler, cfg.likelihood, log=print)
    trace = s.run()
    assert len(trace.chosen) == 3 and np.isfinite(s.best_energy)
    worlds.stop()


@pytest.mark.multiworld
def test_world_slot_does_not_matter(isaac, tmp_path):
    n = 8
    scene, worlds, lik, cfg, _ = _run(n, 1, tmp_path)
    rng = np.random.default_rng(0)
    poses = np.repeat(worlds.truth[None], n, axis=0)
    b = worlds.body_index("block")
    poses[:, b, :2] += rng.normal(0, 0.05, size=(n, 2))  # plausible, non-overlapping-ish hypotheses
    a, _ = lik.evaluate(poses)
    perm = rng.permutation(n)
    c, _ = lik.evaluate(poses[perm])
    assert np.allclose(a.visual[perm], c.visual, atol=1e-6)
    assert np.allclose(a.displacement[perm], c.displacement, atol=1e-4)
    worlds.stop()
