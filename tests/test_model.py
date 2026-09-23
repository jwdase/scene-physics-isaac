"""The model's pieces on CPU: likelihood kernel, composite, proposals, sampler, penetration."""

import numpy as np
import pytest
import warp as wp

from scene_physics.config import LikelihoodConfig, SamplerConfig
from scene_physics.inference.proposals import Proposer, resample_indices
from scene_physics.inference.sampler import GibbsImportanceSampler
from scene_physics.likelihood import composite
from scene_physics.likelihood.composite import Channels
from scene_physics.likelihood.physics_terms import PenetrationEstimator, total_displacement, total_rotation
from scene_physics.likelihood.pointcloud import PointCloudLikelihood, reference_score
from scene_physics.scene import Mesh, Prior, Scene, SceneObject
from scene_physics.camera import DEFAULT_CAMERA

wp.config.quiet = True


def box(h):
    pts = np.array([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


def test_pointcloud_kernel_matches_reference():
    rng = np.random.default_rng(0)
    h, w = 12, 16
    obs = rng.normal(size=(h, w, 3)) * 0.05
    obs[0, :3] = np.nan  # invalid observed depth
    rendered = np.stack([obs + rng.normal(size=obs.shape) * s for s in (0.0, 0.01, 0.1)])
    rendered[1, 5, 5] = np.nan
    lik = PointCloudLikelihood(obs, h, w, device="cpu")
    got = lik.score(rendered)
    want = [reference_score(obs, r) for r in rendered]
    assert np.allclose(got, want, rtol=1e-5, atol=1e-3)
    assert lik.baseline == pytest.approx(reference_score(obs, obs), rel=1e-5)
    assert got[0] == pytest.approx(lik.baseline, rel=1e-6)  # perfect render
    assert got[0] > got[1] > got[2]  # worse match scores lower


def test_rank_composite_matches_old_formula():
    ch = Channels(visual=np.array([-5.0, -1.0, -3.0]), displacement=np.array([0.2, 0.1, 0.0]),
                  rotation=np.array([0.0, 0.3, 0.1]), penetration=np.zeros(3))
    cfg = LikelihoodConfig()
    w = composite.rank_composite(ch, cfg)
    # old: rankdata(average)-1 for each channel (lower-is-better channels negated), each / its sum
    sim = np.array([0, 2, 1]) / 3
    disp = np.array([0, 1, 2]) / 3
    rot = np.array([2, 0, 1]) / 3
    assert np.allclose(w, sim + disp + rot)  # penetration is off by default
    # the old code weighted a zero (stub) penetration channel by 1: tied ranks add 1/N to every world
    old = composite.rank_composite(ch, LikelihoodConfig(lambda_penetration=1.0))
    assert np.allclose(old, sim + disp + rot + 1 / 3)
    assert composite.weights(Channels(*(np.zeros(1) for _ in range(4))), cfg).shape == (1,)


def test_energy_weights_positive_and_ordered():
    cfg = LikelihoodConfig(combine="energy")
    ch = Channels(visual=np.array([-100.0, -10.0]), displacement=np.array([0.0, 0.0]),
                  rotation=np.zeros(2), penetration=np.zeros(2))
    w = composite.weights(ch, cfg)
    assert (w > 0).all() and w[1] > w[0]


def test_resample_keeps_elite_and_handles_degenerate_weights():
    rng = np.random.default_rng(0)
    idx = resample_indices(np.array([0.1, 0.7, 0.2]), rng)
    assert idx[0] == 1 and len(idx) == 3
    idx = resample_indices(np.zeros(4), rng)  # all zero -> uniform, no crash
    assert len(idx) == 4
    assert list(resample_indices(np.array([1.0]), rng)) == [0]


def test_xy_proposer_keeps_height_and_orientation_and_bounds():
    q = [0, 0, 0.38268343, 0.92387953]
    prior = Prior(means=np.array([[0.4, 0.4, 0.75, *q], [-0.4, 0.0, 0.75, *q]]), pos_std=0.3, bounds=(-0.5, 0.5, -0.5, 0.5))
    p = Proposer(prior, 64, 10, np.random.default_rng(0))
    poses = p.initial()
    assert poses.shape == (64, 7)
    assert np.allclose(poses[:, 2], 0.75) and np.allclose(poses[:, 3:], q)
    assert (np.abs(poses[:, :2]) <= 0.5 + 1e-12).all()
    idx = np.arange(64)
    new = p.propose(poses, idx)
    assert np.allclose(new[0], poses[0])  # elite untouched
    for _ in range(9):
        new = p.propose(new, idx)
    # like the old ExpDecayProposal the schedule also advances on the initial proposal: 11 updates
    assert p.pos_std == pytest.approx(0.3 * 0.5 ** (2 * 11 / 10), rel=1e-9)


def test_xyzq_proposer_perturbs_rotation():
    prior = Prior(means=np.array([[0, 0, 0.75, 0, 0, 0, 1.0]]), pos_std=0.05, rot_std=0.2)
    p = Proposer(prior, 16, 10, np.random.default_rng(1), geometry_="xyzq")
    poses = p.initial()
    assert np.allclose(poses[0], prior.means[0])  # row 0 stays at the candidate (old XYZQProposer)
    assert np.abs(poses[1:, 3:5]).max() > 1e-3


class FakeLikelihood:
    """visual = -1000 * distance of the hidden block from its truth; physics does nothing."""

    def __init__(self, scene, n):
        self.body_names = [o.name for o in scene.dynamic_objects]
        self.num_worlds = n
        self.truth = scene.truth_poses(self.body_names)

    def evaluate(self, poses):
        d = np.linalg.norm(poses[:, 0, :2] - self.truth[0, :2], axis=1)
        z = np.zeros(self.num_worlds)
        return Channels(visual=-1000 * d, displacement=z, rotation=z, penetration=z), poses


def test_sampler_moves_towards_truth():
    block = SceneObject("block", [0.2, -0.1, 0.75, 0, 0, 0, 1], box(0.04), role="hidden")
    jug = SceneObject("jug", [-0.2, 0.2, 0.8, 0, 0, 0, 1], box(0.05), role="observed")
    block.prior = Prior(means=np.array([[0.0, 0.0, 0.75, 0, 0, 0, 1]]), pos_std=0.15, bounds=(-0.5, 0.5, -0.5, 0.5))
    jug.prior = Prior(means=jug.pose[None])
    scene = Scene("toy", {"block": block, "jug": jug}, DEFAULT_CAMERA)
    lik = FakeLikelihood(scene, 32)
    s = GibbsImportanceSampler(scene, lik, SamplerConfig(num_worlds=32, iterations=25, seed=3), LikelihoodConfig(),
                               log=lambda m: None)
    s.run()
    best = s.best()["block"]
    assert np.linalg.norm(best[:2] - block.pose[:2]) < 0.03
    assert np.ptp(s.poses[:, 1], axis=0).max() == 0.0  # the observed jug never moves
    arrays = s.trace.arrays()
    assert arrays["poses"].shape == (26, 32, 1, 7) and arrays["channels"].shape == (26, 32, 4)


def test_physics_terms_and_penetration():
    init = np.zeros((2, 1, 7)); init[..., 6] = 1
    final = init.copy(); final[1, 0, 0] = 0.1
    assert np.allclose(total_displacement(init, final), [0.0, 0.1])
    assert np.allclose(total_rotation(init, final), [0.0, 0.0])
    est = PenetrationEstimator([box(0.05), box(0.05)], [], num_worlds=2, samples=4000, device="cpu")
    poses = np.zeros((2, 2, 7)); poses[..., 6] = 1
    poses[0, 1, 0] = 0.05  # half overlap: 0.1 x 0.1 x 0.05 m inside each other, counted for both
    poses[1, 1, 0] = 0.5  # apart
    vol = est(poses)
    assert vol[1] == 0.0
    assert vol[0] == pytest.approx(2 * 0.1 * 0.1 * 0.05, rel=0.1)
