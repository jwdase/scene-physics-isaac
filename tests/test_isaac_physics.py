"""PhysX tests (need Isaac Sim): ``uv run pytest -m isaac tests/test_isaac_physics.py``.

The two-world test is a *physics* isolation check (2 worlds, a few hundred steps), not inference.
"""

from pathlib import Path

import numpy as np
import pytest

from scene_physics.config import PhysicsConfig, SettleConfig
from scene_physics.scene import Mesh, Scene, SceneObject, importers, usd_io
from scene_physics.camera import DEFAULT_CAMERA

pytestmark = pytest.mark.isaac
LEGACY_STAGE = Path("/home/jdase/Desktop/Coding/data-pipeline/stage/scene001.usdc")


def box(hx, hy, hz):
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(pts, tris)


def stack_scene():
    table = SceneObject("table", [0, 0, 0.35, 0, 0, 0, 1], box(0.5, 0.5, 0.35), dynamic=False)
    a = SceneObject("block_a", [0.0, 0.0, 0.72, 0, 0, 0, 1], box(0.05, 0.05, 0.02))
    b = SceneObject("block_b", [0.0, 0.0, 0.76, 0, 0, 0, 1], box(0.03, 0.03, 0.02))  # resting on a
    c = SceneObject("block_c", [0.25, 0.1, 0.80, 0, 0, 0, 1], box(0.03, 0.03, 0.03))  # 7 cm above the table
    return Scene("stack", {o.name: o for o in (table, a, b, c)}, DEFAULT_CAMERA)


def test_settle_stack(isaac, tmp_path):
    from scene_physics.isaac.simulator import SceneSimulator

    path = usd_io.write_scene(stack_scene(), tmp_path / "scene.usd")
    sim = SceneSimulator(path, PhysicsConfig())
    report = sim.settle(SettleConfig(max_seconds=3.0))
    print("\n" + report.table())
    motion = {m.name: m for m in report.motions}
    assert report.at_rest
    assert motion["block_a"].displacement < 0.002 and motion["block_b"].displacement < 0.002
    assert motion["block_c"].displacement == pytest.approx(0.07, abs=0.003)  # dropped onto the table
    final = dict(zip(report.names, report.settled))
    assert final["block_c"][2] == pytest.approx(0.73, abs=0.003)
    assert max(m.residual for m in report.motions) < 1e-3
    sim.stop()


@pytest.mark.skipif(not LEGACY_STAGE.exists(), reason="legacy stage file not present")
def test_real_scene_is_nearly_at_rest(isaac, tmp_path):
    from scene_physics.isaac.simulator import SceneSimulator

    scene = importers.from_usd_export(LEGACY_STAGE, name="scene001")
    path = usd_io.write_scene(scene, tmp_path / "scene.usd")
    sim = SceneSimulator(path, PhysicsConfig())
    report = sim.settle(SettleConfig())
    print("\n" + report.table())
    assert report.at_rest
    assert max(m.displacement for m in report.motions) < 0.01
    sim.stop()


def test_two_worlds_isolated_and_deterministic(isaac):
    from scene_physics.isaac.simulator import ParallelWorlds

    scene = stack_scene()
    worlds = ParallelWorlds(scene, num_worlds=2, physics=PhysicsConfig())
    base = np.stack([worlds.truth, worlds.truth])
    alt = base.copy()
    alt[1, worlds.body_index("block_c"), :3] = [0.0, 0.0, 0.85]  # world 1: drop c onto the stack
    a = worlds.rollout(base, 1.0)
    b = worlds.rollout(base, 1.0)
    c = worlds.rollout(alt, 1.0)
    d = worlds.rollout(base, 1.0)
    # PhysX is deterministic up to broadphase history effects at the micrometre level (the old XPBD
    # solver jittered by millimetres); displacement scores are measured in millimetres to centimetres.
    print(f"\n|a-b|={np.abs(a - b).max():.2e}  world0 a vs c={np.abs(a[0] - c[0]).max():.2e}  |a-d|={np.abs(a - d).max():.2e}")
    assert np.abs(a - b).max() < 1e-5, "same input must give the same result"
    assert np.abs(a[0] - c[0]).max() < 1e-5, "world 0 must not feel world 1"
    assert np.abs(a - d).max() < 1e-5
    assert np.abs(c[1] - a[1]).max() > 0.01, "world 1 must respond to its own change"
    worlds.stop()
