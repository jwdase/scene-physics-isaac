"""Boot check for Isaac Sim 6.1: start the app, build a stage, step PhysX.

Run headless (no display needed):
    .venv/bin/python smoke_test.py
Run with the viewport:
    .venv/bin/python smoke_test.py --gui
"""

import os
import sys

HEADLESS = "--gui" not in sys.argv

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": HEADLESS})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=np.array([0.0, 0.0, 2.0]),
        size=0.5,
    )
)

world.reset()

start_z = cube.get_world_pose()[0][2]
for _ in range(180):
    world.step(render=not HEADLESS)
end_z = cube.get_world_pose()[0][2]

print(f"\n[smoke] cube z: {start_z:.3f} -> {end_z:.3f}")
if end_z < start_z - 0.5:
    print("[smoke] PASS - app booted and PhysX advanced the sim")
    code = 0
else:
    print("[smoke] FAIL - cube did not fall; physics did not step")
    code = 1

# Isaac Sim 6.1 aborts (SIGABRT) inside simulation_app.close() when the async
# engine still has a coroutine in flight: "Destroying busy TaskGroup!". The abort
# is a C-level assert, so it cannot be caught -- it lands after all simulation work
# is done, but it would make this script exit 134 on a successful run. Flush and
# hard-exit instead so the exit code reflects the actual result.
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
