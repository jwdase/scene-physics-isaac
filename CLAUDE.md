# CLAUDE.md

Guidance for Claude Code in this repository. Read `README.md` first (workflow and layout),
`docs/DESIGN.md` (why PhysX, the scene format) and `docs/MIGRATION.md` (old Scene-Physics / data-pipeline).

## Running things

- Everything goes through uv: `uv run sp <command>`, `uv run pytest`. The lockfile pins the exact Isaac
  Sim 6.1 environment in `.venv`; `uv sync` only (re)installs this package. Never let uv re-resolve the
  simulator; if a dependency must change, update `pyproject.toml` constraints and check
  `uv sync --dry-run` first (an exact sync once wanted to uninstall `onnxruntime-gpu`).
- Non-interactive Isaac Sim runs need `OMNI_KIT_ACCEPT_EULA=YES` (the user has accepted the EULA).
- `uv run pytest` runs pure tests only. `uv run pytest -m isaac` boots Kit (single world, fine locally).
  `-m multiworld` is for the inference server.
- **Do not run multi-world inference on the local RTX 3070** (the user's instruction); `sp infer
  --num-worlds 1` smoke tests are fine. Rendering and single-scene physics are fine locally.

## Isaac Sim rules learned the hard way

- Kit must start **before** anything imports `pxr` or `warp`: the venv has a standalone OpenUSD 0.26 and
  warp 1.19, Kit ships its own (warp 1.16); importing ours first crashes Kit or mixes versions. The CLI
  boots Kit first for simulator commands; `tests/conftest.py` does it in `pytest_configure` for `-m isaac`.
  Modules under `scene_physics/isaac/` import omni and may only be imported after `isaac.app.launch()`.
- Kit parses `sys.argv`; `isaac.app.launch()` hides it. Exit through `isaac.app.shutdown()` (hard exit):
  `SimulationApp.close()` can abort in 6.1.
- Choose the PhysX device (`SimulationManager.set_device`) before anything plays the timeline, or PhysX
  silently starts on the CPU pipeline and GPU tensors are rejected.
- Replicator `orchestrator.step()` advances the timeline (capture-on-play); always pass `delta_time=0.0`.
- A path-traced frame requested before any ray-traced frame comes back black; `isaac/rtx.py` warms up in
  ray-tracing mode.
- With the GPU pipeline PhysX writes to Fabric, not USD: read poses through the tensor views, and write
  them back to USD explicitly (`SceneSimulator.write_poses_to_usd`).
- A file Kit has opened stays in the layer registry; do not rewrite it in the same process (simulate a
  temporary copy, as `migrate_legacy` does).
- PhysX reports small velocities for bodies at rest; rest detection uses pose change.
- PhysX is deterministic to ~1e-6 m (broadphase history); chaotic (interpenetrating) hypotheses diverge.

## Conventions

- Poses are `[x, y, z, qx, qy, qz, qw]` (scalar-last) everywhere; USD stores `(w, x, y, z)`: convert only
  via `scene_physics.geometry` / `usd_io`.
- Z-up, metres. Objects live under `/World/Objects/<name>`; the Xform is the rigid body and its transform
  is the ground truth; geometry + collider in `<name>/mesh`.
- The model's view is the Warp ray caster (`render/raycast.py`); the stimulus is RTX. Both use
  `CameraSpec` (vertical FOV, pixel centres at +0.5). RTX segmentation agrees with the ray caster on
  ~99.7% of pixels (anti-aliased edges).
- The physics config (`configs/default.toml [physics]`) is the single source of truth for authoring and
  inference.

## Working here

- Work on a branch; commit with a descriptive message after each coherent step.
- After changing authoring/rendering code, run `uv run sp check` on a real scene and look at `check.png`.
