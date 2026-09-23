# scene-physics-isaac

Can a probabilistic model that *simulates physics* explain how people infer where a hidden object is?
Given one image of a tabletop in which an object is occluded, the model proposes where the hidden object
could be in N parallel physics worlds, scores every proposal by how well it explains the image and how
little it moves when physics runs (a stable arrangement stays put), and refines the proposals by
importance sampling. The same scenes go to people, to neural networks and to this model, and their
answers are compared.

This repository merges the two earlier ones (**Scene-Physics**: the model on standalone NVIDIA Newton;
**data-pipeline**: scene data handling and Blender rendering) into one package, `scene_physics`, on
**Isaac Sim 6.1 with PhysX**. See [`docs/DESIGN.md`](docs/DESIGN.md) for why PhysX, and
[`docs/MIGRATION.md`](docs/MIGRATION.md) for how old commands, files and model behaviour map to the new ones.

## Setup

```bash
uv sync                      # the lockfile pins the exact Isaac Sim 6.1 environment already in .venv
export OMNI_KIT_ACCEPT_EULA=YES   # after you have accepted the NVIDIA Omniverse EULA once
bash scripts/fetch_assets.sh      # table/floor textures + HDRI for the renders (CC0, ~80 MB, gitignored)
uv run pytest                     # pure tests (no simulator)
uv run pytest -m isaac            # tests that boot Isaac Sim (single world)
```

Python 3.12, a CUDA GPU. Isaac Sim warns that an RTX 3070 is below its recommended spec; authoring and
rendering work fine on it. Multi-world inference is meant for the remote server.

## Workflow: authoring a scene

Scenes are authored by hand, in Blender (export USD, Z-up, metres) or directly in Isaac Sim.

```bash
uv run sp import path/to/scene053.usdc --exp data-set   # -> authoring/data-set/scene053.usd
uv run sp check data-set/scene053                        # physics + occlusion check (see below)
uv run sp open data-set/scene053                         # fix things in the Isaac Sim editor, Ctrl+S
uv run sp check data-set/scene053 --gui                  # or watch the physics live
uv run sp finalize data-set/scene053                     # -> data/data-set/scene053/
```

To change a finished scene, bring it back for editing with `sp import data/data-set/scene041` (roles,
tags and priors come along), then `sp check` / `sp open` / `sp finalize --overwrite` as above.

`sp check` simulates the scene under **exactly the physics the model uses** and reports how far every
object moves before coming to rest, renders the scene from its camera before and after physics, and
measures how much of each object the camera sees (objects under 10% visible are *hidden*). It writes
`authoring/data-set/scene053.check/`:

```
check.png        before physics | after physics | segmentation with hidden objects outlined in red
settle.gif       the settle, from the scene camera
report.txt       motion per object, visibility per object, target, warnings
```

`sp finalize` settles the scene (the settled poses become ground truth), measures visibility, sets
roles, picks the target (or `--target NAME`), writes the priors, renders the stimulus, and validates:

```
data/data-set/scene053/
  scene.usd            the scene: geometry, physics, ground truth, roles, priors, camera, metadata
  render/rgb.png       the stimulus (path traced, 1280x960 for the 640x480 camera)
  render/segmentation.png   object ids (0 = background), id map embedded in the PNG
  render/overlay.png   QA view
  render/target.png    the target's full silhouette in its colour
  render/depth.png     depth to the image plane, 16-bit millimetres (for RGB-D models)
```

Other commands: `sp info <scene> [--priors] [--json out.json]`, `sp validate`, `sp list [exp]`,
`sp render <scene>`, `sp priors <scene> --rule occluder`, `sp roles <scene> --measure --target NAME`
(fix roles / the target in place; priors follow), `sp migrate <old scene folders>`.
Scenes can be named as paths or as `<experiment>/<scene>`.

## The scene file

One USD file per scene; open it in Isaac Sim and press Play and it simulates. Everything the old
`_truth / _makeup / _priors / target.json` files held lives on the prims:

```
/World                 sp:scene:{experiment, condition, cue, status, source, physicsConfig, ...}
  /PhysicsScene        PhysX settings the scene was settled with
  /Objects/<name>      transform = ground-truth pose
                       sp:role (static | observed | hidden), sp:tags (target, occluder, decoy, probe, ...)
                       sp:prior:*, sp:visibility:*, sp:segmentationId, sp:settle:*
  /Camera              the camera the stimulus and the model share
```

Object *kind* (static vs dynamic) comes from the physics schemas, *visibility role* is measured,
*annotations* are free tags, and *what the sampler infers* is an inference setting
(`sampler.sample_roles`), so sampling the visible objects later is a config change.

```python
from scene_physics.scene import load_scene
scene = load_scene("data-set/scene017")
scene.target.name, scene.target.pose, scene["jug04"].tags, scene.camera.fov_degree
```

## Inference (remote server)

```bash
uv run sp infer data-set/scene017 --num-worlds 64 --iterations 50
NUM_WORLDS=64 NUM_EPOCHS=50 uv run sp infer data-set/scene017      # old environment variables work too
uv run sp lossmap data-set/scene017 --resolution 0.01 --num-worlds 64
uv run pytest -m multiworld                                         # sanity checks with several worlds
```

Each run writes `results/<experiment>/<scene>/<timestamp>/`: `summary.json` (best hypothesis, errors vs
truth, posterior mean and spread, a truth check, timing), `trace.npz` (every proposal and score),
`config.json`, `sampling.txt`, `reconstruction.usd` (render the model's answer with `sp render`), and
plots. Set `SCENE_PHYSICS_DATA` / `SCENE_PHYSICS_RESULTS` to put data or results on other storage.

Configuration: `configs/default.toml` (physics, settle, render, authoring, likelihood, sampler), per
experiment `experiments/<exp>.toml`, and `--config extra.toml` for a run.

### Setting up the server

```bash
git clone git@github.com:jwdase/scene-physics-isaac.git && cd scene-physics-isaac
git checkout rewrite/isaac-sim
uv sync                                   # installs the pinned Isaac Sim 6.1 environment (~25 GB)
export OMNI_KIT_ACCEPT_EULA=YES
rsync -a <workstation>:scene-physics-isaac/data/ data/      # or: export SCENE_PHYSICS_DATA=/scratch/...
uv run pytest -m isaac && uv run pytest -m multiworld       # the simulator works, worlds are independent
uv run sp infer data/data-set/scene0* --num-worlds 64       # one Isaac Sim session for all scenes
```

`sp infer` and `sp lossmap` start Isaac Sim with RTX rendering switched off (the model sees the scene
through its own Warp ray caster), so they only need CUDA and PhysX; the stimulus renders (`sp finalize`,
`sp render`) need an RTX GPU and are best made on the workstation.

Per iteration the cost is one PhysX rollout (2.5 s of simulated time, 300 steps; about 0.7 s whatever N
is) plus the ray-cast render and likelihood of all N worlds (about 1 ms per world on the workstation).

## Layout

```
src/scene_physics/
  scene/        the data model and scene.usd format (plain OpenUSD; no simulator needed)
  authoring/    import / check / finalize / render / migrate, visibility and validation
  isaac/        everything that needs Isaac Sim: app lifecycle, PhysX simulator (1 or N worlds),
                RTX renderer, studio look
  render/       Warp ray caster (the model's view), image outputs
  likelihood/   3DP3 point-cloud likelihood, physics terms, composite
  inference/    proposals, Gibbs importance sampler, sp infer, sp lossmap, reports
  camera.py, geometry.py, config.py, paths.py, cli.py
configs/  experiments/  assets/  scripts/  tests/  docs/
authoring/  data/  results/     (gitignored)
```
