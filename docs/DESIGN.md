# Design: scene-physics-isaac

This repository replaces two older ones:

* **Scene-Physics**: the model (importance sampling over hidden-object poses, scored by a point-cloud
  likelihood plus forward physics), built on standalone NVIDIA Newton (XPBD) with a Blender
  render path.
* **data-pipeline**: the per-scene data handling (turning Blender exports into
  `_physics.usdc`/`_truth.json`/`_makeup.json`/`_priors.json`, renders, target silhouettes,
  inference and loss-map drivers).

Both now live in one package, `scene_physics`, running on **Isaac Sim 6.1**.

---

## 1. Why Isaac Sim + PhysX (and not Newton)

Isaac Sim 6.1 ships two physics backends: **PhysX 5** (the default) and an **experimental Newton 1.5
integration** (MuJoCo-Warp, plus XPBD/VBD added in 6.1). The model needs *reliable* rigid-body
behaviour on tabletop scenes: a physically stable arrangement must stay still, and an implausible one
must move in a smooth, repeatable way. It also has to evaluate N hypotheses in parallel worlds on one
GPU.

| Requirement | PhysX 5 (GPU, TGS) | Newton in Isaac Sim 6.1 |
|---|---|---|
| Maturity in Isaac Sim | Default production engine | Marked *experimental* in the 6.0/6.1 docs; "API and features may change"; cannot apply USD edits during playback |
| Resting contact / stacking | TGS solver designed for stacking and high mass ratios; stabilization; per-body solver iterations | MuJoCo soft contacts allow penetration by design; the old XPBD path needed a box-collider workaround for tunnelling and blew up on interpenetrating proposals |
| Interpenetrating proposals (common: the sampler drops the hidden object anywhere) | `maxDepenetrationVelocity` caps how fast overlapping bodies separate, so bad hypotheses move smoothly instead of exploding | No equivalent cap in XPBD; MuJoCo contact buffer (`nconmax`, default 200/world) **silently drops** excess contacts |
| Concave objects (mugs, bowls, jugs) | Convex decomposition or SDF colliders for dynamic bodies | MuJoCo collides meshes by convex hull only (cavities are lost) |
| Determinism | Same inputs give the same results on the same GPU (to ~1e-6 m in practice); *enhanced determinism* makes each island independent of the other islands (other worlds) | Old XPBD path was measured non-deterministic (atomic contact accumulation), which caused the "wandering max" in the likelihood plots |
| N parallel worlds | Env cloning with physics replication and **env IDs** (co-located worlds that cannot collide), batched GPU pose I/O through `omni.physics.tensors` | Also supported, but through the experimental integration |

**Measured on this machine (RTX 3070, `.scratch/` experiments during the rewrite):**

* A real hand-authored scene (`data-pipeline/stage/scene001.usdc`) under PhysX GPU/TGS at 120 Hz with
  convex-decomposition colliders comes to rest within 0.5 s of simulated time: the loaf settles 3.9 mm
  and tilts 4 degrees, every other object moves under 0.5 mm, and a further 2 s moves nothing (< 0.06 mm).
  240 steps take 0.7 s. All 52 legacy `data-set` scenes were run through the same check (see `sp migrate`).
* Two cloned worlds: repeated rollouts from the same input agree to ~2 micrometres (bit-identical when
  the preceding history is the same), and changing everything in world 1 moves world 0's result by at
  most ~2 micrometres, often exactly 0 (co-located env-ID worlds and spaced grid worlds behave the same).
  The old XPBD path varied by millimetres.

Decision: **PhysX**, GPU pipeline, TGS solver, enhanced determinism, convex decomposition for dynamic
objects, triangle-mesh colliders for static ones. Newton remains one config switch away should its
Isaac integration mature (differentiability, 10k+ worlds), since the scene format is plain UsdPhysics.

Sources: Isaac Sim Newton backend docs
(<https://docs.isaacsim.omniverse.nvidia.com/latest/physics/newton_physics.html>), Isaac Sim 6.1 release
notes (<https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html>), PhysX 5 rigid body
dynamics and GPU docs (<https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html>,
<https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html>), Isaac Lab 3.0 release
notes on MJWarp's `nconmax` contact dropping (<https://github.com/isaac-sim/IsaacLab/releases>), and the
Isaac Sim `physics-simulation` skill bundled in the install (`.venv/.../isaacsim/skills`).

## 2. What changed, file by file

| Old (Scene-Physics / data-pipeline) | New |
|---|---|
| `<scene>_physics.usdc` | `scene.usd`: one file that is simultaneously the physics scene, the ground truth and the metadata |
| `<scene>_truth.json` | the object prims' own transforms (the settled pose) |
| `<scene>_makeup.json` (static/observed/hidden + occluder/target/decoy/probe/cue/condition) | `sp:role` + `sp:tags` on each object prim, `sp:scene:*` attributes on `/World` |
| `<scene>_priors.json`, `_modified_priors.json` | `sp:prior:*` attributes on each object prim (generated by a named rule, hand-editable) |
| `results/target.json`, `seg_labels.json`, `segmentation_labels.json` | `sp:segmentationId` + `sp:visibility:*` on each object; the id↔name map is also embedded in `segmentation.png` |
| `config/<exp>.json` (camera) | `experiments/<exp>.toml` for authoring defaults; each scene also stores its own `/World/Camera` |
| Blender render (`render.png`, `seg_raw.png`, decode step) | Isaac RTX render: `render/rgb.png`, `render/segmentation.png` (direct instance ids, no colour decoding) |
| Newton `SensorTiledCamera` depth | `render/raycast.py`: a Warp ray caster over the same meshes, N worlds at once |
| XPBD forward sim | PhysX rollouts in N cloned worlds (`isaac/simulator.py`) |

## 3. Scene format

A finished scene is a directory:

```
data/<experiment>/<scene>/
  scene.usd          # everything below lives in this one file
  render/
    rgb.png          # stimulus (RTX)
    segmentation.png # uint8 object ids, 0 = background; PNG text chunk "labels" = {name: id}
    overlay.png      # QA: segmentation over the stimulus, hidden objects outlined
    target.png       # amodal silhouette of the target, filled with its colour (as before)
    depth.png        # depth to the image plane, 16-bit millimetres (RGB-D models)
```

`scene.usd` (Z-up, metres, `defaultPrim = /World`):

```
/World                      sp:scene:name, :experiment, :condition, :cue, :description, :source, ...
  /PhysicsScene             PhysxSceneAPI: TGS, GPU dynamics, 120 Hz, enhanced determinism
  /Objects
    /<object>               Xform = rigid body; its transform IS the ground-truth pose
                            RigidBodyAPI + MassAPI (+PhysxRigidBodyAPI) for dynamic objects only
                            sp:role      = static | observed | hidden
                            sp:tags      = [target, occluder, decoy, probe, reference, ...]
                            sp:asset     = library asset / base name (e.g. "jug04")
                            sp:segmentationId, sp:visibility:{fraction,visiblePixels,silhouettePixels}
                            sp:prior:{positions,orientations,weights,posStd,rotStd,bounds}
                            sp:settle:{displacement,rotation}
      /mesh                 render + collision geometry (CollisionAPI, convexDecomposition or none)
      /Looks/...            the object's material
  /Camera                   the scene camera (pinhole; vertical FOV, resolution in sp:camera:*)
```

**Object types.** The old three-way `static/observed/hidden` makeup mixed two different things. The new
format keeps them separate:

1. *Physical kind* comes from the physics schemas: an object with `RigidBodyAPI` is dynamic, one
   without is a static collider (`sp:role = static`). Nothing to keep in sync by hand.
2. *Visibility role* (`observed` / `hidden`) is **measured** from the scene camera at finalize time
   (visible fraction of the object's silhouette, threshold in the experiment config) and stored with the
   measurement. An author can pin it by setting `sp:role` before finalizing; a disagreement with the
   measurement is reported.
3. *Experimental annotations* (target, occluder, decoy, probe, reference, cue, condition) are free-form
   tags that analysis code reads; inference never depends on them.
4. *What the sampler proposes for* is an inference setting (`sample_roles`, default `["hidden"]`), so
   sampling visible objects later is a config change, not a data change.

## 4. Authoring workflow (manual scenes)

```
Blender export  ->  sp import  -\
                                 >-  sp check  ->  (sp open: fix in Isaac Sim, repeat)  ->  sp finalize
sp new + sp add (library)  -----/
```

Scenes can still come from Blender, or be built without it: `sp library extract` turns the objects of
finished scenes into small physics-ready asset files (same size and look as in the experiments),
`sp new <exp>/<name>` starts a scene with the table and the experiment camera, and `sp add` drops
library objects onto the table; `sp open` is then used to arrange them.

* `sp import <export.usdc> --exp data-set --name scene053`: converts a Blender USD export (or an
  Isaac-authored stage) into the canonical structure under `authoring/<exp>/<name>.usd`.
* `sp check <scene>`: simulates the scene under the current physics config (the same PhysX settings the
  model uses) and reports how far every object moves; renders from the scene camera; reports each
  object's visible fraction; writes `check/` with before/after renders, a GIF of the settle, the
  segmentation overlay and a text report.
* `sp open <scene>`: the scene in the Isaac Sim editor (PhysX forced, scene camera active) for edits.
* `sp finalize <scene>`: settles (unless `--no-settle`), measures visibility, assigns roles and priors,
  renders the stimulus and segmentation, validates, and writes `data/<exp>/<name>/`.

## 5. Inference

```
scene.usd --> observation: ray-cast depth of the ground truth (the model's visual input)
          --> ParallelWorlds(N): scene objects cloned into N co-located PhysX worlds
for each iteration (Gibbs over objects with a sampled role):
    propose poses for the chosen object in all N worlds          (numpy, cheap)
    visual   = 3DP3 point-cloud log-likelihood of the proposal render   (Warp ray caster + kernel)
    physics  = PhysX rollout from the proposals; total displacement / rotation of all dynamic bodies
    (optional) penetration = sampled interpenetration volume at t=0
    weights  = composite (rank, as before, or energy/softmax)
    resample + Gaussian noise, keep the elite in world 0, track the best
```

The sampler, proposal kernels, schedules and rank composite are ports of the old code, so results are
comparable; the documented differences are in `docs/MIGRATION.md`.
