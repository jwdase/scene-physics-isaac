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
* The legacy objects that move under PhysX move the same way with convex-hull, convex-decomposition
  and SDF colliders (e.g. a bee figurine tips 68–69° under all three), so the motion comes from the
  authored poses, not the collider approximation; see `docs/LEGACY_AUDIT.md`. Decomposition is also
  the fastest of the three to settle.
* Two cloned worlds: repeated rollouts from the same input agree to ~2 micrometres (bit-identical when
  the preceding history is the same), and changing everything in world 1 moves world 0's result by at
  most ~2 micrometres, often exactly 0 (co-located env-ID worlds and spaced grid worlds behave the same).
  The old XPBD path varied by millimetres.

**Measured on the inference server (MIT ORCD `node1702`: H100 80 GB, driver 590.48, PhysX GPU,
November 2026; the pinned wheels run unchanged inside `scripts/isaac-sim.def`):**

* `data-set/scene017` (6 dynamic objects, convex decomposition) with **64 worlds**: building the worlds
  and the likelihood takes 3.6 s, and an iteration takes 0.85 s — 0.84 s of it the PhysX rollout (2.5 s
  of simulated time, 300 steps, all 64 worlds at once, so the cost per world is almost free) and 6.5 ms
  the ray-cast render plus the point-cloud likelihood of all 64 worlds (0.1 ms per world). 50 iterations
  take 44 s. A brute-force loss map (51x51 = 2601 cells, 64 per batch) takes 59 s.
* The two-world isolation / determinism checks and the world-slot test pass unchanged on this GPU
  (`pytest -m "isaac or multiworld"`): a hypothesis scores the same whichever world slot it lands in
  (visual to 1e-6, displacement to 1e-4 m).
* Kit boots in ~20 s with warm caches (the first boot takes ~2 min while it builds them).
* A whole-dataset sweep (`scripts/dataset-infer.sbatch`, `node3909`: **A100 80 GB PCIe**, November
  2026): 64 worlds and 50 iterations cost a median 1.05 s an iteration (0.57-1.37 s depending on the
  scene) against the H100's 0.85 s, plus a median 4.2 s to build the worlds, so a scene takes about
  65 s. The 65 scenes of `data-set-3` took 1 h 12 min of wall time in one Kit session, the 7 of
  `data-set-4` 10 min. Kit's **first** boot on a node it has not run on takes ~317 s, not ~20 s, while
  the 25 GB environment and the caches come off `$SP_POOL` over NFS; a second job on the same node
  boots in 13 s. Size `sbatch --time` for the cold boot.

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

**What the visual channel can and cannot say about a hidden object.** The likelihood sums over the
*observed* pixels, so a hypothesis is penalised only where it changes what the camera already sees: a
pose in which the target stays completely out of sight scores exactly 0, the same as the truth. For a
target that is 0% visible this makes the visual channel flat over whole regions, and the physics
channel is flat too wherever the table supports the object, so the posterior concentrates on the
*largest* consistent region rather than on the truth. Measured on `data-set/scene017` (`sp lossmap
--resolution 0.02`, 2601 poses): 229 of them hide the coffeemug perfectly, in three disconnected
plateaus — 179 cells behind `jug01`, 26 cells behind `jug04` (the truth's own hiding place) and 24
behind the shark. A 64-world, 50-iteration `sp infer` run converges into the largest plateau, 73 cm
from the truth, with a 10 cm posterior spread. The run is working as designed; the distance to truth
is a statement about the scene's ambiguity, not about the sampler, and it is what makes these scenes
worth putting to people. Report `summary.json`'s posterior spread and the loss map alongside any
error against truth. The 65-scene `data-set-3` sweep behaves the same way: the best hypothesis is a
median 2.7 cm from truth, but four scenes (050, 051, 056, 069) settle 42-79 cm away with a 10-13 cm
spread, and the posterior mean is far from truth wherever a second hiding place is large enough to
hold most of the particles.
