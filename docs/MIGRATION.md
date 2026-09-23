# Migrating from Scene-Physics + data-pipeline

## Commands

| Old | New |
|---|---|
| `uv run src/data_pipeline/gen_scene.py --origin scene001 --exp data-set --scene scene052` | `sp import stage/scene001.usdc --exp data-set --name scene052`, then `sp check data-set/scene052`, then `sp finalize data-set/scene052` |
| `uv run eval/data-set/resolve_scene.py scene033` (raw export dropped into a scene folder) | `sp import data-pipeline/data/data-set/scene033 --exp data-set` (folders are auto-detected) |
| `uv run eval/data-set/gen_png.py` / `gen_target_png.py` | `sp render data-set/<scene>` (also part of `sp finalize`); `target.png` is produced alongside `rgb.png` |
| `python -m data_pipeline.simulation --exp data-set --scene scene001` (Newton viewer) | `sp check data-set/scene001 --gui` (watch it settle), or `sp open data-set/scene001` (full Isaac Sim editor) |
| `python -m data_pipeline.inference --exp ... --scene ...` / `simulation/sim_sampling.py scene002` | `sp infer data-set/scene002` (`--num-worlds`, `--iterations`, or the old `NUM_WORLDS` / `NUM_EPOCHS` variables) |
| `python -m data_pipeline.loss_map` + `plot_loss_map` | `sp lossmap data-set/<scene> --resolution 0.01` |
| whole legacy dataset | `sp migrate data-pipeline/data/data-set/scene* --exp data-set` |
| `data_gen/scene_gen.py` (procedural Drop-&-Settle) | not ported: every scene is hand-authored now. The pieces it needs (PhysX settling, exact visibility from the ray caster, the scene writer) are all here if it is wanted again. |

## Files

| Old | New |
|---|---|
| `data/<exp>/<scene>/data/<scene>_physics.usdc` | `data/<exp>/<scene>/scene.usd` |
| `<scene>_truth.json` | object transforms in `scene.usd` (`sp info --json` writes a JSON view) |
| `<scene>_makeup.json` | `sp:role` / `sp:tags` on objects, `sp:scene:condition` / `sp:scene:cue` on `/World` |
| `<scene>_priors.json`, `_modified_priors.json` | `sp:prior:*` on objects; regenerate with `sp priors`, or edit the attributes |
| `results/render.png` | `render/rgb.png` |
| `results/seg_raw.png` + `seg_labels.json` + decode step | gone: RTX instance ids are exact |
| `results/segmentation.png`, `segmentation_labels.json` | `render/segmentation.png` (the `{name: id}` map is embedded in the PNG and in `sp:segmentationId`) |
| `results/segmentation_overlay.png` | `render/overlay.png` (hidden objects outlined) |
| `results/target.png`, `target.json` | `render/target.png`; the per-object numbers from `target.json` are `sp:visibility:*` |
| `results/preview_depth.png` (some scenes) | `render/depth.png` for every scene: 16-bit millimetres to the image plane |
| `config/<exp>.json` | `experiments/<exp>.toml` (same camera values) |
| inference outputs next to the scene (`results/sampling.txt`, `*_positions.npy`, `final_world.usdc`, plots) | `results/<exp>/<scene>/<timestamp>/` (`sampling.txt`, `trace.npz`, `reconstruction.usd`, `summary.json`, plots) |

## What behaves differently in the model

These are deliberate; each is listed so results can be compared with runs of the old code.

1. **Physics engine.** Newton XPBD (32 iterations, 8 substeps at 60 Hz, box collider for the table,
   convex hulls) became PhysX TGS on the GPU (120 Hz, 16 position / 4 velocity iterations, convex
   *decomposition* for dynamic objects, the table's own triangle mesh). The rollout length is unchanged
   (2.5 s, the old 50 frames x 0.05 s). Friction 0.8, restitution 0 and density 1000 kg/m^3 are unchanged;
   the old rolling/torsional friction (0.1) is approximated by angular damping 0.1. PhysX caps how fast
   overlapping bodies separate (1 m/s), so a hypothesis that puts the hidden object inside another
   object moves smoothly instead of exploding.
2. **Determinism.** Repeating a rollout gives the same answer to ~2 micrometres (the old XPBD solver
   jittered visibly, which caused the "wandering max likelihood" plots). Hypotheses that overlap other
   objects are chaotic, so they can still differ a lot between repeats; they are all implausible anyway.
3. **Observation / rendering for the likelihood.** Newton's `SensorTiledCamera` became a Warp ray
   caster with the same pinhole convention (vertical FOV, pixel centres, looking down -Z). Newton's
   infinite ground plane is replaced by a large floor quad at z = 0 (the same surface in view).
4. **Point-cloud likelihood.** Same formula and defaults (variance 0.001, outlier probability 0.001,
   outlier volume 1, 7x7 window, -100 sentinel, NaN observed pixels ignored, baseline subtracted).
5. **Composite.** The rank composite is the same, with one exception: the old penetration term was a
   stub returning zeros but still carried weight 1, so its tied ranks added a constant 1/N to every
   world's weight, which slightly flattened resampling. The penetration term now exists (a sampled overlap
   volume) and is off by default (`lambda_penetration = 0`), so the constant is gone. `combine = "energy"`
   is new: magnitudes are kept and the weights are a softmax.
6. **Best hypothesis.** The old code kept the highest *rank composite* ever seen, but rank composites only
   compare worlds within one iteration (the top world always gets about the same value), so the "best"
   was mostly the first iteration's top world. The best is now tracked with the absolute energy score
   (`visual/1000 - displacement/0.01 m - rotation/0.1 rad`, configurable). The final particle cloud and its
   weights are also saved, as a posterior to compare with human responses.
7. **Resampling.** Whole hypotheses (every sampled object in a world) are resampled together; only the
   object being updated gets noise. With a single hidden object this is exactly the old behaviour.
8. **Priors.** The default rule is the data-pipeline one (one candidate behind every observed object, at
   the hidden object's height) but the candidate's orientation is now the hidden object's own; before,
   each candidate borrowed the orientation of the observed object it sat behind, and the XY proposer
   never changed it.
9. **Roles.** `observed` / `hidden` are measured from the camera (visible fraction below 10% = hidden) and
   stored with the measurement. Migrated scenes keep their original roles and report disagreements.
10. **Ground truth.** `sp finalize` settles the scene under the model's physics before storing truth, so
    the truth is a rest state for the model (the old manual pipeline stored the Blender poses as they were;
    in `stage/scene001` the loaf then settles 3.9 mm and tilts 4 degrees). `--no-settle` keeps authored poses;
    `sp migrate` always keeps them and records how far each object would move (`sp:settle:*`).
