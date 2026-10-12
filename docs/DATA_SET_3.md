# data-set-3

data-set-3 is data-set-2 (70 hand-authored scenes) reworked so that every scene satisfies two constraints
that data-set-2 violates:

1. **The target is fully hidden.** In data-set-2, 18 targets show in the stimulus (scene017's loaf is 63%
   visible) and many more are hidden by a hair. A visible target gives no reason to reason about where
   it could be.
2. **The scene is at rest under the model's physics.** In data-set-2, objects in 53 of 70 scenes move
   more than 5 mm or 2° when simulated (bowls floating a centimetre above the table, loaves sunk 7 mm into
   it, a glass floating 6 cm up, a loaf passing 2 cm into a vase), so every inference rollout started
   from a truth that was not a rest state.

Each scene keeps its data-set-2 design: the same objects, target, occluders and camera, and the same
physical relationship between the target and the objects around it (a bowl resting on the hidden block,
a loaf leaning on it, a vase tilted onto the hidden coaster, ...). Edits are as small as the constraints
allow; the table below lists every change.

<!-- RESULTS -->

## Camera exploration

Occlusion depends only on where the eye is: field of view and look-at only frame the picture. 420 eye
positions (azimuth ±30°, elevation 4–30°, distance 1.1–2.4 m around the table centre; FOV a dolly zoom of
the data-set lens so the table keeps its size) were swept over all 70 scenes with the Warp ray caster, and
the best candidates were re-checked at the stimulus resolution with the 2-pixel margin and rendered with
RTX to judge whether they still look like ordinary photographs of a table.

| camera (azimuth, elevation, distance) | eye height | targets hidden (2 px margin) | objects the design shows that get hidden | physical cues that lose > 40% of their visible area |
|---|---|---|---|---|
| **0°, 12°, 1.7 m (data-set / data-set-2)** | 1.15 m | 45 | 0 | 0 |
| 0°, 10°, 2.0 m | 1.15 m | 50 | 1 | 2 |
| 0°, 8°, 1.7 m | 1.04 m | 53 | 6 | 2 |
| 0°, 8°, 1.5 m | 1.01 m | 56 | 6 | 2 |
| ±10° azimuth (any elevation) | | ~20 | | |

* The scenes are designed for the frontal view: turning the camera by 10° hides fewer than half as many
  targets.
* Lowering the camera hides more targets (55–56 at 8° instead of 45 at 12°), and the lower views still
  look natural, but the camera cannot tell the target from the rest: it also hides objects the designs
  show, including two physical cues (the phone resting on the hidden cup in scene031, the coaster on the
  hidden cup in scene067).
* Moving the hidden target is invisible in the stimulus, whereas un-hiding a buried cue means visibly
  moving objects. **data-set-3 therefore keeps the data-set camera** and hides targets by editing the
  scenes. Only 30 targets stay hidden when the camera moves by ±1° / ±15 cm: many data-set-2 designs hide
  the target by a hair, which is why data-set-3 requires a margin.
* Two designs (scene017, scene040) cannot be realised from that camera at all (see the table); they use
  the camera 11 cm lower (8°, 1.5 m, same framing), where small edits hide them and nothing else in the
  scene gets buried.

## Physics

`configs/default.toml` (and docs/DESIGN.md) chose PhysX 5 GPU/TGS at 120 Hz with convex decomposition for
dynamic objects and the table's triangle mesh. Relaxing all 70 data-set-2 scenes (below) under three
variants:

| | data-set default | table as a box | box + 32/8 iterations (**data-set-3**) |
|---|---|---|---|
| come to rest | 69 | 70 | 70 |
| verified rest state (2.5 s rollout moves < 1 mm) | 66 | 69 | 69 |
| robust to a small push (±1 mm, 0.5°: every object back within 5 mm) | 62 | 66 | 67 |
| 90th percentile displacement after a push | 8.3 mm | 3.0 mm | 3.2 mm |
| time per physics step | 1× | 0.95× | 1.3× |

The table collides as its bounding box: a solid block whose top face is exactly the table top (the top
is flat at z = 0.735 m and is the mesh's highest point; four scenes have a table tilted by 0.6–1.7°, which
the box keeps). A triangle mesh is a surface without thickness: resting contacts on it are noisier and a
deep overlap can be pushed the wrong way. The solver runs 32 position / 8 velocity iterations instead of
16/4, which settles stacked and leaning contacts (a bowl on a block, a coaster on a cup) more reliably. A
240 Hz step fixed some scenes and broke others and was not adopted. Both changes live in
`experiments/data-set-3.toml`, so inference on data-set-3 uses them automatically while data-set keeps
the physics its scenes were settled with.

## How the scenes were made

```bash
uv run sp rework data/data-set-2/scene0* --exp data-set-3      # -> data/data-set-3/sceneNNN/
```

`sp rework` (`src/scene_physics/authoring/rework.py`) applies the hand edits of
`experiments/data-set-3/edits.toml` (authoring errors, the two camera exceptions) and then loops, per
scene:

1. **Relax** (`sp relax`, `authoring/relax.py`). Authored poses are rarely a rest state: objects sit a few
   millimetres inside the table or each other, or float above it. Simulated directly under full gravity,
   every overlap becomes a depenetration push in the first step and leaning objects slide off their
   supports. Relaxation instead separates overlaps under almost no gravity (0.3 m/s²) with PhysX's
   depenetration speed capped at 5 cm/s, heavy damping and hard speed caps (10 cm/s, 45°/s), ramps
   gravity to 9.81 m/s² over 2 s, and only then settles under the unmodified model physics. The result
   is verified: a 2.5 s rollout (one inference rollout) from the stored poses must move no object more
   than 1 mm / 0.5°; four rollouts from randomly nudged poses (±1 mm, 0.5°) test that the rest is
   robust. Any object that moves more than 3 cm or 30° on the way is reported (explosion guard): in
   data-set-2 the only such objects were authoring errors (a floating glass, a floating coaster, a coaster
   inside the candle holder), fixed by hand.
2. **Check** the scene against its data-set-2 design:
   * the target is hidden with a **2-pixel margin** at the stimulus resolution (1280×960): its silhouette
     grown by 2 px must be covered by closer objects (`authoring/occlusion.py`; transparent objects such as
     `glass1` do not count as occluders);
   * at least half of it is behind its data-set-2 occluders (it was not simply moved behind something else);
   * its contacts are those of the design (`authoring/contacts.py`, signed distances with hysteresis:
     touching < 2 mm, apart > 4 mm);
   * every other object the design shows keeps most of its visible area (physical cues touching the
     target 75%, other objects 50%).
3. **Hide** (`authoring/hide.py`), if a check fails: the cheapest edit that passes them all with a
   4-pixel margin (extra margin because relaxation can still shift objects by a millimetre). An edit moves
   the target either alone (what rests on it stays put and re-settles) or rigidly together with everything
   physically connected to it (what rests or leans on it), and may move the main occluder a little.
   Candidates go from small moves (≤ 4 cm, 15°) through occluder moves (≤ 4 cm, 15°) to large target moves
   (≤ 16 cm, 90°), cheapest first; moving the hidden target is cheap, moving anything visible (the
   occluder, or a cue that moves with the target) costs three times as much. Then back to 1.

   Every edit is tested by physics: if, after relaxing, something slides, a contact changes, an object the
   design shows loses its visibility, or a visible object ends more than 1.5 cm / 8° from where the edit
   put it (a cue re-settling into a different tilt), the edit is undone and the search continues without
   it. When moving the target alone lets a cue re-settle, only edits that take the cue along remain.

Scenes that pass are finalized (`sp finalize`): the target is the one hidden object and every other
object is observed, exactly as in data-set-2; priors are regenerated (`observed_objects` rule) and the
stimulus, segmentation, depth and target silhouette are rendered.

## Checks on the finished dataset

<!-- CHECKS -->

## Caveats

<!-- CAVEATS -->
