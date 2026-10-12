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
allow; the table below lists every change. The exceptions are scenes 003 and 017: data-set-2 shows their design
(a bowl propped up on a block, as in 001, 002, 005, 053 and 054) from the side, with the block in plain view and
another object as the target. In data-set-3 their arrangement is turned to face the camera, so the bowl hides the
block, and the block is the target.

## Scenes

Target visible: pixels of the target in the 1280×960 data-set-2 stimulus (ray caster). Changes: objects that moved more than 1 cm or 5° from their data-set-2 pose (most of it is physics settling the authored poses). Without the target: simulating the data-set-3 scene with the target removed, the object that moves most (the physical evidence of the hidden object; 'nothing' = a pure occlusion trial).

| scene | target | occluder | relationship (data-set-2) | target visible in data-set-2 | changes | without the target | edit |
|---|---|---|---|---|---|---|---|
| 001 | block | loaf | bowl on block | 0 | loaf 3.2 cm, block (target) 1.7 cm | bowl 1.9 cm | hiding edit |
| 002 | block | loaf | bowl touches block | 1231 | block (target) 10.8 cm, bowl 10.8 cm | bowl 1.0 cm | hiding edit |
| 003 | block | bowl | bowl touches block | 1341 | arrangement turned 90° (the camera looks from the data-set-2 left side), then settling only (< 1 cm, 5°) | bowl 2.9 cm | hand edit (turned, block is the target) |
| 004 | block | grinder | — | 0 | banana 0.3 cm / 6° | banana 2.0 cm | relaxed |
| 005 | block | jug | bowl touches block | 0 | settling only (< 1 cm, 5°) | bowl 2.9 cm | relaxed |
| 006 | block | jug | bowl on block | 0 | settling only (< 1 cm, 5°) | bowl 4.7 cm | relaxed |
| 007 | block | jug | bowl on block | 0 | settling only (< 1 cm, 5°) | bowl 4.7 cm | hiding edit |
| 008 | block | bowl, loaf | bowl on block | 49 | block (target) 3.2 cm | bowl 4.7 cm | hiding edit |
| 009 | block | bowl, loaf | bowl on block | 0 | settling only (< 1 cm, 5°) | bowl 4.7 cm | relaxed |
| 010 | banana | loaf | — | 0 | bowl 1.2 cm | loaf 3.9 cm | relaxed |
| 011 | banana | loaf | — | 0 | bowl 1.2 cm | loaf 3.9 cm | relaxed |
| 012 | banana | jug | — | 561 | banana (target) 7.0 cm, loaf 1.2 cm, bowl 1.2 cm | loaf 2.1 cm | hiding edit |
| 013 | block | loaf | loaf touches block | 0 | settling only (< 1 cm, 5°) | loaf 6.2 cm | relaxed |
| 014 | block | loaf | loaf touches block | 0 | bowl 1.2 cm | loaf 6.3 cm | relaxed |
| 015 | block | jug | — | 0 | bowl 1.2 cm | loaf 8.0 cm | relaxed |
| 016 | block | jug | — | 0 | bowl 1.2 cm | loaf 8.0 cm | relaxed |
| 017 | block | bowl | bowl touches block | 8821 | arrangement turned −90° (the camera looks from the data-set-2 right side), then settling only (< 1 cm, 5°) | bowl 2.5 cm | hand edit (turned, block is the target) |
| 018 | block | bowl | bowl touches block | 0 | settling only (< 1 cm, 5°) | bowl 6.3 cm | relaxed |
| 019 | block | grinder | loaf on block | 0 | settling only (< 1 cm, 5°) | loaf 3.1 cm | relaxed |
| 020 | block | shark | phone on block | 0 | shark 1.1 cm | phone 3.1 cm | hiding edit |
| 021 | block | phone, shark | — | 74 | block (target) 1.9 cm | phone 1.7 cm | hiding edit |
| 022 | block | phone, shark | — | 13 | block (target) 1.1 cm | phone 1.7 cm | hiding edit |
| 023 | banana | grinder | — | 0 | banana (target) 0.7 cm / 9°, phone 0.7 cm / 7° | phone 2.6 cm | relaxed |
| 024 | block | bowl | bowl on block | 0 | settling only (< 1 cm, 5°) | bowl 1.9 cm | relaxed |
| 025 | banana | grinder | — | 274 | phone 7.3 cm / 45°, glass 5.8 cm, banana (target) 1.5 cm / 46° | phone 2.6 cm | hand edit, hiding edit |
| 026 | block | mug | — | 0 | settling only (< 1 cm, 5°) | loaf 1.1 cm | hiding edit |
| 027 | block | grinder, stone coaster | stone coaster on block | 0 | settling only (< 1 cm, 5°) | stone coaster 8.6 cm | relaxed |
| 028 | block | stone coaster | stone coaster touches block | 0 | settling only (< 1 cm, 5°) | stone coaster 3.5 cm | relaxed |
| 029 | block | grinder | loaf touches block | 0 | settling only (< 1 cm, 5°) | loaf 1.7 cm | relaxed |
| 030 | star block | jug | — | 236 | star block (target) 7.0 cm | nothing | hiding edit |
| 031 | cup | bowl | phone on cup | 0 | settling only (< 1 cm, 5°) | phone 9.5 cm | relaxed |
| 032 | block | jug | wineglass on block | 0 | settling only (< 1 cm, 5°) | wineglass 5.8 cm | relaxed |
| 033 | block | grinder | loaf on block | 0 | settling only (< 1 cm, 5°) | loaf 3.1 cm | relaxed |
| 034 | block | mug | — | 0 | settling only (< 1 cm, 5°) | loaf 1.1 cm | hiding edit |
| 035 | block | vase | loaf touches block | 233 | settling only (< 1 cm, 5°) | loaf 6.9 cm | hiding edit |
| 036 | block | vase | loaf touches block | 233 | settling only (< 1 cm, 5°) | loaf 6.9 cm | hiding edit |
| 037 | block | vase | loaf touches block | 1 | vase 1.0 cm | loaf 7.0 cm | hiding edit |
| 038 | block | vase | loaf touches block | 2 | settling only (< 1 cm, 5°) | loaf 7.0 cm | relaxed |
| 039 | block | mug | loaf touches block | 95 | block (target) 9.4 cm / 45°, loaf 4.1 cm / 44° | loaf 7.3 cm | hiding edit |
| 040 | stone coaster | loaf | stone coaster touches vase | 12 | stone coaster (target) 1.6 cm, loaf 1.2 cm / 7°, vase 1.1 cm | vase 0.9 cm | hand edit + camera, hiding edit |
| 041 | stone coaster | block | stone coaster touches vase | 0 | stone coaster (target) 2.4 cm, loaf 1.1 cm / 10° | vase 0.7 cm | hiding edit |
| 042 | stone coaster | mug | vase on stone coaster | 0 | settling only (< 1 cm, 5°) | vase 1.3 cm | relaxed |
| 043 | stone coaster | mug | vase on stone coaster | 0 | loaf 1.1 cm / 10° | vase 1.6 cm | relaxed |
| 044 | stone coaster | mug | vase on stone coaster | 0 | loaf 1.1 cm / 10° | vase 1.6 cm | relaxed |
| 045 | stone coaster | loaf | vase on stone coaster | 0 | stone coaster (target) 2.6 cm, loaf 1.1 cm / 10°, vase 0.2 cm / 11° | vase 1.5 cm | hiding edit |
| 046 | stone coaster | mug | vase on stone coaster | 0 | loaf 1.1 cm / 10° | vase 4.2 cm | relaxed |
| 047 | stone coaster | mug | vase on stone coaster | 0 | loaf 1.1 cm / 10° | vase 4.2 cm | relaxed |
| 048 | stone coaster | block, vase | vase on stone coaster | 63 | settling only (< 1 cm, 5°) | vase 3.8 cm | hiding edit |
| 049 | block | jug | wineglass on block | 0 | settling only (< 1 cm, 5°) | wineglass 3.1 cm | relaxed |
| 050 | block | small bowl | — | 0 | settling only (< 1 cm, 5°) | nothing | relaxed |
| 051 | long block | small bowl | — | 0 | orange 0.4 cm / 18° | nothing | relaxed |
| 052 | long block | small bowl | — | 0 | bung 0.8 cm / 11°, orange 0.4 cm / 18° | nothing | relaxed |
| 053 | block | mug | bowl on block | 0 | loaf 0.9 cm / 10° | bowl 1.4 cm | relaxed |
| 054 | block | bowl | bowl on block | 0 | loaf 0.9 cm / 10° | bowl 1.9 cm | relaxed |
| 055 | banana | jug | loaf touches banana | 0 | bowl 1.4 cm | loaf 2.8 cm | relaxed |
| 056 | block | jug | — | 0 | bowl 1.4 cm | nothing | hiding edit |
| 057 | block | bowl, loaf | bowl on block | 0 | loaf 1.2 cm / 5° | bowl 3.1 cm | hiding edit |
| 058 | block | grinder | phone on block | 0 | settling only (< 1 cm, 5°) | phone 3.1 cm | relaxed |
| 059 | block | jug | loaf on block | 0 | settling only (< 1 cm, 5°) | loaf 3.1 cm | relaxed |
| 060 | block | mug | — | 0 | settling only (< 1 cm, 5°) | loaf 1.1 cm | relaxed |
| 061 | block | mug | — | 0 | stone coaster 4.4 cm | loaf 1.1 cm | hand edit |
| 062 | block | candle holder | stone coaster on block | 0 | block (target) 5.7 cm, stone coaster 5.7 cm | stone coaster 3.1 cm | hand edit |
| 063 | block | shark, stone coaster | stone coaster on block | 0 | settling only (< 1 cm, 5°) | stone coaster 2.7 cm | relaxed |
| 064 | block | shark, stone coaster | stone coaster on block | 0 | settling only (< 1 cm, 5°) | stone coaster 2.7 cm | relaxed |
| 065 | block | mug | — | 0 | banana 0.3 cm / 6° | banana 2.0 cm | relaxed |
| 066 | block | mug | — | 0 | banana 0.4 cm / 6° | banana 2.0 cm | relaxed |
| 067 | cup | bowl | wood coaster on cup | 0 | settling only (< 1 cm, 5°) | wood coaster 9.5 cm | relaxed |
| 068 | cup | jug | wood coaster on cup | 12 | settling only (< 1 cm, 5°) | wood coaster 9.5 cm | hiding edit |
| 069 | cup | jug | — | 39 | jug 3.0 cm, cup (target) 2.0 cm | nothing | hiding edit |
| 070 | block | jug | wineglass on block | 0 | settling only (< 1 cm, 5°) | wineglass 3.1 cm | relaxed |


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
* One design (scene040) cannot be realised from that camera at all (see the table); it uses the camera
  11 cm lower (8°, 1.5 m, same framing), where a small edit hides it and nothing else in the scene gets
  buried. scene017's loaf was a second such target, until the scene was turned to face the camera with the
  block as its target.

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
`experiments/data-set-3/edits.toml` (authoring errors, two designs turned to face the camera with the block
as the target, one camera exception) and then loops, per scene:

1. **Relax** (`sp relax`, `authoring/relax.py`). Authored poses are rarely a rest state: objects sit a few
   millimetres inside the table or each other, or float above it. Simulated directly under full gravity,
   every overlap becomes a depenetration push in the first step and leaning objects slide off their
   supports. Relaxation instead separates overlaps under almost no gravity (0.3 m/s²) with PhysX's
   depenetration speed capped at 5 cm/s, heavy damping and hard speed caps (10 cm/s, 45°/s), ramps
   gravity to 9.81 m/s² over 2 s, and only then settles under the unmodified model physics. The result
   is verified the way inference uses it: a 2.5 s rollout (one inference rollout) started fresh from the
   stored poses must move no object more than 1 mm / 0.5° (a fresh start re-establishes the contacts, which
   can shift a resting object by a millimetre: such states are polished by adopting the fresh rollout's
   rest); four rollouts from randomly nudged poses (±1 mm, 0.5°) test that the rest is robust. Any object that moves more than 3 cm or 30° on the way is reported (explosion guard): in
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
   (≤ 20 cm, 90°), cheapest first; moving the hidden target is cheap, moving anything visible (the
   occluder, or a cue that moves with the target) costs three times as much. Then back to 1.

   Every edit is tested by physics: if, after relaxing, something slides, a contact changes, an object the
   design shows loses its visibility, or a visible object ends more than 1.5 cm / 8° from where the edit
   put it (a cue re-settling into a different tilt), the edit is undone and the search continues without
   it. When moving the target alone lets a cue re-settle, only edits that take the cue along remain.

Scenes that pass are finalized (`sp finalize`): the target is the one hidden object and every other
object is observed, exactly as in data-set-2; priors are regenerated (`observed_objects` rule) and the
stimulus, segmentation, depth and target silhouette are rendered.

## Checks on the finished dataset

* **Target hidden**: the target has 0 pixels in the path-traced segmentation of every stimulus, and the ray caster finds no visible pixel even with the 2-pixel margin in all 70 scenes (66 also with a 4-pixel margin).
* **At rest**: a 2.5 s rollout (one inference rollout) from the stored truth moves no object more than 0.72 mm (median 0.010 mm); in data-set-2 the same rollout moved objects by up to 58 mm (median 6.0 mm).
* **Physical evidence kept**: removing the target makes another object move (≥ 5 mm) in 64 scenes (data-set-2: 61, where 003 and 017 had targets that touch nothing); where both datasets show evidence, it is the same object in 61 of 61. In the other 6 scenes the target touches nothing: they are pure occlusion trials, as in data-set-2.
* `sp validate data/data-set-3/scene0*`: all scenes valid (roles, priors, segmentation ids).
* **No trace of the target**: each stimulus was rendered again with the target removed. Two renders of the same scene differ by up to 11.2/255 (path-tracing noise, 9×9 local mean); without the target, 55 of 70 stimuli stay within twice that; in 003, 005, 013, 016, 017, 024, 025, 028, 039, 042, 043, 044, 051, 053, 055 the target's shadow changes the image (see Caveats).

## Caveats

* **Shadows of the hidden target** (15 scenes): the target is fully occluded, but it still blocks light, and in these scenes its soft shadow falls where the camera sees it (mostly from the rim light behind the table: switching that light off halves the effect). Rendering the hidden object without shadows and reflections (`[render] hidden_shadows = false` in `experiments/data-set-3.toml`, then `sp render` each scene) removes every trace (difference to the scene without the target within path-tracing noise), at the price of lighting that is subtly non-physical. The stimuli are rendered with physical shadows; which one the experiment should use is a design decision.
* **Camera exception**: scene040 uses the data-set camera lowered by 11 cm (8° elevation at 1.5 m, FOV 38.2°, same framing): from p01_f03 no physically valid arrangement hides its target behind its design occluder. Drop it if a single camera matters more.
* **Turned scenes**: 003 and 017 differ from data-set-2 by more than a small edit: the arrangement is turned (90°, −90°) and the target is the block instead of data-set-2's mug and loaf, so their data-set-2 results do not carry over. They were re-made on 2026-10-12 with `sp rework data/data-set-2/scene003 data/data-set-2/scene017 --exp data-set-3 --overwrite` and pass every check above: target 0 px in the stimulus and hidden with a 4-pixel margin, a 2.5 s rollout moves nothing more than 0.45 mm, no perturbation warnings, and without the block the bowl drops (2.9 cm / 18°, 2.5 cm / 21°); the block's shadow shows in both, as the old targets' did. The rework's render of 017 had low-resolution textures (captured before they finished streaming in) and was rendered again with `sp render data-set-3/scene017`: after a rework, look at the stimuli.
* **Fragile rests**: these scenes are at rest, but a ±1 mm / 0.5° nudge of every object moves something more than 5 mm (a cue balanced on the hidden object): 007 (block moves 6.7 mm); 018 (block moves 6.2 mm); 031 (phone moves 31.4 mm); 045 (stone coaster moves 5.2 mm).
* **Observed but not visible**, exactly as in data-set-2: the banana in 024, the key in 032 and 070, the phone and the can in 067, the phone in 068 are 0% visible but observed, so the model is given their poses. Mark them hidden with `sp roles <scene> --set NAME=hidden` if they should be inferred too.
* **Physics config**: data-set-3 uses the box table and 32/8 iterations (`experiments/data-set-3.toml`); data-set keeps its own physics. Adopting the same settings for data-set would mean re-settling its scenes (`sp rework` / `sp finalize`).
