# Composing hidden-object scenes

How to add scenes to a dataset such as data-set-4: one scene at a time, each built around a **physical
relationship** between a hidden target and the objects around it, iterated in a propose → check loop until it
passes every check below. Written for people and for the `scene-composer` agent
(`.claude/agents/scene-composer.md`), which follows this document.

Everything here was learned making data-set-4 (`docs/DATA_SET_4.md`); the numbers are from those runs.

## 1. The bar

A finished scene satisfies all of these (`sp compose` and `sp review` measure them):

| rule | means | measured as |
|---|---|---|
| 1. hidden | no pixel of the target in the 1280×960 stimulus, with a margin | 0 visible px and 0 px inside a 2 px margin (ray caster, stimulus resolution); 0 px in the RTX segmentation |
| 2. at rest | the truth is a rest state of the model's physics, robustly | a 2.5 s rollout from a fresh start moves nothing > 1 mm or 0.5°; 6 random pushes (±1 mm, ±0.5°) move nothing > 5 mm |
| 3. evidence | the target holds something up | without the target, some object moves ≥ 1 cm in one 2.5 s rollout |
| 4. occlusion alone fails | the target is not where an occlusion-only guess looks | the occlusion-only posterior (below): the largest hidden region is a decoy with ≥ 1.5× the mass of the target's region, and ≤ 25% of the mass lies within 4 cm of the target |
| 5. the rest is observed | every other object is visible and in frame | ≥ 10% of each other object visible; every object fully inside the image |

**Why rule 4.** The model's hypotheses for the hidden object are its XY position at its true height and
orientation (`proposer = "xy"`); its prior (`observed_objects`) puts a 10 cm Gaussian behind every visible
object. An observer that only knows what is hidden puts its mass wherever the target would be hidden and not
inside another object — mostly behind the biggest occluder. If the target is there, an occlusion-only model finds
it and the scene does not test physical reasoning. So the target hides in a **tight** spot, and a **decoy** (a
bigger hiding place, with nothing there that could produce the physical evidence) draws the occlusion-only guess.
`compose.occlusion_posterior` computes this posterior for any scene; on data-set-3, 36 of 70 targets sit in the
largest hidden region.

## 2. The loop

```
idea ─► spec (experiments/<exp>/sceneNNN.toml)
     ─► sp preview <spec> --exp <exp>              CPU, 20 s–2 min: placement, occlusion + margin (and where it
                                                    fails), visibility, overlaps, decoy. Repeat until clean, with
                                                    a few pixels to spare (relaxation moves things by millimetres).
     ─► sp compose <specs...> --exp <exp> --no-render
                                                    Isaac Sim, 3–5 min per scene (batch several specs per call):
                                                    relax, rest + robustness, evidence, occlusion, decoy, masses.
                                                    Repeat until PASS.
     ─► sp finalize authoring/<exp>/<name>.usd --exp <exp> --no-settle --target <target> --overwrite
                                                    writes data/<exp>/<name>/ and renders the stimulus (~30 s)
     ─► sp review <exp> <names...> --leak           re-checks the finished files, GIFs with/without the target,
                                                    shadow (leak) test; writes data/<exp>/review.md
     ─► look at render/rgb.png and render/counterfactual.gif; update docs; commit
```

Typical effort: 2–6 `sp compose` rounds per scene, 30–60 minutes of wall time per scene. Outputs of every
attempt: `authoring/<exp>/<name>.compose/` holds `report.txt` (which rule fails), `views.png` (scene camera, side,
top, the scene 2.5 s after removing the target, the occlusion-only map), `preview.png`/`margin.png` from
`sp preview`, and `authoring/<exp>/<name>.usd` (the relaxed, verified poses — what `sp finalize` uses).

**GPU rule.** Only one Isaac Sim process (or any GPU job) at a time on the local RTX 3070 — two at once can
crash the driver (`ERROR_DEVICE_LOST`). `sp preview` runs on the CPU and may run alongside. No multi-world
inference locally.

## 3. Spec reference

```toml
name = "scene011"
title = "Propped-up phone"                    # short, shown in reviews
relationship = "lean back: the phone (occluder) leans on the hidden target (can)"   # "<type>: <sentence>"
idea = """What a viewer sees, why it cannot be like that on its own, where the target must be, what the decoy
offers, and what happens without the target."""
target = "can"                                # the hidden object (one per scene)
touching = [["phone", "can"]]                 # contacts the design needs (checked after relaxation, 6 mm tolerance)
apart = [["left_phone", "right_phone"]]       # pairs that must not touch
gap = 0.0005                                  # m left between objects placed by drop/slide (default)

[[object]]                                    # objects are placed in this order: supports first
name = "phone"
asset = "f10_apple_iphone_4_v2"               # library asset (`sp library list`; table in section 10)
rotate = [["y", 90], ["z", 90], ["x", -20]]   # rotations of the resting asset about WORLD axes, in order (deg)
at = [-0.16, -0.12]                           # footprint centre on the table (m), after rotation
yaw = 0                                       # deg about z, applied after `rotate`
on = "table"                                  # what the drop must land on: "table", an object name, or "any"
slide = [{ dir = [0, 1], until = "can" }]     # then move along dir until touching `until` (or "any"); max = 0.5 m
scale = 1.0                                   # or [sx, sy, sz] in the resting frame (z = height)
# z = 0.75                                    # lowest point height instead of dropping
# offset = [0, 0, 0.001]                      # final small translation
# pose = [x, y, z, qx, qy, qz, qw]            # full pose instead of all of the above
# color = [0.9, 0.1, 0.1]                     # linear RGB override
```

* **Resting frame**: each library asset rests as it did in the scenes it came from (upright, footprint centred,
  bottom at z = 0, long side along x). Phones, coasters, blocks lie flat.
* **Rotation sign**: a positive angle about the axis `d × z` lifts the side of the object that faces the horizontal
  direction `d`. So `["x", a]` lifts the back (+y) side, `["x", -a]` the front, `["y", -a]` the right (+x) side,
  `["y", a]` the left; `[[dx, dy, 0] × z, a]`, i.e. `[[dy, -dx, 0], a]`, lifts the side facing `(dx, dy)`.
  "Tilt back" (top away from the camera) for something standing up = `["x", -a]`; "lean left" = `["y", -a]`.
* **Cookbook**: phone standing portrait, its face to the camera: `[["y", 90], ["z", 90]]`; coaster or block on edge
  facing the camera: `[["x", 90]]`; facing sideways (seen edge-on, for left/right leans): `[["y", 90]]`; upturned
  bowl: `[["x", 180]]`; rectangle block standing on end, thin side along x: `[["y", 90]]` (add `yaw = 90` to turn
  its 5 cm side to the camera).
* **`at` is the footprint (bounding box) centre after rotation**, not the body centre: a mug's body sits ~1.7 cm
  away from its handle side.
* **Drop**: from above everything, down until the first contact; place supports before what rests on them. A drop
  that lands on something other than `on` fails (it used to silently stack objects on their neighbours).
* **Slide**: stops at the first contact with *anything* (the log says "touches X before Y"); the pre-tilted pose
  is kept, and relaxation finds the real lean.
* The placement is only a start: relaxation (low gravity, then the model's physics) settles leans and stacks.

## 4. Geometry rules of thumb (this camera and table)

* Table top z = 0.732 m, edges at ±0.546 m; keep objects within about ±0.45 m and fully in frame. Camera
  `p01_f03` at (0, −1.66, 1.15) looking at (0, 0, 0.8): rays dip ~14° at the table centre (17° at the front,
  12° at the back), i.e. a slope of ~0.25.
* **Hidden behind an occluder of height H** if the target's top is below `H − 0.25·d`, with `d` the horizontal
  distance from the occluder's *far* top edge (for a mug or cup: its far rim; for a loaf: its ridge) to the
  target's far top edge. A can (8.3 cm) can hide close behind a mug (12.4 cm) but not behind a cup (9.4 cm).
* **Width**: the target must be narrower than its occluder by twice (margin + perspective). One stimulus pixel is
  ~1.1 mm at the table centre, so the 2 px margin is ~2–3 mm per side.
* **Perspective**: an object Δ further back looks smaller by Δ/1.7 m and shifts toward the image centre by
  x·Δ/1.7 m; to keep a hidden object centred behind an occluder at x ≠ 0, move it outwards by that much.
* Something raised behind an occluder (on a pedestal) shows only above the occluder's sight line; something
  low hides behind almost anything — which also makes every decoy region larger.
* Gaps under tilted objects let the camera see the table behind them: a hidden object in such a gap is not visible
  but its shadow is (see section 6).

## 5. Beating the occlusion-only guess (rule 4)

* **Tight hiding place**: an occluder only a little larger than the target (phone, coaster, cup, mug, a small
  bowl's rim, the gap between stacked plates). The target's region in the posterior is then small.
* **Strong decoys**
  * *Domes*: an upturned big bowl (`int_kitchen_accessories_le_creuset_bowl_30cm`, `[["x", 180]]`): its hidden
    region is right at its own prior candidate. It held 67% of the mass in 003 and 004.
  * *Wide boxes with a tall visible object right behind them*: `toaster` (37 cm wide) or `toaster_b` (24.5 cm
    tall) with `jug01` (29.5 cm) or `vase_05` standing just behind. The helper's prior candidate falls into the
    box's hidden region: 76–92% of the mass in six scenes (a toaster alone, in 001: 48%). The helper must stick
    out above the box enough to be ≥ 10% visible (`jug01` behind `toaster_b` shows ~19%) and stand ≥ 1 cm clear
    of the box's footprint (or the drop puts it on the box and it falls off).
* **Weak decoys**: gappy silhouettes (`coffee`: the camera sees between its tank and jug), objects that settle
  lower (`b03_loafbread` rolls 7° and sinks 6 mm when relaxed and stops hiding as much as the preview says),
  objects whose hidden region starts far from their centre (`b03_cooking_pot_01`: 16 cm).
* **Prior bookkeeping**: every visible object near the target adds a candidate there (two stacked plates = two
  candidates), so the decoy has to be bigger. Containers hide things inside only at the container floor's
  height (hypotheses keep the target's true height).
* Read the map in `views.png` / `preview.png` (blue = occlusion-only mass, red = the target) and the
  `occlusion only` line ("toaster 88%, mug 12% (target behind mug)").

## 6. Physics pitfalls and fixes

| symptom | cause | fix |
|---|---|---|
| removing the target moves nothing | the leaner stands on its own: tilted less than its tipping angle | tilt past atan(half thickness / COM height): coaster on edge 8.5°, phone 4°, rectangle block on end 21° → use 25–40° |
| removing the target moves nothing | something else also holds it (the occluder: dominoes resting on the mug) | keep ≥ 1 cm between cue and occluder; check `views.png` "without the target" |
| small evidence (≈ 1 cm) | two inward leaners just fall onto each other | a wide spacer (7.7 cm block) so they fall far, or another design |
| hidden support tips or slides when pushed | light or narrow support under a sideways push or an edge load (can under a plate, salt shaker pushed by a mill) | heavy, wide supports (`square_wood_block_v3`, 278 g, on its 7.7 cm face); symmetric loads (leaners from both sides); load through the support's centre (level contacts) |
| object falls off its pedestal when pushed | sphere on a small ring (orange on a can) | a wide ring: an orange in a cup's mouth |
| plate slides when pushed | plate tilted on two rims | equal-height supports, plate level |
| target shows in a thin line under a plate | PhysX contacts sit on convex hulls 1–3 mm off the visual mesh: a slit between plate and front rim; an identical hidden mug shows through it | hidden support slimmer at the same height: `scale = [0.9, 0.9, 1.0]` (lower makes the plate tilt) |
| target shows through an occluder | a mug or cup handle's hole seen side-on | handles point along +x of the asset; turn them toward the camera (`yaw = 270`) or straight back |
| counterweight never balances | PhysX masses come from the convex hulls × 1000 kg/m³: hollow things are heavy (smooth bowl 1.04 kg, big bowl 3.5 kg, mug 418 g) while hideable weights are ≤ 0.3 kg | not feasible with this library (needs authored masses or heavy small assets) |
| leak test flags the scene | the target darkens a visible gap (under a tilted vase or bowl rim) | cover the gap with a front object, or accept: only human viewers see shadows; `[render] hidden_shadows = false` removes them |
| preview passes, compose fails occlusion by 2–6 px | relaxation moves things by millimetres | keep ≥ 4 px spare in the preview (`margin4`) |
| object lands on a neighbour | drop position overlaps a neighbour's footprint (`at` is the bounding box centre) | move it, or set `on` deliberately |

Containers are hollow for PhysX (bowls, mugs, cups; the floor collides ~6 mm above the visual floor); the pot and
juicer are closed (lids); `jug04`, `jug05`, `candle`, `glass1` are partly filled (section 10).

## 7. Iterating fast

* `sp preview` before every `sp compose`: it catches placement, occlusion, visibility and decoy problems in
  seconds, on the CPU, while Isaac Sim may be busy. Its `margin.png` and "right edge 9, top 2" say which way
  to move the target or its occluder.
* Batch: `sp compose a.toml b.toml c.toml` (Isaac Sim starts once, ~30 s). Run the batch in the background and
  preview the next designs on the CPU meanwhile.
* Change one thing per round and say why in the spec (comments are fine).
* Look at `views.png` every round: the side view shows whether a lean really rests where intended; the
  "without the target" panel shows what the target holds; the map shows where an occlusion-only guess goes.
* When geometry is unclear, measure it: `uv run python -c` with `scene_physics.scene.library.read_asset(asset)`
  gives the mesh (extents, profile); a bowl's interior heights decide what fits under it.

## 8. Relationship catalogue

**In data-set-4** (one scene each):

| scene | type | sketch |
|---|---|---|
| 001 | lean back | phone stands tilted back, propped by a can behind it |
| 002 | mutual lean | house of cards: coaster leaning back on a smaller hidden coaster leaning forward |
| 003 | on top (dome) | small upturned bowl tilted, its rim on a block; big upturned bowl as decoy |
| 004 | spacer | top stone plate tilted on a block between two stacked plates |
| 005 | wedge | vase tilted by a battery under the back of its base |
| 006 | side lean | pepper mill leaning left onto a block behind a mug |
| 007 | perched | orange sitting in the mouth of a cup hidden behind a mug |
| 008 | bridge | plate across a mug and a hidden slimmer mug behind it |
| 009 | chain lean | two coasters as dominoes leaning right onto a hidden stop |
| 010 | two-sided spacer | two phones leaning inward, held apart by a hidden block |

**Ideas not yet built** (with what to watch out for):

1. **Chock on a slope**: an orange high on the far inner wall of the big bowl; a small block on the bowl floor
   below the sight line stops it rolling to the centre. Watch: the bowl's interior is a big hidden region — needs
   a strong decoy; the orange must stay visible above the rim.
2. **Ramp**: a stone plate lying with its back edge raised on a block under it (tilted toward the camera); or a
   sideways tilt with a front object covering the gap.
3. **Seesaw**: a long object on a hidden fulcrum behind an occluder, one end down on the table, the other in the
   air. Watch: long thin objects (knife, spoon) are hard to see from this camera.
4. **Crooked lid**: an upturned mug or bowl sitting crooked on a hidden object taller than its interior.
5. **Tipped cup**: a cup on its side, mouth facing away, propped up by an object hidden inside the mouth.
6. **Hidden middle link**: visible leans on hidden leans on visible (a longer domino chain with the hidden piece
   in the middle).
7. **Three-legged tray**: a plate on three supports, one hidden; without it the plate tilts.
8. **Friction sandwich**: something held upright, at an angle it could not keep alone, between the occluder and a
   hidden block.
9. **Ladder foot**: a flat object leaning at a shallow angle against a visible object, its foot (behind a front
   occluder) stopped from sliding out by a hidden chock.
10. **Fruit raised in a bowl**: an orange visibly higher in a bowl than it should be, sitting on a hidden block.
11. **Lean forward**: a tall object behind the occluder leans toward the camera onto the hidden target.
12. **Needs new assets**: books and bookends (a row leaning on a hidden bookend), a cutting board or picture frame
    propped up, a lid propped open, a heavy small weight (counterweights over the table edge), a bread bin (a
    decoy that is not a toaster).

Vary the rest too: left / right / front / back of the table, lean direction (back, left, right, forward), height
(on the table, raised, inside a container), and the decoy (data-set-4 overuses the toaster).

## 9. Finishing a batch

1. `sp finalize ... --no-settle` for every passing scene (the authoring file holds the verified poses), then
   `sp review <exp> <names> --leak`.
2. Look at every `render/rgb.png`: does it look like a normal table? Is the cue (the tilt, the overhang, the gap)
   visible? Watch `render/counterfactual.gif`: does the right thing fall?
3. Update `docs/DATA_SET_<n>.md` (a row per scene: title, relationship, target, what hides it, evidence, decoy,
   leak) and commit the specs and docs on a branch (data under `data/` and `authoring/` is not in git).

## 10. Asset reference

Size in the resting frame (x × y × z, cm), PhysX mass (from the collision hulls at 1000 kg/m³), and for
containers where a dropped cork comes to rest inside (≈ the collision floor; near the top = closed or filled).

| asset | size (cm) | PhysX mass (g) | notes |
|---|---|---|---|
| `_9v_battery` | 2.7 × 1.8 × 5.2 | 23 | standing upright |
| `b03_cocacola_can_cage` | 4.4 × 4.4 × 8.3 | 112 | light: tips under side loads |
| `b03_cooking_pot_01` | 33.4 × 32.7 × 24.1 | 12617 | lid is part of the mesh: closed |
| `b03_loafbread` | 34.6 × 14.8 × 11.1 | 2483 | rolls ~7° and sinks ~6 mm when relaxed |
| `b03_loafbread_v2` | 34.7 × 14.9 × 10.6 | 2600 | |
| `b03_loafbread_v3` | 34.8 × 14.8 × 8.3 | 2220 | |
| `b03_loafbread_v4` | 32.4 × 11.6 × 6.7 | 1293 | |
| `b04_bowl_smooth` | 24.0 × 24.0 × 8.2 | 1041 | hollow (floor at 1.6 cm); upturned interior 7 cm high at the centre |
| `b04_candle_holder_metal` | 11.3 × 11.3 × 21.7 | 390 | gappy (candles) |
| `b04_wineglass` | 6.0 × 6.0 × 20.9 | 191 | opaque red |
| `b05_coffee_grinder` | 16.6 × 11.5 × 18.0 | 1227 | |
| `b05_coffee_grinder_v2` | 18.0 × 14.8 × 19.4 | 1857 | |
| `banana_fix2` | 21.5 × 9.6 × 4.9 | 347 | |
| `banana_fix2_v2` | 22.0 × 9.9 × 7.4 | 502 | |
| `basic_cork` | 3.8 × 2.2 × 2.2 | 14 | rolls |
| `bee` | 16.6 × 15.0 × 16.7 | 822 | toy |
| `bung` | 5.3 × 4.7 × 5.3 | 74 | rubber stopper on its side |
| `candle` | 17.4 × 17.4 × 21.7 | 1769 | jar, filled to 14.5 cm |
| `champagne_cork` | 3.5 × 2.3 × 2.3 | 9 | rolls |
| `coffee` | 32.7 × 16.5 × 34.4 | 11860 | coffee maker; gappy silhouette |
| `coffeecup004_fix` | 10.2 × 8.5 × 9.4 | 193 | hollow (floor 1.6 cm); handle at +x |
| `coffeemug` | 13.0 × 9.6 × 12.4 | 418 | hollow; handle at +x; see-through handle |
| `coffeemug_v2` | 14.9 × 11.0 × 14.3 | 658 | hollow; handle at +x |
| `cylinder01` | 4.0 × 4.0 × 13.3 | 117 | pepper mill |
| `dining_room_table` | 109.2 × 109.2 × 73.2 | static | top at 0.732 m |
| `f10_apple_iphone_4` | 11.6 × 5.9 × 0.9 | 60 | lying flat |
| `f10_apple_iphone_4_v2` | 12.9 × 6.6 × 1.1 | 84 | |
| `f10_apple_iphone_4_v3` | 11.6 × 5.9 × 1.2 | 74 | |
| `glass1` | 7.5 × 7.5 × 15.8 | 156 | transparent (does not hide anything) |
| `glass2` | 5.6 × 5.6 × 10.8 | 71 | wine glass, opaque green |
| `heart` | 23.0 × 7.0 × 7.1 | 296 | |
| `int_kitchen_accessories_le_creuset_bowl_30cm` | 36.5 × 31.0 × 12.9 | 3528 | hollow (floor 2.4 cm); the best dome decoy upturned |
| `int_kitchen_accessories_le_creuset_bowl_30cm_v2` | 36.5 × 31.0 × 10.3 | 2936 | |
| `int_kitchen_accessories_le_creuset_bowl_30cm_v3` | 33.1 × 28.3 × 13.9 | 2772 | |
| `jug01` | 17.6 × 15.2 × 29.5 | 3989 | pitcher; tall decoy helper |
| `jug02` | 11.5 × 11.0 × 18.6 | 1202 | |
| `jug02_v2` | 13.7 × 13.1 × 22.2 | 2020 | |
| `jug03` | 13.7 × 13.7 × 20.8 | 1502 | hollow from 4.5 cm |
| `jug04` | 14.1 × 14.1 × 16.1 | 1669 | filled to 9.3 cm |
| `jug04_v2` | 15.3 × 15.3 × 17.6 | 2145 | |
| `jug04_v3` | 19.1 × 19.1 × 21.8 | 4133 | |
| `jug04_v4` | 18.5 × 18.5 × 21.2 | 3785 | |
| `jug05` | 17.3 × 17.3 × 13.7 | 2230 | filled to 7.2 cm |
| `juicer` | 20.5 × 20.5 × 28.6 | 3020 | closed |
| `kettle` | 19.5 × 14.8 × 25.5 | 2708 | |
| `key_brass` | 5.0 × 1.6 × 0.2 | 1 | too thin to see or to lift anything |
| `knife1` | 17.9 × 1.7 × 0.3 | 8 | barely visible from this camera |
| `orange` | 8.6 × 8.2 × 8.4 | 245 | rolls |
| `pentagon_wood_block` | 8.0 × 7.6 × 2.7 | 115 | |
| `pepper` | 5.0 × 5.0 × 11.9 | 111 | salt shaker; light, tips |
| `rectangle_wood_block` | 7.6 × 5.0 × 3.0 | 115 | domino shape on end |
| `round_coaster_stone` | 10.0 × 10.0 × 1.5 | 102 | disc: rolls on its rim |
| `round_coaster_stone_v2` | 20.1 × 20.1 × 3.2 | 822 | stone plate |
| `shark` | 21.5 × 12.8 × 9.7 | 551 | toy |
| `shark_v2` | 23.3 × 14.1 × 11.5 | 739 | |
| `shark_v3` | 29.6 × 17.9 × 14.6 | 1511 | |
| `spoon1` | 14.9 × 2.9 × 2.0 | 10 | |
| `square_coaster_001_wood` | 10.0 × 10.0 × 1.5 | 134 | card / domino on edge |
| `square_wood_block` | 7.7 × 7.7 × 3.1 | 183 | |
| `square_wood_block_v2` | 5.5 × 5.5 × 2.2 | 68 | |
| `square_wood_block_v3` | 7.7 × 7.7 × 4.7 | 278 | the sturdy hidden support (on its 7.7 face) |
| `square_wood_block_v4` | 7.6 × 5.1 × 3.6 | 113 | |
| `square_wood_block_v5` | 9.9 × 9.9 × 3.1 | 302 | |
| `square_wood_block_v6` | 7.5 × 5.9 × 3.1 | 136 | |
| `square_wood_block_v7` | 10.0 × 9.0 × 3.1 | 276 | |
| `square_wood_block_v8` | 9.1 × 7.7 × 3.1 | 216 | |
| `star_wood_block` | 7.6 × 3.0 × 7.6 | 161 | stands on edge |
| `stelton_emma_tea_vacuum_jug` | 15.7 × 13.7 × 18.4 | 668 | |
| `teakettle_01` | 20.1 × 20.1 × 23.5 | 1449 | |
| `toaster` | 37.3 × 17.0 × 18.0 | 2252 | wide solid box: decoy |
| `toaster_b` | 28.7 × 18.6 × 24.5 | 2090 | tall solid box: decoy |
| `vase_05` | 18.0 × 18.0 × 26.5 | 1305 | wide base (14 cm); decoy helper |
| `vase_05_v2` | 22.7 × 22.7 × 33.3 | 2698 | |
| `vase_05_v3` | 20.9 × 20.9 × 30.8 | 2125 | |

Objects keep their library colour in every scene (one categorical colour per asset), so the same asset looks the
same everywhere. New assets would come from Blender exports through `sp import` and `sp library extract`.
