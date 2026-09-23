# Legacy data audit

What the new checks found in the legacy `data-pipeline` scenes when `sp migrate` converted them
(September 2026): `data-set` (52 scenes) and `data-set-2` (5 scenes). Migration keeps every pose exactly
as authored and records what the model's physics does to each object (`sp:settle:*`), so nothing below
has been changed in the data. Each finding comes with the command that fixes it, if you want it fixed.

## Summary

* All 57 scenes converted and pass `sp validate`; stimuli, segmentation, depth and target images were
  re-rendered in Isaac Sim.
* **Physics.** Of 328 dynamic objects, 275 stay within 5 mm / 2° of their authored pose under the model's
  physics. 36 settle a little (up to 10 mm and 10°), mostly loaves and bowls that Blender left a few
  millimetres into or above the table. 17 move a lot: a stone coaster floating 44 mm above the table
  (scene041), a bee figurine that falls backwards (scene027, scene031), a heart that topples and slides
  (scene015), corks and oranges that roll, a candle holder pushed out of a coaster it intersects (scene042).
* **These are the authored poses, not the collision shapes.** The bee, heart, star block, cork, bowl and
  loaf scenes (027, 015, 019, 035, 001) were re-simulated with convex-hull, convex-decomposition and SDF
  colliders, and the objects move the same way under all three (bee 68–69°, heart 116–131 mm, cork
  8–11 mm, star block 80°, bowl 14 mm, loaf 4°). The exception is scene042's candle holder, which
  intersects a coaster: interpenetrating arrangements have no well-defined outcome (12 mm with hulls,
  47 mm with decomposition, still moving with SDF).
* **Interpenetration.** 14 object pairs overlap at the authored poses, mostly objects sunk 1–7 mm into
  the table (the loaf in scenes 001/051/052 and data-set-2 is 7 mm deep at one end).
* **Roles.** 12 annotations disagree with what the camera sees. In **scene051 the roles are swapped**: the
  makeup marks `jug04_006` as hidden and as the target although it is fully visible, and
  `square_wood_block_013`, which is fully hidden, as observed (its `target.json` names the block). The
  old sampler proposed poses for the makeup's hidden objects, so old runs on scene051 inferred the
  visible jug. Nine other objects marked observed are 0–3% visible: the old model was given their
  poses although the camera cannot see them. In data-set-2/scene002 the hidden target is 29% visible.

## What to do about it

| Finding | Effect on the model | Fix |
|---|---|---|
| scene051 roles swapped | infers the wrong object | `sp roles data-set/scene051 --measure --target square_wood_block_013`, then `sp render data-set/scene051` |
| observed objects the camera cannot see | the model knows poses a viewer cannot | if they should be inferred: `sp roles <scene> --measure` |
| objects that move by themselves | the truth is not a rest state (the `sp infer` truth check reports it); the motion is shared by every hypothesis unless a hypothesis touches the moving object | keep the stimulus as participants saw it, or re-author: `sp import data/data-set/<scene>`, fix it in `sp open`, `sp finalize --overwrite` (the settled poses become the truth, the stimulus is re-rendered) |
| interpenetration | rollouts from the truth start with a depenetration push | as above; `sp check` lists the overlapping pairs |

Re-authoring changes the stimulus, so decide per scene whether matching the human data or a physically
consistent truth matters more.

## How this was measured

* Physics: the default `[physics]` config (PhysX 5 GPU, TGS, 120 Hz, convex decomposition for dynamic
  objects, the table's triangle mesh), simulated until three consecutive 0.1 s windows move less than
  5 mm/s and 0.05 rad/s (at most 5 s), then 2 s more to measure residual motion.
* Visibility: the model's ray caster at 640×480 from each scene's camera; hidden means under 10% of the
  object's silhouette is visible.
* Interpenetration: `authoring.analysis.overlaps` (1024 points inside each object, winding-number tests
  against the others).
* To reproduce for one scene: `sp import data/data-set/scene042`, then `sp check data-set/scene042`
  (report, before/after renders, settle GIF). The tables below come from the `sp:settle:*`,
  `sp:visibility:*` and `sp:role` attributes of the migrated files.

### Objects that move a lot (> 10 mm or > 10°)

| scene | object | role | moves | rotates | what happens |
|---|---|---|---:|---:|---|
| data-set/scene015 | heart | observed | 116 mm | 48 deg | tips / rolls |
| data-set/scene042 | b04_candle_holder_metal_002 | observed | 47 mm | 9 deg | settles |
| data-set/scene041 | round_coaster_stone_004 | observed | 44 mm | 0 deg | floats 44 mm above the table and drops |
| data-set/scene031 | bee | observed | 30 mm | 68 deg | tips / rolls |
| data-set/scene027 | bee | observed | 29 mm | 68 deg | tips / rolls |
| data-set/scene015 | star_wood_block | observed | 14 mm | 80 deg | rests 25 mm above the table (on or against another object) and moves |
| data-set/scene036 | int_kitchen_accessories_le_creuset_bowl_30cm_005 | observed | 14 mm | 2 deg | rests 10 mm above the table (on or against another object) and moves |
| data-set/scene035 | int_kitchen_accessories_le_creuset_bowl_30cm_005 | observed | 14 mm | 2 deg | rests 10 mm above the table (on or against another object) and moves |
| data-set/scene033 | b03_loafbread_008 | observed | 9 mm | 10 deg | tips / rolls |
| data-set/scene034 | b03_loafbread_008 | observed | 9 mm | 10 deg | tips / rolls |
| data-set/scene029 | bung | observed | 8 mm | 11 deg | tips / rolls |
| data-set/scene019 | bung | observed | 8 mm | 11 deg | tips / rolls |
| data-set/scene025 | bung | observed | 8 mm | 11 deg | tips / rolls |
| data-set/scene029 | orange | observed | 3 mm | 18 deg | tips / rolls |
| data-set/scene028 | orange | observed | 3 mm | 18 deg | tips / rolls |
| data-set/scene016 | orange | observed | 3 mm | 18 deg | tips / rolls |
| data-set/scene007 | banana_fix2_004_001 | observed | 0 mm | 10 deg | tips / rolls |

### Objects that settle a little (5–10 mm or 2–10°)

| scene | object | role | moves | rotates | what happens |
|---|---|---|---:|---:|---|
| data-set-2/scene001 | b03_loafbread_008 | observed | 3.9 mm | 4.0 deg | settles |
| data-set-2/scene002 | square_wood_block_013_001 | hidden | 5.6 mm | 0.1 deg | floats 6 mm above the table and drops |
| data-set-2/scene002 | int_kitchen_accessories_le_creuset_bowl_30cm_005_001 | observed | 3.9 mm | 4.7 deg | settles |
| data-set-2/scene002 | b03_loafbread_008_001 | observed | 3.9 mm | 4.0 deg | settles |
| data-set-2/scene003 | square_wood_block_013_002 | observed | 5.6 mm | 0.0 deg | floats 6 mm above the table and drops |
| data-set-2/scene003 | int_kitchen_accessories_le_creuset_bowl_30cm_005_002 | observed | 4.0 mm | 4.6 deg | settles |
| data-set-2/scene003 | b03_loafbread_008_002 | observed | 4.0 mm | 4.1 deg | settles |
| data-set-2/scene004 | square_wood_block_007_001 | hidden | 4.7 mm | 3.6 deg | settles |
| data-set-2/scene004 | banana_fix2_003_001 | observed | 2.7 mm | 5.9 deg | settles |
| data-set-2/scene004 | int_kitchen_accessories_le_creuset_bowl_30cm_001_001 | observed | 3.9 mm | 2.4 deg | settles |
| data-set-2/scene005 | int_kitchen_accessories_le_creuset_bowl_30cm_005_004 | observed | 4.4 mm | 4.9 deg | settles |
| data-set-2/scene005 | square_wood_block_013_005 | hidden | 6.1 mm | 0.9 deg | floats 6 mm above the table and drops |
| data-set/scene001 | b03_loafbread_008 | observed | 3.9 mm | 4.0 deg | settles |
| data-set/scene002 | shark | observed | 2.3 mm | 2.3 deg | settles |
| data-set/scene004 | b03_loafbread_002 | observed | 8.6 mm | 2.9 deg | settles |
| data-set/scene005 | b03_loafbread_003 | observed | 3.7 mm | 2.3 deg | settles |
| data-set/scene006 | int_kitchen_accessories_le_creuset_bowl_30cm_001_001 | observed | 3.9 mm | 2.4 deg | settles |
| data-set/scene006 | square_wood_block_007_001 | hidden | 4.7 mm | 3.6 deg | settles |
| data-set/scene006 | banana_fix2_003_001 | observed | 2.7 mm | 5.2 deg | settles |
| data-set/scene008 | b03_loafbread_005_001 | observed | 6.0 mm | 4.2 deg | settles |
| data-set/scene020 | champagne_cork | observed | 1.5 mm | 5.8 deg | settles |
| data-set/scene021 | champagne_cork | observed | 1.6 mm | 5.9 deg | settles |
| data-set/scene035 | b03_loafbread_008 | observed | 3.5 mm | 5.1 deg | rests 5 mm above the table (on or against another object) and moves |
| data-set/scene036 | b03_loafbread_008 | observed | 3.5 mm | 5.1 deg | rests 5 mm above the table (on or against another object) and moves |
| data-set/scene038 | shark_003 | observed | 4.6 mm | 2.3 deg | rests 2 mm above the table (on or against another object) and moves |
| data-set/scene040 | b03_loafbread_002 | observed | 8.5 mm | 2.8 deg | settles |
| data-set/scene041 | b03_loafbread_002 | observed | 8.5 mm | 2.8 deg | settles |
| data-set/scene042 | b03_loafbread_003 | observed | 3.7 mm | 2.3 deg | settles |
| data-set/scene043 | b03_loafbread_003 | observed | 3.7 mm | 2.3 deg | settles |
| data-set/scene044 | b03_loafbread_003 | observed | 3.7 mm | 2.3 deg | settles |
| data-set/scene045 | square_wood_block_007_001 | hidden | 4.9 mm | 3.6 deg | settles |
| data-set/scene045 | banana_fix2_003_001 | observed | 2.8 mm | 5.8 deg | settles |
| data-set/scene046 | banana_fix2_003_001 | observed | 5.0 mm | 5.6 deg | settles |
| data-set/scene046 | square_wood_block_007_001 | hidden | 6.9 mm | 3.6 deg | settles |
| data-set/scene051 | b03_loafbread_008 | observed | 3.9 mm | 4.0 deg | settles |
| data-set/scene052 | b03_loafbread_008 | observed | 3.9 mm | 4.0 deg | settles |

### Roles that disagree with the camera (hidden = under 10% visible)

| scene | object | legacy role | visible |
|---|---|---|---:|
| data-set/scene016 | glass2 | observed | 0.0% |
| data-set/scene021 | star_wood_block | observed | 0.0% |
| data-set/scene021 | basic_cork | observed | 0.0% |
| data-set/scene023 | b03_cocacola_can_cage | observed | 0.0% |
| data-set/scene024 | pepper | observed | 3.1% |
| data-set/scene024 | basic_cork | observed | 0.0% |
| data-set/scene025 | b03_cocacola_can_cage | observed | 0.0% |
| data-set/scene030 | basic_cork | observed | 0.0% |
| data-set/scene050 | key_brass_001 | observed | 0.0% |
| data-set/scene051 | square_wood_block_013 | observed | 0.0% |
| data-set/scene051 | jug04_006 | hidden | 100.0% |
| data-set-2/scene002 | square_wood_block_013_001 | hidden | 29.1% |

### Interpenetrating at the authored poses (≥ 1% of an object and ≥ 1 cm³)

| scene | objects | overlap |
|---|---|---:|
| data-set/scene001 | b03_loafbread_008 / dining_room_table_018 | 48.3 cm^3 (2%) |
| data-set/scene002 | f10_apple_iphone_4 / square_wood_block_001 | 2.3 cm^3 (3%) |
| data-set/scene007 | banana_fix2_004_001 / dining_room_table_008_001 | 20.7 cm^3 (5%) |
| data-set/scene010 | f10_apple_iphone_4_003 / dining_room_table_012 | 6.8 cm^3 (9%) |
| data-set/scene038 | f10_apple_iphone_4_004 / square_wood_block_001 | 2.5 cm^3 (3%) |
| data-set/scene042 | b04_candle_holder_metal_002 / round_coaster_stone_001 | 6.6 cm^3 (5%) |
| data-set/scene045 | b05_coffee_grinder_004_001 / dining_room_table_007_001 | 8.8 cm^3 (2%) |
| data-set/scene046 | b05_coffee_grinder_004_001 / dining_room_table_007_001 | 8.8 cm^3 (2%) |
| data-set/scene047 | f10_apple_iphone_4_003 / dining_room_table_012 | 6.7 cm^3 (9%) |
| data-set/scene051 | b03_loafbread_008 / dining_room_table_018 | 48.3 cm^3 (2%) |
| data-set/scene052 | b03_loafbread_008 / dining_room_table_018 | 48.3 cm^3 (2%) |
| data-set-2/scene002 | b03_loafbread_008_001 / dining_room_table_018_001 | 54.2 cm^3 (2%) |
| data-set-2/scene002 | vase_05 / dining_room_table_018_001 | 46.2 cm^3 (15%) |
| data-set-2/scene003 | b03_loafbread_008_002 / dining_room_table_018_002 | 48.0 cm^3 (2%) |
