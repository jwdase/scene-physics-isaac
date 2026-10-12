# data-set-4

data-set-4 is a first set of **new** scenes (not derived from data-set-2), composed one at a time with
`sp compose` in a propose → relax → check loop. Every scene is built around a physical relationship
between the hidden target and the object that hides it (or a visible cue resting on it), and each scene
uses a different kind of relationship: leans in several directions, mutual support, things resting on or
under the target, a wedge, a bridge, a chain.

## What every scene satisfies

1. **The target is fully hidden**: no pixel of it shows in the 1280×960 stimulus, with a 2-pixel margin of
   occluder around it (as in data-set-3).
2. **The truth is a verified rest state** under the model's physics (the data-set-3 physics: box table,
   32/8 solver iterations): a 2.5 s inference rollout from a fresh start moves nothing by more than 1 mm,
   and small random pushes (±1 mm, ±0.5°) do not make anything slide or fall (perturbation ≤ 5 mm).
3. **The target leaves physical evidence**: simulated without the target, the object it holds up moves by
   at least 1 cm in one inference rollout (in most scenes by several centimetres).
4. **Occlusion alone does not find the target.** The model's hypotheses for the hidden object are its XY
   position at its true height and orientation, with the prior `observed_objects` (a Gaussian, 10 cm, behind
   every visible object). Weighting every hypothesis on a 1.5 cm grid by that prior and keeping those that
   would be hidden and do not overlap another object gives the posterior of an observer that only knows
   what is hidden. In every scene the largest hidden region under that posterior is a **decoy** — behind or
   under an object that offers far more room to hide than the target's own hiding place (a toaster, a big
   upturned bowl) — holding at least 1.5 times the mass of the target's region, and at most a quarter of
   the mass lies within 4 cm of the target. If physics and occlusion pointed to the same place, an
   occlusion-only guess would find the target and the scene would not test physical reasoning.
5. **Every other object is visible** (at least 10% of it, so its pose is observed) and fully in frame.

Rule 4 shapes the designs: the target hides in a *tight* spot (just behind a phone, a coaster, a cup or a
mug, under the rim of a small bowl, between two plates), and a bigger hiding place elsewhere on the table
attracts an occlusion-only guess. Domes (an upturned bowl) and wide objects with a visible object standing
right behind them (whose prior candidate then falls into the hidden region) make the strongest decoys.

## Scenes

Hides it: the object that covers the target in the stimulus. Without the target: the object that moves most in one inference rollout (2.5 s) when the target is taken away. Occlusion only: the occlusion-only posterior (rule 4) — the region holding the most mass (the decoy), the target's own region, and the mass within 4 cm of the target.

| scene | title | relationship | target | hides it | without the target | occlusion only: largest region | target's region | within 4 cm |
|---|---|---|---|---|---|---|---|---|
| 001 | Propped-up phone | lean back | can | phone | phone 7.5 cm / 69° | toaster 48% | 24% | 5% |
| 002 | House of cards | mutual lean | back coaster | front coaster | front coaster 5.2 cm / 68° | toaster 88% | 7% | 4% |
| 003 | Shell game | on top | block | bowl | bowl 3.1 cm / 12° | big bowl 67% | 33% | 3% |
| 004 | Stacked plates | spacer | block | top plate | top plate 2.5 cm / 16° | big bowl 67% | 31% | 9% |
| 005 | Shimmed vase | wedge | battery | vase | vase 1.5 cm / 13° | loaf 60% | 22% | 8% |
| 006 | Leaning pepper mill | side lean | block | mug | mill 1.4 cm / 53° | toaster 88% | 12% | 4% |
| 007 | Perched orange | on top | cup | mug | orange 7.5 cm / 25° | toaster 92% | 8% | 3% |
| 008 | Plate on two mugs | bridge | back mug | front mug | plate 3.5 cm / 74° | toaster 90% | 9% | 5% |
| 009 | Dominoes | chain lean | stop | mug | domino b 5.3 cm / 65° | toaster 89% | 11% | 2% |
| 010 | Leaning phones | spacer | block | mug | left phone 1.4 cm / 12° | toaster 76% | 24% | 0% |

**scene001 — Propped-up phone.** A phone stands upright, tilted back 20 degrees, with nothing visible holding it. A phone cannot stand like that on its own: something hidden behind it props it up. The toaster hides a much larger area, but nothing behind the toaster could hold the phone. Without the can the phone falls flat on its back.

**scene002 — House of cards.** A square coaster stands on its edge, tilted back. Behind it a second, smaller coaster leans forward against it, like two playing cards in a house of cards. Each holds the other up: without the hidden one the visible coaster falls flat on its back. The toaster would hide a coaster far more easily, but could not hold this one up.

**scene003 — Shell game.** Two bowls lie upside down. The big one lies flat and could hide almost anything underneath it; the small one is tilted, its back rim lifted off the table. A bowl turned over sits flat unless something under its rim holds that side up, so the block is under the small bowl, not under the big one. Without the hidden block the small bowl drops flat.

**scene004 — Stacked plates.** Two stone plates lie on top of each other, but the top one is tilted towards the viewer, its back edge raised. A plate stacked on a plate lies flat; something hidden between the two holds the back edge up. The big upturned bowl could hide the block far more easily, but nothing under it could lift the plate. Without the block the top plate drops flat onto the lower one.

**scene005 — Shimmed vase.** A vase leans to the left and forward. A vase with a wide flat base stands upright on a flat table, and nothing visible touches this one: it is tilted because something is wedged under the back right of its base, out of sight behind the vase itself. The loaf hides a far larger area, but nothing behind the loaf could tilt the vase. Without the battery the vase rocks back upright.

**scene006 — Leaning pepper mill.** A pepper mill leans to the left with its top behind a mug. A mill tilted that far cannot stand on its own; it rests on something hidden behind the mug. The toaster hides far more of the table, but it is too far away to hold the mill. Without the block the mill falls over to the left.

**scene007 — Perched orange.** An orange sits up in the air just behind a mug, higher than the mug's rim and not touching it. An orange cannot stay up there on its own: it sits on something hidden behind the mug. The toaster hides far more of the table, but nothing behind it could hold the orange up. Without the cup the orange drops behind the mug.

**scene008 — Plate on two mugs.** A stone plate lies flat on top of a mug, but most of it reaches out behind the mug. Balanced on one mug it would tip off backwards: it also rests on something hidden behind the mug, exactly as high as the mug's rim (a second, slimmer mug). The toaster hides far more of the table, but nothing behind it could hold the plate. Without the hidden mug the plate tips off backwards.

**scene009 — Dominoes.** Two coasters stand on their edges and lean to the right like falling dominoes, the first on the second and the second on something hidden behind the mug. A coaster tilted that far cannot stand on its own. The toaster hides far more of the table, but it is too far away to stop them. Without the hidden block both fall to the right.

**scene010 — Leaning phones.** Two phones stand on end and lean towards each other, the left one to the right and the right one to the left, but their tops do not touch: something hidden behind the mug keeps them apart and holds both up. The toaster hides far more of the table, but nothing behind it could hold the phones. Without the block both fall inwards.


## How the scenes were made (and how to make more)

The full process, with the pitfalls met along the way, ideas for new relationships and an asset table, is in
[`docs/COMPOSING_SCENES.md`](COMPOSING_SCENES.md); the `scene-composer` agent (`.claude/agents/`) follows it.
In short:

A scene is a spec in `experiments/data-set-4/sceneNNN.toml`: the objects from the library (`sp library list`)
in placement order, each with a rotation of its resting pose, a footprint position, and optional slides:

```toml
[[object]]
name = "phone"
asset = "f10_apple_iphone_4_v2"
rotate = [["y", 90], ["z", 90], ["x", -20]]   # world-axis rotations of the resting asset, in order (deg)
at = [-0.16, -0.12]                           # footprint centre on the table (m)
slide = [{ dir = [0, 1], until = "can" }]     # then move along dir until it touches the can
```

Each object is dropped from above until it touches what is below it (`on` names what that must be, so a
misplaced object fails instead of landing on a neighbour), then slid until contact; contacts are found by
ray casting surface samples against the other meshes. `sp compose <spec> --exp data-set-4` then

* relaxes the arrangement (`sp relax`: low gravity, capped speeds, then the model's physics) and verifies
  the rest state with fresh-start rollouts and perturbations,
* measures the target's occlusion with the margin, the other objects' visibility, the contacts the spec
  asks for (`touching`, `apart`), the physical evidence (a rollout without the target) and the
  occlusion-only posterior (rule 4),
* writes `authoring/data-set-4/<name>.usd` (relaxed poses, roles pinned) and a report with pictures in
  `authoring/data-set-4/<name>.compose/`: the scene camera, a side view, a top view, the scene after a
  rollout without the target, and the occlusion-only map;
* with `--finalize`, a scene that passes every check is written to `data/data-set-4/<name>/` with its
  stimulus.

The loop for a new scene: write the spec, check it in seconds with `sp preview` (CPU: placement, occlusion
and where it fails, visibility, overlaps, decoy), then run `sp compose --no-render` (physics), read
`report.txt` and look at `views.png`, edit the spec, repeat. `sp review data-set-4 --leak` re-checks the
finished scenes and writes `data/data-set-4/review.md` and a with/without-target GIF per scene.

## Files

* `experiments/data-set-4/sceneNNN.toml`: the specs (the design, its idea and relationship, in words).
* `data/data-set-4/sceneNNN/`: the finished scenes; `render/rgb.png` is the stimulus and
  `render/counterfactual.gif` shows the scene simulated with (left) and without (right) the target.
* `authoring/data-set-4/sceneNNN.compose/`: the last `sp compose` report and pictures.

## Checks

* **Hidden**: the target has 0 pixels in the path-traced segmentation of every stimulus; the ray caster finds no visible pixel with the 2-pixel margin in all 10 scenes (8 also with 4 pixels).
* **At rest**: a 2.5 s rollout from the stored truth moves nothing by more than 0.06 mm (median 0.007 mm); after small random pushes nothing ends more than 3.5 mm away.
* **Physical evidence**: without the target, the object it holds moves 1.4–7.5 cm (median 3.3 cm) in one rollout.
* **Occlusion only**: in every scene the largest occlusion-only region is a decoy; the mass within 4 cm of the target is 0%–9%.
* `sp validate data/data-set-4/scene0*`: all scenes valid.
* **No trace of the target**: rendered again with the target invisible, 4 of 10 stimuli stay within twice the path-tracing noise (9.4/255); in 003, 005, 006, 008, 009, 010 the target's shadow changes the image (see Caveats).

## The same check on data-set-3

Run over the 70 data-set-3 scenes, the occlusion-only check finds the target inside the largest hidden region in 36 of them, and 26 pass rule 4: in about half of data-set-3, an observer that only knows what is hidden would look in the right place. (`compose.occlusion_posterior` on any finished scene gives these numbers.)


## Caveats

* **Shadows**: the hidden target still blocks light. In 003, 005, 006, 008, 009 and 010 that changes the
  stimulus, most visibly in the thin gaps under the tilted bowl (003) and the tilted vase (005), which are darker
  with the target there. The model's own view (the ray caster) has no lighting, so this only matters for human
  viewers; `[render] hidden_shadows = false` removes it (as discussed for data-set-3), at the price of lighting
  that is subtly non-physical. The stimuli use physical lighting.
* **Decoys repeat**: a toaster is the decoy in seven scenes, a big upturned bowl in two, the loaf in one. The library has few big
  solid objects, and only those (or domes) hold more occlusion-only mass than a tight hiding place; new assets
  (a bread bin, a cutting board, books) would make the decoys as varied as the relationships.
* The hidden objects are the ones a tight hiding place allows: cans, cups, small blocks, a battery. Rule 4 rules out large targets in this camera view: anything big enough to be interesting to hide is hidden best by the biggest object on the table.
* Visual meshes sit a few millimetres into the table where PhysX's convex hulls are slightly smaller than the mesh (the loaf, the orange; as in data-set-3).
* Scene 008's hidden mug is the library mug made 10% slimmer (same height, so the plate lies level): the same mug
  shows through the hair-thin gap between the plate and the front mug's rim, and a lower one tilts the plate.

