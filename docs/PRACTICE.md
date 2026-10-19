# Practice scenes for the human study

The study website (testing-website) starts with four practice trials, with feedback, in a fixed order chosen by
how visible the target's physical evidence is: **salient, minute, none, salient**. Practice scenes are never test
scenes. Two come from data-set-3 (held out of the test set since the study began); the experiment `practice`
adds the two the held-out scenes could not provide, composed with `sp compose` (`docs/COMPOSING_SCENES.md`):

| order | scene | cue | target | hidden by | the cue |
|---|---|---|---|---|---|
| 1 | data-set-3/scene007 | salient | block | jug04 ("vase") | the big bowl sits 4.7 cm off the table, on the block hidden behind jug04 |
| 2 | practice/scene001 | minute | stone coaster | bowl | the big bowl stands 5° off level, the back right of its foot on the coaster hidden under it |
| 3 | practice/scene002 | none | can | jug01 ("jug", a pitcher) | nothing: the can stands behind the pitcher and touches nothing |
| 4 | data-set-3/scene002 | salient | block | loaf | the big bowl is tilted 18°, its rim on the block behind the loaf |

data-set-3/scene054 (a bowl tilted 11.7° on a block under its rim), the third earlier practice scene, is between
salient and minute and is no longer used; it stays out of the test set so that every participant's test set is
the same.

## The composed scenes

`experiments/practice.toml` uses the data-set-3/4 camera and physics. The specs are
`experiments/practice/scene001.toml` and `scene002.toml`. Unlike data-set-4, they do not have to pass every
composing rule (`docs/COMPOSING_SCENES.md`, section 1): the target is hidden with a margin (rule 1), the truth is
a verified, robust rest state (rule 2) and every other object is visible (rule 5), but the evidence (rule 3) is
deliberately small in scene001 and absent in scene002, and no decoy is required (rule 4).

| | scene001 (minute) | scene002 (none) |
|---|---|---|
| target hidden | 0 px, also inside the 2 px margin (bowl covers 100%) | 0 px, also inside the 2 px margin (pitcher covers 100%) |
| rest state | verified and robust (pushes move nothing more than 1.2 mm) | verified and robust (at most 1.3 mm) |
| others visible | loaf 62%, jug 78%, bowl, mug, banana 100% | toaster 85%, pitcher, mug, banana 100% |
| without the target | the bowl settles flat: 8 mm, 5° | nothing moves |
| occlusion only | the bowl holds 67% of the mass | toaster 47%, mug 28%, pitcher 25% |
| leak (target's shadow) | 13.2 | 13.9 |

Leak: the largest 9×9 local colour difference (0–255) between the stimulus rendered with and without the
target, as in `sp review --leak`. data-set-3's repeated renders of the same scene differ by up to 11.2, and up to
twice that is path-tracing noise; the difference maps show no shadow shape, so neither stimulus carries a trace
of its target (which matters most for scene002, where a shadow would be the cue it must not have). The leak was
measured with the renders only, in one stage per scene: `sp review` itself ran out of GPU memory in PhysX on the
workstation, whose desktop holds half of the 8 GB card. Its physics checks are those of `sp compose` above, and
`sp finalize --no-settle` measured the stored poses at rest (nothing moves more than 0.09 mm).

Made with:

```bash
uv run sp preview experiments/practice/scene001.toml experiments/practice/scene002.toml --exp practice
uv run sp compose experiments/practice/scene001.toml experiments/practice/scene002.toml --exp practice --no-render
uv run sp finalize authoring/practice/scene001.usd --exp practice --no-settle --target coaster --overwrite
uv run sp finalize authoring/practice/scene002.usd --exp practice --no-settle --target can --overwrite
```

## Study pictures

Every practice scene also needs the reveal shown after its trial:

```bash
uv run sp study-renders data-set-3/scene007 practice/scene001 practice/scene002 data-set-3/scene002 --reveal
```
