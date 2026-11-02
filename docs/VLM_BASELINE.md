# VLM and robotics-model baseline

The study's scenes, given to vision-language models and robotics models through their APIs, so their answers
can be compared with the participants' and with the physics model's. `scripts/vlm_baseline.py` runs it, with
the settings and every prompt in `configs/vlm_baseline.toml`.

| task | what the model does | scored | models |
|---|---|---|---|
| `occluder` | the participants' task: names the object it thinks blocks the hidden target, with a point on it | like a click on the website: correct when the chosen object hides at least 5% of the target's outline | Claude Opus 5.5, GPT-6, Gemini 3.1 Pro, Gemini Robotics-ER 2, NVIDIA Cosmos |
| `coord` | points to the hidden target's centre in the image | distance to the centre of the target's full silhouette; whether the point falls inside it | Gemini Robotics-ER 2, NVIDIA Cosmos |

## Setup

The stimuli are the testing-website's build (`../testing-website/stimuli`, made by `uv run study-prepare`
there): the same files participants are shown. Put the API keys in `configs/vlm_keys.env` (git-ignored; the
template is `configs/vlm_keys.env.example`):

```bash
cp configs/vlm_keys.env.example configs/vlm_keys.env    # then fill in the keys
```

```
ANTHROPIC_API_KEY=     # Claude Opus 5.5
OPENAI_API_KEY=        # GPT-6
GEMINI_API_KEY=        # Gemini 3.1 Pro and Gemini Robotics-ER 2
NVIDIA_API_KEY=        # Cosmos on build.nvidia.com (nvapi-...)
```

The script declares its own dependencies (anthropic, openai, google-genai, numpy, pillow); `uv run` runs it
in an isolated environment and never touches the Isaac Sim venv. It needs no GPU and no Isaac Sim.

## Commands

```bash
uv run scripts/vlm_baseline.py check                     # one short call per model: key, model ID, image input
uv run scripts/vlm_baseline.py show occluder             # the exact conversation a model gets (no API calls)
uv run scripts/vlm_baseline.py show coord

# a pilot on 3 scenes first: read the replies in results/vlm/<task>/<model>.jsonl
uv run scripts/vlm_baseline.py run occluder --limit 3
uv run scripts/vlm_baseline.py run coord --limit 3

# task 1, the occluder task: all 72 test scenes, all five models
uv run scripts/vlm_baseline.py run occluder
# or one model at a time
uv run scripts/vlm_baseline.py run occluder --models opus
uv run scripts/vlm_baseline.py run occluder --models gpt
uv run scripts/vlm_baseline.py run occluder --models gemini
uv run scripts/vlm_baseline.py run occluder --models gemini-robotics
uv run scripts/vlm_baseline.py run occluder --models cosmos

# task 2, the 2D coordinate task: the robotics models
uv run scripts/vlm_baseline.py run coord                 # gemini-robotics and cosmos

uv run scripts/vlm_baseline.py summary                   # accuracy / error per model -> results/vlm/summary.csv
```

Options of `run`: `--scenes ds3-scene001,ds4-scene003` (only these), `--limit N`, `--repeats N`
(independent queries per scene, for a distribution of answers; the records carry the repeat number),
`--workers N` (scenes in parallel), `--out-dir`. `run` resumes: it skips the scene and repeat pairs already in
the results, so after an interruption or failed queries the same command finishes the rest. To test the
pipeline offline, `--models mock` answers at random.

## What a model gets

Every test scene is its own conversation, the same for every model, word for word:

1. The instructions, adapted from the website's first page (the task, the steps of every image, the answer
   format).
2. The 4 practice scenes, in the website's order (salient, minute, none, salient; `docs/PRACTICE.md`). Each is
   a turn: the target card ("This is the target: the block. It is hidden in the next image."), the empty
   table, the image. The model answers, and the next turn starts with the participants' feedback ("Correct!
   The target was hidden by the vase." or "Not this time: you chose the bowl. ...") and the reveal picture.
   In the coordinate task the feedback says whether the point was on the target and where its centre was.
3. The test scene, the same way. Its answer is the result.

So a test query is 5 API calls, the last with about 20 images. The practice is run live before every test
scene, so each query has its own practice answers, as each participant does.

The images are the website's files: `card.jpg`, `empty.jpg`, `image.png` (1280 x 960), `reveal.jpg`.
Answers end with a line of JSON, `{"object": "vase", "point": [y, x]}` (occluder) or `{"point": [y, x]}`
(coord), with `[y, x]` normalized to 0-1000, Gemini Robotics' native pointing format, so image resizing by
an API cannot shift it.

**No code.** No model gets a tool: no code execution for Claude or Gemini, no tools for GPT. The models
reason (thinking, or a visible chain of thought), and the prompt says not to write or run code. Their reasoning
is saved: Claude's and Gemini's thinking summaries, GPT's reasoning summary, Cosmos's `<think>` section.

**Differences from the participants.** No time limits (participants see the target 3 s, the table 2 s and
the image 5-10 s). The answer is a name and a point, not a click: participants' clicks are read with the
click map, and so are the models' points, but a model's point is less exact than a mouse. The feedback that
replaces "Time ran out" is "No answer could be read from your reply."

## Scoring

**Occluder.** The point is read with the website's click map (`map.json`), exactly like a click. The name
decides when the point misses the named object: the only object with that name, or, when several objects
share it (two vases in ds3-005), the one nearest the point. A point on an object whose name matches nothing
("jar" for the vase) still chooses that object. Each result records how it was resolved (`resolved_by`), the
object at the point and the objects matching the name, so the answers can be rescored with another rule. With
the true occluder's name and a point on it, all 76 scenes (72 test, 4 practice) score correct.

**Coord.** `error_px` is the distance in stimulus pixels from the point to the centre of the target's full
silhouette (`render/target.png`, the manifest's `centroid`); `in_silhouette` and `in_bbox` whether it falls on
the target; `on_hiding_object` whether it lands on an object that hides the target (the occluder task's
answer, read from the coordinate). `error_px_axes_swapped` reads the point as `[x, y]`: a model that ignores
the `[y, x]` order shows up as a much smaller swapped error.

## Results

```
results/vlm/<task>/<model>.jsonl          one record per scene and repeat: the result, and every trial's
                                          prompt, reply, reasoning, parsed answer, score, stop reason, usage
results/vlm/<task>/<model>.runs.jsonl     each run's config, git commit, practice and scenes
results/vlm/<task>/<model>.errors.jsonl   queries that failed (rerun to retry them)
results/vlm/summary.csv
```

## Models

As of 2026-11-02. Check the IDs with `check` before a run; preview models change and are retired (Gemini
Robotics-ER 1.6 was on 2026-08-31), so the model string and the time of each run are in the results.

| key | model | settings |
|---|---|---|
| `opus` | `claude-opus-5-5` | adaptive thinking (always on), effort `high` |
| `gpt` | `gpt-6-astra` (GPT-6 flagship; `gpt-6.1-terra` is the newer, faster visual model) | reasoning effort `high` |
| `gemini` | `gemini-3.1-pro-preview` | thinking level `high` |
| `gemini-robotics` | `gemini-robotics-er-2-preview` | thinking level `high` |
| `cosmos` | `nvidia/cosmos-reason2-8b` on `integrate.api.nvidia.com` | temperature 0.6, top-p 0.95, top-k 20 |

* **Cosmos.** NVIDIA's hosted API serves Cosmos Reason 2. Its successor, the Cosmos 3 reasoner
  (`nvidia/cosmos3-nano-reasoner`), can only be downloaded: serve it with vLLM or NIM on the inference server
  (not the local 3070) and set `model` and `base_url` in `[models.cosmos]`. NVIDIA does not document Cosmos's
  pointing convention; it may answer `[x, y]` despite the prompt (see `error_px_axes_swapped`).
* **Refusals.** Claude's server-side fallback to another model is deliberately off: a fallback would answer
  as a different model. A refusal is saved with `stop: "refusal"` and an empty reply, and scores as no answer.
* **Repeats.** Opus 5.5 does not accept a temperature; its answers vary between queries anyway. Use
  `--repeats` for every model alike.
