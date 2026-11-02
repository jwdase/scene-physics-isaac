# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic",
#     "openai",
#     "google-genai",
#     "numpy",
#     "pillow",
# ]
# ///
"""Query VLMs and robotics models with the hidden-object study's stimuli (docs/VLM_BASELINE.md).

Two tasks, on the testing-website's built stimuli:

* ``occluder``: the participants' task. The model sees the target card, the empty table and the image, and
  names the object it thinks hides the target (plus a point on it, which says which one when several objects
  are of the same kind). Scored like a participant's click: correct when the chosen object hides at least
  ``correct_min_coverage`` of the target's outline.
* ``coord``: the model points to the hidden target's centre. Scored by the distance to the centre of the
  target's full silhouette and whether the point falls inside it.

Every test scene is its own conversation: the intro, the practice trials with the feedback participants get
(right or wrong, what hid the target, the reveal picture), then the test trial. The models get no tools, so
they cannot run code; they may reason (thinking, or visible chain of thought).

This script has its own dependencies (the block above): ``uv run scripts/vlm_baseline.py`` runs it in an
isolated environment and never touches the Isaac Sim venv.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import tomllib
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "vlm_baseline.toml"
TASKS = ("occluder", "coord")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


# ---------------------------------------------------------------------------------------------- config


@dataclass
class ModelSpec:
    key: str
    provider: str
    model: str
    api_key_env: str = ""
    effort: str | None = None
    max_tokens: int = 16000
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    extra_body: dict = field(default_factory=dict)
    vertexai: bool = False  # google: Agent Platform (formerly Vertex AI) instead of the Gemini API

    @classmethod
    def from_toml(cls, key: str, d: dict) -> "ModelSpec":
        known = set(cls.__dataclass_fields__) - {"key"}
        unknown = set(d) - known
        if unknown:
            sys.exit(f"[models.{key}]: unknown settings {sorted(unknown)}")
        return cls(key=key, **d)


@dataclass
class Config:
    path: Path
    raw: dict
    stimuli: Path
    out_dir: Path
    keys_file: Path
    workers: int
    retries: int
    practice: list[str]
    correct_min_coverage: float
    tasks: dict[str, list[str]]
    models: dict[str, ModelSpec]
    prompts: dict


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    base = path.parent
    return Config(
        path=path,
        raw=raw,
        stimuli=(base / raw["stimuli"]).resolve(),
        out_dir=(base / raw["out_dir"]).resolve(),
        keys_file=(base / raw.get("keys_file", "vlm_keys.env")).resolve(),
        workers=int(raw.get("workers", 4)),
        retries=int(raw.get("retries", 4)),
        practice=list(raw.get("practice", [])),
        correct_min_coverage=float(raw.get("correct_min_coverage", 0.05)),
        tasks={t: list(raw["tasks"][t]["models"]) for t in TASKS},
        models={k: ModelSpec.from_toml(k, v) for k, v in raw["models"].items()},
        prompts=raw["prompts"],
    )


def load_keys(path: Path) -> None:
    """KEY=value lines from the keys file into the environment (a variable already set wins)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (s.strip() for s in line.split("=", 1))
        value = value.strip("'\"")
        if value and not os.environ.get(name):
            os.environ[name] = value


# --------------------------------------------------------------------------------------------- stimuli


@dataclass
class Scene:
    phase: str  # practice | test
    entry: dict  # the manifest entry
    dir: Path  # stimuli/<phase>/<id>
    labels: np.ndarray  # click map: (H, W) object ids, 0 = nothing selectable

    @property
    def id(self) -> str:
        return self.entry["id"]

    @property
    def width(self) -> int:
        return self.entry["width"]

    @property
    def height(self) -> int:
        return self.entry["height"]

    @property
    def target_label(self) -> str:
        return self.entry["target"]["label"]

    def objects(self) -> list[dict]:
        return self.entry["objects"]

    def object(self, object_id: int) -> dict | None:
        return next((o for o in self.objects() if o["id"] == object_id), None)

    def hiding(self, min_coverage: float) -> list[dict]:
        """The objects that count as hiding the target, most coverage first (as the website's feedback)."""
        return sorted((o for o in self.objects() if o["target_coverage"] >= min_coverage),
                      key=lambda o: -o["target_coverage"])

    def silhouette(self) -> np.ndarray:
        """The target's full (amodal) silhouette, from the source scene's render/target.png."""
        img = np.array(PILImage.open(Path(self.entry["source"]) / "render" / "target.png"))
        return img[..., 3] > 0 if img.ndim == 3 and img.shape[2] == 4 else img > 0

    def to_pixels(self, point) -> tuple[float, float]:
        """[y, x] normalized 0-1000 -> (x, y) in stimulus pixels."""
        y, x = point
        return x / 1000 * self.width, y / 1000 * self.height

    def to_normalized(self, x: float, y: float) -> list[int]:
        return [round(y / self.height * 1000), round(x / self.width * 1000)]


def decode_click_map(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    rle = data["rle"]
    flat = np.repeat(np.asarray(rle[0::2], dtype=np.int32), np.asarray(rle[1::2], dtype=np.int64))
    if flat.size != data["width"] * data["height"]:
        raise ValueError(f"{path}: click map runs do not cover the image")
    return flat.reshape(data["height"], data["width"])


def load_stimuli(cfg: Config) -> tuple[list[Scene], list[Scene]]:
    manifest_path = cfg.stimuli / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"{manifest_path} not found: run `uv run study-prepare` in testing-website first")
    manifest = json.loads(manifest_path.read_text())

    def scenes(phase: str) -> list[Scene]:
        out = []
        for e in manifest["sets"][phase]["scenes"]:
            d = cfg.stimuli / phase / e["id"]
            out.append(Scene(phase, e, d, decode_click_map(d / "map.json")))
        return out

    practice, test = scenes("practice"), scenes("test")
    if cfg.practice:
        by_id = {s.id: s for s in practice}
        missing = [p for p in cfg.practice if p not in by_id]
        if missing:
            sys.exit(f"practice scenes {missing} are not in the stimuli (have {sorted(by_id)})")
        practice = [by_id[p] for p in cfg.practice]
    return practice, test


# --------------------------------------------------------------------------------------- conversation


@dataclass
class Image:
    path: Path


Part = str | Image


def trial_parts(cfg: Config, task: str, scene: Scene, header: str) -> list[Part]:
    p, t = cfg.prompts, cfg.prompts[task]
    target = scene.target_label
    return [
        header,
        p["card"].format(target=target),
        Image(scene.dir / "card.jpg"),
        p["empty"],
        Image(scene.dir / "empty.jpg"),
        t["image"].format(target=target),
        Image(scene.dir / "image.png"),
    ]


def join_labels(labels: list[str]) -> str:
    """"vase", "vase and the bowl", "vase, the bowl and the mug" (the website's wording)."""
    return labels[0] if len(labels) < 2 else f"{', the '.join(labels[:-1])} and the {labels[-1]}"


def feedback_parts(cfg: Config, task: str, scene: Scene, score: dict) -> list[Part]:
    p, t = cfg.prompts, cfg.prompts[task]
    if task == "occluder":
        where = join_labels(list(dict.fromkeys(o["label"] for o in scene.hiding(cfg.correct_min_coverage))))
        if score.get("object_id") is None:
            text = f'{p["unreadable"]} {t["missing"].format(where=where)}'
        elif score["correct"]:
            text = t["correct"].format(where=where)
        else:
            text = t["wrong"].format(where=where, chosen=score["object_label"])
    else:
        target_point = json.dumps(scene.to_normalized(*scene.entry["target"]["centroid"]))
        if score.get("point") is None:
            text = f'{p["unreadable"]} {t["missing"].format(target_point=target_point)}'
        else:
            text = t["hit" if score["in_silhouette"] else "miss"].format(target_point=target_point)
    return [text, p["reveal"], Image(scene.dir / "reveal.jpg")]


def transcript_text(parts: list[Part]) -> str:
    return "\n".join(f"[image: {p.path}]" if isinstance(p, Image) else p for p in parts)


# -------------------------------------------------------------------------------------------- parsing


def parse_answer(text: str) -> dict | None:
    """The last JSON object in a reply ({"object": ..., "point": [y, x]}); a JSON list of such objects (the
    robotics models' pointing format) gives its first element. None if there is none."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    for match in reversed(list(re.finditer(r"[\[{]", text))):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            value = value[0]
        if isinstance(value, dict) and ("point" in value or "object" in value):
            return value
    return None


def valid_point(value) -> list[float] | None:
    if (isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)
            and all(math.isfinite(v) and 0 <= v <= 1000 for v in value)):
        return [float(v) for v in value]
    return None


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ("the", "a", "an")]


def label_matches(answer: str, label: str) -> bool:
    """The model's name for an object matches a scene label: one's words contain the other's
    ("red vase" ~ "vase", "bread" ~ "loaf of bread")."""
    a, b = words(answer), words(label)
    if not a or not b:
        return False
    return set(b) <= set(a) or set(a) <= set(b)


def distance_to_object(scene: Scene, object_id: int, x: float, y: float) -> float:
    ys, xs = np.nonzero(scene.labels == object_id)
    if xs.size == 0:
        return math.inf
    return float(np.min(np.hypot(xs + 0.5 - x, ys + 0.5 - y)))


def object_at(scene: Scene, x: float, y: float) -> int:
    xi, yi = math.floor(x), math.floor(y)
    if 0 <= xi < scene.width and 0 <= yi < scene.height:
        return int(scene.labels[yi, xi])
    return 0


def score_occluder(cfg: Config, scene: Scene, answer: dict | None) -> dict:
    """Which object the answer chooses, and whether it is correct.

    The point is read with the website's click map, exactly like a participant's click. The name decides when
    the point misses the named object: the only object with that name, or of several the one nearest the point.
    A point on an object whose name does not match anything still chooses that object."""
    answer = answer or {}
    name = answer.get("object") if isinstance(answer.get("object"), str) else None
    point = valid_point(answer.get("point"))
    at_point = nearest = None
    method = None
    if point is not None:
        x, y = scene.to_pixels(point)
        at_point = object_at(scene, x, y) or None
    candidates = [o["id"] for o in scene.objects() if name and label_matches(name, o["label"])]
    if at_point is not None and at_point in candidates:
        chosen, method = at_point, "point and name"
    elif len(candidates) == 1:
        chosen, method = candidates[0], "name"
    elif candidates and point is not None:
        nearest = min(candidates, key=lambda i: distance_to_object(scene, i, *scene.to_pixels(point)))
        chosen, method = nearest, "name, nearest to the point"
    elif at_point is not None:
        chosen, method = at_point, "point"
    else:
        chosen = None
    obj = scene.object(chosen) if chosen is not None else None
    coverage = obj["target_coverage"] if obj else 0.0
    return {
        "object_name": name,
        "point": point,
        "object_id": chosen,
        "object_label": obj["label"] if obj else None,
        "object_scene_name": obj["name"] if obj else None,
        "coverage": coverage,
        "correct": bool(obj is not None and coverage >= cfg.correct_min_coverage),
        "resolved_by": method,
        "object_at_point": at_point,
        "objects_matching_name": candidates,
    }


def score_coord(cfg: Config, scene: Scene, answer: dict | None) -> dict:
    point = valid_point((answer or {}).get("point"))
    tx, ty = scene.entry["target"]["centroid"]
    out = {"point": point, "target_point": scene.to_normalized(tx, ty), "target_px": [tx, ty]}
    if point is None:
        return out | {"point_px": None, "error_px": None, "error_px_axes_swapped": None, "in_silhouette": False,
                      "in_bbox": False,
                      "object_at_point": None, "on_hiding_object": False}
    x, y = scene.to_pixels(point)
    x0, y0, x1, y1 = scene.entry["target"]["bbox"]
    sil = scene.silhouette()
    xi, yi = math.floor(x), math.floor(y)
    inside = bool(0 <= xi < scene.width and 0 <= yi < scene.height and sil[yi, xi])
    at = object_at(scene, x, y) or None
    hiding_ids = {o["id"] for o in scene.hiding(cfg.correct_min_coverage)}
    return out | {
        "point_px": [x, y],
        "error_px": float(math.hypot(x - tx, y - ty)),
        # the same point read as [x, y]: a model that ignores the [y, x] order lands here
        "error_px_axes_swapped": float(math.hypot(point[0] / 1000 * scene.width - tx, point[1] / 1000 * scene.height - ty)),
        "in_silhouette": inside,
        "in_bbox": bool(x0 <= x <= x1 and y0 <= y <= y1),
        "object_at_point": at,
        "on_hiding_object": at in hiding_ids,
    }


def score(cfg: Config, task: str, scene: Scene, answer: dict | None) -> dict:
    return score_occluder(cfg, scene, answer) if task == "occluder" else score_coord(cfg, scene, answer)


# ------------------------------------------------------------------------------------------- providers


@dataclass
class Reply:
    text: str
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    stop: str | None = None


def image_bytes(img: Image) -> tuple[bytes, str]:
    return img.path.read_bytes(), MIME[img.path.suffix.lower()]


def data_url(img: Image) -> str:
    data, mime = image_bytes(img)
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class AnthropicChat:
    """Messages API; adaptive thinking (always on for Opus 5.5) with summarized thinking kept for analysis."""

    def __init__(self, spec: ModelSpec):
        import anthropic

        self.spec = spec
        self.client = anthropic.Anthropic(api_key=os.environ[spec.api_key_env], max_retries=0)
        self.messages: list[dict] = []

    def send(self, parts: list[Part]) -> Reply:
        content = []
        for p in parts:
            if isinstance(p, Image):
                data, mime = image_bytes(p)
                content.append({"type": "image", "source": {"type": "base64", "media_type": mime,
                                                            "data": base64.b64encode(data).decode()}})
            else:
                content.append({"type": "text", "text": p})
        messages = self.messages + [{"role": "user", "content": content}]
        kwargs = dict(
            model=self.spec.model,
            max_tokens=self.spec.max_tokens,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            cache_control={"type": "ephemeral"},  # each turn's prefix is the previous request
        )
        if self.spec.effort:
            kwargs["output_config"] = {"effort": self.spec.effort}
        with self.client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        # thinking blocks go back unchanged on the next turn
        self.messages = messages + [{"role": "assistant", "content": msg.content}]
        return Reply(
            text="".join(b.text for b in msg.content if b.type == "text"),
            reasoning="\n\n".join(b.thinking for b in msg.content if b.type == "thinking"),
            usage=msg.usage.model_dump(mode="json"),
            stop=msg.stop_reason,
        )


class OpenAIChat:
    """Responses API; the conversation continues on the server through previous_response_id."""

    def __init__(self, spec: ModelSpec):
        import openai

        self.spec = spec
        self.client = openai.OpenAI(api_key=os.environ[spec.api_key_env], max_retries=0, timeout=1800)
        self.previous: str | None = None

    def send(self, parts: list[Part]) -> Reply:
        content = [{"type": "input_image", "image_url": data_url(p), "detail": "high"} if isinstance(p, Image)
                   else {"type": "input_text", "text": p} for p in parts]
        kwargs = dict(model=self.spec.model, input=[{"role": "user", "content": content}],
                      max_output_tokens=self.spec.max_tokens)
        if self.spec.effort:
            kwargs["reasoning"] = {"effort": self.spec.effort, "summary": "auto"}
        if self.previous:
            kwargs["previous_response_id"] = self.previous
        resp = self.client.responses.create(**kwargs)
        self.previous = resp.id
        reasoning = [s.text for item in resp.output if item.type == "reasoning" for s in (item.summary or [])]
        return Reply(
            text=resp.output_text or "",
            reasoning="\n\n".join(reasoning),
            usage=resp.usage.model_dump(mode="json") if resp.usage else {},
            stop=resp.status,
        )


class GoogleChat:
    """google-genai chat session (keeps the history, thought signatures included). No tools are configured,
    so the model cannot run code."""

    def __init__(self, spec: ModelSpec):
        from google import genai
        from google.genai import types

        self.types = types
        # kept on self: the chat only holds the client's API module, and a collected client closes its connection
        self.client = genai.Client(api_key=os.environ[spec.api_key_env], vertexai=spec.vertexai)
        thinking = types.ThinkingConfig(include_thoughts=True,
                                        **({"thinking_level": spec.effort} if spec.effort else {}))
        config = types.GenerateContentConfig(thinking_config=thinking, max_output_tokens=spec.max_tokens)
        self.chat = self.client.chats.create(model=spec.model, config=config)

    def send(self, parts: list[Part]) -> Reply:
        types = self.types
        message = [types.Part.from_bytes(data=image_bytes(p)[0], mime_type=image_bytes(p)[1])
                   if isinstance(p, Image) else types.Part.from_text(text=p) for p in parts]
        resp = self.chat.send_message(message)
        cand = resp.candidates[0] if resp.candidates else None
        out_parts = (cand.content.parts if cand and cand.content and cand.content.parts else [])
        return Reply(
            text="".join(p.text for p in out_parts if p.text and not p.thought),
            reasoning="\n\n".join(p.text for p in out_parts if p.text and p.thought),
            usage=resp.usage_metadata.model_dump(mode="json") if resp.usage_metadata else {},
            stop=str(cand.finish_reason) if cand else None,
        )


THINK = re.compile(r"<think>(.*?)</think>", re.S)


class OpenAIChatCompletions:
    """Any OpenAI-compatible chat-completions server (NVIDIA's hosted API, vLLM, NIM). The history keeps the
    answers without their <think> sections, as reasoning models' chat templates expect."""

    def __init__(self, spec: ModelSpec):
        import openai

        self.spec = spec
        self.client = openai.OpenAI(api_key=os.environ.get(spec.api_key_env) or "none", base_url=spec.base_url,
                                    max_retries=0, timeout=1800)
        self.messages: list[dict] = []

    def send(self, parts: list[Part]) -> Reply:
        content = [{"type": "image_url", "image_url": {"url": data_url(p)}} if isinstance(p, Image)
                   else {"type": "text", "text": p} for p in parts]
        messages = self.messages + [{"role": "user", "content": content}]
        kwargs = dict(model=self.spec.model, messages=messages, max_tokens=self.spec.max_tokens)
        if self.spec.temperature is not None:
            kwargs["temperature"] = self.spec.temperature
        if self.spec.top_p is not None:
            kwargs["top_p"] = self.spec.top_p
        if self.spec.extra_body:
            kwargs["extra_body"] = self.spec.extra_body
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        raw = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or "\n\n".join(THINK.findall(raw))
        text = THINK.sub("", raw).split("</think>")[-1].strip()
        self.messages = messages + [{"role": "assistant", "content": text}]
        return Reply(text=text, reasoning=reasoning or "",
                     usage=resp.usage.model_dump(mode="json") if resp.usage else {},
                     stop=resp.choices[0].finish_reason)


class MockChat:
    """No API: a random object name and a random point, to test the pipeline and the scoring offline."""

    def __init__(self, spec: ModelSpec, seed: int):
        self.rng = random.Random(seed)

    def send(self, parts: list[Part]) -> Reply:
        names = ["vase", "bowl", "mug", "loaf of bread", "jug", "coffee grinder", "toy shark", "coaster"]
        point = [self.rng.randint(300, 700), self.rng.randint(200, 800)]
        answer = {"object": self.rng.choice(names), "point": point}
        return Reply(text=f"Mock reasoning.\n{json.dumps(answer)}", stop="mock")


def make_chat(spec: ModelSpec, seed: int):
    if spec.provider != "mock" and spec.provider != "openai-chat" and not os.environ.get(spec.api_key_env):
        raise RuntimeError(f"{spec.api_key_env} is not set (configs/vlm_keys.env)")
    match spec.provider:
        case "anthropic":
            return AnthropicChat(spec)
        case "openai":
            return OpenAIChat(spec)
        case "google":
            return GoogleChat(spec)
        case "openai-chat":
            return OpenAIChatCompletions(spec)
        case "mock":
            return MockChat(spec, seed)
    raise ValueError(f"[models.{spec.key}]: unknown provider {spec.provider!r}")


PERMANENT = {400, 401, 403, 404, 422}


def send_with_retries(chat, parts: list[Part], retries: int) -> Reply:
    """One turn, retried on transient errors. Chats only record a turn once it succeeds, so a retry resends it."""
    for attempt in range(retries):
        try:
            return chat.send(parts)
        except Exception as exc:  # the SDKs' error classes differ; their status codes do not
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status in PERMANENT or attempt == retries - 1:
                raise
            time.sleep(min(10 * 2**attempt, 120) * (0.5 + random.random()))
    raise AssertionError("unreachable")


# -------------------------------------------------------------------------------------------- running


def episode(cfg: Config, task: str, spec: ModelSpec, practice: list[Scene], test: Scene, repeat: int) -> dict:
    """One test query: intro, practice trials with feedback, test trial, in one fresh conversation."""
    p = cfg.prompts
    chat = make_chat(spec, seed=zlib.crc32(f"{test.id}/{repeat}".encode()))
    pending: list[Part] = [p[task]["intro"].format(n_practice=len(practice))]
    trials = []
    started = time.time()
    for i, scene in enumerate(practice, 1):
        parts = pending + trial_parts(cfg, task, scene, p["practice_header"].format(i=i, n=len(practice)))
        reply = send_with_retries(chat, parts, cfg.retries)
        answer = parse_answer(reply.text)
        result = score(cfg, task, scene, answer)
        trials.append(trial_record("practice", scene, parts, reply, answer, result))
        pending = feedback_parts(cfg, task, scene, result)
    parts = pending + trial_parts(cfg, task, test, p["test_header"])
    reply = send_with_retries(chat, parts, cfg.retries)
    answer = parse_answer(reply.text)
    result = score(cfg, task, test, answer)
    trials.append(trial_record("test", test, parts, reply, answer, result))
    return {
        "task": task,
        "model_key": spec.key,
        "provider": spec.provider,
        "model": spec.model,
        "settings": {k: v for k, v in vars(spec).items() if k not in ("key", "provider", "model", "api_key_env")},
        "scene": test.id,
        "repeat": repeat,
        "target": test.target_label,
        "answered": answer is not None,
        "result": result,
        "practice_correct": [t["result"].get("correct", t["result"].get("in_silhouette")) for t in trials[:-1]],
        "trials": trials,
        "seconds": round(time.time() - started, 1),
        "finished": now(),
    }


def trial_record(phase: str, scene: Scene, parts: list[Part], reply: Reply, answer, result: dict) -> dict:
    return {
        "phase": phase,
        "scene": scene.id,
        "prompt": transcript_text(parts),
        "reply": reply.text,
        "reasoning": reply.reasoning,
        "answer": answer,
        "result": result,
        "stop": reply.stop,
        "usage": reply.usage,
    }


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_models(cfg: Config, task: str, names: str | None) -> list[ModelSpec]:
    keys = names.split(",") if names else cfg.tasks[task]
    unknown = [k for k in keys if k not in cfg.models]
    if unknown:
        sys.exit(f"unknown models {unknown}; configured: {sorted(cfg.models)}")
    return [cfg.models[k] for k in keys]


def select_scenes(test: list[Scene], ids: str | None, limit: int | None) -> list[Scene]:
    if ids:
        by_id = {s.id: s for s in test}
        missing = [i for i in ids.split(",") if i not in by_id]
        if missing:
            sys.exit(f"unknown test scenes {missing}")
        test = [by_id[i] for i in ids.split(",")]
    return test[:limit] if limit else test


def cmd_run(cfg: Config, args) -> None:
    practice, test = load_stimuli(cfg)
    scenes = select_scenes(test, args.scenes, args.limit)
    out = cfg.out_dir / args.task
    out.mkdir(parents=True, exist_ok=True)
    for spec in select_models(cfg, args.task, args.models):
        path = out / f"{spec.key}.jsonl"
        done = {(r["scene"], r["repeat"]) for r in read_records(path)}
        jobs = [(s, r) for s in scenes for r in range(args.repeats) if (s.id, r) not in done]
        print(f"{args.task} / {spec.key} ({spec.model}): {len(jobs)} queries to run, {len(done)} already in {path}")
        if not jobs:
            continue
        info = {"started": now(), "git": git_commit(), "config": cfg.raw | {"models": {spec.key: vars(spec)}},
                "practice": [s.id for s in practice], "scenes": [s.id for s, _ in jobs]}
        with (out / f"{spec.key}.runs.jsonl").open("a") as f:
            f.write(json.dumps(info, default=str) + "\n")
        lock = threading.Lock()
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers or cfg.workers) as pool:
            futures = {pool.submit(episode, cfg, args.task, spec, practice, s, r): (s, r) for s, r in jobs}
            for n, fut in enumerate(as_completed(futures), 1):
                scene, rep = futures[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    failures += 1
                    print(f"  [{n}/{len(jobs)}] {scene.id} #{rep}: FAILED {type(exc).__name__}: {exc}")
                    with lock, (out / f"{spec.key}.errors.jsonl").open("a") as f:
                        f.write(json.dumps({"scene": scene.id, "repeat": rep, "error": repr(exc), "at": now()}) + "\n")
                    continue
                with lock, path.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                print(f"  [{n}/{len(jobs)}] {scene.id} #{rep}: {describe(args.task, rec)}")
        if failures:
            print(f"  {failures} queries failed (see {spec.key}.errors.jsonl); run the same command again to retry them")


def describe(task: str, rec: dict) -> str:
    r = rec["result"]
    if task == "occluder":
        if r["object_id"] is None:
            what = "names no object in the scene" if rec["answered"] else "no answer read"
            return f"{what} ({rec['seconds']} s)"
        return f"{r['object_label']} ({'correct' if r['correct'] else 'wrong'}, by {r['resolved_by']}, {rec['seconds']} s)"
    if r["point"] is None:
        return f"no point read ({rec['seconds']} s)"
    return f"{r['error_px']:.0f} px from the centre{', on the target' if r['in_silhouette'] else ''} ({rec['seconds']} s)"


def cmd_show(cfg: Config, args) -> None:
    """The conversation a model gets for one test scene, with example feedback (no API calls)."""
    practice, test = load_stimuli(cfg)
    scene = select_scenes(test, args.scenes, 1)[0]
    p = cfg.prompts
    pending: list[Part] = [p[args.task]["intro"].format(n_practice=len(practice))]
    for i, s in enumerate(practice, 1):
        print(f"===== user (turn {i}) =====")
        print(transcript_text(pending + trial_parts(cfg, args.task, s, p["practice_header"].format(i=i, n=len(practice)))))
        hiding = {o["id"] for o in s.hiding(cfg.correct_min_coverage)}
        wrong = next(o for o in s.objects() if o["id"] not in hiding)
        print(f"===== model answers; the feedback below is for a wrong answer (the {wrong['label']}) =====")
        example = score(cfg, args.task, s, {"object": wrong["label"], "point": [0, 0]})
        pending = feedback_parts(cfg, args.task, s, example)
    print(f"===== user (turn {len(practice) + 1}, the test) =====")
    print(transcript_text(pending + trial_parts(cfg, args.task, scene, p["test_header"])))


def cmd_check(cfg: Config, args) -> None:
    """One short call per model with one stimulus: is the key set, the model ID right, image input accepted?"""
    practice, test = load_stimuli(cfg)
    scene = (practice or test)[0]
    keys = args.models.split(",") if args.models else sorted({k for t in TASKS for k in cfg.tasks[t]})
    for key in keys:
        spec = cfg.models[key]
        try:
            chat = make_chat(spec, seed=0)
            reply = chat.send(["Name the objects on the table in this image, in one short sentence.",
                               Image(scene.dir / "image.png")])
            print(f"OK    {key:16} {spec.model}: {reply.text.strip()[:150]!r}")
        except Exception as exc:
            print(f"FAIL  {key:16} {spec.model}: {type(exc).__name__}: {str(exc)[:300]}")
            if spec.provider == "openai-chat":
                list_served_models(spec)


def list_served_models(spec: ModelSpec) -> None:
    try:
        import openai

        client = openai.OpenAI(api_key=os.environ.get(spec.api_key_env) or "none", base_url=spec.base_url)
        ids = sorted(m.id for m in client.models.list())
        matches = [i for i in ids if "cosmos" in i.lower()] or ids
        print(f"      models on {spec.base_url}: {matches[:20]}")
    except Exception as exc:
        print(f"      could not list the models on {spec.base_url}: {exc}")


def cmd_summary(cfg: Config, args) -> None:
    rows = []
    for task in TASKS if not args.task else (args.task,):
        for path in sorted((cfg.out_dir / task).glob("*.jsonl")):
            if path.name.endswith((".runs.jsonl", ".errors.jsonl")):
                continue
            recs = read_records(path)
            if not recs:
                continue
            r0 = recs[0]
            row = {"task": task, "model_key": r0["model_key"], "model": r0["model"], "queries": len(recs),
                   "scenes": len({r["scene"] for r in recs}),
                   "answered": sum(r["answered"] for r in recs) / len(recs)}
            res = [r["result"] for r in recs]
            if task == "occluder":
                row["accuracy"] = sum(x["correct"] for x in res) / len(res)
                row["resolved_by_point_and_name"] = sum(x["resolved_by"] == "point and name" for x in res) / len(res)
            else:
                errs = [x["error_px"] for x in res if x["error_px"] is not None]
                row["median_error_px"] = float(np.median(errs)) if errs else None
                swapped = [x["error_px_axes_swapped"] for x in res if x["error_px"] is not None]
                row["median_error_px_axes_swapped"] = float(np.median(swapped)) if swapped else None
                row["in_silhouette"] = sum(x["in_silhouette"] for x in res) / len(res)
                row["on_hiding_object"] = sum(x["on_hiding_object"] for x in res) / len(res)
            practice = [c for r in recs for c in r["practice_correct"] if c is not None]
            row["practice_correct"] = sum(practice) / len(practice) if practice else None
            rows.append(row)
    if not rows:
        sys.exit(f"no results under {cfg.out_dir}")
    for row in rows:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (cfg.out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {cfg.out_dir / 'summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="query the models; resumes, so rerunning retries what failed")
    run.add_argument("task", choices=TASKS)
    run.add_argument("--models", help="comma-separated model keys (default: the task's list in the config)")
    run.add_argument("--scenes", help="comma-separated test scene ids (default: all 72)")
    run.add_argument("--limit", type=int, help="only the first N test scenes")
    run.add_argument("--repeats", type=int, default=1, help="independent queries per scene (default 1)")
    run.add_argument("--workers", type=int, help="scenes in parallel (default: the config's)")
    run.add_argument("--out-dir", type=Path, help="write results here instead of the config's out_dir")

    show = sub.add_parser("show", help="print the conversation a model gets for one scene (no API calls)")
    show.add_argument("task", choices=TASKS)
    show.add_argument("--scenes", help="the test scene to show (default: the first)")

    check = sub.add_parser("check", help="one short call per model: key, model ID, image input")
    check.add_argument("--models", help="comma-separated model keys (default: every model of both tasks)")

    summary = sub.add_parser("summary", help="accuracy / error per model -> summary.csv")
    summary.add_argument("task", nargs="?", choices=TASKS)
    summary.add_argument("--out-dir", type=Path)

    args = ap.parse_args()
    cfg = load_config(args.config.resolve())
    if getattr(args, "out_dir", None):
        cfg.out_dir = args.out_dir.resolve()
    load_keys(cfg.keys_file)
    {"run": cmd_run, "show": cmd_show, "check": cmd_check, "summary": cmd_summary}[args.command](cfg, args)


if __name__ == "__main__":
    main()
