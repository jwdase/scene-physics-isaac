"""Repository layout.

```
<root>/
  configs/default.toml        defaults for every config section
  experiments/<exp>.toml      per-experiment overrides (camera, thresholds, description)
  assets/                     environments (tracked); objects/, textures/, hdri/ (built or fetched)
  authoring/<exp>/<scene>.usd scenes being authored (gitignored)
  data/<exp>/<scene>/         finished scenes: scene.usd + render/ (gitignored)
  results/<exp>/<scene>/<run> inference outputs (gitignored)
```

``SCENE_PHYSICS_ROOT`` overrides the root; ``SCENE_PHYSICS_DATA`` / ``SCENE_PHYSICS_RESULTS`` move
just the (large) data or results trees, e.g. onto scratch storage on the remote server.
"""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    env = os.environ.get("SCENE_PHYSICS_ROOT")
    if env:
        return Path(env).resolve()
    # src/scene_physics/paths.py -> repo root is two levels above the package directory.
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return root() / "configs"


def experiments_dir() -> Path:
    return root() / "experiments"


def assets_dir() -> Path:
    return root() / "assets"


def authoring_dir() -> Path:
    return root() / "authoring"


def data_dir() -> Path:
    env = os.environ.get("SCENE_PHYSICS_DATA")
    return Path(env).resolve() if env else root() / "data"


def results_dir() -> Path:
    env = os.environ.get("SCENE_PHYSICS_RESULTS")
    return Path(env).resolve() if env else root() / "results"


def scene_dir(experiment: str, scene: str) -> Path:
    """Directory of a finished scene: ``data/<experiment>/<scene>/``."""
    return data_dir() / experiment / scene


def resolve_scene(spec: str | Path, prefer: str = "data") -> Path:
    """Accept a path to a scene file/directory, or ``<experiment>/<scene>`` under ``data/`` or
    ``authoring/``. ``prefer`` decides which tree wins when the scene exists in both (authoring
    commands edit the authoring copy; everything else reads the finished scene).

    Returns the path of the scene USD file.
    """
    p = Path(spec)
    trees = [data_dir(), authoring_dir()] if prefer == "data" else [authoring_dir(), data_dir()]
    candidates = [p] + [t / p for t in trees]
    for c in candidates:
        if c.is_file():
            return c.resolve()
        if c.is_dir():
            for name in ("scene.usd", "scene.usda", "scene.usdc"):
                if (c / name).is_file():
                    return (c / name).resolve()
        for suffix in (".usd", ".usda", ".usdc"):
            f = c.with_suffix(suffix)
            if f.is_file():
                return f.resolve()
    raise FileNotFoundError(
        f"Could not find a scene for {str(spec)!r} (tried it as a path and under {data_dir()} and "
        f"{authoring_dir()})"
    )
