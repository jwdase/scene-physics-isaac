"""Typed configuration, loaded from TOML.

``configs/default.toml`` holds the defaults. An experiment file (``experiments/<exp>.toml``) or a
run-specific file only needs the keys it changes; they are merged section by section on top of the
defaults. Unknown keys are an error, so a typo cannot silently fall back to a default.

The **physics** section is the single source of truth for the simulator: authoring (``sp check`` /
``sp finalize``) and inference rollouts apply exactly the same settings, so a scene that is at rest when
it is authored is at rest for the model.
"""

from __future__ import annotations

import copy
import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scene_physics import paths
from scene_physics.camera import CameraSpec


@dataclass
class PhysicsConfig:
    # scene
    dt: float = 1.0 / 120.0
    gravity: float = 9.81
    solver: str = "TGS"  # TGS | PGS
    device: str = "cuda:0"  # PhysX GPU pipeline; "cpu" for the CPU pipeline
    enhanced_determinism: bool = True
    stabilization: bool = True
    ccd: bool = False
    disable_sleeping: bool = True  # history-independent rollouts
    # contact material (all objects share one material, as in the old pipeline)
    static_friction: float = 0.8
    dynamic_friction: float = 0.8
    restitution: float = 0.0
    # rigid bodies
    density: float = 1000.0  # kg/m^3, used when an object has no authored mass
    max_depenetration_velocity: float = 1.0  # m/s; caps separation speed of overlapping bodies
    position_iterations: int = 16
    velocity_iterations: int = 4
    linear_damping: float = 0.0
    angular_damping: float = 0.1  # stands in for the old rolling friction so rolling objects settle
    # collision geometry
    dynamic_approximation: str = "convexDecomposition"  # convexDecomposition | convexHull | sdf
    static_approximation: str = "none"  # triangle mesh
    max_convex_hulls: int = 32
    hull_vertex_limit: int = 64  # PhysX GPU contact generation needs <= 64 vertices per hull
    sdf_resolution: int = 256
    contact_offset: float | None = None  # None -> PhysX default
    rest_offset: float | None = None
    # ground plane below everything (objects that fall off the table land on it)
    ground_plane: bool = True


@dataclass
class SettleConfig:
    max_seconds: float = 5.0
    min_seconds: float = 0.5
    check_every: int = 12  # physics steps between rest checks
    rest_linear_velocity: float = 0.005  # m/s
    rest_angular_velocity: float = 0.05  # rad/s
    stability_seconds: float = 2.0  # extra simulated time used to measure residual motion
    moved_threshold_m: float = 0.005  # report objects that move more than this ...
    moved_threshold_deg: float = 2.0  # ... or rotate more than this while settling


@dataclass
class RenderConfig:
    mode: str = "pathtracing"  # pathtracing | raytracing
    samples: int = 128  # path-tracing samples per pixel (final)
    preview_samples: int = 32
    scale: float = 2.0  # stimulus resolution multiplier on the camera resolution (1280x960 for 640x480)
    denoise: bool = True
    environment: str = "studio"  # look layer applied at render time (assets/environments/<name>.usda)
    gif_fps: int = 30
    gif_frames: int = 90


@dataclass
class AuthoringConfig:
    hidden_threshold: float = 0.1  # visible fraction below which a dynamic object is "hidden"
    prior_rule: str = "observed_objects"
    prior_pos_std: float = 0.1
    prior_rot_std: float = 0.1
    prior_z: str = "truth"  # truth | table_rest
    prior_orientation: str = "truth"  # truth | identity


@dataclass
class LikelihoodConfig:
    # 3DP3 point-cloud likelihood (same defaults as the old JAX implementation)
    variance: float = 0.001
    outlier_prob: float = 0.001
    outlier_volume: float = 1.0
    filter_size: int = 3
    render_scale: float = 1.0  # likelihood render resolution relative to the camera
    # physics rollout
    rollout_seconds: float = 2.5  # old default: 50 frames x 0.05 s
    # composite
    combine: str = "rank"  # rank (old behaviour) | energy
    lambda_visual: float = 1.0
    lambda_displacement: float = 1.0
    lambda_rotation: float = 1.0
    lambda_penetration: float = 0.0  # off by default (the old term was a stub)
    # energy-mode temperatures: log w = sum(lambda * channel / scale)
    visual_scale: float = 1000.0
    displacement_scale: float = 0.01  # metres
    rotation_scale: float = 0.1  # radians
    penetration_scale: float = 1e-6  # cubic metres
    penetration_samples: int = 512  # interior points per object for the penetration estimate


@dataclass
class SamplerConfig:
    num_worlds: int = 15
    iterations: int = 50
    seed: int = 42
    proposer: str = "xy"  # xy (old default) | xyzq
    schedule: str = "exp_decay"  # exp_decay | none
    num_halvings: float = 2.0
    sample_roles: list[str] = field(default_factory=lambda: ["hidden"])
    gibbs_order: str = "random"  # random (old behaviour) | sweep


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    settle: SettleConfig = field(default_factory=SettleConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    authoring: AuthoringConfig = field(default_factory=AuthoringConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    camera: CameraSpec | None = None  # authoring default camera (each scene stores its own)
    experiment: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, CameraSpec):
                out[f.name] = value.to_dict()
            elif dataclasses.is_dataclass(value):
                out[f.name] = dataclasses.asdict(value)
            else:
                out[f.name] = value
        return out


_SECTIONS = {
    "physics": PhysicsConfig,
    "settle": SettleConfig,
    "render": RenderConfig,
    "authoring": AuthoringConfig,
    "likelihood": LikelihoodConfig,
    "sampler": SamplerConfig,
}


def _apply(config: Config, data: dict[str, Any], source: str) -> None:
    for key, value in data.items():
        if key in _SECTIONS:
            section = getattr(config, key)
            names = {f.name for f in dataclasses.fields(section)}
            for k, v in value.items():
                if k not in names:
                    raise KeyError(f"{source}: unknown key [{key}].{k}")
                setattr(section, k, v)
        elif key == "camera":
            config.camera = CameraSpec.from_dict(value)
        elif key in ("experiment", "description"):
            setattr(config, key, value)
        else:
            raise KeyError(f"{source}: unknown section or key {key!r}")


def load_config(*files: str | Path | None, overrides: dict | None = None) -> Config:
    """Defaults, then ``configs/default.toml``, then each file in order, then ``overrides``."""
    config = Config()
    default = paths.configs_dir() / "default.toml"
    for path in (default, *files):
        if path is None:
            continue
        path = Path(path)
        with open(path, "rb") as fh:
            _apply(config, tomllib.load(fh), str(path))
    if overrides:
        _apply(config, overrides, "overrides")
    return config


def load_experiment(exp: str, *files: str | Path | None, overrides: dict | None = None) -> Config:
    """Config for an experiment: defaults + ``experiments/<exp>.toml`` (+ extra files)."""
    exp_file = paths.experiments_dir() / f"{exp}.toml"
    if not exp_file.exists():
        known = sorted(p.stem for p in paths.experiments_dir().glob("*.toml"))
        raise FileNotFoundError(f"No experiment config {exp_file} (known: {', '.join(known) or 'none'})")
    config = load_config(exp_file, *files, overrides=overrides)
    config.experiment = config.experiment or exp
    return config


def copy_config(config: Config) -> Config:
    return copy.deepcopy(config)
