"""In-memory scene model (pure numpy; no USD or Isaac Sim required).

A :class:`Scene` is what every stage of the pipeline passes around: the authoring workflow fills it in,
``scene.usd`` stores it (see :mod:`scene_physics.scene.usd_io`), and inference reads it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from scene_physics import geometry
from scene_physics.camera import CameraSpec


@dataclass
class Mesh:
    """Triangle mesh in its object's body frame."""

    points: np.ndarray  # (V, 3) float32
    triangles: np.ndarray  # (T, 3) int32

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float32).reshape(-1, 3)
        self.triangles = np.ascontiguousarray(self.triangles, dtype=np.int32).reshape(-1, 3)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def extents(self) -> np.ndarray:
        lo, hi = self.aabb
        return hi - lo

    def transformed(self, pose: np.ndarray) -> np.ndarray:
        """World-space vertices at ``pose``."""
        return geometry.transform_points(pose, self.points).astype(np.float32)

    def world_aabb(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = self.transformed(pose)
        return v.min(axis=0), v.max(axis=0)

    @staticmethod
    def from_polygons(points, face_counts, face_indices) -> "Mesh":
        """Fan-triangulate a USD-style polygon mesh."""
        counts = np.asarray(face_counts, dtype=np.int64)
        indices = np.asarray(face_indices, dtype=np.int64)
        if len(counts) and np.all(counts == 3):
            return Mesh(points, indices.reshape(-1, 3))
        tris = []
        offset = 0
        for c in counts:
            face = indices[offset : offset + c]
            for k in range(1, c - 1):
                tris.append((face[0], face[k], face[k + 1]))
            offset += c
        return Mesh(points, np.array(tris, dtype=np.int64).reshape(-1, 3))

    @staticmethod
    def merge(meshes: list["Mesh"]) -> "Mesh":
        pts, tris, offset = [], [], 0
        for m in meshes:
            pts.append(m.points)
            tris.append(m.triangles + offset)
            offset += len(m.points)
        return Mesh(np.concatenate(pts), np.concatenate(tris))


@dataclass
class Prior:
    """Prior over one object's pose: a mixture of Gaussians around candidate poses.

    Same content as one entry of the old ``<scene>_priors.json``.
    """

    means: np.ndarray  # (K, 7) candidate poses [x, y, z, qx, qy, qz, qw]
    pos_std: float = 0.1
    rot_std: float = 0.1
    bounds: tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5)  # x_min, x_max, y_min, y_max
    weights: np.ndarray | None = None  # (K,), uniform when None
    rule: str = "manual"

    def __post_init__(self):
        self.means = np.asarray(self.means, dtype=float).reshape(-1, 7)
        if self.weights is None:
            self.weights = np.full(len(self.means), 1.0 / max(len(self.means), 1))
        self.weights = np.asarray(self.weights, dtype=float)
        self.bounds = tuple(float(b) for b in self.bounds)

    def to_legacy_dict(self) -> dict:
        x_min, x_max, y_min, y_max = self.bounds
        return {
            "position": self.means.tolist(),
            "pos_std": self.pos_std,
            "rot_std": self.rot_std,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }

    @staticmethod
    def from_legacy_dict(d: dict) -> "Prior":
        """Read an entry of the old ``_priors.json`` (single mean or list of candidate means)."""
        pos = np.asarray(d["position"], dtype=float)
        return Prior(
            means=pos.reshape(-1, 7),
            pos_std=float(d.get("pos_std", 0.1)),
            rot_std=float(d.get("rot_std", 0.1)),
            bounds=(d.get("x_min", -0.5), d.get("x_max", 0.5), d.get("y_min", -0.5), d.get("y_max", 0.5)),
            rule="legacy",
        )


@dataclass
class Visibility:
    fraction: float
    visible_pixels: int
    silhouette_pixels: int


@dataclass
class SceneObject:
    name: str
    pose: np.ndarray  # ground-truth world pose (7,)
    mesh: Mesh
    dynamic: bool = True
    role: str | None = None  # static | observed | hidden (None = not yet measured)
    role_source: str = "measured"  # measured | authored
    tags: list[str] = field(default_factory=list)
    asset: str = ""
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    mass: float | None = None  # kg; None -> density from the physics config
    segmentation_id: int | None = None
    visibility: Visibility | None = None
    prior: Prior | None = None
    settle_displacement: float | None = None
    settle_rotation_deg: float | None = None
    settle_residual: float | None = None
    material: dict[str, Any] = field(default_factory=dict)  # render hints (roughness, metallic, ...)
    extra: dict[str, Any] = field(default_factory=dict)  # unrecognised sp:* attributes, preserved

    def __post_init__(self):
        self.pose = geometry.canonical_pose(np.asarray(self.pose, dtype=float).reshape(7))
        if not self.asset:
            self.asset = base_asset_name(self.name)
        if not self.dynamic:
            self.role = "static"

    @property
    def world_points(self) -> np.ndarray:
        return self.mesh.transformed(self.pose)

    def world_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mesh.world_aabb(self.pose)


@dataclass
class Scene:
    name: str
    objects: dict[str, SceneObject]
    camera: CameraSpec
    metadata: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # ------------------------------------------------------------------ access
    def __iter__(self) -> Iterator[SceneObject]:
        return iter(self.objects.values())

    def __getitem__(self, name: str) -> SceneObject:
        return self.objects[name]

    def __contains__(self, name: str) -> bool:
        return name in self.objects

    @property
    def static_objects(self) -> list[SceneObject]:
        return [o for o in self if not o.dynamic]

    @property
    def dynamic_objects(self) -> list[SceneObject]:
        return [o for o in self if o.dynamic]

    def with_role(self, *roles: str) -> list[SceneObject]:
        return [o for o in self if o.role in roles]

    def with_tag(self, tag: str) -> list[SceneObject]:
        return [o for o in self if tag in o.tags]

    @property
    def target(self) -> SceneObject | None:
        tagged = self.with_tag("target")
        if tagged:
            return tagged[0]
        hidden = self.with_role("hidden")
        return hidden[0] if len(hidden) == 1 else None

    @property
    def support(self) -> SceneObject | None:
        """The static object the others rest on (the table): largest static footprint."""
        statics = self.static_objects
        if not statics:
            return None

        def footprint(o):
            lo, hi = o.world_aabb()
            return (hi[0] - lo[0]) * (hi[1] - lo[1])

        return max(statics, key=footprint)

    def support_bounds(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        """XY extent of the support surface ``(x_min, x_max, y_min, y_max)``."""
        s = self.support
        if s is None:
            return (-0.5, 0.5, -0.5, 0.5)
        lo, hi = s.world_aabb()
        return (lo[0] + margin, hi[0] - margin, lo[1] + margin, hi[1] - margin)

    def support_top(self) -> float:
        s = self.support
        return 0.0 if s is None else float(s.world_aabb()[1][2])

    def truth_poses(self, names: list[str] | None = None) -> np.ndarray:
        names = names or [o.name for o in self.dynamic_objects]
        return np.stack([self.objects[n].pose for n in names])

    def assign_segmentation_ids(self) -> dict[str, int]:
        """Ids 1..K in sorted-name order (the old Blender convention); 0 is background."""
        ids = {name: i + 1 for i, name in enumerate(sorted(self.objects))}
        for name, i in ids.items():
            self.objects[name].segmentation_id = i
        return ids

    def segmentation_labels(self) -> dict[str, int]:
        if any(o.segmentation_id is None for o in self):
            self.assign_segmentation_ids()
        return {"background": 0, **{o.name: int(o.segmentation_id) for o in self}}

    def summary(self) -> str:
        lines = [f"Scene {self.name}  ({len(self.objects)} objects, camera {self.camera.name} "
                 f"{self.camera.width}x{self.camera.height} fov {self.camera.fov_degree:g})"]
        for key in ("experiment", "condition", "cue", "status", "source"):
            if self.metadata.get(key):
                lines.append(f"  {key}: {self.metadata[key]}")
        header = f"  {'object':44s} {'role':9s} {'tags':18s} {'visible':>8s}  {'position (m)':>26s}"
        lines.append(header)
        for o in self:
            vis = f"{o.visibility.fraction:8.3f}" if o.visibility else " " * 8
            pos = "(" + ", ".join(f"{v:7.3f}" for v in o.pose[:3]) + ")"
            lines.append(f"  {o.name:44s} {str(o.role or '-'):9s} {','.join(o.tags) or '-':18s} {vis}  {pos:>26s}")
            if o.prior is not None and o.role == "hidden":
                lines.append(f"      prior: {len(o.prior.means)} candidate(s), pos_std {o.prior.pos_std:g}, "
                             f"bounds {tuple(round(b, 3) for b in o.prior.bounds)}, rule {o.prior.rule}")
        return "\n".join(lines)


_INSTANCE_SUFFIX = re.compile(r"(_\d{3})+$")


def base_asset_name(name: str) -> str:
    """``jug04_006`` -> ``jug04``; ``b04_wineglass_ref_003`` -> ``b04_wineglass``; ``_9v_battery`` -> ``9v_battery``.

    Blender appends ``_NNN`` instance suffixes (sometimes repeatedly); a leading underscore was added
    to make digit-leading names valid USD identifiers.
    """
    base = _INSTANCE_SUFFIX.sub("", name)
    base = re.sub(r"_ref$", "", base)
    if base.startswith("_") and base[1:2].isdigit():
        base = base[1:]
    return base


def safe_prim_name(name: str) -> str:
    """Coerce to a valid USD prim identifier (``9v_battery`` -> ``_9v_battery``)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    return s
