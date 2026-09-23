"""The scene camera: a pinhole defined by eye / target / up, vertical FOV and resolution.

One definition is shared by every consumer so the stimulus and the model see the same image:

* the RTX render (a ``UsdGeom.Camera`` authored from :meth:`CameraSpec.usd_attributes`),
* the Warp ray caster used for the likelihood and visibility (:meth:`CameraSpec.ray_directions`),
* projection of world points into the image (:meth:`CameraSpec.project`).

Conventions (identical to the old Newton ``SensorTiledCamera`` / Blender camera):

* camera looks down its local ``-Z``, ``+Y`` is up, ``+X`` is right;
* the vertical field of view is ``fov_degree``; pixels are square; the principal point is the image centre;
* continuous image coordinates put pixel ``(i, j)`` in ``[i, i+1) x [j, j+1)`` (``x`` right, ``y`` down),
  so its centre is ``(i + 0.5, j + 0.5)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# The camera shared by the data-set / exp01-04 experiments ("p01_f03": eye-level front, 34 deg FOV).
P01_F03 = {
    "name": "p01_f03",
    "width": 640,
    "height": 480,
    "fov_degree": 34.0,
    "eye": [0.0, -1.6628509212474696, 1.1534498743901909],
    "target": [0.0, 0.0, 0.8],
    "up": [0.0, 0.0, 1.0],
}


@dataclass(frozen=True)
class CameraSpec:
    width: int = 640
    height: int = 480
    fov_degree: float = 34.0
    eye: tuple = (0.0, -1.6628509212474696, 1.1534498743901909)
    target: tuple = (0.0, 0.0, 0.8)
    up: tuple = (0.0, 0.0, 1.0)
    max_depth: float = 4.0
    near: float = 0.01
    name: str = "camera"
    extra: dict = field(default_factory=dict, compare=False)

    # ------------------------------------------------------------------ intrinsics
    @property
    def fov_rad(self) -> float:
        return float(np.radians(self.fov_degree))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def focal_px(self) -> float:
        """Focal length in pixels (vertical FOV)."""
        return (self.height / 2.0) / np.tan(self.fov_rad / 2.0)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """OpenCV-style K (x right, y down, principal point at the image centre)."""
        f = self.focal_px
        return np.array([[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]])

    def scaled(self, factor: float) -> "CameraSpec":
        """Same framing at a different resolution (the FOV is vertical, so framing is preserved)."""
        return replace(self, width=int(round(self.width * factor)), height=int(round(self.height * factor)))

    # ------------------------------------------------------------------ extrinsics
    @property
    def rotation(self) -> np.ndarray:
        """Camera -> world rotation; columns are the camera's right, up and back (-forward) axes."""
        eye = np.asarray(self.eye, dtype=float)
        forward = np.asarray(self.target, dtype=float) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(self.up, dtype=float))
        if np.linalg.norm(right) < 1e-9:  # looking straight along `up`
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        return np.column_stack([right, cam_up, -forward])

    @property
    def camera_to_world(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = np.asarray(self.eye, dtype=float)
        return m

    # ------------------------------------------------------------------ rays / projection
    def ray_directions_camera(self) -> np.ndarray:
        """Unit ray directions in the camera frame through every pixel centre, ``(H, W, 3)``."""
        i = np.arange(self.width) + 0.5
        j = np.arange(self.height) + 0.5
        x = (i - self.width / 2.0) / self.focal_px
        y = -(j - self.height / 2.0) / self.focal_px
        xx, yy = np.meshgrid(x, y)
        dirs = np.stack([xx, yy, -np.ones_like(xx)], axis=-1)
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    def ray_directions(self) -> np.ndarray:
        """Unit ray directions in the world frame, ``(H, W, 3)``."""
        return self.ray_directions_camera() @ self.rotation.T

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return (points - np.asarray(self.eye, dtype=float)) @ self.rotation

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points ``(N, 3)`` -> continuous pixel coords ``(N, 2)`` and depth along the view axis.

        Points behind the camera get NaN coordinates.
        """
        cam = self.world_to_camera(points)
        depth = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = self.width / 2.0 + self.focal_px * cam[:, 0] / depth
            py = self.height / 2.0 - self.focal_px * cam[:, 1] / depth
        behind = depth <= 1e-9
        px[behind] = np.nan
        py[behind] = np.nan
        return np.stack([px, py], axis=-1), depth

    # ------------------------------------------------------------------ USD
    def usd_attributes(self, focal_length: float = 24.0) -> dict:
        """Aperture/focal values for a ``UsdGeom.Camera`` that reproduces this pinhole exactly."""
        vertical = 2.0 * focal_length * np.tan(self.fov_rad / 2.0)
        return {
            "focalLength": float(focal_length),
            "verticalAperture": float(vertical),
            "horizontalAperture": float(vertical * self.aspect),
            "clippingRange": (float(self.near), float(max(self.max_depth * 10.0, 100.0))),
        }

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "width": int(self.width),
            "height": int(self.height),
            "fov_degree": float(self.fov_degree),
            "eye": [float(v) for v in self.eye],
            "target": [float(v) for v in self.target],
            "up": [float(v) for v in self.up],
            "max_depth": float(self.max_depth),
        }
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSpec":
        """Accepts the new format and the old camera-config JSON (``id``/``position`` keys are kept)."""
        known = {"name", "id", "width", "height", "fov_degree", "eye", "target", "up", "max_depth", "near"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fov_degree=float(d["fov_degree"]),
            eye=tuple(float(v) for v in d["eye"]),
            target=tuple(float(v) for v in d["target"]),
            up=tuple(float(v) for v in d.get("up", (0.0, 0.0, 1.0))),
            max_depth=float(d.get("max_depth", 4.0)),
            near=float(d.get("near", 0.01)),
            name=str(d.get("name", d.get("id", "camera"))),
            extra=extra,
        )

    @classmethod
    def load(cls, path) -> "CameraSpec":
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d.get("camera", d))


DEFAULT_CAMERA = CameraSpec.from_dict(P01_F03)
