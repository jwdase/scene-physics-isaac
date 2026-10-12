"""How well an object is hidden, with a safety margin, and cameras on an orbit around the table.

``visible pixels == 0`` at the model resolution is a fragile definition of "hidden": the stimulus is
rendered at twice that resolution with anti-aliased edges, and a target that is hidden by a hair is
revealed by the slightest change. :func:`occlusion` therefore also counts **margin** pixels: the target's
silhouette grown by ``margin_px`` pixels, where nothing closer than the target covers the image. A target
with zero margin pixels is hidden with at least that many pixels of occluder around it.

Transparent objects (``opacity < 1`` in their look, e.g. ``glass1``) do not hide anything in the RTX
stimulus, so they are left out of the occluders.

Pure (Warp ray caster); no simulator needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from scene_physics.camera import CameraSpec
from scene_physics.render import raycast
from scene_physics.scene.model import Scene


@dataclass
class Occlusion:
    name: str
    visible: int  # target pixels in front of everything (opaque) at the given resolution
    margin: int  # pixels of the silhouette grown by margin_px that no closer opaque object covers
    silhouette: int  # amodal silhouette pixels
    margin_px: int
    in_frame: bool  # the whole silhouette is inside the image

    @property
    def hidden(self) -> bool:
        return self.margin == 0 and self.in_frame

    @property
    def fraction(self) -> float:
        return self.visible / self.silhouette if self.silhouette else 0.0


def transparent_objects(scene: Scene) -> list[str]:
    from scene_physics.scene import looks

    return [o.name for o in scene if float((o.material or looks.look_for(o.asset)).get("opacity", 1.0)) < 1.0]


class OcclusionMeter:
    """Repeated occlusion measurements of one scene (fixed meshes; poses and camera can change)."""

    def __init__(self, scene: Scene, camera: CameraSpec | None = None, *, device: str | None = None):
        self.scene = scene
        self.camera = camera or scene.camera
        self.rc = raycast.Raycaster.from_scene(scene, num_worlds=1, camera=self.camera, device=device, floor=False)
        self.names = self.rc.object_names
        self.dynamic = self.names[: self.rc.num_dynamic]
        self.transparent = [self.names.index(n) for n in transparent_objects(scene) if n in self.names]

    def set_camera(self, camera: CameraSpec) -> None:
        import warp as wp

        rc = self.rc
        rays = camera.ray_directions().reshape(-1, 3).astype(np.float32)
        rc.camera, rc.height, rc.width = camera, camera.height, camera.width
        rc.rays = wp.array(rays, dtype=wp.vec3f, device=rc.device)
        rc.eye = wp.vec3f(*[float(v) for v in camera.eye])
        fwd = np.asarray(camera.target, dtype=float) - np.asarray(camera.eye, dtype=float)
        rc.cos_axis = (rays @ (fwd / np.linalg.norm(fwd))).reshape(camera.height, camera.width).astype(np.float32)
        self.camera = camera

    def poses(self, overrides: dict[str, np.ndarray] | None = None) -> np.ndarray:
        poses = self.scene.truth_poses(self.dynamic)
        if overrides:
            for k, n in enumerate(self.dynamic):
                if n in overrides:
                    poses[k] = overrides[n]
        return poses[None]

    def measure(self, name: str, *, margin_px: int = 2, poses: dict[str, np.ndarray] | None = None) -> Occlusion:
        """Occlusion of object ``name`` at the scene poses (``poses`` overrides some of them)."""
        k = self.names.index(name)
        p = self.poses(poses)
        h, w = self.camera.height, self.camera.width
        d_o = self.rc.render(p, hidden=[k, *self.transparent]).distance.numpy().reshape(h, w)
        d_o = np.where(np.isnan(d_o), np.inf, d_o)
        d_t = self.rc.render(p, solo=k).distance.numpy().reshape(h, w)
        d_t = np.where(np.isnan(d_t), np.inf, d_t)
        sil = np.isfinite(d_t)
        visible = int((sil & ~(d_o < d_t)).sum())
        if margin_px > 0:
            d_min = ndimage.minimum_filter(d_t, size=2 * margin_px + 1, mode="constant", cval=np.inf)
        else:
            d_min = d_t
        grown = np.isfinite(d_min)
        margin = int((grown & ~(d_o < d_min)).sum())
        border = np.zeros_like(sil)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        return Occlusion(name, visible, margin, int(sil.sum()), margin_px, not bool((sil & border).any()))


def occlusion(scene: Scene, name: str, *, camera: CameraSpec | None = None, margin_px: int = 2,
              scale: float = 2.0) -> Occlusion:
    """One-off measurement at ``scale`` x the camera resolution (2 = the stimulus resolution)."""
    cam = (camera or scene.camera).scaled(scale)
    return OcclusionMeter(scene, cam).measure(name, margin_px=margin_px)


# ------------------------------------------------------------------------------------------- cameras


def orbit_camera(azimuth_deg: float, elevation_deg: float, distance: float, *, look_at=(0.0, 0.0, 0.8),
                 fov_degree: float | None = None, reference: CameraSpec | None = None, name: str | None = None,
                 width: int | None = None, height: int | None = None) -> CameraSpec:
    """A camera on a sphere around ``look_at`` (azimuth 0 = in front, on the -y side; elevation above
    the horizontal). Without ``fov_degree`` the FOV is a dolly zoom of ``reference``: objects at the look-at
    point keep the size they have in the reference view.
    """
    from scene_physics.camera import DEFAULT_CAMERA

    ref = reference or DEFAULT_CAMERA
    look = np.asarray(look_at, dtype=float)
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = look + distance * np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    if fov_degree is None:
        ref_dist = float(np.linalg.norm(np.asarray(ref.eye, dtype=float) - np.asarray(ref.target, dtype=float)))
        fov_degree = float(np.degrees(2 * np.arctan(np.tan(np.radians(ref.fov_degree / 2)) * ref_dist / distance)))
    return CameraSpec(width=width or ref.width, height=height or ref.height, fov_degree=float(fov_degree),
                      eye=tuple(float(v) for v in eye), target=tuple(float(v) for v in look), up=(0.0, 0.0, 1.0),
                      max_depth=max(ref.max_depth, distance + 2.0),
                      name=name or f"orbit_az{azimuth_deg:+.0f}_el{elevation_deg:.0f}_d{distance:.2f}")


def orbit_of(camera: CameraSpec) -> tuple[float, float, float]:
    """(azimuth_deg, elevation_deg, distance) of a camera around its look-at point."""
    v = np.asarray(camera.eye, dtype=float) - np.asarray(camera.target, dtype=float)
    d = float(np.linalg.norm(v))
    return float(np.degrees(np.arctan2(v[0], -v[1]))), float(np.degrees(np.arcsin(v[2] / d))), d
