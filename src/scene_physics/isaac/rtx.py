"""RTX renders from the scene camera: the stimulus image, instance segmentation and depth.

    renderer = StimulusRenderer(scene_labels, camera, render_cfg)
    frame = renderer.capture(quality="final")      # path traced
    frame.rgb, frame.segmentation, frame.depth

Segmentation comes straight from RTX instance ids (no colour-decoding step as in the Blender
pipeline), remapped to each scene object's stable ``sp:segmentationId`` (1..K in sorted-name order,
0 = background, which includes the floor, wall and anything that is not a scene object).
"""

from __future__ import annotations

from dataclasses import dataclass

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd

from scene_physics.camera import CameraSpec
from scene_physics.config import RenderConfig
from scene_physics.isaac import app as isaac_app
from scene_physics.scene import schema as S


@dataclass
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    segmentation: np.ndarray  # (H, W) uint16 scene segmentation ids, 0 = background
    depth: np.ndarray  # (H, W) float32 distance to the image plane (inf = background)


def set_render_mode(mode: str, samples: int, denoise: bool = True) -> None:
    s = carb.settings.get_settings()
    if mode == "pathtracing":
        s.set("/rtx/rendermode", "PathTracing")
        s.set("/rtx/pathtracing/spp", int(min(samples, 64)))
        s.set("/rtx/pathtracing/totalSpp", int(samples))
        s.set("/rtx/pathtracing/maxBounces", 6)
        s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
        s.set("/rtx/pathtracing/optixDenoiser/enabled", bool(denoise))
        s.set("/rtx/pathtracing/clampSpp", 0)
    else:
        s.set("/rtx/rendermode", "RaytracedLighting")
    s.set("/rtx/post/aa/op", 3)  # TAA / DLAA-quality anti-aliasing on the beauty pass


class StimulusRenderer:
    """Render product on ``/World/Camera`` with rgb, instance-id segmentation and depth annotators."""

    def __init__(self, labels: dict[str, int], camera: CameraSpec, cfg: RenderConfig, *, scale: float | None = None,
                 camera_path: str = S.CAMERA):
        self.cfg = cfg
        self.labels = labels  # object name -> segmentation id
        cam = camera.scaled(cfg.scale if scale is None else scale)
        self.width, self.height = cam.width, cam.height
        self.product = rep.create.render_product(camera_path, (self.width, self.height))
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False})
        self._depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        for a in (self._rgb, self._seg, self._depth):
            a.attach(self.product)
        self._warm = False

    def _object_id(self, prim_path: str) -> int:
        prefix = S.OBJECTS + "/"
        if not prim_path.startswith(prefix):
            return 0
        name = prim_path[len(prefix):].split("/", 1)[0]
        return int(self.labels.get(name, 0))

    def capture(self, quality: str = "final", samples: int | None = None) -> Frame:
        """``quality``: ``final`` (path traced, ``cfg.samples``), ``preview`` (path traced, fewer
        samples) or ``fast`` (real-time ray tracing, for animations)."""
        # delta_time=0: the timeline must not advance while we capture, otherwise Replicator's
        # capture-on-play would step (or even initialise) physics behind our back.
        if not self._warm:
            # A path-traced frame requested before any ray-traced frame comes back black (measured on Isaac
            # Sim 6.1), and the first frames stay empty while shaders compile and textures stream: warm up
            # in ray-tracing mode until the annotator delivers a lit image.
            set_render_mode("raytracing", 1)
            for _ in range(12):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                data = np.asarray(self._rgb.get_data())
                if data.size and data[..., :3].max() > 0:
                    break
            self._warm = True
        if quality == "fast":
            set_render_mode("raytracing", 1)
            subframes = 4
        else:
            spp = samples or (self.cfg.samples if quality == "final" else self.cfg.preview_samples)
            mode = self.cfg.mode if quality == "final" else "pathtracing"
            set_render_mode(mode, spp, self.cfg.denoise)
            subframes = max(1, int(np.ceil(spp / 64))) if mode == "pathtracing" else 8
        rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0)
        rgb = np.asarray(self._rgb.get_data())[..., :3].copy()
        seg_data = self._seg.get_data()
        raw = np.asarray(seg_data["data"])
        id_to_path = seg_data["info"].get("idToLabels", {})
        lut = {int(k): self._object_id(str(v)) for k, v in id_to_path.items()}
        seg = np.zeros(raw.shape, dtype=np.uint16)
        for inst, obj_id in lut.items():
            if obj_id:
                seg[raw == inst] = obj_id
        depth = np.asarray(self._depth.get_data(), dtype=np.float32).copy()
        return Frame(rgb=rgb, segmentation=seg, depth=depth)

    def destroy(self) -> None:
        for a in (self._rgb, self._seg, self._depth):
            try:
                a.detach()
            except Exception:
                pass
        try:
            self.product.destroy()
        except Exception:
            pass


def settle_frames(n: int = 3) -> None:
    isaac_app.update(n)


def current_stage():
    return omni.usd.get_context().get_stage()
