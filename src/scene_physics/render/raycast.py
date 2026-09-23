"""Warp ray caster: depth, point clouds and object ids for many worlds in one launch.

This is the model's "eye" (it replaces Newton's ``SensorTiledCamera``). It renders exactly the same
pinhole as the scene camera (:class:`~scene_physics.camera.CameraSpec`) against the same triangle
meshes, so the observed point cloud and every hypothesis render are directly comparable. It also gives
exact visibility: the first-hit object id per pixel (visible segmentation) and single-object renders
(amodal silhouettes).

Object index space: dynamic bodies ``0 .. B-1`` (their meshes are in body frames and move with
per-world poses), then static meshes ``B .. B+S-1`` (world frame, shared by all worlds). Pixels that
hit nothing within ``max_depth`` get id ``-1`` and NaN points.

Works on CPU (tests) and CUDA, inside or outside Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from scene_physics.camera import CameraSpec
from scene_physics.scene.model import Mesh, Scene


@wp.kernel(enable_backward=False)
def _raycast(
    mesh_ids: wp.array(dtype=wp.uint64),
    num_dynamic: int,
    num_objects: int,
    poses: wp.array(dtype=wp.transformf),
    rays: wp.array(dtype=wp.vec3f),
    eye: wp.vec3f,
    max_t: float,
    solo: int,
    hidden: wp.array(dtype=wp.int32),
    t_out: wp.array2d(dtype=wp.float32),
    id_out: wp.array2d(dtype=wp.int32),
):
    w, p = wp.tid()
    d = rays[p]
    best_t = max_t
    best_id = int(-1)
    for k in range(num_objects):
        if solo >= 0 and k != solo:
            continue
        if hidden[k] != 0:
            continue
        o = eye
        dd = d
        if k < num_dynamic:
            inv = wp.transform_inverse(poses[w * num_dynamic + k])
            o = wp.transform_point(inv, eye)
            dd = wp.transform_vector(inv, d)
        q = wp.mesh_query_ray(mesh_ids[k], o, dd, best_t)
        if q.result:
            best_t = q.t
            best_id = k
    if best_id < 0:
        t_out[w, p] = wp.nan
    else:
        t_out[w, p] = best_t
    id_out[w, p] = best_id


@wp.kernel(enable_backward=False)
def _to_transforms(src: wp.array2d(dtype=wp.float32), dst: wp.array(dtype=wp.transformf)):
    i = wp.tid()
    dst[i] = wp.transformf(wp.vec3f(src[i, 0], src[i, 1], src[i, 2]),
                           wp.quatf(src[i, 3], src[i, 4], src[i, 5], src[i, 6]))


@wp.kernel(enable_backward=False)
def _points(
    t: wp.array2d(dtype=wp.float32),
    rays: wp.array(dtype=wp.vec3f),
    eye: wp.vec3f,
    out: wp.array2d(dtype=wp.vec3f),
):
    w, p = wp.tid()
    tt = t[w, p]
    if wp.isnan(tt):
        out[w, p] = wp.vec3f(wp.nan, wp.nan, wp.nan)
    else:
        out[w, p] = eye + rays[p] * tt


@dataclass
class RenderOutput:
    distance: wp.array  # (N, H*W) float32 distance along each ray, NaN = background
    ids: wp.array  # (N, H*W) int32 object index, -1 = background
    height: int
    width: int
    raycaster: "Raycaster"

    @property
    def num_worlds(self) -> int:
        return self.distance.shape[0]

    def points_wp(self) -> wp.array:
        """World-space point cloud ``(N, H*W)`` of vec3 (NaN = background)."""
        rc = self.raycaster
        out = wp.empty(self.distance.shape, dtype=wp.vec3f, device=rc.device)
        wp.launch(_points, dim=self.distance.shape, inputs=[self.distance, rc.rays, rc.eye], outputs=[out],
                  device=rc.device)
        return out

    def points(self) -> np.ndarray:
        return self.points_wp().numpy().reshape(self.num_worlds, self.height, self.width, 3)

    def id_image(self) -> np.ndarray:
        return self.ids.numpy().reshape(self.num_worlds, self.height, self.width)

    def depth(self) -> np.ndarray:
        """Depth along the optical axis (the convention of RTX ``distance_to_image_plane``)."""
        rc = self.raycaster
        t = self.distance.numpy().reshape(self.num_worlds, self.height, self.width)
        return t * rc.cos_axis[None]


class Raycaster:
    """Renders a scene's meshes from one camera for ``num_worlds`` pose hypotheses at once."""

    def __init__(
        self,
        camera: CameraSpec,
        dynamic_meshes: list[Mesh],
        static_meshes_world: list[Mesh] | None = None,
        *,
        num_worlds: int = 1,
        device: str | None = None,
        max_depth: float | None = None,
    ):
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        self.camera = camera
        self.height, self.width = camera.height, camera.width
        self.num_worlds = num_worlds
        self.num_dynamic = len(dynamic_meshes)
        static_meshes_world = static_meshes_world or []
        self._meshes = [self._wp_mesh(m) for m in list(dynamic_meshes) + list(static_meshes_world)]
        self.num_objects = len(self._meshes)
        self.mesh_ids = wp.array([m.id for m in self._meshes], dtype=wp.uint64, device=self.device)
        rays = camera.ray_directions().reshape(-1, 3).astype(np.float32)
        self.rays = wp.array(rays, dtype=wp.vec3f, device=self.device)
        fwd = np.asarray(camera.target, dtype=float) - np.asarray(camera.eye, dtype=float)
        fwd /= np.linalg.norm(fwd)
        self.cos_axis = (rays @ fwd).reshape(self.height, self.width).astype(np.float32)
        self.eye = wp.vec3f(*[float(v) for v in camera.eye])
        self.max_depth = float(max_depth if max_depth is not None else camera.max_depth)
        self._no_hidden = wp.zeros(self.num_objects, dtype=wp.int32, device=self.device)
        self._identity = wp.array(np.tile([0, 0, 0, 0, 0, 0, 1], (max(num_worlds * self.num_dynamic, 1), 1)).astype(np.float32),
                                  dtype=wp.transformf, device=self.device)

    def _wp_mesh(self, mesh: Mesh) -> wp.Mesh:
        pts = wp.array(np.ascontiguousarray(mesh.points, dtype=np.float32), dtype=wp.vec3f, device=self.device)
        idx = wp.array(np.ascontiguousarray(mesh.triangles.reshape(-1), dtype=np.int32), dtype=wp.int32, device=self.device)
        return wp.Mesh(points=pts, indices=idx)

    # ------------------------------------------------------------------ construction from a scene
    @classmethod
    def from_scene(cls, scene: Scene, *, num_worlds: int = 1, camera: CameraSpec | None = None,
                   device: str | None = None, dynamic_names: list[str] | None = None,
                   floor: bool = True) -> "Raycaster":
        """Dynamic objects in ``dynamic_names`` order (default: the scene's dynamic objects), then statics.

        ``floor`` adds a large quad at ``z = 0`` (the ground the table stands on).
        """
        names = dynamic_names or [o.name for o in scene.dynamic_objects]
        dyn = [scene[n].mesh for n in names]
        statics = [Mesh(o.world_points, o.mesh.triangles) for o in scene.static_objects]
        static_names = [o.name for o in scene.static_objects]
        if floor:
            s = 20.0
            statics.append(Mesh(np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]]),
                                np.array([[0, 1, 2], [0, 2, 3]])))
            static_names.append("__floor__")
        rc = cls(camera or scene.camera, dyn, statics, num_worlds=num_worlds, device=device)
        rc.object_names = list(names) + static_names
        return rc

    # ------------------------------------------------------------------ rendering
    def _pose_array(self, poses) -> wp.array:
        n = self.num_worlds * self.num_dynamic
        if self.num_dynamic == 0:
            return self._identity
        if isinstance(poses, wp.array):
            if poses.dtype == wp.transformf:
                return poses
            if str(poses.device) != str(wp.get_device(self.device)):
                poses = poses.to(self.device)
            src = poses.reshape((n, 7))
            dst = wp.empty(n, dtype=wp.transformf, device=self.device)
            wp.launch(_to_transforms, dim=n, inputs=[src], outputs=[dst], device=self.device)
            return dst
        arr = np.ascontiguousarray(np.asarray(poses, dtype=np.float32).reshape(n, 7))
        return wp.array(arr, dtype=wp.transformf, device=self.device)

    def render(self, poses=None, *, solo: int = -1, hidden: list[int] | None = None) -> RenderOutput:
        """Render all worlds.

        ``poses``: ``(num_worlds, num_dynamic, 7)`` numpy, or a warp array of ``num_worlds*num_dynamic``
        transforms (world-major), e.g. straight from the PhysX tensor API. ``solo``: render only that
        object (amodal silhouette). ``hidden``: object indices to leave out.
        """
        if poses is None and self.num_dynamic:
            raise ValueError("poses are required when the scene has dynamic objects")
        pose_arr = self._pose_array(poses)
        hid = self._no_hidden
        if hidden:
            h = np.zeros(self.num_objects, dtype=np.int32)
            h[list(hidden)] = 1
            hid = wp.array(h, dtype=wp.int32, device=self.device)
        shape = (self.num_worlds, self.height * self.width)
        t = wp.empty(shape, dtype=wp.float32, device=self.device)
        ids = wp.empty(shape, dtype=wp.int32, device=self.device)
        wp.launch(
            _raycast,
            dim=shape,
            inputs=[self.mesh_ids, self.num_dynamic, self.num_objects, pose_arr, self.rays, self.eye,
                    self.max_depth, int(solo), hid],
            outputs=[t, ids],
            device=self.device,
        )
        return RenderOutput(t, ids, self.height, self.width, self)


def render_scene_truth(scene: Scene, *, device: str | None = None, camera: CameraSpec | None = None,
                       floor: bool = True) -> tuple[Raycaster, RenderOutput]:
    """Single-world render of the scene at its ground-truth poses (the model's observation)."""
    rc = Raycaster.from_scene(scene, num_worlds=1, camera=camera, device=device, floor=floor)
    names = rc.object_names[: rc.num_dynamic]
    poses = scene.truth_poses(names)[None] if names else None
    return rc, rc.render(poses)


def visibility(scene: Scene, *, device: str | None = None, camera: CameraSpec | None = None) -> dict[str, dict]:
    """Per-object visible / silhouette pixel counts from the scene camera (exact, no anti-aliasing).

    Returns ``{name: {"visible": int, "silhouette": int, "fraction": float}}`` plus the visible id image
    under the key ``"__ids__"`` (object-index image, -1 background) and the index->name list.
    """
    rc, out = render_scene_truth(scene, device=device, camera=camera, floor=False)
    ids = out.id_image()[0]
    names = rc.object_names
    poses = scene.truth_poses(names[: rc.num_dynamic])[None] if rc.num_dynamic else None
    result: dict = {}
    for k, name in enumerate(names):
        solo = rc.render(poses, solo=k).id_image()[0]
        sil = int((solo == k).sum())
        vis = int((ids == k).sum())
        result[name] = {"visible": vis, "silhouette": sil, "fraction": (vis / sil) if sil else 0.0,
                        "amodal_mask": solo == k, "visible_mask": ids == k}
    result["__ids__"] = ids
    result["__names__"] = names
    return result
