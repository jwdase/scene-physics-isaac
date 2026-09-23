"""Pose and rotation helpers.

Conventions used everywhere in the package:

* A **pose** is a 7-vector ``[x, y, z, qx, qy, qz, qw]`` (quaternion scalar-last). This matches the old
  Scene-Physics code, SciPy, Warp and the PhysX tensor API (``RigidBodyView.get_transforms``).
* USD stores orientations scalar-first (``Gf.Quatf(w, x, y, z)``); convert only through
  :func:`quat_xyzw_to_wxyz` / :func:`quat_wxyz_to_xyzw`.
* World frame is Z-up, metres.

All functions accept a single pose ``(7,)`` or a batch ``(..., 7)``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def canonical_quat(q: np.ndarray) -> np.ndarray:
    """Flip quaternions (xyzw) into the ``qw >= 0`` hemisphere and normalise them.

    Quaternions already unit to float32 precision are left as they are, so values read from a file
    (stored as float32) are written back bit-identically.
    """
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(np.abs(norm - 1.0) > 1e-6, q / norm, q)
    return np.where(q[..., 3:4] < 0.0, -q, q)


def canonical_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.array(pose, dtype=float, copy=True)
    pose[..., 3:] = canonical_quat(pose[..., 3:])
    return pose


def make_pose(position, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    return np.concatenate([np.asarray(position, dtype=float), np.asarray(quat_xyzw, dtype=float)])


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """``(..., 7)`` pose -> ``(..., 4, 4)`` homogeneous matrix (column-vector convention)."""
    pose = np.asarray(pose, dtype=float)
    batch = pose.shape[:-1]
    flat = pose.reshape(-1, 7)
    mats = np.tile(np.eye(4), (flat.shape[0], 1, 1))
    mats[:, :3, :3] = Rotation.from_quat(flat[:, 3:]).as_matrix()
    mats[:, :3, 3] = flat[:, :3]
    return mats.reshape(*batch, 4, 4)


def matrix_to_pose(mat: np.ndarray) -> np.ndarray:
    """``(..., 4, 4)`` rigid matrix (no scale) -> ``(..., 7)`` pose."""
    mat = np.asarray(mat, dtype=float)
    batch = mat.shape[:-2]
    flat = mat.reshape(-1, 4, 4)
    quat = Rotation.from_matrix(flat[:, :3, :3]).as_quat()
    out = np.concatenate([flat[:, :3, 3], quat], axis=-1)
    return canonical_pose(out).reshape(*batch, 7)


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply one pose ``(7,)`` to points ``(N, 3)`` (body frame -> world frame)."""
    pose = np.asarray(pose, dtype=float)
    return Rotation.from_quat(pose[3:]).apply(np.asarray(points, dtype=float)) + pose[:3]


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a * b``: apply ``b`` first, then ``a`` (both ``(..., 7)``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = Rotation.from_quat(a[..., 3:].reshape(-1, 4))
    rb = Rotation.from_quat(b[..., 3:].reshape(-1, 4))
    pos = ra.apply(b[..., :3].reshape(-1, 3)) + a[..., :3].reshape(-1, 3)
    quat = (ra * rb).as_quat()
    shape = np.broadcast_shapes(a.shape, b.shape)
    return canonical_pose(np.concatenate([pos, quat], axis=-1)).reshape(shape)


def inverse(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    r = Rotation.from_quat(pose[..., 3:].reshape(-1, 4)).inv()
    pos = -r.apply(pose[..., :3].reshape(-1, 3))
    return canonical_pose(np.concatenate([pos, r.as_quat()], axis=-1)).reshape(pose.shape)


def rotation_angle(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between quaternions (xyzw); sign-invariant."""
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    q_a = q_a / np.linalg.norm(q_a, axis=-1, keepdims=True)
    q_b = q_b / np.linalg.norm(q_b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(q_a * q_b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def position_error(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(p_a)[..., :3] - np.asarray(p_b)[..., :3], axis=-1)


def perturb_rotations(quats: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Left-multiply each quaternion (xyzw, ``(N, 4)``) by a random axis-angle rotation."""
    quats = np.asarray(quats, dtype=float)
    if std <= 0.0 or len(quats) == 0:
        return quats.copy()
    delta = Rotation.from_rotvec(rng.normal(0.0, std, size=(len(quats), 3)))
    return canonical_quat((delta * Rotation.from_quat(quats)).as_quat())


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])
