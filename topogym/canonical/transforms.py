"""Frame transforms for the canonical spec (numpy only).

Conventions (see :mod:`topogym.canonical.spec`):

- **World and body**: REP-103, right-handed, z up; body x forward,
  y left, z up. Metres and radians.
- **Camera**: OpenCV, x right, y down, z forward (the optical axis).
  Depth is z-depth, the distance along the optical axis, not the range
  along the ray.
- **Quaternions**: ``(qx, qy, qz, qw)``, scalar last, unit norm,
  Hamilton product; ``q`` rotates vectors from the child frame into the
  parent frame.
- **pose7**: ``(x, y, z, qx, qy, qz, qw)``, the child frame's origin and
  orientation in the parent frame, i.e. ``T_parent_child``.
- **Pinhole intrinsics**: pixel centres at integer coordinates, so the
  principal point of a ``W x H`` image is ``((W - 1) / 2, (H - 1) / 2)``.

Everything accepts batches: leading dimensions broadcast.
"""

from __future__ import annotations

import math

import numpy as np

#: Rotation taking vectors from the REP-103 body frame into the OpenCV
#: camera frame of a camera looking along the body's +x:
#: x_cv = -y_body, y_cv = -z_body, z_cv = x_body.
R_OPENCV_FROM_REP103 = np.array([[0.0, -1.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [1.0, 0.0, 0.0]])
R_REP103_FROM_OPENCV = R_OPENCV_FROM_REP103.T

#: The same rotation as a quaternion: T_body_cam's orientation for a
#: camera mounted looking forward.
Q_REP103_FROM_OPENCV = np.array([-0.5, 0.5, -0.5, 0.5])

IDENTITY_POSE7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def rep103_to_opencv(v) -> np.ndarray:
    """Vectors (..., 3) from REP-103 body axes to OpenCV camera axes."""
    return np.asarray(v, dtype=np.float64) @ R_OPENCV_FROM_REP103.T


def opencv_to_rep103(v) -> np.ndarray:
    """Vectors (..., 3) from OpenCV camera axes to REP-103 body axes."""
    return np.asarray(v, dtype=np.float64) @ R_REP103_FROM_OPENCV.T


# -- quaternions ----------------------------------------------------------------


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conjugate(q) -> np.ndarray:
    """The inverse of a unit quaternion."""
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vectors (..., 3) by unit quaternions (..., 4)."""
    return np.einsum("...ij,...j->...i", quat_to_matrix(q),
                     np.asarray(v, dtype=np.float64))


def quat_to_matrix(q) -> np.ndarray:
    """Unit quaternions (..., 4) to rotation matrices (..., 3, 3)."""
    x, y, z, w = np.moveaxis(quat_normalize(q), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def matrix_to_quat(m) -> np.ndarray:
    """Rotation matrices (..., 3, 3) to unit quaternions (..., 4), with
    ``qw >= 0``."""
    m = np.asarray(m, dtype=np.float64)
    batch = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    out = np.empty((len(m), 4))
    for k, r in enumerate(m):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                 (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [0.25 * s, (r[0, 1] + r[1, 0]) / s,
                 (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s,
                 (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s,
                 0.25 * s, (r[1, 0] - r[0, 1]) / s]
        q = np.array(q)
        out[k] = -q if q[3] < 0 else q
    return quat_normalize(out).reshape(*batch, 4)


def quat_from_euler(roll, pitch, yaw) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw, then pitch, then roll) to a quaternion,
    REP-103 (roll about x, pitch about y, yaw about z)."""
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    return np.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], axis=-1)


def quat_to_euler(q) -> np.ndarray:
    """Unit quaternions to ``(roll, pitch, yaw)`` (inverse of
    :func:`quat_from_euler`; pitch clipped to [-pi/2, pi/2])."""
    x, y, z, w = np.moveaxis(quat_normalize(q), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def quat_from_yaw(yaw) -> np.ndarray:
    return quat_from_euler(np.zeros_like(np.asarray(yaw, float)),
                           np.zeros_like(np.asarray(yaw, float)), yaw)


# -- pose7 -----------------------------------------------------------------------


def pose7(position, quat) -> np.ndarray:
    return np.concatenate([np.asarray(position, dtype=np.float64),
                           quat_normalize(quat)], axis=-1)


def pose7_to_matrix(p) -> np.ndarray:
    """pose7 (..., 7) to homogeneous transforms (..., 4, 4)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros(p.shape[:-1] + (4, 4))
    out[..., :3, :3] = quat_to_matrix(p[..., 3:])
    out[..., :3, 3] = p[..., :3]
    out[..., 3, 3] = 1.0
    return out


def matrix_to_pose7(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.concatenate([m[..., :3, 3], matrix_to_quat(m[..., :3, :3])],
                          axis=-1)


def pose7_compose(a, b) -> np.ndarray:
    """``T_ac = T_ab * T_bc``: ``a`` is ``T_ab``, ``b`` is ``T_bc``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    q = quat_normalize(quat_multiply(a[..., 3:], b[..., 3:]))
    t = a[..., :3] + quat_rotate(a[..., 3:], b[..., :3])
    return np.concatenate([t, q], axis=-1)


def pose7_inverse(p) -> np.ndarray:
    """``T_ba`` from ``T_ab``."""
    p = np.asarray(p, dtype=np.float64)
    qi = quat_conjugate(quat_normalize(p[..., 3:]))
    return np.concatenate([-quat_rotate(qi, p[..., :3]), qi], axis=-1)


def pose7_apply(p, points) -> np.ndarray:
    """Map points (..., 3) from the child frame into the parent frame."""
    p = np.asarray(p, dtype=np.float64)
    return quat_rotate(p[..., 3:], points) + p[..., :3]


def pose7_relative(a, b) -> np.ndarray:
    """``T_ab`` from two poses in a common frame (``T_wa``, ``T_wb``)."""
    return pose7_compose(pose7_inverse(a), b)


# -- cameras ---------------------------------------------------------------------


def intrinsics_from_fov(width: int, height: int, hfov: float | None = None,
                        vfov: float | None = None) -> dict:
    """Pinhole intrinsics from a field of view (radians) and image size.

    Give either FOV; square pixels are assumed for the other. Returns
    ``{"fx", "fy", "cx", "cy", "width", "height"}``.
    """
    if hfov is None and vfov is None:
        raise ValueError("give hfov or vfov")
    fx = fy = None
    if hfov is not None:
        fx = (width / 2.0) / math.tan(hfov / 2.0)
    if vfov is not None:
        fy = (height / 2.0) / math.tan(vfov / 2.0)
    fx = fx if fx is not None else fy
    fy = fy if fy is not None else fx
    return {"fx": fx, "fy": fy, "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0, "width": int(width),
            "height": int(height)}


def intrinsics_matrix(k: dict) -> np.ndarray:
    return np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]],
                     [0.0, 0.0, 1.0]])


def project(k: dict, points) -> np.ndarray:
    """OpenCV-camera points (..., 3) to pixel coordinates (..., 2)."""
    p = np.asarray(points, dtype=np.float64)
    z = p[..., 2]
    return np.stack([k["fx"] * p[..., 0] / z + k["cx"],
                     k["fy"] * p[..., 1] / z + k["cy"]], axis=-1)


def backproject_depth(depth, k: dict, *, frame: str = "opencv",
                      T_world_cam=None) -> np.ndarray:
    """Points (H, W, 3) from a z-depth image (H, W) in metres.

    ``frame="opencv"`` returns camera-frame points; ``"rep103"`` the
    same points in REP-103 axes (x forward, y left, z up). With
    ``T_world_cam`` (a pose7 of the OpenCV camera frame in the world),
    points are returned in the world frame instead. Invalid depths
    (non-finite or <= 0) give NaN points.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    valid = np.isfinite(d) & (d > 0)
    z = np.where(valid, d, np.nan)
    pts = np.stack([(u - k["cx"]) * z / k["fx"],
                    (v - k["cy"]) * z / k["fy"], z], axis=-1)
    if T_world_cam is not None:
        return pose7_apply(T_world_cam, pts)
    if frame == "rep103":
        return opencv_to_rep103(pts)
    if frame != "opencv":
        raise ValueError(f"frame must be 'opencv' or 'rep103', got {frame!r}")
    return pts
