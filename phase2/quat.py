"""Minimal quaternion math for building animation keyframes (xyzw convention, matching glTF)."""
from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a * b, both xyzw. Applying the result rotates by b first, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def conjugate(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    return multiply(multiply(q, qv), conjugate(q))[:3]


def from_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle_rad / 2.0)])


def from_axis_angle_degrees(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    return from_axis_angle(axis, np.radians(angle_deg))


def combine(*deltas: np.ndarray) -> np.ndarray:
    """Compose rotations expressed in the same frame (e.g. several world-space
    deltas on one joint) into one, applied in the order given: the first argument
    first, then the next on top of it, and so on."""
    result = IDENTITY
    for d in deltas:
        result = multiply(d, result)
    return result


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
