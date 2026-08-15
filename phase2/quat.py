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


def angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle in degrees between two rotations, shortest-path (abs(dot) handles the
    double-cover sign ambiguity -- q and -q represent the same rotation)."""
    q1, q2 = normalize(q1), normalize(q2)
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)))


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation, shortest path. t=0 -> q1, t=1 -> q2 exactly."""
    q1, q2 = normalize(q1), normalize(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:  # antipodal quats are the same rotation; flip to take the short way
        q2, dot = -q2, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly identical: sin-ratio formula divides by ~sin(0), unstable
        return normalize(q1 + t * (q2 - q1))
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    a = np.sin((1.0 - t) * theta_0) / sin_theta_0
    b = np.sin(t * theta_0) / sin_theta_0
    return a * q1 + b * q2


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
