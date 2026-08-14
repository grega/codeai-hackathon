"""The reward function.

Phase 4 of the experience is about the reward: the training screen exposes
these weights as sliders so a child can change what earns points, retrain, and
watch a different behaviour emerge. Any real Trainer should score with this
module so that the sliders keep meaning what the UI says they mean.
"""

from __future__ import annotations

import math
from typing import Iterable

from schemas import (
    ARTICULATED_BONES,
    BONE_TREE,
    DEFAULT_REWARD_WEIGHTS,
    IDENTITY_QUAT,
)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Quaternion / vector helpers
# --------------------------------------------------------------------------

def quat_mul(a: Iterable[float], b: Iterable[float]) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q: Iterable[float], v: Vec3) -> Vec3:
    """Rotate vector ``v`` by quaternion ``q``."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_from_euler(x_deg: float, y_deg: float, z_deg: float) -> Quat:
    """Build a quaternion from XYZ Euler angles in degrees.

    Authoring poses by hand in quaternions is miserable; this is what the mock
    poser (and probably your prompt templates) should use.
    """
    hx, hy, hz = (math.radians(a) / 2.0 for a in (x_deg, y_deg, z_deg))
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    return (
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz,
    )


def quat_angle(a: Iterable[float], b: Iterable[float]) -> float:
    """Shortest angle in radians between two rotations, in [0, pi]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = abs(ax * bx + ay * by + az * bz + aw * bw)
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def quat_slerp(a: Iterable[float], b: Iterable[float], t: float) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:  # take the short way round
        bx, by, bz, bw, dot = -bx, -by, -bz, -bw, -dot
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:  # nearly parallel — lerp and normalise
        rx, ry, rz, rw = (ax + (bx - ax) * t, ay + (by - ay) * t,
                          az + (bz - az) * t, aw + (bw - aw) * t)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        s0 = math.sin((1.0 - t) * theta) / sin_theta
        s1 = math.sin(t * theta) / sin_theta
        rx, ry, rz, rw = (ax * s0 + bx * s1, ay * s0 + by * s1,
                          az * s0 + bz * s1, aw * s0 + bw * s1)

    length = math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw) or 1.0
    return (rx / length, ry / length, rz / length, rw / length)


# --------------------------------------------------------------------------
# Forward kinematics
# --------------------------------------------------------------------------

def world_positions(pose: dict[str, list[float]]) -> dict[str, Vec3]:
    """Resolve a pose to world-space joint positions.

    Used by the spatial reward terms (is the hand above the head?) and handy for
    anyone writing a real trainer. BONE_TREE is ordered parents-before-children,
    so one pass suffices.
    """
    positions: dict[str, Vec3] = {}
    rotations: dict[str, Quat] = {}

    for bone, (parent, offset) in BONE_TREE.items():
        local = tuple(pose.get(bone, IDENTITY_QUAT))
        if parent is None:
            parent_rot: Quat = IDENTITY_QUAT
            parent_pos: Vec3 = (0.0, 0.0, 0.0)
        else:
            parent_rot = rotations[parent]
            parent_pos = positions[parent]

        rotated = quat_rotate(parent_rot, offset)
        positions[bone] = (parent_pos[0] + rotated[0],
                           parent_pos[1] + rotated[1],
                           parent_pos[2] + rotated[2])
        rotations[bone] = quat_mul(parent_rot, local)

    return positions


# --------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------

def per_joint_error(current: dict[str, list[float]],
                    target: dict[str, list[float]]) -> dict[str, float]:
    """Normalised 0..1 error per articulated joint. Drives the joint heat colours."""
    return {
        bone: quat_angle(current.get(bone, IDENTITY_QUAT),
                         target.get(bone, IDENTITY_QUAT)) / math.pi
        for bone in ARTICULATED_BONES
    }


def pose_distance(current: dict[str, list[float]],
                  target: dict[str, list[float]]) -> float:
    """How far a pose is from a target, 0 (identical) .. 1 (maximally different).

    Half the average joint error and half the *worst* joint error. Plain
    averaging would be misleading here: most poses only move three or four of
    the eleven articulated joints, so a completely wrong arm barely moves a
    mean taken over the whole body — the score would sit near 0.9 the whole way
    through training and the learning curve would look flat. Counting the worst
    joint too means "one limb is still wrong" actually shows up.
    """
    errors = per_joint_error(current, target)
    if not errors:
        return 0.0
    mean = sum(errors.values()) / len(errors)
    worst = max(errors.values())
    return 0.5 * mean + 0.5 * worst


# --------------------------------------------------------------------------
# Reward terms — each returns 0..1, higher is better
# --------------------------------------------------------------------------

def _term_pose_match(current, target, previous) -> float:
    return 1.0 - pose_distance(current, target)


def _term_arm_height(current, target, previous) -> float:
    """Points for getting the hands up. The most legible term for a child:
    turn it up and the avatar reaches for the sky whatever the target says."""
    pos = world_positions(current)
    head_y = pos["head"][1]
    hips_y = pos["hips"][1]
    span = max(head_y - hips_y, 1e-6)
    scores = []
    for hand in ("L_hand", "R_hand"):
        # 0 when the hand is at hip height, 1 when it is a full torso above the head
        scores.append(_clamp01((pos[hand][1] - hips_y) / (span * 2.0)))
    return sum(scores) / len(scores)


def _term_symmetry(current, target, previous) -> float:
    """Points for the left and right sides mirroring each other."""
    pos = world_positions(current)
    total = 0.0
    pairs = (("L_hand", "R_hand"), ("L_elbow", "R_elbow"),
             ("L_foot", "R_foot"), ("L_knee", "R_knee"))
    for left, right in pairs:
        lx, ly, lz = pos[left]
        rx, ry, rz = pos[right]
        # mirror the right joint through the YZ plane, then compare
        d = math.dist((lx, ly, lz), (-rx, ry, rz))
        total += _clamp01(1.0 - d / 1.2)
    return total / len(pairs)


def _term_stillness(current, target, previous) -> float:
    """Points for not thrashing. Rewards smooth movement between steps."""
    if not previous:
        return 1.0
    return _clamp01(1.0 - pose_distance(current, previous) * 4.0)


TERMS = {
    "pose_match": _term_pose_match,
    "arm_height": _term_arm_height,
    "symmetry": _term_symmetry,
    "stillness": _term_stillness,
}

#: Shown next to each slider in the UI.
TERM_LABELS = {
    "pose_match": "Match the target pose",
    "arm_height": "Get the hands up high",
    "symmetry": "Keep both sides the same",
    "stillness": "Move smoothly",
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def reward(current: dict[str, list[float]],
           target: dict[str, list[float]],
           weights: dict[str, float] | None = None,
           previous: dict[str, list[float]] | None = None,
           ) -> tuple[float, dict[str, float]]:
    """Score a pose against a target.

    Returns ``(total, breakdown)`` where total is 0..1 and breakdown is the
    unweighted value of each term — the UI shows both, so a child can see
    *which* part of the reward they are earning, not just the number.

    Weights are normalised, so turning every slider up is the same as turning
    every slider down: it is the balance between terms that matters.
    """
    weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    breakdown = {name: fn(current, target, previous)
                 for name, fn in TERMS.items()}

    total_weight = sum(max(0.0, weights.get(n, 0.0)) for n in TERMS)
    if total_weight <= 0:
        # All sliders at zero: nothing is being rewarded. Say so rather than
        # dividing by zero — the UI surfaces this as a hint.
        return 0.0, breakdown

    total = sum(breakdown[n] * max(0.0, weights.get(n, 0.0))
                for n in TERMS) / total_weight
    return _clamp01(total), breakdown
