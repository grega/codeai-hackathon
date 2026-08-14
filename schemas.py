"""The data contract.

Everything that crosses a provider boundary is defined here. If you are
implementing a real Rigger/Poser/Trainer, this file is the only vocabulary you
need — see CONTRACT.md and docs/adding-a-provider.md.

This module stands alone at the bottom of the import graph. Keep it that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Skeleton
# --------------------------------------------------------------------------
# 16 bones, fixed names, fixed hierarchy. This set is CLOSED: a pose mentioning
# any other name is a hard error, not a silent drop. Cross-team drift in bone
# naming is the single most likely way this prototype breaks, so we fail loudly.

# name -> (parent or None, rest offset from parent in metres)
BONE_TREE: dict[str, tuple[str | None, tuple[float, float, float]]] = {
    "hips":       (None,          (0.00,  0.00, 0.0)),
    "spine":      ("hips",        (0.00,  0.25, 0.0)),
    "neck":       ("spine",       (0.00,  0.25, 0.0)),
    "head":       ("neck",        (0.00,  0.15, 0.0)),

    "L_shoulder": ("spine",       (-0.18, 0.20, 0.0)),
    "L_elbow":    ("L_shoulder",  (-0.25, 0.00, 0.0)),
    "L_hand":     ("L_elbow",     (-0.22, 0.00, 0.0)),

    "R_shoulder": ("spine",       (0.18,  0.20, 0.0)),
    "R_elbow":    ("R_shoulder",  (0.25,  0.00, 0.0)),
    "R_hand":     ("R_elbow",     (0.22,  0.00, 0.0)),

    "L_hip":      ("hips",        (-0.10, -0.05, 0.0)),
    "L_knee":     ("L_hip",       (0.00, -0.35, 0.0)),
    "L_foot":     ("L_knee",      (0.00, -0.35, 0.0)),

    "R_hip":      ("hips",        (0.10, -0.05, 0.0)),
    "R_knee":     ("R_hip",       (0.00, -0.35, 0.0)),
    "R_foot":     ("R_knee",      (0.00, -0.35, 0.0)),
}

BONES: list[str] = list(BONE_TREE)
BONE_SET: frozenset[str] = frozenset(BONES)

#: Bones a learner is allowed to move during training. Feet/hands are IK-ish
#: leaves that follow their parents; letting the policy drive them makes the
#: search space needlessly large for no visible gain.
ARTICULATED_BONES: list[str] = [
    "spine", "neck", "head",
    "L_shoulder", "L_elbow",
    "R_shoulder", "R_elbow",
    "L_hip", "L_knee",
    "R_hip", "R_knee",
]

IDENTITY_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

#: The rest pose: every bone at identity rotation. A Pose is always expressed
#: as local rotations *relative to rest*, so this is the neutral pose too.
REST_POSE: dict[str, list[float]] = {b: list(IDENTITY_QUAT) for b in BONES}

MAX_KEYFRAMES = 240
MAX_CLIP_DURATION = 30.0
QUAT_TOLERANCE = 1e-3


class ProviderError(Exception):
    """Raised by a provider when it cannot complete the request.

    ``user_message`` is rendered verbatim to a child, so write it for an
    11-year-old: "I couldn't read that drawing — try one with clearer lines."
    Anything else (stack traces, model errors, API keys) never leaves the server.
    """

    def __init__(self, user_message: str, detail: str | None = None):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail


class ContractError(ValueError):
    """A provider returned something that violates this module's schema."""


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass
class Rig:
    """A rigged avatar.

    ``format`` decides how the browser renders it:
      - "procedural": the viewport builds geometry from the bone tree itself,
        so it stands up with only this bone list. This is what the mocks return.
      - "glb": the viewport loads ``glb_bytes`` via GLTFLoader. Bone names in
        the GLB must match BONES exactly.

    Both paths hit the same viewport code, so switching a real rigger on
    requires no frontend change.
    """

    format: str = "procedural"
    skeleton: list[str] = field(default_factory=lambda: list(BONES))
    glb_bytes: bytes | None = None
    #: free-form, shown in the UI ("I saw a person with long arms")
    notes: str = ""

    def to_json(self, glb_url: str | None = None) -> dict[str, Any]:
        return {
            "format": self.format,
            "skeleton": self.skeleton,
            "bone_tree": {k: {"parent": v[0], "offset": list(v[1])}
                          for k, v in BONE_TREE.items()},
            "glb_url": glb_url,
            "notes": self.notes,
        }


@dataclass
class Keyframe:
    t: float                       # seconds from clip start
    pose: dict[str, list[float]]   # bone name -> [x, y, z, w]

    def to_json(self) -> dict[str, Any]:
        return {"t": self.t, "pose": self.pose}


@dataclass
class Clip:
    """A named animation. A single pose is just a 1-keyframe clip, which is why
    phase 2 (generate a pose) and phase 3 (hit a target pose) speak one type."""

    name: str
    keyframes: list[Keyframe]
    fps: int = 24
    id: str | None = None
    prompt: str = ""
    loop: bool = False

    @property
    def duration(self) -> float:
        return self.keyframes[-1].t if self.keyframes else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "fps": self.fps,
            "loop": self.loop,
            "duration": self.duration,
            "keyframes": [k.to_json() for k in self.keyframes],
        }


@dataclass
class TrainConfig:
    """Knobs the training screen exposes. ``reward_weights`` is the pedagogical
    payload — changing what earns points is the whole lesson."""

    episodes: int = 300
    #: How much of an improving change to keep. Low values look like careful
    #: nudging but need thousands of episodes to cross a large pose change, so
    #: the default is deliberately bold enough to converge inside one demo.
    learning_rate: float = 0.6
    exploration: float = 0.8
    seed: int = 0
    reward_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> "TrainConfig":
        data = data or {}
        weights = dict(DEFAULT_REWARD_WEIGHTS)
        for k, v in (data.get("reward_weights") or {}).items():
            if k in weights:
                weights[k] = _clamp(float(v), 0.0, 1.0)
        return cls(
            episodes=int(_clamp(int(data.get("episodes", 300)), 1, 5000)),
            learning_rate=_clamp(float(data.get("learning_rate", 0.6)), 0.001, 1.0),
            exploration=_clamp(float(data.get("exploration", 0.8)), 0.0, 1.0),
            seed=int(data.get("seed", 0)),
            reward_weights=weights,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    """One training episode, streamed to the browser over SSE."""

    episode: int
    reward: float
    best_reward: float
    pose: dict[str, list[float]]
    per_joint_error: dict[str, float] = field(default_factory=dict)
    exploration: float = 0.0
    done: bool = False
    note: str = ""

    #: Progress from the starting pose to the target, 0..1. This is the headline
    #: number the child reads ("68% there"), and it is deliberately NOT the
    #: reward: reward is what the algorithm maximises and depends on the slider
    #: weights, whereas match always answers "how close to the target am I".
    #: When the sliders are set to reward something other than copying the
    #: target, the two diverge — which is exactly the lesson.
    match: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


#: The reward terms the UI exposes as sliders. Keys are stable; see rewards.py.
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "pose_match": 1.0,   # how close every joint is to the target
    "arm_height": 0.0,   # bonus for hands above the head
    "symmetry": 0.0,     # bonus for mirrored left/right limbs
    "stillness": 0.0,    # penalty for large moves between steps
}


# --------------------------------------------------------------------------
# Validation — every provider return value goes through these before it
# reaches the store, so a bad implementation fails at its own boundary.
# --------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float):
    return max(lo, min(hi, v))


def validate_pose(pose: Any, *, where: str = "pose") -> dict[str, list[float]]:
    """Return a normalised copy of ``pose``, or raise ContractError.

    Missing bones are filled from rest (a provider need only send what it
    moves). Unknown bones are an error.
    """
    if not isinstance(pose, dict):
        raise ContractError(f"{where}: expected an object of bone -> quaternion, "
                            f"got {type(pose).__name__}")

    unknown = sorted(set(pose) - BONE_SET)
    if unknown:
        raise ContractError(
            f"{where}: unknown bone name(s) {unknown}. "
            f"Valid bones are: {', '.join(BONES)}")

    out = {b: list(IDENTITY_QUAT) for b in BONES}
    for bone, quat in pose.items():
        if not isinstance(quat, (list, tuple)) or len(quat) != 4:
            raise ContractError(
                f"{where}.{bone}: expected a quaternion [x, y, z, w], got {quat!r}")
        try:
            x, y, z, w = (float(c) for c in quat)
        except (TypeError, ValueError):
            raise ContractError(f"{where}.{bone}: quaternion must be 4 numbers, "
                                f"got {quat!r}") from None
        if not all(math.isfinite(c) for c in (x, y, z, w)):
            raise ContractError(f"{where}.{bone}: quaternion contains NaN/inf")

        length = math.sqrt(x * x + y * y + z * z + w * w)
        if length < QUAT_TOLERANCE:
            raise ContractError(f"{where}.{bone}: zero-length quaternion {quat!r}")
        out[bone] = [x / length, y / length, z / length, w / length]
    return out


def validate_clip(clip: Any, *, where: str = "clip") -> Clip:
    """Return a validated Clip, or raise ContractError.

    Accepts either a Clip instance or the equivalent plain dict, so a provider
    can return whichever is convenient.
    """
    if isinstance(clip, Clip):
        name, fps, keyframes = clip.name, clip.fps, clip.keyframes
        prompt, loop, cid = clip.prompt, clip.loop, clip.id
    elif isinstance(clip, dict):
        name = clip.get("name", "untitled")
        fps = clip.get("fps", 24)
        keyframes = clip.get("keyframes", [])
        prompt = clip.get("prompt", "")
        loop = bool(clip.get("loop", False))
        cid = clip.get("id")
    else:
        raise ContractError(f"{where}: expected a Clip or dict, "
                            f"got {type(clip).__name__}")

    if not keyframes:
        raise ContractError(f"{where}: must have at least one keyframe")
    if len(keyframes) > MAX_KEYFRAMES:
        raise ContractError(f"{where}: {len(keyframes)} keyframes exceeds the "
                            f"limit of {MAX_KEYFRAMES}")
    if not isinstance(fps, int) or not 1 <= fps <= 120:
        raise ContractError(f"{where}.fps: expected an int in 1..120, got {fps!r}")

    out: list[Keyframe] = []
    last_t = -1.0
    for i, kf in enumerate(keyframes):
        t = kf.t if isinstance(kf, Keyframe) else kf.get("t")
        pose = kf.pose if isinstance(kf, Keyframe) else kf.get("pose")
        try:
            t = float(t)
        except (TypeError, ValueError):
            raise ContractError(f"{where}.keyframes[{i}].t: expected a number, "
                                f"got {t!r}") from None
        if not math.isfinite(t) or t < 0:
            raise ContractError(f"{where}.keyframes[{i}].t: must be >= 0, got {t}")
        if t <= last_t and i > 0:
            raise ContractError(
                f"{where}.keyframes[{i}].t = {t} is not after the previous "
                f"keyframe at {last_t}. Keyframe times must strictly increase.")
        if t > MAX_CLIP_DURATION:
            raise ContractError(f"{where}.keyframes[{i}].t = {t} exceeds the "
                                f"{MAX_CLIP_DURATION}s clip limit")
        last_t = t
        out.append(Keyframe(t=t, pose=validate_pose(
            pose, where=f"{where}.keyframes[{i}].pose")))

    return Clip(id=cid, name=str(name)[:80], keyframes=out, fps=fps,
                prompt=str(prompt)[:400], loop=loop)


def validate_rig(rig: Any, *, where: str = "rig") -> Rig:
    if not isinstance(rig, Rig):
        raise ContractError(f"{where}: expected a Rig, got {type(rig).__name__}")
    if rig.format not in ("procedural", "glb"):
        raise ContractError(
            f"{where}.format: expected 'procedural' or 'glb', got {rig.format!r}")
    if rig.format == "glb" and not rig.glb_bytes:
        raise ContractError(f"{where}: format is 'glb' but glb_bytes is empty")

    missing = sorted(BONE_SET - set(rig.skeleton))
    unknown = sorted(set(rig.skeleton) - BONE_SET)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if unknown:
            problems.append(f"unknown {unknown}")
        raise ContractError(
            f"{where}.skeleton: {'; '.join(problems)}. A rig must expose exactly "
            f"the {len(BONES)} contract bones.")
    return rig


def validate_episode(ep: Any, *, where: str = "episode") -> Episode:
    if not isinstance(ep, Episode):
        raise ContractError(f"{where}: expected an Episode, "
                            f"got {type(ep).__name__}")
    if ep.episode < 0:
        raise ContractError(f"{where}.episode: must be >= 0, got {ep.episode}")
    for name in ("reward", "best_reward"):
        v = getattr(ep, name)
        if not math.isfinite(v):
            raise ContractError(f"{where}.{name}: must be a finite number, got {v}")
    ep.pose = validate_pose(ep.pose, where=f"{where}.pose")
    return ep


def schema_json() -> dict[str, Any]:
    """Served at GET /api/schema so the browser never hard-codes bone names."""
    return {
        "bones": BONES,
        "articulated_bones": ARTICULATED_BONES,
        "bone_tree": {k: {"parent": v[0], "offset": list(v[1])}
                      for k, v in BONE_TREE.items()},
        "rest_pose": REST_POSE,
        "reward_weights": DEFAULT_REWARD_WEIGHTS,
        "limits": {
            "max_keyframes": MAX_KEYFRAMES,
            "max_clip_duration": MAX_CLIP_DURATION,
        },
    }
