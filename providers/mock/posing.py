"""Mock poser: keyword-matches the prompt against hand-authored clips.

An unknown prompt still produces *something* — a deterministic pose seeded from
the prompt text — so a live demo never dead-ends in front of a classroom.

TODO(real): implement providers/real/posing.py with a class `RealPoser`
subclassing providers.base.Poser, satisfying `Poser.pose()`. Send the prompt and
the bone list (schemas.BONES) to an LLM, ask for a JSON clip in the documented
shape, and return it. schemas.validate_clip() will tell you loudly if the model
invents a bone name.
"""

from __future__ import annotations

import hashlib
import time

from providers.base import Poser, Progress
from rewards import quat_from_euler
from schemas import ARTICULATED_BONES, Clip, Keyframe, ProviderError, Rig

# Rotation conventions for this skeleton. Worth reading before authoring a pose
# — they are not guessable, and getting one backwards is the usual reason a pose
# comes out looking broken. All angles are degrees, applied XYZ.
#
#   ARMS. The left arm rests along -X, the right along +X.
#     raise:        L_shoulder z = -90 is straight up;  R_shoulder z = +90
#                   (past 90 the arm crosses the midline, so stay under ~110)
#     swing fwd:    rotate about Y. +Y swings the LEFT arm forward and the
#                   RIGHT arm backward, which is what you want for a walk.
#     (rotating an arm about X does nothing — it is already the X axis.)
#
#   LEGS. Both legs rest along -Y.
#     swing fwd:    NEGATIVE x rotation (-30 = thigh forward)
#     knee bend:    POSITIVE x rotation (knees bend backwards)
#     spread out:   L_hip z = -25 opens left, R_hip z = +25 opens right
#
#   SPINE. +X leans forward, z tilts side to side.

def _kf(t: float, **bones_degrees) -> Keyframe:
    """Author a keyframe as `bone=(x_deg, y_deg, z_deg)`."""
    return Keyframe(t=t, pose={bone: list(quat_from_euler(*angles))
                               for bone, angles in bones_degrees.items()})


def _wave() -> list[Keyframe]:
    """Right arm up, hand swinging side to side."""
    return [
        _kf(0.0, R_shoulder=(0, 0, 85), R_elbow=(0, 0, 30)),
        _kf(0.4, R_shoulder=(0, 0, 95), R_elbow=(0, 0, -20)),
        _kf(0.8, R_shoulder=(0, 0, 85), R_elbow=(0, 0, 30)),
        _kf(1.2, R_shoulder=(0, 0, 95), R_elbow=(0, 0, -20)),
        _kf(1.6, R_shoulder=(0, 0, 85), R_elbow=(0, 0, 30)),
    ]


def _arms_up() -> list[Keyframe]:
    """Both arms overhead, waving. The canonical 'waves arms in the air'."""
    return [
        _kf(0.0, L_shoulder=(0, 0, -80), R_shoulder=(0, 0, 80),
            L_elbow=(0, 0, -15), R_elbow=(0, 0, 15)),
        _kf(0.5, L_shoulder=(0, 0, -65), R_shoulder=(0, 0, 95),
            L_elbow=(0, 0, -25), R_elbow=(0, 0, 25), spine=(0, 0, 6)),
        _kf(1.0, L_shoulder=(0, 0, -95), R_shoulder=(0, 0, 65),
            L_elbow=(0, 0, -25), R_elbow=(0, 0, 25), spine=(0, 0, -6)),
        _kf(1.5, L_shoulder=(0, 0, -80), R_shoulder=(0, 0, 80),
            L_elbow=(0, 0, -15), R_elbow=(0, 0, 15)),
    ]


def _jump() -> list[Keyframe]:
    return [
        # crouch: thighs forward, knees bent, lean in
        _kf(0.0, L_hip=(-35, 0, 0), R_hip=(-35, 0, 0),
            L_knee=(70, 0, 0), R_knee=(70, 0, 0), spine=(20, 0, 0)),
        # launch: straighten out, arms up
        _kf(0.35, L_shoulder=(0, 0, -75), R_shoulder=(0, 0, 75),
            L_knee=(5, 0, 0), R_knee=(5, 0, 0)),
        _kf(0.7, L_shoulder=(0, 0, -95), R_shoulder=(0, 0, 95)),
        # land
        _kf(1.1, L_hip=(-30, 0, 0), R_hip=(-30, 0, 0),
            L_knee=(60, 0, 0), R_knee=(60, 0, 0), spine=(15, 0, 0)),
        _kf(1.5),
    ]


def _dance() -> list[Keyframe]:
    return [
        _kf(0.0, L_shoulder=(0, 0, -100), R_shoulder=(0, 0, 30),
            spine=(0, 0, 10), L_hip=(-20, 0, 0)),
        _kf(0.4, L_shoulder=(0, 0, -30), R_shoulder=(0, 0, 100),
            spine=(0, 0, -10), R_hip=(-20, 0, 0)),
        _kf(0.8, L_shoulder=(0, 0, -100), R_shoulder=(0, 0, 30),
            spine=(0, 0, 10), L_hip=(-20, 0, 0)),
        _kf(1.2, L_shoulder=(0, 0, -30), R_shoulder=(0, 0, 100),
            spine=(0, 0, -10), R_hip=(-20, 0, 0)),
        _kf(1.6, L_shoulder=(0, 0, -100), R_shoulder=(0, 0, 30),
            spine=(0, 0, 10)),
    ]


def _walk() -> list[Keyframe]:
    return [
        # left leg forward, right leg trailing with a bent knee; arms opposite
        _kf(0.0, L_hip=(-30, 0, 0), R_hip=(30, 0, 0), R_knee=(25, 0, 0),
            L_shoulder=(0, 25, 0), R_shoulder=(0, 25, 0)),
        _kf(0.5, L_hip=(30, 0, 0), R_hip=(-30, 0, 0), L_knee=(25, 0, 0),
            L_shoulder=(0, -25, 0), R_shoulder=(0, -25, 0)),
        _kf(1.0, L_hip=(-30, 0, 0), R_hip=(30, 0, 0), R_knee=(25, 0, 0),
            L_shoulder=(0, 25, 0), R_shoulder=(0, 25, 0)),
    ]


def _t_pose() -> list[Keyframe]:
    return [_kf(0.0)]


def _star_jump() -> list[Keyframe]:
    return [
        _kf(0.0),
        # arms and legs out on the diagonal
        _kf(0.4, L_shoulder=(0, 0, -45), R_shoulder=(0, 0, 45),
            L_hip=(0, 0, -25), R_hip=(0, 0, 25)),
        _kf(0.8),
    ]


def _bow() -> list[Keyframe]:
    return [
        _kf(0.0),
        _kf(0.6, spine=(60, 0, 0), neck=(20, 0, 0)),
        _kf(1.4, spine=(60, 0, 0), neck=(20, 0, 0)),
        _kf(2.0),
    ]


#: keyword -> (clip name, keyframe builder). First match wins, so put more
#: specific phrases first.
_LIBRARY: list[tuple[tuple[str, ...], str, callable]] = [
    (("arms in air", "arms up", "both arms", "reach up", "hands up",
      "arms in the air"), "Arms in the air", _arms_up),
    (("star jump", "jumping jack"), "Star jump", _star_jump),
    (("wave", "waving", "hello", "hi ", "greet"), "Wave", _wave),
    (("jump", "hop", "leap"), "Jump", _jump),
    (("dance", "dancing", "boogie", "wiggle"), "Dance", _dance),
    (("walk", "walking", "step", "march"), "Walk", _walk),
    (("bow", "bowing", "thank you"), "Bow", _bow),
    (("t-pose", "t pose", "stand", "neutral", "rest", "still"), "Stand", _t_pose),
]

_STAGES = [
    (0.2, "Reading your words..."),
    (0.55, "Imagining the pose..."),
    (0.85, "Moving the joints..."),
]


def _fallback(prompt: str) -> list[Keyframe]:
    """Deterministic nonsense pose derived from the prompt.

    Same words always give the same pose, which matters: a child typing the same
    thing twice and getting a different avatar would undermine the whole lesson
    about inputs mapping to outputs.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    keyframes = []
    for k in range(3):
        bones = {}
        for i, bone in enumerate(ARTICULATED_BONES):
            byte = digest[(i + k * 7) % len(digest)]
            # -70..70 degrees, mostly around the Z axis so it reads as a pose
            angle = (byte / 255.0) * 140.0 - 70.0
            bones[bone] = (angle * 0.3, 0.0, angle)
        keyframes.append(_kf(k * 0.6, **bones))
    return keyframes


class MockPoser(Poser):
    def pose(self, prompt: str, rig: Rig, progress: Progress) -> Clip:
        text = (prompt or "").strip().lower()
        if not text:
            raise ProviderError("Tell me what you'd like your avatar to do!",
                                detail="empty prompt")

        for fraction, message in _STAGES:
            progress(fraction, message)
            time.sleep(0.3)

        for keywords, name, builder in _LIBRARY:
            if any(word in text for word in keywords):
                return Clip(name=name, prompt=prompt, keyframes=builder(),
                            loop=True)

        return Clip(name=prompt[:40].strip().capitalize() or "Made-up move",
                    prompt=prompt, keyframes=_fallback(text), loop=True)
