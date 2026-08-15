"""Procedural animation generators.

Each generator returns {node_index: (times, quaternions)} suitable for
gltf_utils.add_rotation_animation. Keyframes are the joint's rest rotation
composed with a small time-varying offset, so frame 0 matches the bind pose.

These two (`generate_wave`/`generate_spin`) are hand-written references -- they
encode the world-axis rotation approach worked out by hand and verified against
this specific rig. `generate()` tries the LLM first (llm_animator.generate, which
writes the same kind of code against the same helpers) and falls back to these
keyword-matched ones if the LLM is unavailable or produces something invalid.
"""
from __future__ import annotations

import numpy as np

import llm_animator
import quat
from gltf_utils import GLTF2, Joint, extract_pose, find_joint, local_rotation_for_world_delta

FPS = 30


def _keyframe_times(duration: float) -> list[float]:
    n = max(2, int(duration * FPS))
    return list(np.linspace(0.0, duration, n))


def generate_wave(gltf: GLTF2, joints: list[Joint], duration: float = 2.0) -> dict[int, tuple[list[float], list[np.ndarray]]]:
    """This rig rests in a T-pose (arm already extended out to the side), so a wave
    is just an elbow bend -- lifting the forearm up around the *world* Z axis, with a
    smaller oscillation layered on top for the side-to-side motion. World axis, not
    local: this bone's own local axes don't line up with anything intuitive, which is
    what sent the arm behind the model's back the first time around.

    Bending the forearm carries the hand along rigidly, including whatever twist was
    baked into its T-pose rest orientation -- which pointed the palm out to the side,
    not forward. A second, independent wrist correction (also computed via a world
    axis, this time the forearm's *current* bent orientation so the twist axis tracks
    the limb) fixes the palm to face forward, verified by simulating finger positions
    at candidate angles and checking which one points the palm normal at +Z (forward,
    per this rig's toe-bone direction).
    """
    forearm = find_joint(joints, "rightforearm")
    hand = find_joint(joints, "righthand")
    if forearm is None or hand is None:
        raise ValueError("could not find right forearm/hand joints on this rig")

    times = _keyframe_times(duration)
    bend_angle = np.radians(-100)  # verified against this rig: negative lifts the hand up
    wave_amplitude = np.radians(20)
    wrist_correction = np.radians(90)  # verified: faces the palm forward (+Z) after the bend

    forearm_quats = []
    hand_quats = []
    for t in times:
        bend = bend_angle + wave_amplitude * np.sin(2 * np.pi * 2.5 * t / duration)
        forearm_delta = quat.from_axis_angle((0, 0, 1), bend)
        forearm_local = local_rotation_for_world_delta(gltf, forearm, forearm_delta)
        forearm_quats.append(forearm_local)

        # Temporarily apply the bend so the wrist correction is computed relative to
        # the *animated* forearm orientation, not its rest pose, then restore it --
        # gltf.nodes must keep the original bind pose for the saved file.
        gltf.nodes[forearm.node_index].rotation = forearm_local.tolist()
        wrist_delta = quat.from_axis_angle((0, 1, 0), wrist_correction)
        hand_quats.append(local_rotation_for_world_delta(gltf, hand, wrist_delta))
        gltf.nodes[forearm.node_index].rotation = forearm.rest_rotation.tolist()

    return {
        forearm.node_index: (times, forearm_quats),
        hand.node_index: (times, hand_quats),
    }


def generate_spin(gltf: GLTF2, joints: list[Joint], duration: float = 2.0) -> dict[int, tuple[list[float], list[np.ndarray]]]:
    hips = find_joint(joints, "hips")
    if hips is None:
        raise ValueError("could not find a hips joint on this rig")

    times = _keyframe_times(duration)
    quats = []
    for t in times:
        angle = 2 * np.pi * (t / duration)
        delta = quat.from_axis_angle((0, 1, 0), angle)
        quats.append(local_rotation_for_world_delta(gltf, hips, delta))

    return {hips.node_index: (times, quats)}


GENERATORS = {
    "wave": generate_wave,
    "spin": generate_spin,
}


def generate(
    prompt: str, gltf: GLTF2, joints: list[Joint], use_llm: bool = True
) -> tuple[str, dict[int, tuple[list[float], list[np.ndarray]]]]:
    llm_error = None
    if use_llm:
        try:
            return "generated", llm_animator.generate(prompt, gltf, joints)
        except Exception as e:
            llm_error = str(e)
            print(f"[animator] LLM generation failed, falling back to procedural keywords: {e}")

    lower = prompt.lower()
    for keyword, fn in GENERATORS.items():
        if keyword in lower:
            return keyword, fn(gltf, joints)

    if llm_error is not None:
        raise RuntimeError(
            f"could not animate {prompt!r}: the LLM failed ({llm_error}), and no hardcoded "
            f"fallback matches either (known keywords: {list(GENERATORS)}). Try rephrasing, "
            f"or re-run -- LLM output is non-deterministic."
        )
    raise ValueError(
        f"no procedural generator matches prompt {prompt!r}; known keywords: {list(GENERATORS)}"
    )


_POSE_EPSILON_DEGREES = 2.0


def _moved_joints(
    pose_a: dict[str, np.ndarray],
    pose_b: dict[str, np.ndarray],
    epsilon_degrees: float = _POSE_EPSILON_DEGREES,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Joints present in both poses whose rotation differs by more than
    `epsilon_degrees` -- drops floating-point/re-export noise (observed ~1.2 degrees
    on untouched joints between rigged_human.glb and frame_1.glb) while keeping real
    motion (observed ~23-51 degrees on the shoulders/arms in that pair)."""
    result = {}
    for name, start_q in pose_a.items():
        end_q = pose_b.get(name)
        if end_q is None:
            continue
        if quat.angle_between(start_q, end_q) > epsilon_degrees:
            result[name] = (start_q, end_q)
    return result


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def _transition_fallback(
    start_pose: dict[str, np.ndarray],
    end_pose: dict[str, np.ndarray],
    duration: float = 2.0,
) -> dict[str, tuple[list[float], list[np.ndarray]]]:
    """Deterministic no-LLM baseline: every moved joint slerps start->end pose on the
    same smoothstep-eased timeline. Rig-agnostic -- operates on whatever joint names
    are in start_pose/end_pose, unlike generate_wave/generate_spin's hardcoded names."""
    times = _keyframe_times(duration)
    tracks = {}
    for name, start_q in start_pose.items():
        end_q = end_pose[name]
        quats = [quat.slerp(start_q, end_q, _smoothstep(t / duration)) for t in times]
        tracks[name] = (times, quats)
    return tracks


def generate_transition(
    joints_a: list[Joint],
    joints_b: list[Joint],
    duration: float = 2.0,
    style_prompt: str | None = None,
    use_llm: bool = True,
) -> tuple[str, dict[int, tuple[list[float], list[np.ndarray]]]]:
    pose_a = extract_pose(joints_a)
    pose_b = extract_pose(joints_b)
    moved = _moved_joints(pose_a, pose_b)
    if not moved:
        raise ValueError(
            "pose-b matches pose-a (within 2 degrees) on every non-leaf joint -- nothing to animate"
        )
    start_pose = {name: s for name, (s, _e) in moved.items()}
    end_pose = {name: e for name, (_s, e) in moved.items()}

    tracks_by_name = None
    if use_llm:
        try:
            tracks_by_name = llm_animator.generate_transition(start_pose, end_pose, duration, style_prompt)
        except Exception as e:
            print(f"[animator] LLM transition generation failed, falling back to slerp baseline: {e}")

    if tracks_by_name is None:
        tracks_by_name = _transition_fallback(start_pose, end_pose, duration)

    by_name = {j.name: j.node_index for j in joints_a}
    tracks = {by_name[name]: value for name, value in tracks_by_name.items() if name in by_name}
    return "pose_transition", tracks
