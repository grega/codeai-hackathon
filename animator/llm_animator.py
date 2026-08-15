"""Has the LLM write the *code* for a procedural animation, not raw keyframe numbers.

Raw dense numeric output (per-frame quaternions) is exactly what LLMs are unreliable
at; a few lines of trig calling our own kinematics helpers is exactly what they're
good at. The model gets the joint hierarchy plus the same world-axis rotation helper
we built and verified by hand for animator.generate_wave, and writes an animate()
function against it -- so it inherits the "reason in world axes, not arbitrary bone
-local axes" fix from that debugging session instead of repeating the mistake.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
from pygltflib import GLTF2

import llm_client
import quat
from gltf_utils import Joint, find_joint, local_rotation_for_world_delta

# Finger/eye sub-joints add ~50 entries of noise for prompts that rarely need them.
_SKIP_NAME_PARTS = ("thumb", "index", "ring", "pinky", "eye", "toe_end", "handmiddle2", "handmiddle3", "handmiddle4")


def _describe_skeleton(joints: list[Joint]) -> str:
    by_index = {j.node_index: j for j in joints}
    lines = []
    for j in joints:
        if any(part in j.name.lower() for part in _SKIP_NAME_PARTS):
            continue
        parent = by_index.get(j.parent_index)
        lines.append(f"- node {j.node_index}: {j.name} (parent: {parent.name if parent else None})")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You write short Python functions that animate a rigged 3D character by rotating its
skeleton joints. You are given a fixed set of helpers -- use ONLY these, do not import
anything or access the filesystem/network:

- `axis_angle(axis: tuple[float, float, float], angle_degrees: float)`
  Builds a rotation around a WORLD-space axis, angle in DEGREES (not radians -- e.g.
  pass 30 for a 30 degree rotation, not 0.52). World axes for this character: +Y = up,
  +Z = forward (the direction the character faces), +X = the character's left side,
  -X = the character's right side.
- `local_rotation_for_world_delta(gltf, joint, world_delta_quaternion)`
  Converts a world-space rotation into the joint's own local rotation space, composed
  on top of its rest pose. ALWAYS go through this function to rotate a joint -- never
  guess a joint's local axes directly, they are arbitrary per-bone and will not do
  what you expect.
- `find_joint(joints, name_substring)` -- case-insensitive substring lookup by name.
- `combine(*deltas)` -- composes several world-space rotations (each built with
  `axis_angle`) into one, applied in the order given. Use this whenever a single
  joint needs more than one simultaneous motion (e.g. lift AND sway at once) --
  do NOT invent a workaround or split one motion across unrelated joints because you
  think you can't combine rotations; you can, this is how.
- `np` (numpy) -- only for `np.sin`/`np.cos`/`np.pi`/`np.linspace` to shape motion over
  time. Those still take radians as normal (e.g. `2 * np.pi * t / duration`), that is
  unrelated to `axis_angle`'s degrees argument -- do not mix the two up.

Write exactly one function:

    def animate(gltf, joints, duration):
        times = [...]        # floats in [0, duration], at least 2, ~30/sec is plenty
        tracks = {}           # joint.node_index (an int) -> (times, [quaternion, ...])
        ...
        return tracks

Worked example -- spin the character around by rotating the hips (copy this pattern
exactly: `find_joint` returns a Joint object, but the dict key must be its
`.node_index`, and every rotation goes through `local_rotation_for_world_delta`):

    def animate(gltf, joints, duration):
        hips = find_joint(joints, "hips")
        times = list(np.linspace(0, duration, 60))
        quats = []
        for t in times:
            angle_deg = 360 * (t / duration)
            world_delta = axis_angle((0, 1, 0), angle_deg)
            quats.append(local_rotation_for_world_delta(gltf, hips, world_delta))
        return {hips.node_index: (times, quats)}

Second example -- combining two simultaneous motions (lift + side-to-side sway) on
one joint, using `combine`:

    lift = axis_angle((0, 0, 1), 100)
    sway = axis_angle((0, 1, 0), 15 * np.sin(2 * np.pi * t / duration))
    world_delta = combine(lift, sway)
    quats.append(local_rotation_for_world_delta(gltf, joint, world_delta))

Rules:
- Dict keys in `tracks` MUST be `joint.node_index` (an int), never the Joint object.
- Only use joint names that literally appear in the skeleton list below -- do not
  invent or guess names like "LeftElbow" (this rig calls it "LeftForeArm").
- Every track's quaternion list must be the same length as its times list.
- Keep it anatomically plausible: prefer 2-4 joints with rotation ranges of tens of
  degrees, not hundreds, unless the action clearly needs more (e.g. a full spin).
- If you animate both a joint AND its parent joint, their rotations COMPOSE: the
  child inherits the parent's motion, then its own delta applies on top of that. Do
  NOT give a child joint the inverse/negative of its parent's rotation "to correct
  it" -- that cancels the parent's motion and freezes everything below it in place
  (only do this deliberately, e.g. keeping a hand facing forward while the forearm
  above it swings -- not as a default habit).
- This rig's REST POSE is a T-POSE: both arms already extend roughly horizontally out
  to the sides, not hanging down. A rotation "raises" an arm relative to that already-
  extended position, so ~80-100 degrees on the shoulder/upper-arm alone is enough to
  bring a hand overhead. If you ALSO add a large rotation on the forearm (its child)
  on top of that, the two compose and add up along the limb -- this easily overshoots
  past vertical and swings the hand behind the body/head. When a parent joint's
  rotation already accounts for most of the intended motion, keep the child's own
  delta small (a few degrees to tens, not another 80-100).
- Output ONLY the function definition. No markdown fences, no example call, no prose.
"""

TRANSITION_SYSTEM_PROMPT = """\
You write short Python functions that animate a rigged 3D character transitioning
from one known pose to another known pose. You are given a fixed set of helpers --
use ONLY these, do not import anything or access the filesystem/network:

- `slerp(q1, q2, t)` -- spherical interpolation between two quaternions (xyzw,
  matching glTF). `t=0.0` returns exactly `q1`, `t=1.0` returns exactly `q2`, values
  in between blend smoothly along the shortest rotational path. This is the ONLY way
  you build a rotation in this sandbox -- you never construct a quaternion from
  scratch or guess an axis/angle here, you only interpolate between the two given
  endpoints for each joint.
- `np` (numpy) -- only for `np.linspace`/`np.sin`/`np.cos`/`np.clip` etc. to shape
  timing/easing curves over `[0, 1]`. Trig here still takes radians as normal, but
  you should rarely need it -- most easing curves below are plain polynomials in
  `u = t / duration`, no trig required.

You are called as:

    def animate(start_pose, end_pose, duration):
        ...

`start_pose` and `end_pose` are both `dict[str, quaternion]`: the SAME set of joint
names (this rig's actual joint names, already known-good -- you do not need to look
them up or guess them), mapping each name to its rotation quaternion in `start_pose`
(the beginning of the transition) and `end_pose` (where that joint must end up).
Both dicts are given to you already correct -- your only job is to decide the TIMING
of the interpolation between them, not the rotation values themselves.

Return a dict `tracks`: `joint_name (str) -> (times, quats)`, where `times` is a
list of floats in `[0, duration]` (at least 2, non-decreasing) and `quats` is a
list of quaternions of the same length.

Worked example 1 -- no style guidance given, plain smooth ease in/out (smoothstep):

    def animate(start_pose, end_pose, duration):
        times = list(np.linspace(0, duration, 30))
        tracks = {}
        for name, start_q in start_pose.items():
            end_q = end_pose[name]
            quats = []
            for t in times:
                u = t / duration
                eased = u * u * (3 - 2 * u)  # smoothstep: slow start, slow finish
                quats.append(slerp(start_q, end_q, eased))
            tracks[name] = (times, quats)
        return tracks

Worked example 2 -- style guidance "make it snappy": fast start, sharp arrival, no
lingering ease-out:

    def animate(start_pose, end_pose, duration):
        times = list(np.linspace(0, duration, 30))
        tracks = {}
        for name, start_q in start_pose.items():
            end_q = end_pose[name]
            quats = []
            for t in times:
                u = t / duration
                eased = 1 - (1 - u) ** 3  # ease-out-cubic: quick arrival, "snappy"
                quats.append(slerp(start_q, end_q, eased))
            tracks[name] = (times, quats)
        return tracks

Rules:
- Dict keys in `tracks` MUST be the joint name (a string, exactly as given in
  `start_pose`/`end_pose`) -- NOT a node index. This is the opposite convention
  from other animation tools you may have used; do not use an integer key here.
- `tracks` MUST include every name that appears in `start_pose` -- do not drop a
  joint because it seems minor; each one has a real, verified rotation change
  between the two poses and must actually move.
- Whatever easing curve `eased(u)` you use, it MUST equal exactly `0.0` at `u = 0`
  and exactly `1.0` at `u = 1` (the last entry in `times` should be `duration`, or
  very close to it). The FINAL keyframe of every track is checked against
  `end_pose[name]` and rejected if it lands more than a few degrees off -- an
  animation that approaches the target but doesn't reach it, or overshoots and
  settles somewhere else, will fail validation and you'll be asked to retry.
- If you use `np.linspace(0, duration, n)`, keep numpy's default `endpoint=True`
  so the last sample lands exactly at `duration` -- do not pass `endpoint=False`.
- Different joints MAY use different timing (e.g. stagger start times so joints
  move in sequence rather than all at once) -- `times` does not have to be
  identical across joints. But every joint's own last keyframe must still land on
  `end_pose[name]`.
- Do not use oscillating functions (`np.sin`/`np.cos`) as the easing curve unless
  you guarantee they settle at exactly `1.0` by the final sample -- a bounce or
  wobble that hasn't damped out by the last keyframe will fail the landing check.
- Output ONLY the function definition. No markdown fences, no example call, no
  prose.
"""

_FORBIDDEN = re.compile(r"__|import\s|open\s*\(|exec\s*\(|eval\s*\(|\bos\.|\bsys\.|subprocess")

_SANDBOX_BUILTINS = {
    "range": range, "len": len, "list": list, "dict": dict, "tuple": tuple,
    "float": float, "int": int, "min": min, "max": max, "abs": abs,
    "enumerate": enumerate, "zip": zip, "True": True, "False": False, "None": None,
}


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


class GenerationError(RuntimeError):
    pass


def _call_with_timeout(fn, args: tuple, seconds: float):
    # signal.alarm-based timeouts only work on the main thread; this runs inside
    # webview's background worker thread (see main.py), so it needs a mechanism that
    # works from any thread. A leaked worker thread on timeout is an acceptable
    # tradeoff here -- there's no way to forcibly kill a running Python thread.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args)
    try:
        return future.result(timeout=seconds)
    except FutureTimeoutError:
        raise TimeoutError(f"animation code did not finish within {seconds}s") from None
    finally:
        # wait=False: if fn is genuinely hung (e.g. an infinite loop), don't block the
        # caller waiting for it too -- we're already giving up on it above.
        pool.shutdown(wait=False)


def _validate_tracks(tracks, joints: list[Joint]) -> None:
    valid_nodes = {j.node_index for j in joints}
    if not isinstance(tracks, dict) or not tracks:
        raise GenerationError(f"animate() must return a non-empty dict, got: {tracks!r}")
    for node_index, value in tracks.items():
        if node_index not in valid_nodes:
            raise GenerationError(f"node {node_index} is not a joint on this rig")
        times, quats = value
        if len(times) != len(quats) or len(times) < 2:
            raise GenerationError(f"node {node_index}: times/quats length mismatch or too short")


def _run_generated_code(code: str, gltf: GLTF2, joints: list[Joint], duration: float):
    if _FORBIDDEN.search(code):
        raise GenerationError(f"generated code used a disallowed construct:\n{code}")

    # Deliberately minimal builtins/globals: this executes text an LLM wrote, so the
    # sandbox is the actual security boundary, not just a style choice.
    sandbox_globals = {
        "__builtins__": _SANDBOX_BUILTINS,
        "np": np,
        "axis_angle": quat.from_axis_angle_degrees,
        "combine": quat.combine,
        "find_joint": find_joint,
        "local_rotation_for_world_delta": local_rotation_for_world_delta,
    }
    try:
        exec(code, sandbox_globals)
    except Exception as e:
        raise GenerationError(f"generated code failed to parse/execute:\n{code}\n\nerror: {e}") from e

    animate_fn = sandbox_globals.get("animate")
    if animate_fn is None:
        raise GenerationError(f"generated code did not define animate():\n{code}")

    try:
        tracks = _call_with_timeout(animate_fn, (gltf, joints, duration), seconds=10)
    except Exception as e:
        # Must land as GenerationError -- anything else escapes the retry loop below
        # uncaught, silently skipping the "send the error back and retry" step.
        raise GenerationError(f"animate() raised while running:\n{code}\n\nerror: {e!r}") from e

    _validate_tracks(tracks, joints)
    return tracks


def generate(prompt: str, gltf: GLTF2, joints: list[Joint], duration: float = 2.0, retries: int = 1):
    user_prompt = (
        f'Action to animate: "{prompt}"\n\n'
        f"Skeleton (node index: name (parent)):\n{_describe_skeleton(joints)}\n\n"
        f"duration = {duration}"
    )
    last_error = None
    for _ in range(retries + 1):
        if last_error is not None:
            user_prompt += f"\n\nYour previous attempt failed with:\n{last_error}\nFix it."
        raw = llm_client.complete(SYSTEM_PROMPT, user_prompt)
        code = _extract_code(raw)
        try:
            return _run_generated_code(code, gltf, joints, duration)
        except GenerationError as e:
            last_error = str(e)
    raise GenerationError(f"LLM failed to produce a working animation after {retries + 1} attempts: {last_error}")


_TRANSITION_LANDING_TOLERANCE_DEGREES = 5.0


def _validate_transition_tracks(
    tracks,
    start_pose: dict[str, np.ndarray],
    end_pose: dict[str, np.ndarray],
) -> None:
    if not isinstance(tracks, dict) or not tracks:
        raise GenerationError(f"animate() must return a non-empty dict, got: {tracks!r}")

    missing = set(start_pose) - set(tracks)
    if missing:
        raise GenerationError(f"animate() did not include every moved joint, missing: {sorted(missing)}")

    for name, value in tracks.items():
        if name not in start_pose:
            raise GenerationError(f"{name!r} is not one of the joints given in start_pose/end_pose")
        times, quats = value
        if len(times) != len(quats) or len(times) < 2:
            raise GenerationError(f"{name!r}: times/quats length mismatch or too short")
        if list(times) != sorted(times):
            raise GenerationError(f"{name!r}: times must be non-decreasing")

        landed = np.asarray(quats[-1], dtype=np.float64)
        target = end_pose[name]
        angle = quat.angle_between(landed, target)
        if angle > _TRANSITION_LANDING_TOLERANCE_DEGREES:
            raise GenerationError(
                f"{name!r}: last frame is {angle:.1f} degrees off the target end pose "
                f"(tolerance {_TRANSITION_LANDING_TOLERANCE_DEGREES}) -- the animation must "
                f"actually reach end_pose[name] by the final keyframe, not just move toward it"
            )


def _run_generated_transition_code(
    code: str,
    start_pose: dict[str, np.ndarray],
    end_pose: dict[str, np.ndarray],
    duration: float,
):
    if _FORBIDDEN.search(code):
        raise GenerationError(f"generated code used a disallowed construct:\n{code}")

    sandbox_globals = {
        "__builtins__": _SANDBOX_BUILTINS,
        "np": np,
        "slerp": quat.slerp,
    }
    try:
        exec(code, sandbox_globals)
    except Exception as e:
        raise GenerationError(f"generated code failed to parse/execute:\n{code}\n\nerror: {e}") from e

    animate_fn = sandbox_globals.get("animate")
    if animate_fn is None:
        raise GenerationError(f"generated code did not define animate():\n{code}")

    try:
        tracks = _call_with_timeout(animate_fn, (start_pose, end_pose, duration), seconds=10)
    except Exception as e:
        raise GenerationError(f"animate() raised while running:\n{code}\n\nerror: {e!r}") from e

    _validate_transition_tracks(tracks, start_pose, end_pose)
    return tracks


def generate_transition(
    start_pose: dict[str, np.ndarray],
    end_pose: dict[str, np.ndarray],
    duration: float = 2.0,
    style_prompt: str | None = None,
    retries: int = 1,
) -> dict[str, tuple[list[float], list[np.ndarray]]]:
    joint_names = "\n".join(f"- {name}" for name in start_pose)
    style_line = (
        f'Style guidance: "{style_prompt}"'
        if style_prompt
        else "Style guidance: none given -- use a plain smooth ease in/out (see worked example 1)."
    )
    user_prompt = (
        f"Joints moving in this transition (start_pose and end_pose both have exactly "
        f"these {len(start_pose)} keys):\n{joint_names}\n\n"
        f"{style_line}\n\n"
        f"duration = {duration}"
    )
    last_error = None
    for _ in range(retries + 1):
        if last_error is not None:
            user_prompt += f"\n\nYour previous attempt failed with:\n{last_error}\nFix it."
        raw = llm_client.complete(TRANSITION_SYSTEM_PROMPT, user_prompt)
        code = _extract_code(raw)
        try:
            return _run_generated_transition_code(code, start_pose, end_pose, duration)
        except GenerationError as e:
            last_error = str(e)
    raise GenerationError(
        f"LLM failed to produce a working transition after {retries + 1} attempts: {last_error}"
    )
