// Pose/clip maths shared by the viewport and the training screen.
// Mirrors the server-side helpers in rewards.py — keep the two in step.

import * as THREE from "three";

const _a = new THREE.Quaternion();
const _b = new THREE.Quaternion();
const _out = new THREE.Quaternion();

/** Sample a clip at time `t` (seconds), interpolating between keyframes. */
export function samplePose(clip, t) {
  const frames = clip.keyframes;
  if (!frames?.length) return {};
  if (frames.length === 1) return frames[0].pose;

  const duration = clip.duration || frames[frames.length - 1].t;
  const time = clip.loop && duration > 0 ? t % duration : Math.min(t, duration);

  let index = 0;
  while (index < frames.length - 2 && frames[index + 1].t <= time) index += 1;

  const from = frames[index];
  const to = frames[index + 1];
  const span = to.t - from.t;
  const alpha = span > 0 ? (time - from.t) / span : 0;

  const pose = {};
  for (const bone of Object.keys(from.pose)) {
    const q0 = from.pose[bone];
    const q1 = to.pose[bone] || q0;
    _a.set(q0[0], q0[1], q0[2], q0[3]);
    _b.set(q1[0], q1[1], q1[2], q1[3]);
    _out.slerpQuaternions(_a, _b, alpha);
    pose[bone] = [_out.x, _out.y, _out.z, _out.w];
  }
  return pose;
}

/** Angle in radians between two quaternions, 0..PI. */
export function quatAngle(a, b) {
  const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, Math.max(-1, dot)));
}

/**
 * The keyframe that best represents a clip — the one furthest from rest.
 *
 * This must match characteristic_pose() in providers/mock/training.py, because
 * it is what the training screen shows as "the target" and the server trains
 * towards. If they disagree, a child sees the avatar learning a pose that isn't
 * the one on screen.
 */
export function characteristicPose(clip, restPose) {
  if (!clip?.keyframes?.length) return { ...restPose };
  let best = clip.keyframes[0];
  let bestDistance = -1;
  for (const frame of clip.keyframes) {
    const distance = poseDistance(frame.pose, restPose);
    if (distance > bestDistance) {
      bestDistance = distance;
      best = frame;
    }
  }
  return best.pose;
}

/** Mirrors pose_distance() in rewards.py: half the mean error, half the worst. */
export function poseDistance(current, target, bones) {
  const names = bones || Object.keys(target);
  if (!names.length) return 0;
  let sum = 0;
  let worst = 0;
  for (const bone of names) {
    const error = quatAngle(current[bone] || [0, 0, 0, 1],
                            target[bone] || [0, 0, 0, 1]) / Math.PI;
    sum += error;
    worst = Math.max(worst, error);
  }
  return 0.5 * (sum / names.length) + 0.5 * worst;
}
