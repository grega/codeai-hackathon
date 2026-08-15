"""Mock trainer: a (1+1) hill-climber with decaying exploration.

This is a real (if very simple) learning algorithm, not a scripted fake. It
keeps a current best pose, jitters it, keeps the jitter if the reward improved,
and shrinks the jitter over time. That produces an honest noisy-but-rising
learning curve, and — importantly for the lesson — it genuinely responds to the
reward sliders: turn `pose_match` down and `arm_height` up and it will learn to
reach for the sky instead of copying the target.

TODO(real): implement providers/real/training.py with a class `RealTrainer`
subclassing providers.base.Trainer, satisfying `Trainer.train()`. Keep scoring
with rewards.reward() so the sliders keep meaning what the UI says.

NOTE: do not sleep in your trainer. Yield episodes as fast as you can compute
them; the server paces the stream so the child's speed control works.
"""

from __future__ import annotations

import random
import threading
from typing import Iterator

from providers.base import Trainer
from rewards import (
    per_joint_error,
    pose_distance,
    quat_from_euler,
    quat_mul,
    quat_slerp,
    reward,
)
from schemas import (
    ARTICULATED_BONES,
    Clip,
    Episode,
    IDENTITY_QUAT,
    REST_POSE,
    Rig,
    TrainConfig,
)


def characteristic_pose(clip: Clip) -> dict[str, list[float]]:
    """The keyframe that best represents a clip — the one furthest from rest.

    A wave is mostly 'arm up'; which frame of the wave you train toward barely
    matters, but picking the most extreme one gives the clearest target for a
    child to compare against.
    """
    if not clip.keyframes:
        return dict(REST_POSE)
    return max(clip.keyframes,
               key=lambda kf: pose_distance(kf.pose, REST_POSE)).pose


def _jitter(pose: dict[str, list[float]], sigma_deg: float,
            rng: random.Random) -> dict[str, list[float]]:
    """Perturb a couple of random joints by a small random rotation.

    A couple, rather than all eleven articulated bones at once: most of the body
    is already correct at any given moment, so a whole-body jitter makes almost
    every candidate worse than what we have and the climb stalls at its starting
    score. Moving one or two joints is both far more effective and a much better
    story for a child — it tries a small change and keeps it if that helped.
    """
    out = dict(pose)
    for bone in rng.sample(ARTICULATED_BONES, k=rng.randint(1, 3)):
        noise = quat_from_euler(rng.gauss(0, sigma_deg),
                                rng.gauss(0, sigma_deg * 0.4),
                                rng.gauss(0, sigma_deg))
        out[bone] = list(quat_mul(pose.get(bone, IDENTITY_QUAT), noise))
    return out


class MockTrainer(Trainer):
    def train(self, rig: Rig, target: Clip, cfg: TrainConfig,
              stop: threading.Event) -> Iterator[Episode]:
        rng = random.Random(cfg.seed or None)
        goal = characteristic_pose(target)

        current = dict(REST_POSE)
        current_reward, _ = reward(current, goal, cfg.reward_weights)
        best_reward = current_reward
        previous = None

        # How far the starting pose is from the goal, so `match` can express
        # progress as "0% = where I started, 100% = the target".
        baseline = max(pose_distance(REST_POSE, goal), 1e-6)

        def match_of(pose) -> float:
            return max(0.0, min(1.0, 1.0 - pose_distance(pose, goal) / baseline))

        for episode in range(1, cfg.episodes + 1):
            if stop.is_set():
                # The child pressed stop. Report where we got to and return.
                yield Episode(episode=episode - 1, reward=current_reward,
                              best_reward=best_reward, pose=current,
                              per_joint_error=per_joint_error(current, goal),
                              exploration=0.0, done=True, note="Stopped",
                              match=match_of(current))
                return

            # Exploration decays from cfg.exploration to near zero: big wild
            # guesses early, small careful adjustments later. This shape is the
            # thing we want a child to notice.
            # The floor matters: decaying exploration all the way to zero leaves
            # the learner unable to fix whatever it still has wrong, and it
            # plateaus visibly short of the target.
            progress_fraction = episode / cfg.episodes
            sigma = cfg.exploration * 40.0 * (1.0 - progress_fraction) + 4.0

            candidate = _jitter(current, sigma, rng)
            candidate_reward, breakdown = reward(
                candidate, goal, cfg.reward_weights, previous=current)

            note = ""
            if candidate_reward > current_reward:
                # Accept, but move only part of the way — that's the learning rate.
                blend = cfg.learning_rate
                current = {
                    bone: list(quat_slerp(current.get(bone, IDENTITY_QUAT),
                                          candidate.get(bone, IDENTITY_QUAT),
                                          blend))
                    for bone in current
                }
                current_reward, breakdown = reward(
                    current, goal, cfg.reward_weights, previous=previous)
                if current_reward > best_reward:
                    best_reward = current_reward
                    # if episode > 1:
                        # note = "New best!"

            previous = current
            yield Episode(
                episode=episode,
                # The score of what it TRIED this episode, not the score it kept.
                # Reporting the kept score would make `reward` a monotonic line
                # identical to best_reward, and the chart's "score each try"
                # would be a lie — the whole point of showing it is that trying
                # things out is noisy.
                reward=candidate_reward,
                best_reward=best_reward,
                pose=current,
                per_joint_error=per_joint_error(current, goal),
                exploration=round(sigma / 36.0, 3),
                done=episode == cfg.episodes,
                note=note,
                match=match_of(current),
            )
