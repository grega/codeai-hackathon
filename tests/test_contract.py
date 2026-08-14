"""The contract, as executable checks.

These run against WHICHEVER provider is configured, so each team runs the same
suite against their own implementation:

    pytest                             # checks the mocks
    PROVIDER_POSING=real pytest        # checks your real poser

If your provider passes this file, the app will work with it. If it fails, the
message names the rule you broke.
"""

from __future__ import annotations

import math
import threading

import pytest

import providers
from rewards import per_joint_error, pose_distance, quat_from_euler, world_positions
from schemas import (
    ARTICULATED_BONES,
    BONE_SET,
    Clip,
    ContractError,
    Episode,
    Keyframe,
    REST_POSE,
    TrainConfig,
    validate_clip,
    validate_pose,
    validate_rig,
)

PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d4944415478da63f8cfc0500f0004000100ff8f5c"
    "5c5f0000000049454e44ae426082")


def noop_progress(fraction: float, message: str = "") -> None:
    """Providers must tolerate a progress callback that does nothing."""


def _hand_height(pose) -> float:
    pos = world_positions(pose)
    return (pos["L_hand"][1] + pos["R_hand"][1]) / 2


@pytest.fixture(scope="module")
def rig():
    return validate_rig(providers.get_rigger().rig(PIXEL_PNG, "image/png",
                                                   noop_progress))


@pytest.fixture(scope="module")
def clip(rig):
    return validate_clip(providers.get_poser().pose("waves arms in the air",
                                                    rig, noop_progress))


# --------------------------------------------------------------------------
# Rigger
# --------------------------------------------------------------------------

class TestRigger:
    def test_returns_every_contract_bone(self, rig):
        assert set(rig.skeleton) == BONE_SET

    def test_format_is_renderable(self, rig):
        assert rig.format in ("procedural", "glb")
        if rig.format == "glb":
            assert rig.glb_bytes, "format='glb' requires glb_bytes"

    def test_reports_progress(self):
        seen = []
        providers.get_rigger().rig(PIXEL_PNG, "image/png",
                                   lambda f, m="": seen.append((f, m)))
        assert seen, "call progress() so the UI can show a loading state"
        assert all(0 <= f <= 1 for f, _ in seen), "progress must be 0..1"


# --------------------------------------------------------------------------
# Poser
# --------------------------------------------------------------------------

class TestPoser:
    def test_produces_at_least_one_keyframe(self, clip):
        assert clip.keyframes

    def test_only_uses_contract_bones(self, clip):
        for frame in clip.keyframes:
            assert set(frame.pose) <= BONE_SET

    def test_quaternions_are_normalised(self, clip):
        for frame in clip.keyframes:
            for bone, q in frame.pose.items():
                length = math.sqrt(sum(c * c for c in q))
                assert length == pytest.approx(1.0, abs=1e-3), \
                    f"{bone} quaternion is not unit length"

    def test_keyframe_times_strictly_increase(self, clip):
        times = [f.t for f in clip.keyframes]
        assert times == sorted(set(times)), "keyframe times must strictly increase"
        assert times[0] >= 0

    def test_is_deterministic(self, rig):
        """The same words must give the same move.

        A child typing the same thing twice and getting a different result would
        undermine the point of the exercise. If your poser calls an LLM, pin the
        temperature to 0 or cache by prompt.
        """
        poser = providers.get_poser()
        a = validate_clip(poser.pose("waves arms in the air", rig, noop_progress))
        b = validate_clip(poser.pose("waves arms in the air", rig, noop_progress))
        assert [f.t for f in a.keyframes] == [f.t for f in b.keyframes]
        for fa, fb in zip(a.keyframes, b.keyframes):
            for bone in fa.pose:
                assert fa.pose[bone] == pytest.approx(fb.pose[bone], abs=1e-6)

    def test_unknown_prompt_still_returns_something(self, rig):
        """A live demo must never dead-end on an unexpected prompt."""
        result = providers.get_poser().pose("zzzq florble", rig, noop_progress)
        assert validate_clip(result).keyframes

    def test_moves_the_avatar(self, clip):
        """A pose that is identical to rest means nothing happened."""
        assert max(pose_distance(f.pose, REST_POSE) for f in clip.keyframes) > 0.01


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

class TestTrainer:
    def test_yields_valid_episodes(self, rig, clip):
        cfg = TrainConfig(episodes=25, seed=1)
        episodes = list(providers.get_trainer().train(rig, clip, cfg,
                                                      threading.Event()))
        assert episodes
        for ep in episodes:
            assert isinstance(ep, Episode)
            validate_pose(ep.pose)
            assert 0 <= ep.reward <= 1, "reward must be 0..1"
            assert math.isfinite(ep.best_reward)

    def test_episode_numbers_increase(self, rig, clip):
        cfg = TrainConfig(episodes=25, seed=1)
        numbers = [ep.episode for ep in
                   providers.get_trainer().train(rig, clip, cfg, threading.Event())]
        assert numbers == sorted(numbers)

    def test_best_reward_never_decreases(self, rig, clip):
        cfg = TrainConfig(episodes=40, seed=1)
        best = [ep.best_reward for ep in
                providers.get_trainer().train(rig, clip, cfg, threading.Event())]
        assert all(b <= a for b, a in zip(best, best[1:])), \
            "best_reward is a high-water mark and must never go down"

    def test_honours_stop(self, rig, clip):
        """The child pressed stop: return promptly, do not run to completion."""
        stop = threading.Event()
        cfg = TrainConfig(episodes=100_000, seed=1)
        produced = 0
        for _ in providers.get_trainer().train(rig, clip, cfg, stop):
            produced += 1
            if produced == 5:
                stop.set()
            assert produced < 500, "trainer ignored stop.is_set()"

    def test_actually_learns(self, rig, clip):
        """Reward at the end must beat reward at the start.

        Loose on purpose — how well it learns is your business, but a flat line
        means the child sees nothing happen.
        """
        cfg = TrainConfig(episodes=300, seed=5)
        episodes = list(providers.get_trainer().train(rig, clip, cfg,
                                                      threading.Event()))
        assert episodes[-1].best_reward > episodes[0].reward

    def test_responds_to_reward_weights(self, rig):
        """The sliders must change what is learned — this is the whole lesson.

        Trains twice towards a NEUTRAL target: once rewarding pose-matching,
        once rewarding hand height. Matching should leave the hands where they
        started; rewarding height should raise them. The target is deliberately
        neutral so the two rewards genuinely pull in different directions —
        against an arms-up target, matching it would raise the hands anyway and
        the test would pass without proving anything.
        """
        trainer = providers.get_trainer()
        target = validate_clip(Clip(name="neutral",
                                    keyframes=[Keyframe(t=0.0,
                                                        pose=dict(REST_POSE))]))
        rest_height = _hand_height(REST_POSE)
        heights = {}
        for label, weights in {
            "match": {"pose_match": 1.0, "arm_height": 0.0,
                      "symmetry": 0.0, "stillness": 0.0},
            "hands": {"pose_match": 0.0, "arm_height": 1.0,
                      "symmetry": 0.0, "stillness": 0.0},
        }.items():
            cfg = TrainConfig(episodes=250, seed=3, reward_weights=weights)
            final = list(trainer.train(rig, target, cfg, threading.Event()))[-1]
            heights[label] = _hand_height(final.pose)

        assert heights["hands"] > rest_height + 0.05, \
            "rewarding hand height did not raise the hands"
        assert heights["hands"] > heights["match"], \
            "changing the reward weights had no effect on what was learned"


# --------------------------------------------------------------------------
# Validation — the rules that protect every provider from every other one
# --------------------------------------------------------------------------

class TestValidation:
    def test_unknown_bone_is_rejected(self):
        with pytest.raises(ContractError, match="unknown bone"):
            validate_pose({"left_arm": [0, 0, 0, 1]})

    def test_missing_bones_fall_back_to_rest(self):
        pose = validate_pose({"spine": list(quat_from_euler(0, 0, 30))})
        assert set(pose) == BONE_SET
        assert pose["L_hand"] == [0.0, 0.0, 0.0, 1.0]

    def test_quaternions_are_normalised_on_the_way_in(self):
        pose = validate_pose({"spine": [0, 0, 0, 5]})
        assert pose["spine"] == [0.0, 0.0, 0.0, 1.0]

    def test_zero_quaternion_is_rejected(self):
        with pytest.raises(ContractError, match="zero-length"):
            validate_pose({"spine": [0, 0, 0, 0]})

    def test_out_of_order_keyframes_are_rejected(self):
        with pytest.raises(ContractError, match="not after"):
            validate_clip({"name": "x", "keyframes": [
                {"t": 1.0, "pose": {}}, {"t": 0.5, "pose": {}}]})

    def test_empty_clip_is_rejected(self):
        with pytest.raises(ContractError, match="at least one keyframe"):
            validate_clip({"name": "x", "keyframes": []})


class TestRewards:
    def test_identical_poses_score_perfectly(self):
        pose = validate_pose({"L_shoulder": list(quat_from_euler(0, 0, -80))})
        assert pose_distance(pose, pose) == pytest.approx(0.0, abs=1e-9)

    def test_per_joint_error_covers_articulated_bones(self):
        errors = per_joint_error(REST_POSE, REST_POSE)
        assert set(errors) == set(ARTICULATED_BONES)

    def test_forward_kinematics_matches_the_bone_tree(self):
        """Rest pose: the head sits above the hips and the arms hang either side."""
        pos = world_positions(REST_POSE)
        assert pos["head"][1] > pos["hips"][1]
        assert pos["L_hand"][0] < 0 < pos["R_hand"][0]
        assert pos["L_foot"][1] < pos["hips"][1]

    def test_raising_an_arm_raises_the_hand(self):
        """Guards the rotation convention the pose library is authored against."""
        raised = validate_pose({"L_shoulder": list(quat_from_euler(0, 0, -80))})
        assert world_positions(raised)["L_hand"][1] > \
            world_positions(REST_POSE)["L_hand"][1]

    def test_zero_weights_do_not_divide_by_zero(self):
        from rewards import reward
        total, breakdown = reward(REST_POSE, REST_POSE,
                                  {k: 0.0 for k in breakdown_keys()})
        assert total == 0.0
        assert breakdown


def breakdown_keys():
    from schemas import DEFAULT_REWARD_WEIGHTS
    return DEFAULT_REWARD_WEIGHTS.keys()
