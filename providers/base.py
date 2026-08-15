"""The three integration points.

If you are adding real functionality to this prototype, you are implementing
one of these three classes and nothing else. See docs/adding-a-provider.md for
a worked example.

Rules (enforced by tests/test_contract.py):
  1. Speak only schemas.py types, in and out.
  2. Import from schemas and rewards only — leave app, store, jobs and the
     other providers alone.
  3. Raise ProviderError(user_message=...) on failure. The user_message is
     shown verbatim to a child; anything else stays on the server.
  4. You run on a worker thread and may be slow, but you must call progress()
     and must remain interruptible.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable, Iterator

from schemas import Clip, Episode, Rig, TrainConfig

#: progress(fraction 0..1, message shown to the user)
Progress = Callable[[float, str], None]


class Rigger(ABC):
    """Phase 1: a rendered character image becomes a rigged avatar."""

    @abstractmethod
    def rig(self, image_bytes: bytes, mime: str, progress: Progress) -> Rig:
        """Turn a rendered character image into a Rig.

        Return ``Rig(format="procedural", ...)`` to let the browser draw the
        figure from the bone tree, or ``Rig(format="glb", glb_bytes=...)`` with
        a real skinned mesh whose bone names match schemas.BONES exactly.

        The frontend renders both identically, so you can start procedural and
        switch to GLB later while the UI stays as it is.
        """


class Poser(ABC):
    """Phase 2: words become a pose or a short animation."""

    @abstractmethod
    def pose(self, prompt: str, rig: Rig, progress: Progress) -> Clip:
        """Turn a prompt like "waves arms in the air" into a Clip.

        A single pose is a 1-keyframe clip. Poses are LOCAL rotations relative
        to rest, as quaternions [x, y, z, w]. You need only include the bones
        you move; the rest are filled from the rest pose.

        Only names in schemas.BONES are accepted — an unknown bone raises,
        loudly and immediately.
        """


class Trainer(ABC):
    """Phases 3 and 4: the avatar learns to hit a target pose."""

    @abstractmethod
    def train(self, rig: Rig, target: Clip, cfg: TrainConfig,
              stop: threading.Event) -> Iterator[Episode]:
        """Yield one Episode per training episode.

        Yield as you go: each Episode is pushed to the browser the moment it is
        produced, which is what makes the learning curve appear live.

        You MUST check ``stop.is_set()`` between episodes and return promptly
        when it is set — the child pressed stop.

        Score with rewards.reward() so that the reward sliders in the UI keep
        meaning what they say.
        """
