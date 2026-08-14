"""Training run lifecycle and pacing.

The Trainer provider is a pure generator that yields episodes as fast as it can
compute them. Everything to do with *when* those episodes reach the browser
lives here, so a provider never has to think about it: pacing, the speed
control, pausing, stopping, and letting a browser attach late without missing
the start of the curve.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Iterator

import config
from schemas import (
    Clip,
    ContractError,
    Episode,
    ProviderError,
    TrainConfig,
    validate_episode,
)
from store import new_id


class TrainingRun:
    """One training session. Owns the worker thread and the episode history.

    The history list is the single source of truth for the stream: the worker
    appends, readers walk it by index. That means several viewers (or one that
    reconnects) all see exactly the same episodes exactly once. Episodes are
    capped at 5000 by TrainConfig, so keeping them all is a few MB at worst.
    """

    def __init__(self, avatar_id: str, target_clip: Clip, cfg: TrainConfig):
        self.id = new_id("run")
        self.avatar_id = avatar_id
        self.target_clip = target_clip
        self.cfg = cfg

        self.status = "pending"   # pending | running | paused | done | stopped | error
        self.error: str | None = None
        self.speed = 1.0
        self.best_reward = 0.0
        self.episode = 0

        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._cond = threading.Condition()
        self._history: list[Episode] = []
        self._finished = False
        self._thread: threading.Thread | None = None

    # -- control ---------------------------------------------------------
    def start(self, trainer, rig) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, args=(trainer, rig),
                                        name=f"train-{self.id}", daemon=True)
        self.status = "running"
        self._thread.start()

    def pause(self) -> None:
        if self.status == "running":
            self.status = "paused"
            self._resume.clear()

    def resume(self) -> None:
        if self.status == "paused":
            self.status = "running"
            self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()  # unblock a paused worker so it can notice the stop

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, min(20.0, float(speed)))

    @property
    def finished(self) -> bool:
        return self.status in ("done", "stopped", "error")

    # -- worker ----------------------------------------------------------
    def _run(self, trainer, rig) -> None:
        """Pull episodes from the provider, pace them, publish them."""
        try:
            for ep in trainer.train(rig, self.target_clip, self.cfg, self._stop):
                validate_episode(ep)

                # Pacing lives here, not in the provider. A paused run blocks on
                # _resume; the speed control changes the interval mid-stream.
                self._resume.wait()
                if self._stop.is_set() and not ep.done:
                    self._publish(ep)
                    break

                self._publish(ep)
                time.sleep(1.0 / max(0.1, config.EPISODE_RATE * self.speed))

            self.status = "stopped" if self._stop.is_set() else "done"
        except ProviderError as exc:
            self.status, self.error = "error", exc.user_message
            print(f"[run {self.id}] provider error: {exc.detail or exc}")
        except ContractError as exc:
            self.status = "error"
            self.error = ("The training service sent back something I didn't "
                          "understand. Check the server log.")
            print(f"[run {self.id}] CONTRACT VIOLATION: {exc}")
        except Exception:  # noqa: BLE001 - last line of defence
            self.status, self.error = "error", "Training stopped unexpectedly."
            print(f"[run {self.id}] unexpected error:\n{traceback.format_exc()}")
        finally:
            with self._cond:
                self._finished = True
                self._cond.notify_all()

    def _publish(self, ep: Episode) -> None:
        self.episode = ep.episode
        self.best_reward = max(self.best_reward, ep.best_reward)
        with self._cond:
            self._history.append(ep)
            self._cond.notify_all()

    # -- stream ----------------------------------------------------------
    def events(self) -> Iterator[Episode]:
        """Yield every episode of this run, from the beginning, exactly once.

        Starting from the beginning matters: the browser attaches its
        EventSource a moment after POSTing the run, and losing the first few
        episodes would put a visible notch in the learning curve.
        """
        index = 0
        while True:
            with self._cond:
                while index >= len(self._history) and not self._finished:
                    self._cond.wait(timeout=1.0)
                if index >= len(self._history) and self._finished:
                    return
                batch = self._history[index:]
                index = len(self._history)
            yield from batch

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "avatar_id": self.avatar_id,
            "status": self.status,
            "error": self.error,
            "speed": self.speed,
            "episode": self.episode,
            "best_reward": self.best_reward,
            "config": self.cfg.to_json(),
            "target_clip_id": self.target_clip.id,
        }
