"""In-memory registries for avatars, clips, training runs and behaviours.

PROTOTYPE SCOPE: single process, single user, wiped on restart. Uploaded images
are the only thing that touches disk. Swapping this for SQLite later means
changing this file only — nothing else reaches into the dicts.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import config
from schemas import Clip, Rig, TrainConfig


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class Avatar:
    id: str
    rig: Rig
    image_path: str | None = None
    name: str = "My avatar"

    def to_json(self) -> dict[str, Any]:
        glb_url = f"/api/avatars/{self.id}/glb" if self.rig.format == "glb" else None
        return {
            "id": self.id,
            "name": self.name,
            "image_url": f"/api/avatars/{self.id}/image" if self.image_path else None,
            "rig": self.rig.to_json(glb_url=glb_url),
        }


@dataclass
class Behaviour:
    """A named, trained behaviour — the bridge from 'a pose' to 'a thing my
    avatar does', which is what the playground triggers."""

    id: str
    name: str
    clip: Clip
    avatar_id: str
    trained: bool = False
    best_reward: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "avatar_id": self.avatar_id,
            "trained": self.trained,
            "best_reward": self.best_reward,
            "clip": self.clip.to_json(),
        }


class Store:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.avatars: dict[str, Avatar] = {}
        self.clips: dict[str, Clip] = {}
        self.behaviours: dict[str, Behaviour] = {}
        self.runs: dict[str, Any] = {}  # run_id -> TrainingRun (see training.py)

    # -- avatars ---------------------------------------------------------
    def add_avatar(self, rig: Rig, image_bytes: bytes | None = None,
                   mime: str = "image/png") -> Avatar:
        avatar_id = new_id("av")
        image_path = None
        if image_bytes:
            config.ensure_dirs()
            ext = {"image/png": ".png", "image/jpeg": ".jpg",
                   "image/webp": ".webp"}.get(mime, ".png")
            path = config.UPLOAD_DIR / f"{avatar_id}{ext}"
            path.write_bytes(image_bytes)
            image_path = str(path)

        avatar = Avatar(id=avatar_id, rig=rig, image_path=image_path)
        with self._lock:
            self.avatars[avatar_id] = avatar
        return avatar

    def get_avatar(self, avatar_id: str) -> Avatar | None:
        return self.avatars.get(avatar_id)

    # -- clips -----------------------------------------------------------
    def add_clip(self, clip: Clip) -> Clip:
        clip.id = clip.id or new_id("clip")
        with self._lock:
            self.clips[clip.id] = clip
        return clip

    def get_clip(self, clip_id: str) -> Clip | None:
        return self.clips.get(clip_id)

    # -- behaviours ------------------------------------------------------
    def add_behaviour(self, name: str, clip: Clip, avatar_id: str,
                      trained: bool = False, best_reward: float = 0.0) -> Behaviour:
        behaviour = Behaviour(id=new_id("bhv"), name=name, clip=clip,
                              avatar_id=avatar_id, trained=trained,
                              best_reward=best_reward)
        with self._lock:
            self.behaviours[behaviour.id] = behaviour
        return behaviour

    def list_behaviours(self, avatar_id: str | None = None) -> list[Behaviour]:
        items = list(self.behaviours.values())
        if avatar_id:
            items = [b for b in items if b.avatar_id == avatar_id]
        return items

    # -- runs ------------------------------------------------------------
    def add_run(self, run: Any) -> Any:
        with self._lock:
            self.runs[run.id] = run
        return run

    def get_run(self, run_id: str) -> Any | None:
        return self.runs.get(run_id)


#: The single process-wide store.
store = Store()
