"""Configuration. Everything is an env var with a sane default so the app runs
with no setup at all.

The three PROVIDER_* vars are the switches that swap mock implementations for
real ones — see CONTRACT.md.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"

# "mock" | "real" — see providers/__init__.py
PROVIDER_RIGGING = os.environ.get("PROVIDER_RIGGING", "mock")
PROVIDER_POSING = os.environ.get("PROVIDER_POSING", "mock")
PROVIDER_TRAINING = os.environ.get("PROVIDER_TRAINING", "mock")

#: Episodes per second pushed to the browser at speed 1.0. The training screen
#: multiplies this by its speed control.
EPISODE_RATE = float(os.environ.get("EPISODE_RATE", "20"))

#: Max upload size for a sketch.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))

#: How long a provider gets before the job runner gives up on it.
PROVIDER_TIMEOUT = float(os.environ.get("PROVIDER_TIMEOUT", "120"))


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
