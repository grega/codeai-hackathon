"""Configuration. Everything is an env var with a sane default so the app runs
with no setup at all.

The three PROVIDER_* vars are the switches that swap mock implementations for
real ones — see CONTRACT.md.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent

# Local development defaults, from .env in the repo root.
#
# load_dotenv does not overwrite variables that are already set, so the
# precedence is: real environment > .env > the defaults below. That is what lets
# the same file work on Heroku — config vars arrive as real environment
# variables and win, and .env is not in the slug anyway.
#
# This has to run before the os.environ reads below. `flask run` also loads .env
# by itself once python-dotenv is installed, but gunicorn and pytest do not,
# which is why it is explicit here.
load_dotenv(ROOT / ".env")
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

#: Path to a rigged GLB the mock rigger should serve instead of the procedural
#: figure. Lets the whole GLB path be exercised before a real rigger exists:
#:   MOCK_RIG_GLB=tests/fixtures/mixamo-style.glb flask --app app run
MOCK_RIG_GLB = os.environ.get("MOCK_RIG_GLB", "").strip()

#: How long a provider gets before the job runner gives up on it.
PROVIDER_TIMEOUT = float(os.environ.get("PROVIDER_TIMEOUT", "120"))


# --------------------------------------------------------------------------
# Bedrock prompt endpoint
# --------------------------------------------------------------------------
# Every default here is the locked-down one. The endpoint stays switched off
# and refuses every model until someone deliberately configures both.

#: Bearer token callers must present. Unset means the route returns 404.
#: This is a server-side secret — it must never be sent to the browser.
LLM_API_TOKEN = os.environ.get("LLM_API_TOKEN", "").strip()

#: Comma-separated Bedrock model IDs the endpoint may invoke. Empty refuses
#: everything, so a misconfigured deployment cannot be pointed at an
#: expensive model. Find the exact IDs for your account and region with:
#:   aws bedrock list-inference-profiles --region <region>
#:   aws bedrock list-foundation-models  --region <region>
BEDROCK_ALLOWED_MODELS = tuple(
    m.strip() for m in os.environ.get("BEDROCK_ALLOWED_MODELS", "").split(",")
    if m.strip()
)

#: boto3 reads AWS_DEFAULT_REGION itself; BEDROCK_REGION overrides it when
#: Bedrock lives somewhere other than the rest of the account's resources.
BEDROCK_REGION = (os.environ.get("BEDROCK_REGION")
                  or os.environ.get("AWS_DEFAULT_REGION", ""))

#: Ceiling on max_tokens, whatever the caller asks for.
BEDROCK_MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "4096"))

#: Longest prompt accepted, in characters.
BEDROCK_MAX_PROMPT_CHARS = int(os.environ.get("BEDROCK_MAX_PROMPT_CHARS", "20000"))

#: Seconds to wait for a model response.
BEDROCK_TIMEOUT = float(os.environ.get("BEDROCK_TIMEOUT", "60"))

#: Per-caller and whole-deployment request caps. The daily one bounds the bill.
LLM_RATE_PER_MINUTE = int(os.environ.get("LLM_RATE_PER_MINUTE", "10"))
LLM_RATE_PER_DAY = int(os.environ.get("LLM_RATE_PER_DAY", "500"))


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
