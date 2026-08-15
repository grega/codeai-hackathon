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

#: Server-to-server auto-rigging service. The browser never receives this URL.
RIGGING_SERVICE_URL = os.environ.get("RIGGING_SERVICE_URL", "").strip().rstrip("/")

#: Overall deadline for classify, mesh generation, joint inference, rigging,
#: and the final GLB download.
RIGGING_SERVICE_TIMEOUT = float(
    os.environ.get("RIGGING_SERVICE_TIMEOUT", "300"))

#: Delay between remote mesh/rig task status checks.
RIGGING_POLL_INTERVAL = float(
    os.environ.get("RIGGING_POLL_INTERVAL", "5"))

#: Episodes per second pushed to the browser at speed 1.0. The training screen
#: multiplies this by its speed control.
EPISODE_RATE = float(os.environ.get("EPISODE_RATE", "20"))

#: Max upload size. Sized for a sketch PNG originally; also covers the
#: llm-animator endpoint's GLB uploads (input + pose_b can be ~3.5MB each).
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))

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


# --------------------------------------------------------------------------
# Sketch-render feature (POST /api/renders)
# --------------------------------------------------------------------------
# A purpose-built, browser-facing endpoint — unlike the prompt endpoint above,
# the frontend is meant to call this one. It takes a fixed shape (an uploaded
# drawing plus a short prompt) and always invokes the same model, so
# there's no caller-selectable model_id and therefore no allowlist. It still
# shares BEDROCK_REGION/AWS credentials and fails closed the same way: no
# region configured means bedrock._get_client() refuses before anything runs.

#: Stability's Control Sketch service — image-conditioned, so the drawing's
#: lines actually shape the output rather than just informing a text prompt.
BEDROCK_RENDER_MODEL_ID = os.environ.get(
    "BEDROCK_RENDER_MODEL_ID", "us.stability.stable-image-control-sketch-v1:0")

#: This endpoint has no bearer token — every visitor's browser can reach it,
#: like the rest of the avatar API — so it needs its own caps to bound the
#: bill. Tighter than LLM_RATE_PER_* because image generation costs more per
#: call than a short text completion.
RENDER_RATE_PER_MINUTE = int(os.environ.get("RENDER_RATE_PER_MINUTE", "5"))
RENDER_RATE_PER_DAY = int(os.environ.get("RENDER_RATE_PER_DAY", "50"))


# --------------------------------------------------------------------------
# T-pose feature (POST /api/avatars/<id>/tpose)
# --------------------------------------------------------------------------
# Also browser-facing and also fixed-shape (an avatar's saved drawing, no
# caller input), but always two Bedrock calls: a pose transform, then a
# background removal. Its own model ids so each stage can be retuned or
# swapped without touching the free-text render feature above.

#: Stage 1 — redraws the avatar in a forward-facing T-pose. Same Stability
#: Control Sketch service as BEDROCK_RENDER_MODEL_ID, kept as a separate
#: setting since the two features may want to diverge later.
BEDROCK_TPOSE_MODEL_ID = os.environ.get(
    "BEDROCK_TPOSE_MODEL_ID", "us.stability.stable-image-control-sketch-v1:0")

#: Stage 2 — strips the background, returning a PNG with a real alpha
#: channel. Stability's own Remove Background service, invoked the same way
#: as BEDROCK_TPOSE_MODEL_ID. (Amazon Nova Canvas can also do this, but isn't
#: available in every account/region — Stability's version is used here
#: since it's already required for stage 1.)
BEDROCK_BG_REMOVAL_MODEL_ID = os.environ.get(
    "BEDROCK_BG_REMOVAL_MODEL_ID", "us.stability.stable-image-remove-background-v1:0")

#: Tighter than RENDER_RATE_PER_* since each call here is two Bedrock
#: invocations rather than one.
TPOSE_RATE_PER_MINUTE = int(os.environ.get("TPOSE_RATE_PER_MINUTE", "3"))
TPOSE_RATE_PER_DAY = int(os.environ.get("TPOSE_RATE_PER_DAY", "30"))


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
