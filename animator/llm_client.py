"""OpenAI-compatible client, pointed at either a locally hosted MLX model
(`python -m mlx_vlm.server` / `mlx_lm.server`, no API key needed) or a hosted
provider reached through OpenRouter's OpenAI-compatible endpoint (needs
OPENROUTER_API_KEY in the environment). Same protocol either way, so the app code
that calls `complete()` doesn't need to know which one it's talking to.
"""
from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8811/v1"
DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"


def complete(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> str:
    # Read the module globals at call time, not as bound default-argument values --
    # main.py overrides DEFAULT_MODEL/DEFAULT_BASE_URL after import based on --model/
    # --base-url, and default args would freeze the pre-override values instead.
    resolved_base_url = base_url or DEFAULT_BASE_URL
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY") or "not-needed"
    client = OpenAI(base_url=resolved_base_url, api_key=resolved_key)
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
