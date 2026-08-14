"""Guards for the LLM endpoint.

The endpoint spends money on someone else's AWS account, so the design is
fail-closed at every step:

  * No token configured -> the route does not exist (404).
  * Wrong token         -> the route does not exist (404).
  * No model allowlist  -> every model is refused.

404 rather than 401 on an auth failure is deliberate. An unauthenticated
caller cannot tell the endpoint apart from a typo'd URL, so scanners find
nothing to come back to. It does mean a developer with a bad token sees a
confusing 404 — the server log says which it was.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import deque
from functools import wraps
from typing import Callable

from flask import jsonify, request

import config
from logs import log


def _hash(value: str) -> str:
    """Identify a caller in logs and rate-limit buckets without storing the token."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class RateLimiter:
    """Sliding-window limiter with a per-caller and a global bucket.

    The global bucket is the one that matters: per-caller limits stop one
    person hammering the endpoint, but a shared token handed round a team is
    a single caller as far as this is concerned. The global cap is what bounds
    the AWS bill.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callers: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()

    def check(self, identity: str) -> str | None:
        """Return None if allowed, or a message explaining the refusal."""
        now = time.monotonic()
        with self._lock:
            window = self._callers.setdefault(identity, deque())
            _evict(window, now - 60.0)
            _evict(self._global, now - 86400.0)

            if len(window) >= config.LLM_RATE_PER_MINUTE:
                return (f"Rate limit: {config.LLM_RATE_PER_MINUTE} requests per "
                        f"minute. Try again shortly.")
            if len(self._global) >= config.LLM_RATE_PER_DAY:
                return (f"Daily cap of {config.LLM_RATE_PER_DAY} requests reached "
                        f"for this deployment.")

            window.append(now)
            self._global.append(now)
            return None

    def snapshot(self) -> dict[str, int]:
        now = time.monotonic()
        with self._lock:
            _evict(self._global, now - 86400.0)
            return {
                "used_today": len(self._global),
                "daily_limit": config.LLM_RATE_PER_DAY,
                "per_minute_limit": config.LLM_RATE_PER_MINUTE,
            }


def _evict(window: deque[float], cutoff: float) -> None:
    while window and window[0] < cutoff:
        window.popleft()


limiter = RateLimiter()


def _bearer_token() -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return None


def require_token(view: Callable) -> Callable:
    """Gate a route behind LLM_API_TOKEN and the rate limiter."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        expected = config.LLM_API_TOKEN
        if not expected:
            log("[llm] refused: LLM_API_TOKEN is not set, endpoint disabled")
            return jsonify({"error": "Not found."}), 404

        supplied = _bearer_token()
        # compare_digest on every path, so a missing header and a wrong token
        # take the same time and neither is distinguishable from the outside.
        if not secrets.compare_digest(supplied or "", expected):
            log(f"[llm] refused: bad or missing token from "
                f"{request.remote_addr}")
            return jsonify({"error": "Not found."}), 404

        # Bucket by token AND client address: a shared token still gets
        # per-machine limits, and one caller can't exhaust everyone's quota.
        identity = f"{_hash(expected)}:{request.remote_addr}"
        refusal = limiter.check(identity)
        if refusal:
            log(f"[llm] rate limited: {identity}")
            return jsonify({"error": refusal}), 429

        return view(*args, **kwargs)

    return wrapper
