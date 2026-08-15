"""Amazon Bedrock client, via the Converse API.

Converse is model-agnostic — Anthropic, Meta, Mistral and Amazon Nova all take
the same request and return the same response shape — which is what lets the
caller name the model per request without this module knowing anything about
model families.

Credentials come from the environment (`AWS_ACCESS_KEY_ID` and friends); boto3
reads them itself, so nothing here touches a secret.

If you want Anthropic-specific features that Converse does not expose — the
full Messages API surface, adaptive thinking, prompt caching — use the
`AnthropicBedrockMantle` client from the `anthropic` SDK instead. It is a
different client with `anthropic.`-prefixed model IDs, and it would be a second
provider here rather than a change to this one.
"""

from __future__ import annotations

import base64
import json
import threading

import config
from logs import log

_client = None
_client_lock = threading.Lock()


class BedrockError(Exception):
    """Failure to report to the caller. ``status`` is the HTTP code to send."""

    def __init__(self, message: str, status: int = 502, detail: str = ""):
        super().__init__(detail or message)
        self.message = message
        self.status = status
        self.detail = detail


def _get_client():
    """Build the boto3 client once, lazily.

    Lazily because boto3 is only needed when the endpoint is actually used —
    the app has to start and serve the avatar experience on a machine with no
    AWS credentials at all.
    """
    global _client
    with _client_lock:
        if _client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise BedrockError(
                    "The model service isn't installed on this server.",
                    status=503,
                    detail="boto3 is missing; pip install -r requirements.txt",
                ) from exc

            region = config.BEDROCK_REGION
            if not region:
                raise BedrockError(
                    "The model service isn't configured.",
                    status=503,
                    detail="Set AWS_DEFAULT_REGION (or BEDROCK_REGION)",
                )

            _client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(
                    read_timeout=config.BEDROCK_TIMEOUT,
                    connect_timeout=10,
                    # Bedrock throttles hard; let botocore back off rather than
                    # surfacing a throttle as a failure on the first attempt.
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
    return _client


def allowed_models() -> list[str]:
    return list(config.BEDROCK_ALLOWED_MODELS)


def check_model(model_id: str) -> None:
    """Reject anything outside the allowlist.

    An empty allowlist refuses everything. That is the intended default: an
    endpoint that will run whatever model string it is handed is an open
    invitation to invoke the most expensive model in the account.
    """
    if not config.BEDROCK_ALLOWED_MODELS:
        raise BedrockError(
            "No models are enabled on this server.",
            status=503,
            detail="BEDROCK_ALLOWED_MODELS is empty; nothing can be invoked",
        )
    if model_id not in config.BEDROCK_ALLOWED_MODELS:
        raise BedrockError(
            f"Model '{model_id}' isn't on the allowed list.",
            status=400,
            detail=f"allowed: {', '.join(config.BEDROCK_ALLOWED_MODELS)}",
        )


def converse(model_id: str, prompt: str, *, system: str | None = None,
             max_tokens: int = 1024, temperature: float | None = None
             ) -> dict:
    """Run one prompt against one Bedrock model and return the reply.

    ``temperature`` is omitted from the request unless explicitly supplied —
    the newer Anthropic models reject sampling parameters outright, so sending
    a default would break exactly the models most people will reach for.
    """
    check_model(model_id)

    inference: dict[str, object] = {
        "maxTokens": max(1, min(int(max_tokens), config.BEDROCK_MAX_TOKENS)),
    }
    if temperature is not None:
        inference["temperature"] = max(0.0, min(float(temperature), 1.0))

    request: dict[str, object] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": inference,
    }
    if system:
        request["system"] = [{"text": system}]

    client = _get_client()
    try:
        response = client.converse(**request)
    except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
        raise _translate(exc) from exc

    blocks = response.get("output", {}).get("message", {}).get("content", [])
    text = "".join(block.get("text", "") for block in blocks)
    usage = response.get("usage", {})

    return {
        "model_id": model_id,
        "text": text,
        "stop_reason": response.get("stopReason"),
        "usage": {
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
        },
    }


def render_sketch(image_bytes: bytes, prompt: str, *,
                   control_strength: float = 0.7,
                   negative_prompt: str | None = None,
                   seed: int = 0,
                   output_format: str = "png") -> dict:
    """Render a line drawing into a finished image via Stability's Control
    Sketch service.

    Unlike ``converse()`` this goes through ``invoke_model`` rather than the
    Converse API — Control Sketch is image-conditioned (the drawing's lines
    shape the output directly), which Converse has no request shape for.
    There is no caller-selectable ``model_id``: this always calls
    ``config.BEDROCK_RENDER_MODEL_ID``.
    """
    payload: dict[str, object] = {
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "prompt": prompt,
        "control_strength": max(0.0, min(float(control_strength), 1.0)),
        "output_format": output_format,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed:
        payload["seed"] = int(seed)

    client = _get_client()
    try:
        response = client.invoke_model(
            modelId=config.BEDROCK_RENDER_MODEL_ID,
            body=json.dumps(payload),
        )
    except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
        raise _translate(exc) from exc

    body = json.loads(response["body"].read())

    # null means success; anything else names what got filtered.
    finish_reason = (body.get("finish_reasons") or [None])[0]
    if finish_reason:
        raise BedrockError(
            "That couldn't be rendered — try a different drawing or prompt.",
            status=422, detail=finish_reason)

    images = body.get("images") or []
    if not images:
        raise BedrockError(
            "The model didn't return an image.", status=502,
            detail=f"empty images, finish_reasons={body.get('finish_reasons')}")

    return {
        "image_bytes": base64.b64decode(images[0]),
        "output_format": output_format,
        "seed": (body.get("seeds") or [None])[0],
    }


def _translate(exc: Exception) -> BedrockError:
    """Map a botocore exception to something safe to return.

    AWS error messages can name account IDs, ARNs and role names, so the
    message the caller sees is written here and the original goes to the log.
    """
    name = type(exc).__name__
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")

    known = {
        "AccessDeniedException": (
            "This server isn't allowed to use that model.", 403),
        "ThrottlingException": (
            "The model service is busy. Try again in a moment.", 429),
        "ValidationException": (
            "That request wasn't valid for this model.", 400),
        "ResourceNotFoundException": (
            "That model isn't available in this region.", 404),
        "ModelTimeoutException": (
            "The model took too long to reply.", 504),
        "ServiceQuotaExceededException": (
            "This account is out of model quota.", 429),
        "ExpiredTokenException": (
            "The server's AWS credentials have expired.", 503),
        "UnrecognizedClientException": (
            "The server's AWS credentials are invalid.", 503),
    }
    message, status = known.get(code or name,
                                ("The model service failed. Try again?", 502))
    log(f"[bedrock] {code or name}: {exc}")
    return BedrockError(message, status=status, detail=f"{code or name}: {exc}")
