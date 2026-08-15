"""Flask app: JSON API + the static frontend, one process.

Routes are deliberately thin. They validate input, hand off to a provider, and
serialise the result — no domain logic lives here. See CONTRACT.md.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

import bedrock
import config
import export
import providers
from auth import RateLimiter, limiter, require_token
from jobs import runner
from logs import log
from rewards import TERM_LABELS
from schemas import (
    Clip,
    ProviderError,
    TrainConfig,
    schema_json,
    validate_clip,
    validate_rig,
)
from store import store
from training import TrainingRun

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

# Keep dict order as declared. The bone tree is written parents-first in
# schemas.py and reads far better that way in the API too; Flask would
# otherwise sort it alphabetically.
app.json.sort_keys = False

STATIC_DIR = Path(__file__).parent / "static"

# Separate bucket from the LLM-endpoint limiter in auth.py: this one guards a
# route with no bearer token, so it's keyed by client address alone.
render_limiter = RateLimiter("RENDER_RATE_PER_MINUTE", "RENDER_RATE_PER_DAY")
tpose_limiter = RateLimiter("TPOSE_RATE_PER_MINUTE", "TPOSE_RATE_PER_DAY")


# --------------------------------------------------------------------------
# Errors — every failure reaches the browser in the same shape, so the frontend
# has exactly one error path to render.
# --------------------------------------------------------------------------

def fail(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.errorhandler(404)
def _not_found(_):
    if request.path.startswith("/api/"):
        return fail("Not found.", 404)
    return send_from_directory(STATIC_DIR, "index.html")


@app.errorhandler(413)
def _too_large(_):
    mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    return fail(f"That image is too big — keep it under {mb}MB.", 413)


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(STATIC_DIR, filename)


# --------------------------------------------------------------------------
# Schema + status
# --------------------------------------------------------------------------

@app.get("/api/schema")
def get_schema():
    """The browser reads bone names and reward terms from here rather than
    hard-coding them, so the contract can only ever be defined in one place."""
    return jsonify({
        **schema_json(),
        "reward_labels": TERM_LABELS,
        "providers": providers.active(),
    })


# --------------------------------------------------------------------------
# Phase 1 — sketch to rigged avatar
# --------------------------------------------------------------------------

@app.post("/api/avatars")
def create_avatar():
    upload = request.files.get("image")
    if upload is None:
        return fail("No image was uploaded.")

    image_bytes = upload.read()
    if not image_bytes:
        return fail("That image was empty. Try drawing something!")
    mime = upload.mimetype or "image/png"

    rigger = providers.get_rigger()

    def work(progress):
        rig = validate_rig(rigger.rig(image_bytes, mime, progress))
        avatar = store.add_avatar(rig, image_bytes=image_bytes, mime=mime)
        return avatar.to_json()

    job = runner.submit(work, message="Waking up your avatar...")
    return jsonify(job.to_json()), 202


@app.get("/api/avatars/<avatar_id>")
def get_avatar(avatar_id: str):
    avatar = store.get_avatar(avatar_id)
    if not avatar:
        return fail("That avatar doesn't exist.", 404)
    return jsonify(avatar.to_json())


@app.get("/api/avatars/<avatar_id>/image")
def get_avatar_image(avatar_id: str):
    avatar = store.get_avatar(avatar_id)
    if not avatar or not avatar.image_path:
        return fail("No drawing saved for that avatar.", 404)
    return send_file(avatar.image_path)


@app.get("/api/avatars/<avatar_id>/glb")
def get_avatar_glb(avatar_id: str):
    avatar = store.get_avatar(avatar_id)
    if not avatar or not avatar.rig.glb_bytes:
        return fail("That avatar has no GLB — it's drawn procedurally.", 404)
    # Without validators a browser heuristically caches a 3MB binary and never
    # asks again — so a swapped fixture keeps rendering the old body with no
    # sign anything is stale. An ETag over the bytes plus no-cache means the
    # browser always revalidates, but pays for the transfer only when the
    # content actually differs (304 otherwise).
    etag = hashlib.sha256(avatar.rig.glb_bytes).hexdigest()[:32]
    if request.if_none_match.contains(etag):
        return Response(status=304, headers={"ETag": f'"{etag}"',
                                             "Cache-Control": "no-cache"})

    return Response(avatar.rig.glb_bytes, mimetype="model/gltf-binary",
                    headers={"ETag": f'"{etag}"', "Cache-Control": "no-cache"})


@app.post("/api/avatars/<avatar_id>/render")
def render_avatar(avatar_id: str):
    """Send the avatar's line drawing plus a prompt to Bedrock and get a
    rendered image back — a purpose-specific endpoint, not the general
    prompt pipe at /api/llm/generate (see the note above that route)."""
    avatar = store.get_avatar(avatar_id)
    if not avatar or not avatar.image_path:
        return fail("That avatar doesn't exist.", 404)

    prompt = (request.json or {}).get("prompt", "").strip()
    if not prompt:
        return fail("Describe how you'd like this rendered.")
    if len(prompt) > config.BEDROCK_MAX_PROMPT_CHARS:
        return fail(f"Prompt is too long — the limit is "
                    f"{config.BEDROCK_MAX_PROMPT_CHARS} characters.", 413)

    refusal = render_limiter.check(request.remote_addr or "unknown")
    if refusal:
        return fail(refusal, 429)

    image_bytes = Path(avatar.image_path).read_bytes()

    def work(progress):
        try:
            # The T-pose transform prefers this render as its source (see
            # tpose_avatar below), so it needs to hold full-body framing
            # itself — a close-up here can't be recovered later.
            result = bedrock.render_sketch(
                image_bytes, f"{prompt}, {bedrock.FULL_BODY_HINT}",
                negative_prompt=bedrock.FULL_BODY_NEGATIVE_HINT)
        except bedrock.BedrockError as exc:
            raise ProviderError(exc.message, detail=exc.detail) from exc
        # The player's actual designed look, not just the raw sketch — later
        # steps (the T-pose transform) prefer this over avatar.image_path.
        store.set_rendered_image(avatar_id, result["image_bytes"])
        return {
            "image_base64": base64.b64encode(result["image_bytes"]).decode("ascii"),
            "output_format": result["output_format"],
            "seed": result["seed"],
        }

    job = runner.submit(work, message="Rendering your character...")
    return jsonify(job.to_json()), 202


@app.post("/api/avatars/<avatar_id>/tpose")
def tpose_avatar(avatar_id: str):
    """Turn the avatar into a forward-facing, T-pose, transparent-background
    PNG, via bedrock.tpose_transform. Fixed shape, no request body — always
    the same transform, so nothing to validate beyond the avatar existing.

    Always built from the avatar's most recent /render output, never the raw
    line drawing: a plain black-on-white sketch gives Bedrock nothing to work
    with for colour or style, and testing found no way to reliably recover a
    good pose from it. Refuses if nothing has been rendered yet rather than
    silently falling back to the sketch."""
    avatar = store.get_avatar(avatar_id)
    if not avatar or not avatar.image_path:
        return fail("That avatar doesn't exist.", 404)
    if not avatar.rendered_image_bytes:
        return fail("Render your avatar first — the posed version is built "
                     "from that.", 400)

    refusal = tpose_limiter.check(request.remote_addr or "unknown")
    if refusal:
        return fail(refusal, 429)

    image_bytes = avatar.rendered_image_bytes

    def work(progress):
        try:
            result = bedrock.tpose_transform(image_bytes)
        except bedrock.BedrockError as exc:
            raise ProviderError(exc.message, detail=exc.detail) from exc
        return {
            "image_base64": base64.b64encode(result["image_bytes"]).decode("ascii"),
            "output_format": result["output_format"],
        }

    job = runner.submit(work, message="Posing your character...")
    return jsonify(job.to_json()), 202


# --------------------------------------------------------------------------
# Phase 2 — prompt to pose
# --------------------------------------------------------------------------

@app.post("/api/avatars/<avatar_id>/poses")
def create_pose(avatar_id: str):
    avatar = store.get_avatar(avatar_id)
    if not avatar:
        return fail("That avatar doesn't exist.", 404)

    prompt = (request.json or {}).get("prompt", "").strip()
    if not prompt:
        return fail("Tell me what you'd like your avatar to do!")

    poser = providers.get_poser()

    def work(progress):
        clip = validate_clip(poser.pose(prompt, avatar.rig, progress))
        clip.prompt = clip.prompt or prompt
        return store.add_clip(clip).to_json()

    job = runner.submit(work, message="Thinking about that move...")
    return jsonify(job.to_json()), 202


@app.get("/api/clips/<clip_id>")
def get_clip(clip_id: str):
    clip = store.get_clip(clip_id)
    if not clip:
        return fail("That move doesn't exist.", 404)
    return jsonify(clip.to_json())


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = runner.get(job_id)
    if not job:
        return fail("That job has expired.", 404)
    return jsonify(job.to_json())


# --------------------------------------------------------------------------
# Phases 3 & 4 — training
# --------------------------------------------------------------------------

@app.post("/api/training/runs")
def create_run():
    body = request.json or {}
    avatar = store.get_avatar(body.get("avatar_id", ""))
    if not avatar:
        return fail("That avatar doesn't exist.", 404)

    target = store.get_clip(body.get("target_clip_id", ""))
    if not target:
        return fail("Pick a move to train towards first.", 404)

    cfg = TrainConfig.from_json(body.get("config"))
    run = TrainingRun(avatar.id, target, cfg)
    store.add_run(run)
    run.set_speed(body.get("speed", 1.0))
    run.start(providers.get_trainer(), avatar.rig)
    return jsonify(run.to_json()), 201


@app.get("/api/training/runs/<run_id>")
def get_run(run_id: str):
    run = store.get_run(run_id)
    if not run:
        return fail("That training run doesn't exist.", 404)
    return jsonify(run.to_json())


# --------------------------------------------------------------------------
# Training output
#
# What a finished run produces, for whatever comes next. Two artifacts because
# they have different audiences: the GLB is for rendering and 3D tools, the
# JSON is for the animation step. See export.py — in particular, do not hand
# the GLB to a language model.
# --------------------------------------------------------------------------

def _export_inputs(run_id: str):
    """Resolve a run to (run, avatar, clip), or raise ExportError."""
    run = store.get_run(run_id)
    if not run:
        raise export.ExportError("That training run doesn't exist.", 404)
    avatar = store.get_avatar(run.avatar_id)
    if not avatar:
        raise export.ExportError("That avatar is no longer around.", 404,
                                 "avatar evicted from the in-memory store")
    return run, avatar, run.target_clip


@app.get("/api/training/runs/<run_id>/export.glb")
def export_run_glb(run_id: str):
    """The learned pose, baked into the avatar's own model."""
    try:
        run, avatar, clip = _export_inputs(run_id)
        data, _ = export.build_glb(run, avatar, clip)
    except export.ExportError as exc:
        if exc.detail:
            log(f"[export] {run_id}: {exc.detail}")
        return fail(exc.user_message, exc.status)

    name = f"{clip.name.replace(' ', '-').lower()}-{run_id}.glb"
    return Response(data, mimetype="model/gltf-binary", headers={
        "Content-Disposition": f'attachment; filename="{name}"',
        "Cache-Control": "no-cache",
    })


@app.get("/api/training/runs/<run_id>/export.json")
def export_run_json(run_id: str):
    """Bones, start pose, end pose, provenance — the animation step's input."""
    try:
        run, avatar, clip = _export_inputs(run_id)
        document = export.build_document(run, avatar, clip)
    except export.ExportError as exc:
        if exc.detail:
            log(f"[export] {run_id}: {exc.detail}")
        return fail(exc.user_message, exc.status)
    return jsonify(document)


@app.get("/api/training/runs/<run_id>/events")
def stream_run(run_id: str):
    run = store.get_run(run_id)
    if not run:
        return fail("That training run doesn't exist.", 404)

    def generate():
        # Flush a comment straight away so the proxy commits to the response
        # and the browser's EventSource opens rather than sitting on a buffer.
        yield ": open\n\n"
        for episode in run.events():
            if episode is None:
                yield ": keepalive\n\n"
                continue
            # The id lets a reconnecting browser tell us where it got to.
            yield (f"id: {episode.episode}\n"
                   f"event: episode\ndata: {json.dumps(episode.to_json())}\n\n")
        yield f"event: end\ndata: {json.dumps(run.to_json())}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # don't let a proxy swallow the stream
        "Connection": "keep-alive",
    })


@app.post("/api/training/runs/<run_id>/<action>")
def control_run(run_id: str, action: str):
    run = store.get_run(run_id)
    if not run:
        return fail("That training run doesn't exist.", 404)

    if action == "stop":
        run.stop()
    elif action == "pause":
        run.pause()
    elif action == "resume":
        run.resume()
    elif action == "speed":
        run.set_speed((request.json or {}).get("speed", 1.0))
    else:
        return fail(f"Unknown action '{action}'.", 404)
    return jsonify(run.to_json())


# --------------------------------------------------------------------------
# Behaviours — the bridge from "a pose" to "a thing my avatar does"
# --------------------------------------------------------------------------

@app.get("/api/behaviours")
def list_behaviours():
    avatar_id = request.args.get("avatar_id")
    return jsonify({"behaviours": [b.to_json()
                                   for b in store.list_behaviours(avatar_id)]})


@app.post("/api/behaviours")
def create_behaviour():
    body = request.json or {}
    avatar = store.get_avatar(body.get("avatar_id", ""))
    if not avatar:
        return fail("That avatar doesn't exist.", 404)

    name = (body.get("name") or "").strip()
    if not name:
        return fail("Give this behaviour a name.")

    clip_id = body.get("clip_id")
    if clip_id:
        clip = store.get_clip(clip_id)
        if not clip:
            return fail("That move doesn't exist.", 404)
    elif body.get("clip"):
        clip = store.add_clip(validate_clip(body["clip"]))
    else:
        return fail("A behaviour needs a move to play.")

    behaviour = store.add_behaviour(
        name=name, clip=clip, avatar_id=avatar.id,
        trained=bool(body.get("trained")),
        best_reward=float(body.get("best_reward", 0.0)))
    return jsonify(behaviour.to_json()), 201


# --------------------------------------------------------------------------
# Bedrock prompt endpoint
#
# A utility for the teams building the real providers: run a prompt against a
# Bedrock model and see what comes back. Guarded by a bearer token, a model
# allowlist and rate limits — see auth.py and bedrock.py.
#
# NOT called by the frontend, and the token must never be shipped to the
# browser: anything the browser holds is public, and this endpoint spends money.
# --------------------------------------------------------------------------

@app.post("/api/llm/generate")
@require_token
def llm_generate():
    body = request.json or {}

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return fail("A 'prompt' is required.")
    if len(prompt) > config.BEDROCK_MAX_PROMPT_CHARS:
        return fail(f"Prompt is too long — the limit is "
                    f"{config.BEDROCK_MAX_PROMPT_CHARS} characters.", 413)

    model_id = (body.get("model_id") or "").strip()
    if not model_id:
        return fail("A 'model_id' is required. GET /api/llm/models lists the "
                    "ones this server allows.")

    system = (body.get("system") or "").strip() or None
    if system and len(system) > config.BEDROCK_MAX_PROMPT_CHARS:
        return fail("System prompt is too long.", 413)

    try:
        temperature = body.get("temperature")
        result = bedrock.converse(
            model_id, prompt,
            system=system,
            max_tokens=int(body.get("max_tokens", 1024)),
            temperature=None if temperature is None else float(temperature),
        )
    except bedrock.BedrockError as exc:
        return fail(exc.message, exc.status)
    except (TypeError, ValueError) as exc:
        return fail(f"Invalid request: {exc}", 400)

    return jsonify(result)


@app.get("/api/llm/models")
@require_token
def llm_models():
    """What this deployment will actually run, plus current rate-limit usage."""
    return jsonify({
        "models": bedrock.allowed_models(),
        "region": config.BEDROCK_REGION or None,
        "limits": {
            **limiter.snapshot(),
            "max_tokens": config.BEDROCK_MAX_TOKENS,
            "max_prompt_chars": config.BEDROCK_MAX_PROMPT_CHARS,
        },
    })


@app.errorhandler(ProviderError)
def _provider_error(exc: ProviderError):
    return fail(exc.user_message, 502)


if __name__ == "__main__":
    config.ensure_dirs()
    app.run(debug=True, threaded=True, port=5000)
