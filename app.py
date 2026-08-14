"""Flask app: JSON API + the static frontend, one process.

Routes are deliberately thin. They validate input, hand off to a provider, and
serialise the result — no domain logic lives here. See CONTRACT.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

import config
import providers
from jobs import runner
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
    return Response(avatar.rig.glb_bytes, mimetype="model/gltf-binary")


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


@app.errorhandler(ProviderError)
def _provider_error(exc: ProviderError):
    return fail(exc.user_message, 502)


if __name__ == "__main__":
    config.ensure_dirs()
    app.run(debug=True, threaded=True, port=5000)
