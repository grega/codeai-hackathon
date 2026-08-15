"""The Draw workflow renders before it creates and rigs an avatar."""

from __future__ import annotations

import base64
import io
import time

import pytest

import app as app_module
import auth
import bedrock
import config
import providers
from jobs import runner
from schemas import BONES, Rig
from store import store

DRAWING = b"input drawing bytes"
RENDERED = b"rendered png bytes"
POSED = b"posed png bytes"


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_image_limiters():
    app_module.render_limiter = auth.RateLimiter(
        "RENDER_RATE_PER_MINUTE", "RENDER_RATE_PER_DAY")
    app_module.tpose_limiter = auth.RateLimiter(
        "TPOSE_RATE_PER_MINUTE", "TPOSE_RATE_PER_DAY")


def wait_for_job(job_id: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job and job.status not in ("queued", "running"):
            return job
        time.sleep(0.01)
    pytest.fail(f"job {job_id} did not finish")


def test_render_accepts_a_drawing_without_creating_or_rigging_an_avatar(
        client, monkeypatch):
    captured = {}
    avatar_ids = set(store.avatars)

    def fake_render(image_bytes, prompt, **kwargs):
        captured.update(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt"),
        )
        return {"image_bytes": RENDERED, "output_format": "png", "seed": 7}

    def unexpected_rigger():
        pytest.fail("rendering must not invoke the rigger")

    monkeypatch.setattr(bedrock, "render_sketch", fake_render)
    monkeypatch.setattr(providers, "get_rigger", unexpected_rigger)

    response = client.post(
        "/api/renders",
        data={"image": (io.BytesIO(DRAWING), "drawing.png"),
              "prompt": "a colorful robot"},
    )

    assert response.status_code == 202
    job = wait_for_job(response.get_json()["id"])
    assert job.status == "done"
    assert captured == {
        "image_bytes": DRAWING,
        "prompt": f"a colorful robot, {bedrock.FULL_BODY_HINT}",
        "negative_prompt": bedrock.FULL_BODY_NEGATIVE_HINT,
    }
    assert base64.b64decode(job.result["image_base64"]) == RENDERED
    assert set(store.avatars) == avatar_ids


def test_avatar_creation_passes_rendered_bytes_to_rigging_unchanged(
        client, monkeypatch, tmp_path):
    captured = {}

    class CapturingRigger:
        def rig(self, image_bytes, mime, progress):
            captured.update(image_bytes=image_bytes, mime=mime)
            progress(0.5, "Rigging...")
            return Rig(format="procedural", skeleton=list(BONES))

    monkeypatch.setattr(providers, "get_rigger", lambda: CapturingRigger())
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/api/avatars",
        data={"image": (
            io.BytesIO(RENDERED), "rendered.png", "image/png")},
    )

    assert response.status_code == 202
    job = wait_for_job(response.get_json()["id"])
    assert job.status == "done"
    assert captured == {"image_bytes": RENDERED, "mime": "image/png"}
    assert (tmp_path / f"{job.result['id']}.png").read_bytes() == RENDERED
    store.avatars.pop(job.result["id"], None)


def test_tpose_uses_the_same_rendered_image_as_rigging(
        client, monkeypatch, tmp_path):
    captured = {}

    class CapturingRigger:
        def rig(self, image_bytes, mime, progress):
            captured["rig"] = image_bytes
            return Rig(format="procedural", skeleton=list(BONES))

    def fake_tpose(image_bytes):
        captured["tpose"] = image_bytes
        return {"image_bytes": POSED, "output_format": "png"}

    monkeypatch.setattr(providers, "get_rigger", lambda: CapturingRigger())
    monkeypatch.setattr(bedrock, "tpose_transform", fake_tpose)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)

    create_response = client.post(
        "/api/avatars",
        data={"image": (
            io.BytesIO(RENDERED), "rendered.png", "image/png")},
    )
    create_job = wait_for_job(create_response.get_json()["id"])
    avatar_id = create_job.result["id"]

    tpose_response = client.post(f"/api/avatars/{avatar_id}/tpose")
    tpose_job = wait_for_job(tpose_response.get_json()["id"])

    assert tpose_job.status == "done"
    assert captured == {"rig": RENDERED, "tpose": RENDERED}
    assert base64.b64decode(tpose_job.result["image_base64"]) == POSED
    store.avatars.pop(avatar_id, None)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"prompt": "robot"}, "No image was uploaded."),
        ({"image": (io.BytesIO(b""), "empty.png"), "prompt": "robot"},
         "That image was empty."),
        ({"image": (io.BytesIO(DRAWING), "drawing.png"), "prompt": ""},
         "Describe how you'd like this rendered."),
    ],
)
def test_render_validates_multipart_input(client, data, message):
    response = client.post("/api/renders", data=data)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_render_rejects_an_oversized_prompt(client, monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_MAX_PROMPT_CHARS", 5)

    response = client.post(
        "/api/renders",
        data={"image": (io.BytesIO(DRAWING), "drawing.png"),
              "prompt": "too long"},
    )

    assert response.status_code == 413
    assert "limit is 5 characters" in response.get_json()["error"]


def test_render_job_exposes_only_the_safe_bedrock_error(
        client, monkeypatch):
    secret = "arn:aws:iam::123456789012:role/RenderRole"

    def fail_render(*args, **kwargs):
        raise bedrock.BedrockError(
            "That couldn't be rendered.", status=403, detail=secret)

    monkeypatch.setattr(bedrock, "render_sketch", fail_render)

    response = client.post(
        "/api/renders",
        data={"image": (io.BytesIO(DRAWING), "drawing.png"),
              "prompt": "robot"},
    )
    job = wait_for_job(response.get_json()["id"])
    wire_response = client.get(f"/api/jobs/{job.id}")

    assert job.status == "error"
    assert job.error == "That couldn't be rendered."
    assert wire_response.get_json()["error"] == "That couldn't be rendered."
    assert secret not in wire_response.get_data(as_text=True)
