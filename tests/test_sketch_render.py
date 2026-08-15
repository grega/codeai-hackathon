"""Sketch rendering and rigging are independent operations."""

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


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_render_limiter():
    app_module.render_limiter = auth.RateLimiter(
        "RENDER_RATE_PER_MINUTE", "RENDER_RATE_PER_DAY")


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

    def fake_render(image_bytes, prompt):
        captured.update(image_bytes=image_bytes, prompt=prompt)
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
        "prompt": "a colorful robot",
    }
    assert base64.b64decode(job.result["image_base64"]) == RENDERED
    assert set(store.avatars) == avatar_ids


def test_avatar_creation_passes_the_uploaded_sketch_to_the_rigger(
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
        data={"image": (io.BytesIO(DRAWING), "drawing.png")},
    )

    assert response.status_code == 202
    job = wait_for_job(response.get_json()["id"])
    assert job.status == "done"
    assert captured == {
        "image_bytes": DRAWING,
        "mime": "image/png",
    }
    store.avatars.pop(job.result["id"], None)


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
