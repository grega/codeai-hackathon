"""A compatible GLB can enter the experience without running phase 1."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import app as app_module
import gltf
import providers
from schemas import BONES
from store import store

FIXTURE = Path(__file__).parent / "fixtures" / "mixamo-style.glb"


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    existing = set(store.avatars)
    with app_module.app.test_client() as test_client:
        yield test_client
    for avatar_id in set(store.avatars) - existing:
        store.avatars.pop(avatar_id, None)


def post_glb(client, data: bytes, filename: str = "avatar.glb"):
    return client.post(
        "/api/avatars/glb",
        data={"glb": (io.BytesIO(data), filename)},
    )


def test_sideload_creates_a_glb_avatar_without_calling_the_rigger(
        client, monkeypatch):
    def unexpected_rigger():
        pytest.fail("sideloading a GLB must not invoke the phase 1 rigger")

    monkeypatch.setattr(providers, "get_rigger", unexpected_rigger)
    glb_bytes = FIXTURE.read_bytes()

    response = post_glb(client, glb_bytes)

    assert response.status_code == 201
    avatar_json = response.get_json()
    assert avatar_json["image_url"] is None
    assert avatar_json["rig"]["format"] == "glb"
    assert avatar_json["rig"]["glb_url"].endswith("/glb")
    assert avatar_json["rig"]["skeleton"] == BONES

    avatar = store.get_avatar(avatar_json["id"])
    assert avatar is not None
    assert avatar.rig.glb_bytes == glb_bytes

    downloaded = client.get(avatar_json["rig"]["glb_url"])
    assert downloaded.status_code == 200
    assert downloaded.mimetype == "model/gltf-binary"
    assert downloaded.data == glb_bytes


def test_sideload_is_not_subject_to_the_sketch_upload_limit(client):
    previous_limit = app_module.app.config["MAX_CONTENT_LENGTH"]
    app_module.app.config["MAX_CONTENT_LENGTH"] = 1024
    try:
        response = post_glb(client, FIXTURE.read_bytes())
    finally:
        app_module.app.config["MAX_CONTENT_LENGTH"] = previous_limit

    assert response.status_code == 201


def test_sideload_accepts_mixamo_names_without_the_mixamorig_prefix(client):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    for node in document["nodes"]:
        name = node.get("name", "")
        if name.startswith("mixamorig:"):
            node["name"] = name.split(":", 1)[1]

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 201


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "No GLB was uploaded."),
        ({"glb": (io.BytesIO(b""), "empty.glb")}, "That GLB was empty."),
        ({"glb": (io.BytesIO(b"not a glb"), "broken.glb")},
         "That GLB can't be used:"),
    ],
)
def test_sideload_rejects_missing_empty_and_malformed_files(
        client, data, message):
    response = client.post("/api/avatars/glb", data=data)

    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_sideload_rejects_an_unskinned_model(client):
    document = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": bone} for bone in BONES],
        "scenes": [{"nodes": list(range(len(BONES)))}],
        "scene": 0,
    }

    response = post_glb(client, gltf.write_glb(document, b""))

    assert response.status_code == 400
    assert "does not contain a skinned skeleton" in response.get_json()["error"]


def test_sideload_rejects_invalid_node_names_cleanly(client):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    document["nodes"][0]["name"] = 42

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 400
    assert "node with an invalid name" in response.get_json()["error"]


def test_sideload_names_the_missing_required_joint(client):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    head_index = gltf.resolve_bones(document)["head"]
    document["nodes"][head_index]["name"] = "unsupported_head_joint"

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 400
    assert "missing required joints: head" in response.get_json()["error"]
