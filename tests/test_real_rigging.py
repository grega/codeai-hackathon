from __future__ import annotations

import json
import struct
from collections.abc import Callable
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

from providers.real.rigging import GlbError, RealRigger, rewrite_glb_joint_names
from schemas import BONES, ProviderError

REMOTE_NAMES = {
    "hips": "joint_pelvis",
    "spine": "joint_spine",
    "neck": "joint_neck",
    "head": "joint_head",
    "L_shoulder": "joint_left_shoulder",
    "L_elbow": "joint_left_elbow",
    "L_hand": "joint_left_hand",
    "R_shoulder": "joint_right_shoulder",
    "R_elbow": "joint_right_elbow",
    "R_hand": "joint_right_hand",
    "L_hip": "joint_left_hip",
    "L_knee": "joint_left_knee",
    "L_foot": "joint_left_foot",
    "R_hip": "joint_right_hip",
    "R_knee": "joint_right_knee",
    "R_foot": "joint_right_foot",
}


def make_glb(
    *,
    missing: str | None = None,
    duplicate: str | None = None,
) -> bytes:
    nodes = [{"name": "joint_root"}, {"name": "joint_chest"}]
    joints = [0, 1]
    for bone in BONES:
        if bone == missing:
            continue
        joints.append(len(nodes))
        nodes.append({"name": REMOTE_NAMES[bone]})
        if bone == duplicate:
            joints.append(len(nodes))
            nodes.append({"name": REMOTE_NAMES[bone]})

    document = {
        "asset": {"version": "2.0"},
        "nodes": nodes,
        "skins": [{"joints": joints}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    chunks = [
        (b"JSON", json_chunk),
        (b"BIN\x00", b"\x01\x02\x03\x04"),
        (b"TEST", b"keep-this-chunk"),
    ]
    body = b"".join(
        struct.pack("<I4s", len(payload), kind) + payload
        for kind, payload in chunks
    )
    return struct.pack("<4sII", b"glTF", 2, len(body) + 12) + body


def read_glb(glb: bytes) -> tuple[dict, list[tuple[bytes, bytes]]]:
    chunks = []
    offset = 12
    document = None
    while offset < len(glb):
        length, kind = struct.unpack_from("<I4s", glb, offset)
        payload = glb[offset + 8:offset + 8 + length]
        chunks.append((kind, payload))
        if kind == b"JSON":
            document = json.loads(payload.rstrip(b" \x00"))
        offset += 8 + length
    assert document is not None
    return document, chunks


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class SequenceOpener:
    def __init__(self, responses: list[tuple[str, dict | bytes | Exception]]):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, *, timeout: float):
        assert timeout > 0
        self.requests.append(request)
        expected_target, response = self.responses.pop(0)
        parsed = urlsplit(request.full_url)
        actual_target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        assert actual_target == expected_target
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            response = json.dumps(response).encode()
        return FakeResponse(response)


def rigger_with(
    responses: list[tuple[str, dict | bytes | Exception]],
    **kwargs,
) -> tuple[RealRigger, SequenceOpener]:
    opener = SequenceOpener(responses)
    rigger = RealRigger(
        "https://rigging.example.test",
        timeout=30,
        poll_interval=0,
        opener=opener,
        **kwargs,
    )
    return rigger, opener


def not_found() -> HTTPError:
    return HTTPError("https://rigging.example.test/missing", 404, "missing", {}, None)


def server_error() -> HTTPError:
    return HTTPError(
        "https://rigging.example.test/results/missing",
        500,
        "internal server error",
        {},
        None,
    )


def successful_responses(
    glb: bytes,
    *,
    cache_miss: HTTPError | None = None,
) -> list[tuple[str, dict | bytes | Exception]]:
    return [
        ("/classify", {"category": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", cache_miss or not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "status": "ok",
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a", {
            "status": "ok",
            "choice": "a",
            "active_image_url": "/results/drawing-1/augmented-a.png",
        }),
        ("/mesh?classify_id=drawing-1", {"status": "completed",
                   "mesh_url": "http://worker.internal/files/mesh.glb"}),
        ("/infer_joints?classify_id=drawing-1",
         {"joint_hints": [{"name": "joint_pelvis"}]}),
        ("/rig?classify_id=drawing-1", {
            "status": "completed",
            "glb_url": "https://assets.example.test/unrigged.glb",
            "rigged_url": "http://worker.internal/files/rigged.glb",
        }),
        ("/files/rigged.glb", glb),
    ]


def test_immediate_cache_hits_return_rewritten_glb():
    original = make_glb()
    rigger, opener = rigger_with(successful_responses(original))
    progress = []

    rig = rigger.rig(
        b"png", "image/png", lambda value, message: progress.append((value, message)))

    assert rig.format == "glb"
    assert rig.skeleton == BONES
    document, chunks = read_glb(rig.glb_bytes)
    skin_names = {
        document["nodes"][index]["name"]
        for index in document["skins"][0]["joints"]
    }
    assert set(BONES) <= skin_names
    assert {"joint_root", "joint_chest"} <= skin_names
    assert chunks[1:] == read_glb(original)[1][1:]
    assert progress[-1][0] == pytest.approx(0.99)
    assert opener.responses == []

    assert opener.requests[0].data == b"png"
    assert opener.requests[0].get_header("Content-type") == "image/png"
    assert all(request.data is None for request in opener.requests[2:5])


def test_existing_rig_is_downloaded_without_restarting_remote_tasks():
    rigger, opener = rigger_with([
        ("/classify", {"category": "humanoid", "classify_id": "cached-1"}),
        ("/results/cached-1/cached-1_rigged.glb", make_glb()),
    ])

    rig = rigger.rig(b"png", "image/png", lambda *_: None)

    assert rig.glb_bytes
    assert len(opener.requests) == 2


def test_result_endpoint_server_error_is_treated_as_cache_miss():
    rigger, opener = rigger_with(
        successful_responses(make_glb(), cache_miss=server_error()))

    rig = rigger.rig(b"png", "image/png", lambda *_: None)

    assert rig.glb_bytes
    assert any(
        urlsplit(request.full_url).path == "/mesh"
        for request in opener.requests
    )


def test_async_mesh_and_rig_tasks_are_polled():
    rigger, opener = rigger_with([
        ("/classify", {
            "result": {"label": "person"}, "classify_id": "drawing/1"}),
        ("/results/drawing%2F1/drawing%2F1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing%2F1", {
            "image_a_url": "/results/drawing%2F1/augmented-a.png",
            "image_b_url": "/results/drawing%2F1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing%2F1&choice=a",
         {"status": "ok", "choice": "a"}),
        ("/mesh?classify_id=drawing%2F1",
         {"status": "queued", "task_id": "mesh 1"}),
        ("/mesh/status/mesh%201", {"status": "meshy", "progress": 10}),
        ("/mesh/status/mesh%201", {"status": "decimating", "progress": 85}),
        ("/mesh/status/mesh%201", {"status": "done",
                            "result": {"url": "/files/mesh.glb"}}),
        ("/infer_joints?classify_id=drawing%2F1",
         {"result": {"joints": {"hips": [0, 0, 0]}}}),
        ("/rig?classify_id=drawing%2F1",
         {"state": "accepted", "job_id": "rig-1"}),
        ("/rig/status/rig-1", {"state": "rigging", "progress": 10}),
        ("/rig/status/rig-1", {"state": "decimating", "progress": 20}),
        ("/rig/status/rig-1", {"state": "inferring_skeleton", "progress": 30}),
        ("/rig/status/rig-1", {"state": "injecting_keyframes", "progress": 40}),
        ("/rig/status/rig-1", {"state": "visualizing", "progress": 50}),
        ("/rig/status/rig-1", {"state": "rigging_blender", "progress": 60}),
        ("/rig/status/rig-1", {"state": "finalizing", "progress": 90}),
        ("/rig/status/rig-1", {"state": "succeeded",
                        "result": {"download_url": "/files/avatar.glb"}}),
        ("/files/avatar.glb", make_glb()),
    ])
    seen = []

    rig = rigger.rig(b"png", "image/png", lambda f, m: seen.append((f, m)))

    assert rig.glb_bytes
    assert sum(request.method == "GET" for request in opener.requests) == 13
    assert any(0.26 < fraction <= 0.52 for fraction, _ in seen)
    assert any(0.70 < fraction <= 0.90 for fraction, _ in seen)


@pytest.mark.parametrize("response", [
    {"classification": "car"},
    {"result": {"category": "animal"}},
    {"is_humanoid": False},
])
def test_non_humanoids_get_a_child_safe_error(response):
    rigger, _ = rigger_with([("/classify", response)])
    with pytest.raises(ProviderError, match="non-humanoid classification") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)
    assert "drawings of people" in caught.value.user_message


def test_malformed_json_is_reported_as_provider_error():
    rigger, _ = rigger_with([("/classify", b"<html>not json</html>")])
    with pytest.raises(ProviderError, match="malformed JSON"):
        rigger.rig(b"png", "image/png", lambda *_: None)


def test_remote_http_error_is_reported_as_provider_error():
    error = HTTPError(
        "https://rigging.example.test/classify", 503, "unavailable", {}, None)
    error.read = lambda _size=-1: b"service unavailable"
    rigger, _ = rigger_with([("/classify", error)])
    with pytest.raises(ProviderError, match="HTTP 503"):
        rigger.rig(b"png", "image/png", lambda *_: None)


def test_failed_remote_task_is_reported():
    rigger, _ = rigger_with([
        ("/classify", {"classification": "human", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a",
         {"status": "ok", "choice": "a"}),
        ("/mesh?classify_id=drawing-1",
         {"status": "failed", "error": "out of GPU memory"}),
    ])
    with pytest.raises(ProviderError, match="out of GPU memory") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)
    assert caught.value.user_message == \
        "I couldn't finish building that avatar. Try the drawing again?"


def test_invalid_download_is_rejected():
    rigger, _ = rigger_with(successful_responses(b"not a glb"))
    with pytest.raises(ProviderError, match="invalid rigged GLB"):
        rigger.rig(b"png", "image/png", lambda *_: None)


@pytest.mark.parametrize("augmentation", [
    {"image_b_url": "/results/drawing-1/augmented-b.png"},
    {
        "image_a_url": "/results/drawing-1/augmented-a.png",
        "image_b_url": "",
    },
])
def test_incomplete_augmentation_response_stops_before_mesh(augmentation):
    rigger, opener = rigger_with([
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", augmentation),
    ])

    with pytest.raises(ProviderError, match="invalid image URLs") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)

    assert "incomplete pose" in caught.value.user_message
    assert len(opener.requests) == 3


def test_incomplete_augmentation_confirmation_stops_before_mesh():
    rigger, opener = rigger_with([
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a",
         {"status": "ok"}),
    ])

    with pytest.raises(ProviderError, match="confirmation was incomplete") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)

    assert "couldn't choose a pose" in caught.value.user_message
    assert len(opener.requests) == 4


@pytest.mark.parametrize("failed_target", ["augment", "confirm"])
def test_augmentation_http_errors_stop_before_mesh(failed_target):
    error = HTTPError(
        "https://rigging.example.test/augment_image",
        503,
        "unavailable",
        {},
        None,
    )
    responses = [
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
    ]
    if failed_target == "augment":
        responses.append(("/augment_image?classify_id=drawing-1", error))
    else:
        responses.extend([
            ("/augment_image?classify_id=drawing-1", {
                "image_a_url": "/results/drawing-1/augmented-a.png",
                "image_b_url": "/results/drawing-1/augmented-b.png",
            }),
            ("/augment_image/confirm?classify_id=drawing-1&choice=a", error),
        ])
    rigger, opener = rigger_with(responses)

    with pytest.raises(ProviderError, match="HTTP 503"):
        rigger.rig(b"png", "image/png", lambda *_: None)

    assert len(opener.requests) == len(responses)


def test_overall_deadline_applies_while_polling():
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    opener = SequenceOpener([
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a",
         {"status": "ok", "choice": "a"}),
        ("/mesh?classify_id=drawing-1",
         {"status": "queued", "task_id": "slow"}),
    ])
    rigger = RealRigger(
        "https://rigging.example.test",
        timeout=1,
        poll_interval=2,
        opener=opener,
        clock=clock,
        sleep=sleep,
    )
    with pytest.raises(ProviderError, match="exceeded 1.000s deadline") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)
    assert "took too long" in caught.value.user_message


def test_http_urls_are_rebuilt_on_the_configured_https_origin():
    rigger = RealRigger("https://public.ngrok-free.app/api")
    assert rigger._service_url(
        "http://127.0.0.1:8000/output/avatar.glb?token=abc"
    ) == "https://public.ngrok-free.app/output/avatar.glb?token=abc"
    assert rigger._service_url(
        "/tasks/123"
    ) == "https://public.ngrok-free.app/api/tasks/123"


def test_glb_rewriter_preserves_non_json_chunks_and_extra_nodes():
    original = make_glb()
    rewritten = rewrite_glb_joint_names(original)
    document, chunks = read_glb(rewritten)
    original_document, original_chunks = read_glb(original)

    assert chunks[1:] == original_chunks[1:]
    assert [node["name"] for node in document["nodes"][:2]] == \
        [node["name"] for node in original_document["nodes"][:2]]
    assert struct.unpack_from("<I", rewritten, 8)[0] == len(rewritten)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"missing": "R_foot"}, "missing contract bones: R_foot"),
    ({"duplicate": "L_elbow"}, "duplicate contract bones: L_elbow"),
])
def test_glb_rewriter_rejects_invalid_contract_skins(kwargs, message):
    with pytest.raises(GlbError, match=message):
        rewrite_glb_joint_names(make_glb(**kwargs))


@pytest.mark.parametrize("mutate", [
    lambda value: b"bad!" + value[4:],
    lambda value: value[:8] + struct.pack("<I", len(value) + 4) + value[12:],
    lambda value: value[:20] + b"!" + value[21:],
])
def test_glb_rewriter_rejects_malformed_files(
    mutate: Callable[[bytes], bytes],
):
    with pytest.raises(GlbError):
        rewrite_glb_joint_names(mutate(make_glb()))
