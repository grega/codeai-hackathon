"""Real sketch-to-GLB rigging provider.

The remote service stays behind this adapter. Its temporary URLs are consumed
immediately and the resulting GLB bytes are returned through the normal Rig
contract, so browsers only ever request /api/avatars/<id>/glb from this app.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import config
from providers.base import Progress, Rigger
from schemas import BONES, ProviderError, Rig

_JSON_CHUNK = b"JSON"
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_GLB_BYTES = 100 * 1024 * 1024
_PENDING_STATUSES = {
    "accepted", "created", "downloading", "in_progress", "meshy", "pending",
    "processing", "queued", "rigging", "running", "started", "uploading",
}
_SUCCESS_STATUSES = {"complete", "completed", "done", "success", "succeeded"}
_FAILED_STATUSES = {"cancelled", "canceled", "error", "failed", "failure"}
_HUMANOID_LABELS = {"biped", "human", "humanoid", "person"}

# The rigging service uses semantic joint names but the Teach contract has its
# own stable vocabulary. Matching is normalized for case and separator style;
# values not listed here (notably root and chest) remain untouched.
_REMOTE_JOINT_ALIASES: dict[str, str] = {
    "joint_pelvis": "hips",
    "joint_hips": "hips",
    "joint_spine": "spine",
    "joint_neck": "neck",
    "joint_head": "head",

    "joint_left_shoulder": "L_shoulder",
    "joint_shoulder_left": "L_shoulder",
    "joint_l_shoulder": "L_shoulder",
    "joint_shoulder_l": "L_shoulder",
    "joint_left_elbow": "L_elbow",
    "joint_elbow_left": "L_elbow",
    "joint_l_elbow": "L_elbow",
    "joint_elbow_l": "L_elbow",
    "joint_left_hand": "L_hand",
    "joint_hand_left": "L_hand",
    "joint_l_hand": "L_hand",
    "joint_hand_l": "L_hand",

    "joint_right_shoulder": "R_shoulder",
    "joint_shoulder_right": "R_shoulder",
    "joint_r_shoulder": "R_shoulder",
    "joint_shoulder_r": "R_shoulder",
    "joint_right_elbow": "R_elbow",
    "joint_elbow_right": "R_elbow",
    "joint_r_elbow": "R_elbow",
    "joint_elbow_r": "R_elbow",
    "joint_right_hand": "R_hand",
    "joint_hand_right": "R_hand",
    "joint_r_hand": "R_hand",
    "joint_hand_r": "R_hand",

    "joint_left_hip": "L_hip",
    "joint_hip_left": "L_hip",
    "joint_l_hip": "L_hip",
    "joint_hip_l": "L_hip",
    "joint_left_knee": "L_knee",
    "joint_knee_left": "L_knee",
    "joint_l_knee": "L_knee",
    "joint_knee_l": "L_knee",
    "joint_left_foot": "L_foot",
    "joint_foot_left": "L_foot",
    "joint_l_foot": "L_foot",
    "joint_foot_l": "L_foot",

    "joint_right_hip": "R_hip",
    "joint_hip_right": "R_hip",
    "joint_r_hip": "R_hip",
    "joint_hip_r": "R_hip",
    "joint_right_knee": "R_knee",
    "joint_knee_right": "R_knee",
    "joint_r_knee": "R_knee",
    "joint_knee_r": "R_knee",
    "joint_right_foot": "R_foot",
    "joint_foot_right": "R_foot",
    "joint_r_foot": "R_foot",
    "joint_foot_r": "R_foot",
}


class GlbError(ValueError):
    """The downloaded file is not a Teach-compatible binary glTF."""


def _normalise_joint_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_").replace(":", "_")


def rewrite_glb_joint_names(glb_bytes: bytes) -> bytes:
    """Rename remote skin joints and verify the complete Teach bone contract.

    Only the JSON chunk is rebuilt. BIN and extension chunks are copied
    byte-for-byte and in their original order.
    """
    if len(glb_bytes) < 12:
        raise GlbError("file is shorter than a GLB header")

    magic, version, declared_length = struct.unpack_from("<4sII", glb_bytes)
    if magic != b"glTF" or version != 2:
        raise GlbError("file is not a version 2 binary glTF")
    if declared_length != len(glb_bytes):
        raise GlbError(
            f"GLB length header says {declared_length}, got {len(glb_bytes)} bytes")

    chunks: list[tuple[bytes, bytes]] = []
    json_index: int | None = None
    offset = 12
    while offset < len(glb_bytes):
        if offset + 8 > len(glb_bytes):
            raise GlbError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<I4s", glb_bytes, offset)
        start = offset + 8
        end = start + chunk_length
        if end > len(glb_bytes):
            raise GlbError("truncated GLB chunk data")
        if chunk_type == _JSON_CHUNK:
            if json_index is not None:
                raise GlbError("GLB contains more than one JSON chunk")
            json_index = len(chunks)
        chunks.append((chunk_type, glb_bytes[start:end]))
        offset = end

    if offset != len(glb_bytes) or json_index is None:
        raise GlbError("GLB has no valid JSON chunk")

    try:
        document = json.loads(
            chunks[json_index][1].rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbError(f"invalid GLB JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise GlbError("GLB JSON root is not an object")

    nodes = document.get("nodes")
    skins = document.get("skins")
    if not isinstance(nodes, list) or not isinstance(skins, list) or not skins:
        raise GlbError("GLB does not contain a skinned skeleton")

    joint_indices: list[int] = []
    for skin_number, skin in enumerate(skins):
        joints = skin.get("joints") if isinstance(skin, dict) else None
        if not isinstance(joints, list):
            raise GlbError(f"skin {skin_number} has no joint list")
        for node_index in joints:
            if (not isinstance(node_index, int) or isinstance(node_index, bool)
                    or not 0 <= node_index < len(nodes)):
                raise GlbError(
                    f"skin {skin_number} references invalid joint {node_index!r}")
            joint_indices.append(node_index)

    for node_index in set(joint_indices):
        node = nodes[node_index]
        if not isinstance(node, dict):
            raise GlbError(f"joint node {node_index} is not an object")
        name = node.get("name")
        if isinstance(name, str):
            replacement = _REMOTE_JOINT_ALIASES.get(
                _normalise_joint_name(name))
            if replacement:
                node["name"] = replacement

    contract_nodes: dict[str, list[int]] = {bone: [] for bone in BONES}
    for node_index in joint_indices:
        name = nodes[node_index].get("name")
        if name in contract_nodes:
            contract_nodes[name].append(node_index)

    duplicates = {
        bone: indices for bone, indices in contract_nodes.items()
        if len(indices) > 1
    }
    missing = [bone for bone, indices in contract_nodes.items() if not indices]
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise GlbError(f"skin has duplicate contract bones: {names}")
    if missing:
        raise GlbError(f"skin is missing contract bones: {', '.join(missing)}")

    json_payload = json.dumps(
        document, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * (-len(json_payload) % 4)
    chunks[json_index] = (_JSON_CHUNK, json_payload)

    body = bytearray()
    for chunk_type, payload in chunks:
        body.extend(struct.pack("<I4s", len(payload), chunk_type))
        body.extend(payload)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + bytes(body)


class RealRigger(Rigger):
    def __init__(
        self,
        service_url: str | None = None,
        timeout: float | None = None,
        poll_interval: float | None = None,
        *,
        opener: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.service_url = (
            config.RIGGING_SERVICE_URL if service_url is None else service_url
        ).strip().rstrip("/")
        self.timeout = (
            config.RIGGING_SERVICE_TIMEOUT if timeout is None else float(timeout)
        )
        self.poll_interval = (
            config.RIGGING_POLL_INTERVAL
            if poll_interval is None else float(poll_interval)
        )
        self._open = opener or urlopen
        self._clock = clock
        self._sleep = sleep

    def rig(self, image_bytes: bytes, mime: str, progress: Progress) -> Rig:
        if not image_bytes:
            raise ProviderError(
                "I didn't get a drawing. Try sketching something first!",
                detail="empty image payload")
        if not self.service_url:
            raise ProviderError(
                "The avatar builder isn't ready right now. Ask a grown-up to check it.",
                detail="RIGGING_SERVICE_URL is not configured")
        if self.timeout <= 0:
            raise ProviderError(
                "Your avatar took too long to build. Try again?",
                detail=f"invalid rigging timeout: {self.timeout}")

        deadline = self._clock() + self.timeout

        progress(0.03, "Looking at your drawing...")
        classification = self._request_json(
            "POST", "/classify", deadline,
            raw_body=image_bytes, content_type=mime)
        label = self._classification_label(classification)
        if label not in _HUMANOID_LABELS:
            raise ProviderError(
                "I can only wake up drawings of people right now. Try drawing "
                "a person with a head, two arms, and two legs!",
                detail=f"non-humanoid classification: {label!r}")

        progress(0.14, "Turning your drawing into a 3D shape...")
        classify_id = self._find_value(classification, "classify_id")
        if not isinstance(classify_id, (str, int)) or isinstance(classify_id, bool):
            raise ProviderError(
                "I couldn't understand the avatar builder. Try again?",
                detail=f"classify response had no classify_id: {classification!r}")
        encoded_id = quote(str(classify_id), safe="")
        classify_query = f"classify_id={encoded_id}"

        progress(0.10, "Checking for an avatar we already built...")
        # This service currently raises 500 when the optional cached file does
        # not exist. Continue through the normal build path for either response.
        cached = self._request_bytes(
            "GET", f"/results/{encoded_id}/{encoded_id}_rigged.glb",
            deadline, _MAX_GLB_BYTES, missing_statuses=(404, 500))
        if cached is not None:
            progress(0.94, "Bringing your avatar home...")
            rig = self._make_rig(cached)
            progress(0.99, "Your avatar is ready!")
            return rig

        mesh_task = self._request_json(
            "POST", f"/mesh?{classify_query}", deadline)
        mesh_url = self._await_url(
            "mesh", mesh_task, deadline, progress,
            progress_range=(0.16, 0.50),
            url_keys=(
                "mesh_url", "model_url", "glb_url", "download_url",
                "result_url", "output_url", "file_url", "url",
            ),
            message="Building the 3D shape...")

        progress(0.55, "Finding the head, arms and legs...")
        joint_response = self._request_json(
            "POST", f"/infer_joints?{classify_query}", deadline)
        joints = self._find_value(
            joint_response, "joint_hints", "joints", "joint_positions",
            "joint_data", "skeleton")
        if joints is None:
            result = joint_response.get("result")
            if isinstance(result, (dict, list, str)):
                joints = result
        if joints is None:
            raise ProviderError(
                "I couldn't find all the joints in that avatar. Try a clearer drawing?",
                detail=f"infer_joints response had no joints: {joint_response!r}")

        progress(0.68, "Connecting the joints...")
        rig_task = self._request_json(
            "POST", f"/rig?{classify_query}", deadline)
        glb_url = self._await_url(
            "rig", rig_task, deadline, progress,
            progress_range=(0.70, 0.90),
            url_keys=(
                "rigged_url", "rigged_glb_url", "rigged_model_url", "rigged_mesh_url",
                "glb_url", "download_url", "result_url", "output_url",
                "file_url", "url",
            ),
            message="Connecting the skeleton...")

        progress(0.94, "Bringing your avatar home...")
        downloaded = self._request_bytes("GET", glb_url, deadline, _MAX_GLB_BYTES)
        rig = self._make_rig(downloaded)
        progress(0.99, "Your avatar is ready!")
        return rig

    @staticmethod
    def _make_rig(downloaded: bytes) -> Rig:
        try:
            glb_bytes = rewrite_glb_joint_names(downloaded)
        except GlbError as exc:
            raise ProviderError(
                "The avatar came back muddled. Try that drawing again?",
                detail=f"invalid rigged GLB: {exc}") from exc

        return Rig(
            format="glb",
            skeleton=list(BONES),
            glb_bytes=glb_bytes,
            notes="I found a person and built a movable 3D avatar.",
        )

    def _classification_label(self, response: dict[str, Any]) -> str:
        humanoid = self._find_value(response, "is_humanoid", "humanoid")
        if isinstance(humanoid, bool):
            return "humanoid" if humanoid else "not_humanoid"
        value = self._find_value(
            response, "classification", "class", "category", "label", "object_type")
        if value is None and isinstance(response.get("result"), str):
            value = response["result"]
        if not isinstance(value, str):
            raise ProviderError(
                "I couldn't understand that drawing. Try one with clearer lines?",
                detail=f"classify response had no label: {response!r}")
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    def _await_url(
        self,
        kind: str,
        response: dict[str, Any],
        deadline: float,
        progress: Progress,
        *,
        progress_range: tuple[float, float],
        url_keys: tuple[str, ...],
        message: str,
    ) -> str:
        task = response
        poll_number = 0
        active_poll_url: str | None = None
        while True:
            url = self._find_value(task, *url_keys)
            if url is None and isinstance(task.get("result"), str):
                url = task["result"]
            if isinstance(url, str) and url.strip():
                return self._service_url(url)

            status_value = self._find_value(task, "status", "state")
            status = (
                str(status_value).strip().lower().replace("-", "_").replace(" ", "_")
                if status_value is not None else ""
            )
            if status in _FAILED_STATUSES:
                reason = self._find_value(
                    task, "error", "detail", "message", "reason")
                raise ProviderError(
                    "I couldn't finish building that avatar. Try the drawing again?",
                    detail=f"{kind} task failed ({status}): {reason!r}")
            if status in _SUCCESS_STATUSES:
                raise ProviderError(
                    "The avatar builder sent back an incomplete result. Try again?",
                    detail=f"{kind} task completed without a download URL: {task!r}")
            if status and status not in _PENDING_STATUSES:
                raise ProviderError(
                    "I couldn't understand the avatar builder. Try again?",
                    detail=f"{kind} task returned unknown status {status!r}: {task!r}")

            poll_url = self._find_value(
                task, "status_url", "poll_url", "task_url", "status_endpoint")
            task_id = self._find_value(
                task, "task_id", "job_id", "request_id", "id")
            if not isinstance(poll_url, str) or not poll_url.strip():
                if isinstance(task_id, (str, int)) and not isinstance(task_id, bool):
                    poll_url = f"/{kind}/status/{quote(str(task_id), safe='')}"
                elif active_poll_url:
                    poll_url = active_poll_url
                else:
                    raise ProviderError(
                        "The avatar builder sent back an incomplete result. Try again?",
                        detail=f"{kind} response had no URL or task id: {task!r}")
            active_poll_url = poll_url

            self._sleep_for_poll(deadline)
            poll_number += 1
            start, end = progress_range
            remote_progress = self._find_value(task, "progress", "percent")
            if isinstance(remote_progress, (int, float)):
                fraction = start + (end - start) * max(
                    0.0, min(100.0, float(remote_progress))) / 100
            else:
                fraction = min(
                    end, start + (end - start) * (1 - 0.72 ** poll_number))
            progress(fraction, message)
            task = self._request_json("GET", active_poll_url, deadline)

    def _sleep_for_poll(self, deadline: float) -> None:
        remaining = self._remaining(deadline)
        delay = min(max(0.0, self.poll_interval), remaining)
        if delay:
            self._sleep(delay)
        self._remaining(deadline)

    def _request_json(
        self,
        method: str,
        path_or_url: str,
        deadline: float,
        *,
        json_body: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        body = None
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif raw_body is not None:
            body = raw_body
            headers["Content-Type"] = content_type or "application/octet-stream"

        raw = self._request_bytes(
            method, path_or_url, deadline, _MAX_JSON_BYTES,
            body=body, headers=headers)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "I couldn't understand the avatar builder. Try again?",
                detail=f"{method} {path_or_url} returned malformed JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ProviderError(
                "I couldn't understand the avatar builder. Try again?",
                detail=f"{method} {path_or_url} returned {type(data).__name__}, not object")
        return data

    def _request_bytes(
        self,
        method: str,
        path_or_url: str,
        deadline: float,
        limit: int,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        missing_statuses: tuple[int, ...] = (),
    ) -> bytes | None:
        request_headers = {
            "User-Agent": "CodeAI-Avatar-Rigger/1.0",
            "ngrok-skip-browser-warning": "true",
            **(headers or {}),
        }
        url = self._service_url(path_or_url)
        request = Request(
            url, data=body, headers=request_headers, method=method.upper())
        try:
            with self._open(request, timeout=self._remaining(deadline)) as response:
                data = response.read(limit + 1)
        except ProviderError:
            raise
        except HTTPError as exc:
            if exc.code in missing_statuses:
                return None
            try:
                detail = exc.read(4096).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 - preserve the original HTTP error
                detail = ""
            raise ProviderError(
                "The avatar builder couldn't finish that request. Try again?",
                detail=f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                "I couldn't reach the avatar builder. Try again?",
                detail=f"{method} {url} failed: {type(exc).__name__}: {exc}") from exc

        if len(data) > limit:
            raise ProviderError(
                "The avatar builder sent back a file that was too big. Try again?",
                detail=f"{method} {url} exceeded {limit} bytes")
        return data

    def _service_url(self, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        base = urlsplit(self.service_url)
        if parsed.scheme == "http":
            # The remote worker often reports its internal HTTP origin. Route
            # the path back through the configured public HTTPS tunnel.
            return urlunsplit((
                base.scheme, base.netloc, parsed.path, parsed.query, parsed.fragment))
        if parsed.scheme:
            return value
        return urljoin(f"{self.service_url}/", value.lstrip("/"))

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise ProviderError(
                "Your avatar took too long to build. Try again?",
                detail=f"rigging service exceeded {self.timeout:.3f}s deadline")
        return remaining

    @staticmethod
    def _find_value(data: Any, *keys: str) -> Any:
        if not isinstance(data, dict):
            return None
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        for value in data.values():
            if isinstance(value, dict):
                found = RealRigger._find_value(value, *keys)
                if found is not None:
                    return found
        return None
