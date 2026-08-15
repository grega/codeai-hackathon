"""Read a GLB, write a posed one back.

Two jobs:

  1. GLB container surgery — parse the header/chunks, edit the JSON, repack.
     Deliberately hand-rolled rather than pulling in pygltflib or trimesh: we
     touch node rotations and `extras` only, never geometry, so the whole thing
     is a few dozen lines and adds nothing to the slug.

  2. The same world-space retarget the browser does, in Python. A contract pose
     is a local rotation per bone in OUR frame, where every bone binds at
     identity; a GLB bone's local frame is whatever its exporter produced. See
     `pose_glb` for the maths, and static/js/viewport.js #applyPoseGlb for the
     browser twin.

WARNING: that twin is a genuine duplication. If you change the retarget here,
change it there — tests/test_export.py pins the shapes but cannot see the
browser. The long-term fix is to compute poses once, server-side, and have the
viewport apply what it is given.
"""

from __future__ import annotations

import json
import struct
from typing import Any

from rewards import quat_mul
from schemas import BONE_TREE, IDENTITY_QUAT, MIXAMO_BONE_MAP

_MAGIC = 0x46546C67          # 'glTF'
_CHUNK_JSON = 0x4E4F534A     # 'JSON'
_CHUNK_BIN = 0x004E4942      # 'BIN\0'


class GlbError(ValueError):
    """The bytes we were handed are not a GLB we can work with."""


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Split a GLB into its JSON document and binary buffer."""
    if len(data) < 12:
        raise GlbError("too short to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _MAGIC:
        raise GlbError("not a GLB (bad magic — a .gltf JSON file needs "
                       "no unpacking, but this code path expects binary)")
    if version != 2:
        raise GlbError(f"unsupported glTF version {version}")

    document: dict[str, Any] | None = None
    binary = b""
    offset = 12
    # Trust the header length over len(data): some exporters append junk.
    end = min(length, len(data))
    while offset + 8 <= end:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        start = offset + 8
        chunk = data[start:start + chunk_len]
        if chunk_type == _CHUNK_JSON:
            document = json.loads(chunk.decode("utf-8"))
        elif chunk_type == _CHUNK_BIN:
            binary = chunk
        offset = start + chunk_len + (-chunk_len % 4)   # chunks are 4-aligned

    if document is None:
        raise GlbError("GLB has no JSON chunk")
    return document, binary


def write_glb(document: dict[str, Any], binary: bytes) -> bytes:
    """Repack a JSON document and buffer into GLB bytes."""
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)         # JSON pads with spaces
    binary = binary + b"\x00" * (-len(binary) % 4)      # BIN pads with zeros

    total = 12 + 8 + len(json_bytes) + (8 + len(binary) if binary else 0)
    out = bytearray()
    out += struct.pack("<III", _MAGIC, 2, total)
    out += struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes
    if binary:
        out += struct.pack("<II", len(binary), _CHUNK_BIN) + binary
    return bytes(out)


# --------------------------------------------------------------------------
# Bone resolution — mirrors normaliseBoneName in static/js/viewport.js
# --------------------------------------------------------------------------

def normalise_bone_name(name: str) -> str:
    """Reduce a bone name to something comparable across exporters.

    A bone authored as "mixamorig:LeftArm" has been seen as `mixamorigLeftArm`
    (three's sanitizeNodeName), `mixamorig_LeftArm` (FBX conversion) and
    `mixamorig_LeftArm_011` (exporter index). Drop one trailing _<digits>, then
    every separator, then lowercase.

    The index is dropped in a SINGLE pass on purpose — stripping all trailing
    digits would fold "Spine_02" and "Spine1_03" onto one key.
    """
    if not name:
        return ""
    trimmed = name
    if "_" in trimmed:
        head, _, tail = trimmed.rpartition("_")
        if head and tail.isdigit():
            trimmed = head
    return "".join(c for c in trimmed if c.isalnum()).lower()


def resolve_bones(document: dict[str, Any]) -> dict[str, int]:
    """Map each contract bone to a glTF node index."""
    nodes = document.get("nodes", [])
    by_name: dict[str, int] = {}
    for index, node in enumerate(nodes):
        name = node.get("name")
        if not name:
            continue
        by_name.setdefault(name, index)
        by_name.setdefault(normalise_bone_name(name), index)

    resolved: dict[str, int] = {}
    for bone in BONE_TREE:
        for candidate in (bone, MIXAMO_BONE_MAP.get(bone)):
            if not candidate:
                continue
            hit = by_name.get(candidate)
            if hit is None:
                hit = by_name.get(normalise_bone_name(candidate))
            if hit is not None:
                resolved[bone] = hit
                break
    return resolved


# --------------------------------------------------------------------------
# Posing
# --------------------------------------------------------------------------

def _conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def _node_rotation(node: dict[str, Any]):
    r = node.get("rotation")
    return tuple(r) if r else IDENTITY_QUAT


def _parents(document: dict[str, Any]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for index, node in enumerate(document.get("nodes", [])):
        for child in node.get("children", []) or []:
            parents[child] = index
    return parents


def _bind_world_rotations(document: dict[str, Any]) -> dict[int, tuple]:
    """World rotation of every node in the file's current (bind) state."""
    nodes = document.get("nodes", [])
    parents = _parents(document)
    roots = [i for i in range(len(nodes)) if i not in parents]

    world: dict[int, tuple] = {}
    stack = [(r, IDENTITY_QUAT) for r in roots]
    while stack:
        index, parent_world = stack.pop()
        here = quat_mul(parent_world, _node_rotation(nodes[index]))
        world[index] = here
        for child in nodes[index].get("children", []) or []:
            stack.append((child, here))
    return world


def contract_world_deltas(pose: dict[str, list[float]]) -> dict[str, tuple]:
    """FK our canonical skeleton to a world rotation per contract bone.

    Our bind is identity everywhere, so a bone's world rotation IS its delta
    from bind — which is exactly what has to be applied to the GLB.
    """
    world: dict[str, tuple] = {}
    for bone, (parent, _offset) in BONE_TREE.items():
        local = tuple(pose.get(bone, IDENTITY_QUAT))
        world[bone] = quat_mul(world[parent], local) if parent else local
    return world


def pose_glb(document: dict[str, Any], pose: dict[str, list[float]],
             bones: dict[str, int] | None = None) -> dict[str, int]:
    """Write a contract pose into a glTF document's node rotations.

    Returns the bone -> node-index map that was used, so a caller can record
    which joints actually moved.

    Walks the node tree top-down carrying each node's POSED world rotation:
    a mapped bone gets `delta * bind_world` and its local is solved against its
    parent's posed world; an unmapped bone (fingers, Spine1/2) keeps its local
    rotation and simply inherits the movement of its parents.
    """
    nodes = document.get("nodes", [])
    if bones is None:
        bones = resolve_bones(document)

    bind_world = _bind_world_rotations(document)
    deltas = contract_world_deltas(pose)
    node_to_bone = {index: bone for bone, index in bones.items()}
    parents = _parents(document)
    roots = [i for i in range(len(nodes)) if i not in parents]

    stack = [(r, IDENTITY_QUAT) for r in roots]
    while stack:
        index, parent_world = stack.pop()
        bone = node_to_bone.get(index)

        if bone is not None:
            # World-space rotations pre-multiply: delta THEN bind orientation.
            desired = quat_mul(deltas[bone], bind_world[index])
            local = quat_mul(_conjugate(parent_world), desired)
            nodes[index]["rotation"] = [round(c, 6) for c in local]
            here = desired
        else:
            here = quat_mul(parent_world, _node_rotation(nodes[index]))

        for child in nodes[index].get("children", []) or []:
            stack.append((child, here))

    return bones


def attach_extras(document: dict[str, Any], payload: dict[str, Any]) -> None:
    """Embed our metadata under asset.extras, namespaced.

    Namespaced because `extras` is a shared free-for-all — another tool in the
    chain may well want to put its own keys there.
    """
    asset = document.setdefault("asset", {})
    extras = asset.setdefault("extras", {})
    extras["avatarTrainer"] = payload
