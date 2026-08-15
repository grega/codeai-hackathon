"""Read the joint hierarchy out of a skinned GLB and write rotation animations back into it."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

import quat
from pygltflib import (
    FLOAT,
    SCALAR,
    VEC4,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    BufferView,
    GLTF2,
)


@dataclass
class Joint:
    node_index: int
    name: str
    parent_index: int | None
    rest_rotation: np.ndarray  # xyzw, identity if the node has no explicit rotation
    children: list[int] = field(default_factory=list)


def load_skeleton(path: str) -> tuple[GLTF2, list[Joint]]:
    gltf = GLTF2().load(path)
    skin = gltf.skins[0]

    parent_of: dict[int, int] = {}
    for i, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parent_of[child] = i

    joints = []
    for j in skin.joints:
        node = gltf.nodes[j]
        rot = np.array(node.rotation, dtype=np.float64) if node.rotation else np.array([0.0, 0.0, 0.0, 1.0])
        joints.append(Joint(
            node_index=j,
            name=node.name or f"joint_{j}",
            parent_index=parent_of.get(j),
            rest_rotation=rot,
            children=list(node.children or []),
        ))
    return gltf, joints


_TRAILING_NUMBER = re.compile(r"_\d+$")


def find_joint(joints: list[Joint], *name_parts: str) -> Joint | None:
    """First joint whose name contains all of name_parts (case-insensitive).

    Falls back to matching with trailing "_NN" suffixes stripped. This rig's own
    naming is irregular (e.g. "LeftLeg_063" but "RightLeg_00", not "_064" as the
    left/right pattern elsewhere would suggest), which is exactly the kind of thing
    an LLM guessing a joint name gets plausibly wrong -- better to still resolve the
    joint than crash on a None.
    """
    for j in joints:
        lname = j.name.lower()
        if all(part.lower() in lname for part in name_parts):
            return j

    stripped_parts = [_TRAILING_NUMBER.sub("", part.lower()) for part in name_parts]
    for j in joints:
        lname = _TRAILING_NUMBER.sub("", j.name.lower())
        if all(part in lname for part in stripped_parts):
            return j
    return None


def _parent_map(gltf: GLTF2) -> dict[int, int]:
    parent_of: dict[int, int] = {}
    for i, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parent_of[child] = i
    return parent_of


def world_rotation(gltf: GLTF2, node_index: int | None) -> np.ndarray:
    """Accumulated world-space rotation of a node, walking up to the scene root.
    Rig bones (esp. auto-rigged/Mixamo-style ones) have local axes that point in
    arbitrary directions, so "rotate around local X" is not predictable -- this lets
    us reason in real world axes instead and convert down to local space afterwards."""
    if node_index is None:
        return quat.IDENTITY
    parent_of = _parent_map(gltf)
    node = gltf.nodes[node_index]
    local_r = np.array(node.rotation, dtype=np.float64) if node.rotation else quat.IDENTITY
    return quat.multiply(world_rotation(gltf, parent_of.get(node_index)), local_r)


def local_rotation_for_world_delta(
    gltf: GLTF2,
    joint: Joint,
    world_delta: np.ndarray,
) -> np.ndarray:
    """Compose `world_delta` (a rotation expressed in world/scene axes, e.g. "tip up
    around world Z") on top of a joint's rest orientation, converted into that joint's
    local rotation space -- the space glTF's `rotation` channel actually animates."""
    parent_world = world_rotation(gltf, joint.parent_index)
    inv_parent = quat.conjugate(parent_world)
    local_delta = quat.multiply(quat.multiply(inv_parent, world_delta), parent_world)
    return quat.multiply(local_delta, joint.rest_rotation)


def add_rotation_animation(
    gltf: GLTF2,
    tracks: dict[int, tuple[list[float], list[np.ndarray]]],
    name: str = "generated",
) -> GLTF2:
    """Append a new Animation to `gltf`. `tracks` maps node_index -> (times, quaternions[xyzw])."""
    blob = bytearray(gltf.binary_blob() or b"")

    def append(data: bytes) -> int:
        pad = (-len(blob)) % 4
        blob.extend(b"\x00" * pad)
        offset = len(blob)
        blob.extend(data)
        return offset

    def add_accessor(values: np.ndarray, type_: str) -> int:
        values = values.astype(np.float32)
        offset = append(values.tobytes())
        bv_index = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=values.nbytes))
        flat = values.reshape(len(values), -1)
        acc_index = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_index,
            componentType=FLOAT,
            count=len(values),
            type=type_,
            min=flat.min(axis=0).tolist(),
            max=flat.max(axis=0).tolist(),
        ))
        return acc_index

    channels: list[AnimationChannel] = []
    samplers: list[AnimationSampler] = []
    for node_index, (times, quats) in tracks.items():
        time_acc = add_accessor(np.array(times, dtype=np.float32), SCALAR)
        rot_acc = add_accessor(np.array(quats, dtype=np.float32), VEC4)
        sampler_index = len(samplers)
        samplers.append(AnimationSampler(input=time_acc, output=rot_acc, interpolation="LINEAR"))
        channels.append(AnimationChannel(
            sampler=sampler_index,
            target=AnimationChannelTarget(node=node_index, path="rotation"),
        ))

    if gltf.animations is None:
        gltf.animations = []
    gltf.animations.append(Animation(name=name, channels=channels, samplers=samplers))

    gltf.set_binary_blob(bytes(blob))
    gltf.buffers[0].byteLength = len(blob)
    return gltf


def save(gltf: GLTF2, path: str) -> None:
    gltf.save_binary(path)
