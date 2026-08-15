"""Training output: a posed GLB plus the document that feeds the next stage.

The geometric proof that the pose is correct lives in the browser (the exported
GLB is loaded and joint positions are measured). These guard the parts that can
be checked without a renderer: the container round-trips, the rotations really
change, the metadata is present and self-consistent, and the refusals are the
ones a caller can act on.
"""

from __future__ import annotations

import json
import threading

import pytest

import export
import gltf
import providers
from rewards import quat_from_euler
from schemas import BONES, REST_POSE, TrainConfig, validate_clip, validate_rig
from store import store
from training import TrainingRun

FIXTURE = "tests/fixtures/mixamo-style.glb"


def noop(fraction: float, message: str = "") -> None:
    pass


@pytest.fixture(scope="module")
def glb_bytes():
    with open(FIXTURE, "rb") as handle:
        return handle.read()


@pytest.fixture
def finished_run(monkeypatch, glb_bytes):
    """A completed run against a GLB-rigged avatar."""
    import config
    monkeypatch.setattr(config, "MOCK_RIG_GLB", FIXTURE)

    rig = validate_rig(providers.get_rigger().rig(b"x", "image/png", noop))
    avatar = store.add_avatar(rig, image_bytes=b"x")
    clip = store.add_clip(validate_clip(
        providers.get_poser().pose("waves arms in the air", rig, noop)))

    run = TrainingRun(avatar.id, clip, TrainConfig(episodes=40, seed=3))
    store.add_run(run)
    for episode in providers.get_trainer().train(
            rig, clip, run.cfg, threading.Event()):
        run._history.append(episode)
        run.best_reward = max(run.best_reward, episode.best_reward)
    return run, avatar, clip


class TestGlbContainer:
    def test_round_trip_preserves_the_document_and_buffer(self, glb_bytes):
        document, binary = gltf.read_glb(glb_bytes)
        again, again_binary = gltf.read_glb(gltf.write_glb(document, binary))
        assert again["nodes"] == document["nodes"]
        assert again_binary == binary

    def test_rejects_non_glb(self):
        with pytest.raises(gltf.GlbError, match="magic"):
            gltf.read_glb(b'{"asset":{"version":"2.0"}}')

    def test_rejects_truncated(self):
        with pytest.raises(gltf.GlbError):
            gltf.read_glb(b"glTF")


class TestBoneResolution:
    def test_resolves_every_contract_bone(self, glb_bytes):
        document, _ = gltf.read_glb(glb_bytes)
        assert set(gltf.resolve_bones(document)) == set(BONES)

    @pytest.mark.parametrize("mangled", [
        "mixamorig:LeftArm",       # as authored
        "mixamorigLeftArm",        # three's sanitizeNodeName
        "mixamorig_LeftArm",       # FBX conversion
        "mixamorig_LeftArm_011",   # ...plus an exporter index
    ])
    def test_normalisation_folds_every_seen_mangling(self, mangled):
        assert gltf.normalise_bone_name(mangled) == "mixamorigleftarm"

    def test_spine_and_spine1_do_not_collide(self):
        """Stripping ALL trailing digits would fold these together and the
        wrong bone would win."""
        assert (gltf.normalise_bone_name("mixamorig_Spine_02")
                != gltf.normalise_bone_name("mixamorig_Spine1_03"))

    def test_matches_the_browser_normaliser(self):
        """Same inputs, same outputs as normaliseBoneName in viewport.js."""
        for name, expected in [
            ("mixamorig:Hips", "mixamorighips"),
            ("mixamorig_RightUpLeg_040", "mixamorigrightupleg"),
            ("L_shoulder", "lshoulder"),
            ("", ""),
        ]:
            assert gltf.normalise_bone_name(name) == expected


class TestPosing:
    def test_posing_changes_node_rotations(self, glb_bytes):
        document, _ = gltf.read_glb(glb_bytes)
        bones = gltf.resolve_bones(document)
        before = [document["nodes"][bones["L_shoulder"]].get("rotation")]

        pose = dict(REST_POSE)
        pose["L_shoulder"] = list(quat_from_euler(0, 0, -90))
        gltf.pose_glb(document, pose, bones)

        after = document["nodes"][bones["L_shoulder"]].get("rotation")
        assert after != before[0]
        assert len(after) == 4

    def test_rest_pose_leaves_the_rig_at_bind(self, glb_bytes):
        """A pose of all-identity must be a no-op, or 'start' is meaningless.

        Compares EFFECTIVE local rotations, not the raw `rotation` field: on a
        matrix-form node an absent `rotation` means "read the matrix", not
        identity, so comparing the field would pass a rig that had been
        silently flattened to identity.
        """
        document, _ = gltf.read_glb(glb_bytes)
        bones = gltf.resolve_bones(document)
        before = {i: gltf._node_rotation(document["nodes"][i])
                  for i in bones.values()}

        gltf.pose_glb(document, dict(REST_POSE), bones)

        for index, original in before.items():
            now = gltf._node_rotation(document["nodes"][index])
            # q and -q are the same rotation; compare on the shorter arc.
            sign = -1.0 if sum(a * b for a, b in zip(now, original)) < 0 else 1.0
            for a, b in zip(now, original):
                assert a * sign == pytest.approx(b, abs=1e-5)

    def test_unmapped_nodes_keep_their_local_rotation(self, glb_bytes):
        """Fingers and Spine1/2 ride along with their parents rather than
        being reset."""
        document, _ = gltf.read_glb(glb_bytes)
        bones = gltf.resolve_bones(document)
        mapped = set(bones.values())
        unmapped = [i for i in range(len(document["nodes"])) if i not in mapped]
        before = {i: document["nodes"][i].get("rotation") for i in unmapped}

        pose = dict(REST_POSE)
        pose["spine"] = list(quat_from_euler(20, 0, 0))
        gltf.pose_glb(document, pose, bones)

        for index in unmapped:
            assert document["nodes"][index].get("rotation") == before[index]


class TestMatrixFormNodes:
    """glTF nodes may carry a 4x4 `matrix` INSTEAD of translation/rotation/scale,
    and when they do the TRS fields are ignored entirely.

    three's GLTFExporter writes `matrix`; Meshy's FBX pipeline writes TRS. A
    posing routine that only writes `rotation` silently does nothing on the
    former: the JSON gains a rotation nobody reads and the model renders at
    bind. Caught only by rendering the result, so it is pinned here.
    """

    def test_the_fixture_really_is_matrix_form(self, glb_bytes):
        """If this ever fails the fixture changed and the tests below stop
        covering the case they exist for."""
        document, _ = gltf.read_glb(glb_bytes)
        bones = gltf.resolve_bones(document)
        assert all("matrix" in document["nodes"][i] for i in bones.values())

    def test_posing_converts_matrix_nodes_to_trs(self, glb_bytes):
        document, _ = gltf.read_glb(glb_bytes)
        bones = gltf.resolve_bones(document)
        pose = dict(REST_POSE)
        pose["L_shoulder"] = list(quat_from_euler(0, 0, -90))
        gltf.pose_glb(document, pose, bones)

        node = document["nodes"][bones["L_shoulder"]]
        assert "matrix" not in node, "matrix left in place — rotation is ignored"
        assert "rotation" in node

    def test_decompose_round_trips_a_known_rotation(self):
        """A 90 degrees z rotation as a column-major matrix."""
        matrix = [0, 1, 0, 0,
                  -1, 0, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, 1]
        translation, rotation, scale = gltf._decompose(matrix)
        expected = quat_from_euler(0, 0, 90)
        for a, b in zip(rotation, expected):
            assert a == pytest.approx(b, abs=1e-6)
        assert translation == [0, 0, 0]
        for s in scale:
            assert s == pytest.approx(1.0, abs=1e-9)

    def test_decompose_keeps_translation_and_scale(self):
        matrix = [2, 0, 0, 0,
                  0, 3, 0, 0,
                  0, 0, 4, 0,
                  5, 6, 7, 1]
        translation, _rotation, scale = gltf._decompose(matrix)
        assert translation == [5, 6, 7]
        assert scale == pytest.approx([2, 3, 4])

    def test_conversion_preserves_the_bind_transform(self, glb_bytes):
        """Rest pose on a matrix rig must leave the joint where it was."""
        document, _ = gltf.read_glb(glb_bytes)
        bones = gltf.resolve_bones(document)
        index = bones["L_elbow"]
        before = gltf._decompose(document["nodes"][index]["matrix"])

        gltf.pose_glb(document, dict(REST_POSE), bones)
        node = document["nodes"][index]

        for a, b in zip(node["rotation"], before[1]):
            assert a == pytest.approx(b, abs=1e-5)
        for a, b in zip(node.get("translation", [0, 0, 0]), before[0]):
            assert a == pytest.approx(b, abs=1e-5)


class TestExportDocument:
    def test_start_is_the_t_pose(self, finished_run):
        run, avatar, clip = finished_run
        doc = export.build_document(run, avatar, clip)
        assert set(doc["poses"]["start"]) == set(BONES)
        assert all(q == [0.0, 0.0, 0.0, 1.0]
                   for q in doc["poses"]["start"].values())

    def test_end_differs_from_start(self, finished_run):
        run, avatar, clip = finished_run
        doc = export.build_document(run, avatar, clip)
        assert doc["poses"]["end"] != doc["poses"]["start"]

    def test_carries_the_vocabulary_needed_to_read_it(self, finished_run):
        """A consumer should need no other file to interpret the poses."""
        run, avatar, clip = finished_run
        doc = export.build_document(run, avatar, clip)
        assert doc["skeleton"]["bones"] == list(BONES)
        assert set(doc["skeleton"]["hierarchy"]) == set(BONES)
        assert "quaternion" in doc["skeleton"]["pose_format"]

    def test_carries_intent_and_provenance(self, finished_run):
        run, avatar, clip = finished_run
        doc = export.build_document(run, avatar, clip)
        assert doc["target"]["prompt"] == "waves arms in the air"
        assert doc["training"]["episodes_run"] > 0
        assert doc["run_id"] == run.id

    def test_is_small_enough_to_hand_to_a_model(self, finished_run):
        """The whole point of the sidecar: the GLB is megabytes, this is not."""
        run, avatar, clip = finished_run
        size = len(json.dumps(export.build_document(run, avatar, clip)))
        assert size < 20_000, f"{size} bytes is too big for a prompt"


class TestExportGlb:
    def test_produces_a_valid_glb_with_metadata(self, finished_run):
        run, avatar, clip = finished_run
        data, payload = export.build_glb(run, avatar, clip)
        assert data[:4] == b"glTF"

        document, _ = gltf.read_glb(data)
        embedded = document["asset"]["extras"]["avatarTrainer"]
        assert embedded["posed"] == "end"
        assert embedded["poses"]["end"] == payload["poses"]["end"]
        assert embedded["schema_version"] == export.SCHEMA_VERSION

    def test_node_rotations_hold_the_end_pose(self, finished_run, glb_bytes):
        """The file should render posed without anyone reading extras."""
        run, avatar, clip = finished_run
        data, _ = export.build_glb(run, avatar, clip)

        original, _ = gltf.read_glb(glb_bytes)
        posed, _ = gltf.read_glb(data)
        bones = gltf.resolve_bones(posed)
        changed = [b for b, i in bones.items()
                   if posed["nodes"][i].get("rotation")
                   != original["nodes"][i].get("rotation")]
        assert changed, "no node rotation changed — the GLB is still at bind"

    def test_records_which_joints_were_driven(self, finished_run):
        run, avatar, clip = finished_run
        _, payload = export.build_glb(run, avatar, clip)
        assert payload["node_map"]["L_shoulder"].startswith("mixamorig")
        assert payload["unmapped_bones"] == []

    def test_procedural_rig_is_refused_clearly(self, monkeypatch, finished_run):
        import config
        run, avatar, clip = finished_run
        monkeypatch.setattr(config, "MOCK_RIG_GLB", "")
        rig = validate_rig(providers.get_rigger().rig(b"x", "image/png", noop))
        plain = store.add_avatar(rig)
        with pytest.raises(export.ExportError) as exc:
            export.build_glb(run, plain, clip)
        assert exc.value.status == 409
        assert "stick figure" in exc.value.user_message

    def test_run_with_no_episodes_is_refused(self, finished_run):
        run, avatar, clip = finished_run
        empty = TrainingRun(avatar.id, clip, TrainConfig(episodes=10))
        with pytest.raises(export.ExportError) as exc:
            export.build_glb(empty, avatar, clip)
        assert exc.value.status == 409
