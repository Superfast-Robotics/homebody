"""The sensor export filters MCAP topics from synthetic ROS messages and leaves the source capture unchanged.

fixtures/capture_native.mcap contains only generated standard ROS 2 messages: two each on the raw
PointCloud2, Imu, registered PointCloud2, Odometry and SLAM epoch channels, plus excluded
/exploration/state and /lowcmd string canaries. All coordinates, timestamps and content are
synthetic. The export tests rewrite this closed MCAP and compare native schemas, channel metadata,
log/publish times, sequence numbers and payloads.
"""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

pytest.importorskip("mcap", reason="offline export requires the declared exploration extra")
from mcap.reader import make_reader
from mcap.writer import Writer
from test_build_scene import room as room  # noqa: PLC0414

from homebody.__main__ import main
from homebody.capture_mcap import REQUIRED_TOPICS, SENSOR_TOPICS, scan
from homebody.capture_package import REQUIRED_CHECKS, package
from homebody.real2sim import runner

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "capture_native.mcap"


def file_row(path, root, identity, role):
    return {"id": identity, "path": path.relative_to(root).as_posix(), "role": role,
            "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "finalized": True}


def write_manifest(root, value):
    (root / "capture_manifest.json").write_text(json.dumps(value))


def refresh_file(root, relative):
    manifest = json.loads((root / "capture_manifest.json").read_text())
    row = next(row for row in manifest["files"] if row["path"] == relative)
    row.update(file_row(root / relative, root, row["id"], row["role"]))
    write_manifest(root, manifest)


def records(path):
    with path.open("rb") as stream:
        return [(schema.name, schema.encoding, schema.data, channel.topic,
                 channel.message_encoding, channel.metadata, message.log_time,
                 message.publish_time, message.sequence, message.data)
                for schema, channel, message in make_reader(stream).iter_messages(log_time_order=False)]


def rewrite(path, *, omit=()):
    rows = records(path)
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="ros2")
        for name, encoding, schema, topic, message_encoding, metadata, log, publish, sequence, payload in rows:
            if topic in omit:
                continue
            schema_id = writer.register_schema(name, encoding, schema)
            channel = writer.register_channel(topic, message_encoding, schema_id, metadata)
            writer.add_message(channel, log, payload, publish, sequence)
        writer.finish()


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "capture"
    (root / "ws").mkdir(parents=True)
    (root / "quality").mkdir()
    (root / "phone/native").mkdir(parents=True)
    (root / "explore/session").mkdir(parents=True)
    shutil.copyfile(FIXTURE, root / "ws/ws_0.mcap")
    (root / "phone/native/video.mkv").write_bytes(b"synthetic native video copy witness")
    (root / "explore/session/provider.txt").write_text("NOT_IN_PACKAGE")
    (root / "explore/session/finalization.json").write_text(json.dumps({"status": "closed", "failure": None}))
    (root / "quality/capture_health.json").write_text(json.dumps({
        "checks": [{"name": name, "outcome": "pass"} for name in sorted(REQUIRED_CHECKS)],
        "collection_acceptance": "pending", "sensor_integrity": "accepted",
        "readiness": {"geometry": {"status": "pending", "gaps": ["NOT_IN_PACKAGE"]}},
    }))
    manifest = json.loads((ROOT / "configs/capture/capture_manifest.template.json").read_text())
    manifest.update(collection_state="finalized", acquisition_profile="full-local", streams=[],
                    quality_report_file_id="health")
    manifest["files"] = [file_row(root / path, root, identity, role) for path, identity, role in [
        ("ws/ws_0.mcap", "bag", "native_bag"),
        ("phone/native/video.mkv", "video", "native_video"),
        ("explore/session/finalization.json", "session_finalization", "metadata_or_observation"),
        ("quality/capture_health.json", "health", "capture_health")]]
    for topic, row in scan(root / "ws/ws_0.mcap")["topics"].items():
        manifest["streams"].append({"id": topic.strip("/").replace("/", "."),
            "kind": "events" if topic == "/homebody/slam_epoch" else "lidar", "status": "recorded",
            "frame_id": "ws_map", "timestamp_status": "known", "bindings": [{"file_id": "bag", "selector": topic}],
            "payload_schema": {"ros_type": row["schema_name"]}, "timestamp_fields": [{
                "name": "bag_receive_time", "clock_id": "host", "unit": "nanoseconds",
                "encoding": "exact integer", "meaning": "receipt time"}]})
    manifest["streams"].append({**manifest["streams"][0], "id": "phone", "kind": "rgb",
        "bindings": [{"file_id": "video", "selector": "video:0"}]})
    manifest["provenance"]["notes"] = "NOT_IN_PACKAGE"
    write_manifest(root, manifest)
    return root


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_export_keeps_only_sensor_streams(capture, tmp_path):
    before = snapshot(capture)
    result = package(capture, tmp_path / "portable")
    original = [row for row in records(capture / "ws/ws_0.mcap") if row[3] in SENSOR_TOPICS]
    assert records(result / "ws/ws_0.mcap") == original
    assert set(scan(result / "ws/ws_0.mcap")["actual_topics"]) == REQUIRED_TOPICS
    assert snapshot(capture) == before
    assert not any(b"NOT_IN_PACKAGE" in data for data in snapshot(result).values())
    manifest = json.loads((result / "capture_manifest.json").read_text())
    assert manifest["sensor_integrity"] == "accepted"
    assert (result / "phone/native/video.mkv").read_bytes() == before["phone/native/video.mkv"]
    jsonschema.validate(manifest, json.loads((ROOT / "configs/capture/capture_manifest.schema.json").read_text()))
    again = package(result, tmp_path / "portable-again")
    assert records(again / "ws/ws_0.mcap") == original


def test_export_cli_accepts_full_capture_without_starting_services(capture, tmp_path):
    assert main(["real2sim", "export", "--exploration-data", str(capture),
                 "--output", str(tmp_path / "portable")]) == 0
    assert (capture / "explore/session/provider.txt").exists()


@pytest.mark.parametrize("state", [None, {"status": "open"}, {"status": "failed"},
                                   {"status": "closed", "failure": "retained write failure"}])
def test_failed_or_unknown_session_finalization_rejects_export(capture, tmp_path, state):
    relative = "explore/session/finalization.json"
    if state is None:
        (capture / relative).unlink()
        manifest = json.loads((capture / "capture_manifest.json").read_text())
        manifest["files"] = [row for row in manifest["files"] if row["path"] != relative]
        write_manifest(capture, manifest)
    else:
        (capture / relative).write_text(json.dumps(state))
        refresh_file(capture, relative)
    before = snapshot(capture)
    with pytest.raises(ValueError, match="Session finalization"):
        package(capture, tmp_path / "portable")
    assert snapshot(capture) == before
    assert not (tmp_path / "portable").exists()


def test_interrupted_task_is_consistent_with_closed_session(capture, tmp_path):
    task = capture / "explore/session/outcome.json"
    task.write_text(json.dumps({"status": "interrupted"}))
    result = package(capture, tmp_path / "portable")
    checks = json.loads((result / "quality/capture_health.json").read_text())["checks"]
    assert {"name": "session_finalization", "outcome": "pass"} in checks
    assert task.exists()


def test_capture_without_session_remains_readable(capture, tmp_path):
    shutil.rmtree(capture / "explore/session")
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    manifest["files"] = [row for row in manifest["files"] if not row["path"].startswith("explore/session/")]
    write_manifest(capture, manifest)
    assert package(capture, tmp_path / "portable").is_dir()


@pytest.mark.parametrize("failure", ["metric_inputs", "bag_reopen", "agent_context"])
def test_failed_original_check_rejects_publication(capture, tmp_path, failure):
    path = capture / "quality/capture_health.json"
    health = json.loads(path.read_text())
    health["checks"] = [row for row in health["checks"] if row["name"] != failure]
    health["checks"].append({"name": failure, "outcome": "fail"})
    path.write_text(json.dumps(health))
    refresh_file(capture, "quality/capture_health.json")
    before = snapshot(capture)
    with pytest.raises(ValueError, match="checks reject export"):
        package(capture, tmp_path / "portable")
    assert not (tmp_path / "portable").exists()
    assert snapshot(capture) == before


def test_actual_channel_inventory_overrules_a_passing_report(capture, tmp_path):
    rewrite(capture / "ws/ws_0.mcap", omit={"/ws/registered_scan_aligned"})
    refresh_file(capture, "ws/ws_0.mcap")
    with pytest.raises(ValueError, match="actual|Actual"):
        package(capture, tmp_path / "portable")
    assert not (tmp_path / "portable").exists()


def test_profile_flag_cannot_hide_model_or_control_topics(capture, tmp_path):
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    manifest["acquisition_profile"] = "sensor-only"
    write_manifest(capture, manifest)
    with pytest.raises(ValueError, match="omits MCAP channels"):
        package(capture, tmp_path / "portable")
    assert not (tmp_path / "portable").exists()


def test_actual_sensor_channels_require_manifest_streams(capture, tmp_path):
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    manifest["streams"] = [row for row in manifest["streams"] if all(
        binding["selector"] != "/homebody/slam_epoch" for binding in row["bindings"])]
    write_manifest(capture, manifest)
    before = snapshot(capture)
    with pytest.raises(ValueError, match="lack manifest streams"):
        package(capture, tmp_path / "portable")
    assert snapshot(capture) == before
    assert not (tmp_path / "portable").exists()


@pytest.mark.parametrize("relative", ["ws/ws_0.mcap", "phone/native/video.mkv", "quality/capture_health.json"])
def test_tampered_bound_bytes_reject_export(capture, tmp_path, relative):
    with (capture / relative).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        package(capture, tmp_path / "portable")


def test_output_reread_rejects_changed_native_timestamps(capture, tmp_path, monkeypatch):
    original = Writer.add_message

    def changed(self, channel_id, log_time, data, publish_time, sequence=0):
        return original(self, channel_id, log_time + 1, data, publish_time, sequence)

    monkeypatch.setattr(Writer, "add_message", changed)
    before = snapshot(capture)
    with pytest.raises(ValueError, match="changed native"):
        package(capture, tmp_path / "portable")
    assert snapshot(capture) == before
    assert not (tmp_path / "portable").exists()
    assert not list(tmp_path.glob(".capture-*"))


@pytest.mark.parametrize("field,value", [("payload_schema", {"ros_type": "wrong/msg/Schema"}),
                                        ("sample_count", 999)])
def test_manifest_must_match_actual_schema_and_counts(capture, tmp_path, field, value):
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    manifest["streams"][0][field] = value
    write_manifest(capture, manifest)
    with pytest.raises(ValueError, match="disagree"):
        package(capture, tmp_path / "portable")
    assert not (tmp_path / "portable").exists()


def test_unclosed_mcap_cannot_be_exported_even_with_matching_hash(capture, tmp_path):
    from mcap.exceptions import McapError

    path = capture / "ws/ws_0.mcap"
    path.write_bytes(path.read_bytes()[:-30])
    refresh_file(capture, "ws/ws_0.mcap")
    before = snapshot(capture)
    with pytest.raises((McapError, ValueError, EOFError)):
        package(capture, tmp_path / "portable")
    assert snapshot(capture) == before
    assert not (tmp_path / "portable").exists()


@pytest.mark.parametrize("portable", [False, True])
def test_reconstruction_accepts_full_or_validated_sensor_package(
        capture, room, tmp_path, monkeypatch, portable):
    if portable:
        capture = package(capture, tmp_path / "portable")
    task = json.loads((room / "task_scene.json").read_text())
    task.update(frame="room", robot_start=[0, 0, 0])
    (room / "task_scene.json").write_text(json.dumps(task))
    semantics = json.loads((room / "semantics.json").read_text())
    semantics["coordinate_frame"].update(units="metres", up_axis="Z", handedness="right")
    (room / "semantics.json").write_text(json.dumps(semantics))
    calls = []

    def agent(command, **kwargs):
        calls.append(command)
        work = Path(kwargs["cwd"])
        shutil.copytree(room, work / "scene", dirs_exist_ok=True)
        (work / "scene/work.log").write_text("NOT_IN_PACKAGE")
        (work / "audit.json").write_text(json.dumps({"ready": True, "checks": [], "limitations": []}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", agent)
    before = snapshot(capture)
    work, output = tmp_path / "work", tmp_path / "scene"
    assert main(["real2sim", "reconstruct", "--exploration-data", str(capture), "--output", str(output),
                 "--task", "Build the synthetic room", "--root", str(ROOT), "--work", str(work),
                 "--agent-command", "synthetic-agent"]) == 0
    assert calls == [["synthetic-agent"]]
    assert snapshot(capture) == before
    assert (work / "audit.json").exists() and (work / "request.txt").exists()
    assert not any(b"NOT_IN_PACKAGE" in data for data in snapshot(output).values())
    assert {path.name for path in output.iterdir()} == {
        "scene.xml", "semantics.json", "task_scene.json", "physics.json", "map.npz", "meshes"}


def test_reconstruction_rejects_failed_capture_before_provider_or_work(capture, tmp_path, monkeypatch):
    path = capture / "quality/capture_health.json"
    health = json.loads(path.read_text())
    health["sensor_integrity"] = "rejected"
    path.write_text(json.dumps(health))
    refresh_file(capture, "quality/capture_health.json")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("No provider call"))
    with pytest.raises(ValueError, match="reject"):
        runner.reconstruct(capture, tmp_path / "output", "room", root=ROOT, work=tmp_path / "work")
    assert not (tmp_path / "work").exists()
