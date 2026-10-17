"""Package the sensor streams of a closed run for reconstruction; the run itself stays in place."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from itertools import pairwise
from pathlib import Path

from homebody.capture import Asset, Capture
from homebody.capture_mcap import REQUIRED_TOPICS, SENSOR_TOPICS, filtered, scan

REQUIRED_CHECKS = frozenset({"bag_inspection_binding", "metric_inputs", "bag_reopen",
                           "native_archive", "archive_hashes", "file_integrity"})
FIELDS = {
    "streams": "id kind status frame_id bindings payload_schema timestamp_status timestamp_fields sample_count first_stamp last_stamp",
    "clocks": "id domain unit origin synchronization segments",
    "frames": "id axes handedness length_unit status",
    "calibrations": "id kind status stream_ids bindings validity uncertainty",
    "transforms": "id source_frame target_frame status representation translation_m quaternion_xyzw bindings validity uncertainty",
    "alignments": "id kind status source_id target_id bindings method validity uncertainty",
}


def checked_file(evidence, row):
    if row.get("finalized") is not True or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
        raise ValueError(f"Capture file lacks finalized byte integrity: {row.get('path')}")
    path = evidence.resolve(Asset(row["path"], sha256=row["sha256"]), verify=True)
    local = evidence.root / row["path"]
    if any(parent.is_symlink() for parent in (local, *local.parents) if parent != evidence.root) or path.stat().st_size != row.get("size_bytes"):
        raise ValueError(f"Capture file size or link mismatch: {row['path']}")
    return path


def accepted(source):
    """Require the bound original check results, then verify current native file bytes."""
    evidence = Capture.open(source)
    manifest = json.loads(evidence.resolve(Asset("capture_manifest.json")).read_text())
    if manifest.get("collection_state") != "finalized":
        raise ValueError("Capture must be finalized")
    files = {row["id"]: row for row in manifest["files"]}
    quality = files.get(manifest.get("quality_report_file_id"))
    if quality is None:
        raise ValueError("Capture lacks its bound capture health report")
    health = json.loads(checked_file(evidence, quality).read_text())
    checks = {row["name"]: row for row in health.get("checks", [])}
    if len(checks) != len(health.get("checks", [])):
        raise ValueError("Duplicate capture health checks")
    missing = sorted(name for name in REQUIRED_CHECKS if checks.get(name, {}).get("outcome") != "pass")
    failed = sorted(name for name, row in checks.items() if row.get("outcome") == "fail")
    if (missing or failed or health.get("sensor_integrity") not in {None, "accepted"}
            or manifest.get("sensor_integrity") not in {None, "accepted"}):
        raise ValueError(f"Capture checks reject export: {sorted(set(missing + failed))}")
    session_paths = ("explore/session/",)
    for prefix in session_paths:
        if not ((evidence.root / prefix).exists() or
                any(row["path"].startswith(prefix) for row in files.values())):
            continue
        finalization = next((row for row in files.values()
                             if row["path"] == prefix + "finalization.json"), None)
        if finalization is None:
            raise ValueError("Session finalization is unknown")
        state = json.loads(checked_file(evidence, finalization).read_text())
        if state.get("status") != "closed" or state.get("failure") is not None:
            raise ValueError("Session finalization failed or remains open")
    for row in files.values():
        if row.get("role") in {"native_bag", "native_video"}:
            checked_file(evidence, row)
    return evidence, manifest, files, health


def package(source, destination):
    """Filter actual bag channels, copy native video bytes, validate, and atomically publish."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if (destination.exists() or destination.is_symlink() or
            destination.resolve().is_relative_to(source) or source.is_relative_to(destination.resolve())):
        raise ValueError("Destination must be a new directory separate from the capture")
    evidence, manifest, files, health = accepted(source)
    rows = {key: [{k: v for k, v in row.items() if k in fields.split()}
                  for row in manifest.get(key, [])] for key, fields in FIELDS.items()}
    bag_ids = {identity for identity, row in files.items() if row["path"].endswith(".mcap")}
    rows["streams"] = [row for row in rows["streams"] if row.get("bindings") and all(
        (binding.get("selector") in SENSOR_TOPICS if binding["file_id"] in bag_ids else
         files[binding["file_id"]].get("role") == "native_video") for binding in row["bindings"])]
    selected = {row["id"] for row in rows["streams"]}
    rows["transforms"] = [row for row in rows["transforms"] if "exploration_start" not in
                          (row["source_frame"], row["target_frame"])]
    rows["alignments"] = [row for row in rows["alignments"] if not row.get("bindings")]
    frames = {row["frame_id"] for row in rows["streams"] if row.get("frame_id")}
    for row in rows["transforms"]:
        frames.update((row["source_frame"], row["target_frame"]))
    rows["frames"] = [row for row in rows["frames"] if row["id"] in frames]
    rows["calibrations"] = [row for row in rows["calibrations"] if set(row.get("stream_ids", [])) & selected]
    for calibration in rows["calibrations"]:
        calibration["stream_ids"] = [name for name in calibration["stream_ids"] if name in selected]
        bindings = []
        for binding in calibration.get("bindings", []):
            file = files[binding["file_id"]]
            if file["path"] == "meta/phone_camera.json" and json.loads(
                    checked_file(evidence, file).read_text()).get("reference_profile_unverified"):
                continue
            bindings.append(binding)
        calibration["bindings"] = bindings
    references = {binding["file_id"] for key in ("streams", "calibrations", "transforms", "alignments")
                  for row in rows[key] for binding in row.get("bindings", [])}
    selected_bags = bag_ids & references
    if not selected_bags:
        raise ValueError("No native metric bag was selected")
    archive_path = source / "phone/native/archive.json"
    timing = {}
    if archive_path.is_file():
        archive_row = next((row for row in files.values() if row["path"] == "phone/native/archive.json"), None)
        if archive_row is None:
            raise ValueError("Native timing archive is not bound to the capture")
        archive = json.loads(checked_file(evidence, archive_row).read_text())
        timing = {row["file"]: row for row in archive.get("segments", [])}
    for clock in rows["clocks"]:
        file = files.get(clock["id"])
        if file is None or file["id"] not in references or file["path"] not in timing:
            continue
        segment = timing[file["path"]]
        anchor = {key: float(segment[key]) for key in ("pts_start_s", "pts_end_s", "first_decode_s")}
        if not all(math.isfinite(v) for v in anchor.values()) or anchor["pts_end_s"] < anchor["pts_start_s"]:
            raise ValueError("Invalid native video timing anchor")
        clock["origin"] = "Segment PTS rebased; segments gives the approximate receiver-clock anchor"
        clock["segments"] = [{"file_id": file["id"], **anchor,
                               "timestamp_kind": "approximate_receiver_clock_not_exposure_time"}]
    result = {"contract_version": "real-to-sim-capture/v1", "profile": "sensor-only",
              "collection_state": "finalized", "acquisition_profile": "sensor-only", **rows, "files": []}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".capture-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "capture"
        staging.mkdir()
        inspections = {}
        for identity in sorted(references):
            file = files[identity]
            original = checked_file(evidence, file)
            target = staging / file["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if identity in bag_ids:
                inspections[identity] = filtered(original, target)
            else:
                shutil.copyfile(original, target)
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if identity not in bag_ids and digest != file["sha256"]:
                raise ValueError("Native copy changed during export")
            checked_file(evidence, file)
            result["files"].append({"id": identity, "path": file["path"],
                "role": file.get("role", "metadata_or_observation"), "size_bytes": target.stat().st_size,
                "sha256": digest, "finalized": True, "source_sha256": file["sha256"]})
        actual_topics = {name for item in inspections.values() for name, row in item["topics"].items() if row["count"]}
        if missing := REQUIRED_TOPICS - actual_topics:
            raise ValueError(f"Actual MCAP channels lack required metric inputs: {sorted(missing)}")
        described_topics = {binding["selector"] for row in rows["streams"]
                            for binding in row["bindings"] if binding["file_id"] in inspections}
        if undescribed := actual_topics - described_topics:
            raise ValueError(f"Actual MCAP channels lack manifest streams: {sorted(undescribed)}")
        for row in rows["streams"]:
            count = 0
            for binding in row["bindings"]:
                if binding["file_id"] in inspections and binding["selector"] not in actual_topics:
                    raise ValueError(f"Stream has no actual MCAP channel: {row['id']}")
                if binding["file_id"] in inspections:
                    actual = inspections[binding["file_id"]]["topics"].get(binding["selector"])
                    if actual is None:
                        continue  # A topic can first appear after the first bag split.
                    count += actual["count"]
                    schema = row.get("payload_schema", {})
                    if not isinstance(schema, dict) or schema.get("ros_type") != actual["schema_name"]:
                        raise ValueError(f"Manifest and native schema disagree: {row['id']}")
                    if actual["header_reversals"]:
                        raise ValueError(f"Native header timestamps reverse: {row['id']}")
            if count and row.get("sample_count", count) != count:
                raise ValueError(f"Manifest and native sample count disagree: {row['id']}")
        for topic in REQUIRED_TOPICS:
            spans = sorted((item["topics"][topic] for item in inspections.values() if topic in item["topics"]),
                           key=lambda row: int(row["first_log_time_ns"]))
            for previous, current in pairwise(spans):
                if ("last_header_stamp_ns" in previous and
                        int(current["first_header_stamp_ns"]) < int(previous["last_header_stamp_ns"])):
                    raise ValueError(f"Native header timestamps reverse between splits: {topic}")
        if manifest.get("acquisition_profile") == "sensor-only":
            for identity in selected_bags:
                if set(scan(evidence.resolve(Asset(files[identity]["path"])))["actual_topics"]) - SENSOR_TOPICS:
                    raise ValueError("Sensor-only manifest omits MCAP channels present in the bag")
        # Keep source acceptance and limitations; the session's text records are not exported.
        report = {"sensor_integrity": "accepted", "collection_acceptance": "pending",
            "checks": [{"name": name, "outcome": "pass"} for name in sorted(REQUIRED_CHECKS)],
            "readiness": {key: {"status": value.get("status", "pending")} for key, value in
                          health.get("readiness", {}).items()},
            "limitations": ["Coverage, frame calibration and cross-sensor synchronization require review."],
            "mcap": inspections, "epoch_messages_preserved": True}
        if (any(row["path"].startswith("explore/session/") for row in files.values()) or
                any(row["name"] == "session_finalization" and row["outcome"] == "pass"
                    for row in health["checks"])):
            report["checks"].append({"name": "session_finalization", "outcome": "pass"})
        target = staging / "quality/capture_health.json"
        target.parent.mkdir()
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        result["files"].append({"id": "capture_health", "path": "quality/capture_health.json",
            "role": "capture_health", "size_bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "finalized": True})
        result["quality_report_file_id"] = "capture_health"
        result["sensor_integrity"] = "accepted"
        (staging / "capture_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        Capture.open(staging).context()
        accepted(staging)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Destination appeared during export")
        staging.rename(destination)
    return destination
