"""Bind closed native iPhone exploration captures to the real-to-sim v1 contract.

Offline metadata only; source messages are left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
CONFIG = REPO / "configs/capture"



def initialize(run):
    manifest = read(CONFIG / "capture_manifest.template.json")
    interval = read(run / "meta/record_interval.json")
    manifest.update(session_id=run.name, started_at_utc=interval['started_at_utc'],
                    collection_state='recording', acquisition_profile='full-local',
                    sensor_integrity='pending',
                    start_time_note='UTC recorder start; sensors have separate clocks.')
    for value in manifest['readiness'].values():
        if value['status'] != 'not_requested':
            value.update(status='pending', gaps=['Recording not finalized or inspected.'])
    shutil.copyfile(CONFIG / 'capture_manifest.schema.json', run / 'capture_manifest.schema.json')
    source = run / 'meta/recorder_source'
    source.mkdir(exist_ok=True)
    shutil.copyfile(CONFIG / 'capture_manifest.template.json', source / 'capture_profile.json')
    execution_source = Path(__file__).resolve().parents[1]
    for path in sorted(execution_source.rglob('*.py')):
        target = source / path.relative_to(execution_source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    # also copy the homebody package source
    package_source = execution_source.parents[1]
    for path in sorted(package_source.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.md', '.sh', '.yaml', '.txt'}:
            target = source / 'homebody' / path.relative_to(package_source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write(run / 'capture_manifest.json', manifest)


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_id(path):
    return "f_" + hashlib.sha256(path.encode()).hexdigest()[:20]


def inspect_video(path, output):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_frames",
        "-show_packets",
        "-show_entries",
        "packet=pts,dts,duration:frame=pts,pkt_pts,pkt_dts,best_effort_timestamp:stream=index,codec_name,width,height,time_base,start_pts,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    data = json.loads(result.stdout or "{}")
    records, streams = data.get("packets_and_frames", []), data.get("streams", [])
    frames = [r for r in records if r.get("type") == "frame"]
    packets = [r for r in records if r.get("type") == "packet"]
    for index, frame in enumerate(frames):
        frame["decoded_index"] = index
        for key in list(frame):
            if key in ("pts", "pkt_pts", "pkt_dts", "best_effort_timestamp"):
                frame[key] = str(frame[key]) if frame[key] is not None else None
    for packet in packets:
        for key in ("pts", "dts", "duration"):
            if key in packet:
                packet[key] = str(packet[key])
    for stream in streams:
        if "start_pts" in stream:
            stream["start_pts"] = str(stream["start_pts"])
        stream["time_base_num"], stream["time_base_den"] = map(int, stream["time_base"].split("/"))
    data.update(
        decoder="ffprobe -show_frames; full segment decoded",
        error=result.stderr or None,
        returncode=result.returncode,
    )
    write(output, data)
    stamps = [p.get("pts") for p in packets]
    valid = bool(stamps) and all(s is not None for s in stamps)
    return {
        "count": len(frames),
        "stream": streams[0] if streams else {},
        "first": str(min(map(int, stamps))) if valid else None,
        "last": str(max(map(int, stamps))) if valid else None,
        "pts_known": valid,
        "reversals": sum(int(b) < int(a) for a, b in itertools.pairwise(stamps)) if valid else None,
        "decoded": bool(frames) and result.returncode == 0 and not result.stderr,
    }


def build(run: Path):
    run = run.resolve()
    if not (run / "meta/settings.json").is_file():
        raise ValueError("This capture needs meta/settings.json from homebody real2sim explore")
    quality = run / "quality"
    quality.mkdir(exist_ok=True)
    inspection = read(quality / "bag_inspection.json", {})
    archive = read(run / "phone/native/archive.json", {})
    interval = read(run / "meta/record_interval.json", {})
    manifest = read(run / 'meta/recorder_source/capture_profile.json') or read(
        CONFIG / "capture_manifest.template.json")
    manifest["calibrations"][0]["notes"] = read(
        CONFIG / "capture_manifest.template.json")["calibrations"][0]["notes"]
    shutil.copyfile(Path(__file__), quality / 'capture_postprocessor.py')
    manifest['provenance']['video_decoder'] = subprocess.run(
        ['ffprobe', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    shutil.copyfile(CONFIG / "capture_manifest.schema.json", run / "capture_manifest.schema.json")
    manifest.update(
        session_id=run.name,
        acquisition_profile="full-local",
        collection_state=("finalized" if interval.get('end_s') is not None
                          and archive.get('complete') else "interrupted"),
        started_at_utc=interval.get('started_at_utc') or (
            datetime.fromtimestamp(interval["start_s"], timezone.utc)  # noqa: UP017 -- Python 3.10 recorder image
            .isoformat().replace('+00:00', 'Z')
            if interval.get("start_s") is not None
            else None
        ),
        start_time_note="Recorder start wall time; sensor and video clocks remain separate.",
    )
    checks, videos = [], {}

    def check(name, outcome, detail):
        checks.append({"name": name, "outcome": outcome, "detail": detail})

    session = run / 'explore/session'
    if session.exists():
        try:
            finalization = read(session / 'finalization.json', {})
            closed = finalization.get('status') == 'closed' and finalization.get('failure') is None
        except (OSError, ValueError, AttributeError):
            closed = False
        check('session_finalization', 'pass' if closed else 'fail',
              'Session evidence must close without a retained recording failure; task interruption is separate.')

    current = [{"path": p.name, "size": p.stat().st_size, "mtime_ns": str(p.stat().st_mtime_ns)}
               for p in sorted((run / 'ws').glob('*.mcap'))]
    bag_matches = bool(current) and current == inspection.get('bag_files')
    check('bag_inspection_binding', 'pass' if bag_matches else 'fail',
          'Split sizes/mtimes must match inspection; rerun check after a source change.')
    required = {'/utlidar/cloud_livox_mid360', '/utlidar/imu_livox_mid360',
                '/ws/laser_odometry_aligned', '/ws/registered_scan_aligned'}
    missing = sorted(required - inspection.get('topics', {}).keys())
    check('metric_inputs', 'fail' if missing else 'pass', {'missing_topics': missing})

    for entry in archive.get("segments", []):
        relative = entry["file"]
        path = run / relative
        if path.is_symlink() or not path.resolve().is_relative_to(run):
            raise ValueError(f"unsafe native segment: {relative}")
        try:
            videos[relative] = inspect_video(
                path, quality / "video" / (file_id(relative) + ".json")
            )
            check(relative, "pass" if videos[relative]["decoded"] else "fail", videos[relative])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            check(relative, "fail", str(exc))
    check(
        "bag_reopen",
        ("pass" if inspection["passed"] else "fail") if inspection else "not_checked",
        "Whole native bag read; see quality/bag_inspection.json.",
    )
    check(
        "native_archive",
        "pass" if archive.get("complete") else "fail",
        {
            "included": len(videos),
            "pending": archive.get("pending_unclosed_segments", []),
            "exposure_coverage_verified": False,
        },
    )
    check(
        "cross_sensor_sync", "not_checked", "No calibrated video exposure-to-LiDAR clock mapping."
    )
    check(
        "coverage_and_dynamic_contamination",
        "not_checked",
        "Inspect the full route; stationary preview does not establish room coverage.",
    )
    check(
        "calibration_and_frame_paths",
        "not_checked",
        "Phone projection/mount and raw IMU axes require measurement; no identity substitutes.",
    )
    def bindings(paths, selector):
        return [{"file_id": file_id(p), "selector": selector} for p in paths]
    chunks = sorted(str(p.relative_to(run)) for p in (run / "ws").glob("*.mcap"))

    def stream(
        name, kind, frame, paths, selector, count, first, last, clock, field, schema, known=True
    ):
        manifest["streams"].append(
            {
                "id": name,
                "kind": kind,
                "status": "recorded",
                "reason": None,
                "timestamp_status": "known" if known else "receive_time_only",
                "timing_notes": "Native timestamps retained; cross-sensor synchronization unverified.",
                "sensor_identity": selector,
                "frame_id": frame,
                "bindings": bindings(paths, selector),
                "payload_schema": schema,
                "timestamp_fields": [
                    {
                        "name": field,
                        "clock_id": clock,
                        "unit": "nanoseconds" if clock in ("host", "mid360") else "ticks",
                        "encoding": "exact integer, decimal string in this manifest",
                        "meaning": field,
                    }
                ],
                "summary_timestamp_field": field,
                "sample_count": count,
                "first_stamp": first,
                "last_stamp": last,
                "observed_rate_hz": None,
            }
        )
        if not count:
            manifest['streams'][-1].update(status='failed', reason='No decodable samples.')

    declared_frames = {f["id"] for f in manifest["frames"]}
    for topic, stats in inspection.get("topics", {}).items():
        sample = stats.get("sample", {})
        frame = sample.get("frame_id") or ("imu_unresolved" if "imu" in topic else None)
        if frame and frame not in declared_frames:
            manifest["frames"].append(
                {"id": frame, "axes": None, "handedness": "unknown", "length_unit": None, "status": "unknown"}
            )
            declared_frames.add(frame)
        source = "source_first_ns" in stats
        clock = "mid360" if topic.startswith(("/ws/", "/utlidar/")) and source else "host"
        kind = (
            "lidar"
            if "PointCloud" in stats["type"]
            else "poses"
            if "Odometry" in stats["type"]
            else "imu"
            if "Imu" in stats["type"]
            else "rgb"
            if "CompressedImage" in stats["type"]
            else "joints"
            if "State" in stats["type"]
            else "events"
        )
        stream(
            "ros_" + topic.strip("/").replace("/", "."),
            kind,
            frame,
            chunks,
            topic,
            stats["n"],
            stats.get("source_first_ns", stats["first_ns"]),
            stats.get("source_last_ns", stats["last_ns"]),
            clock,
            "header.stamp" if source else "bag_receive_time",
            {"ros_type": stats["type"], "sample": sample},
            known=clock == "mid360",
        )
        manifest["streams"][-1]["payload_schema"]["conventions"] = manifest[
            "native_conventions"
        ].get(kind, "See native schema and meta/frames.txt")
        if source:
            manifest["streams"][-1]["timestamp_fields"].append(
                {
                    "name": "bag_receive_time",
                    "clock_id": "host",
                    "unit": "nanoseconds",
                    "encoding": "exact integer in MCAP",
                    "meaning": "Recorder receive/write time",
                }
            )
        span = int(stats["last_ns"]) - int(stats["first_ns"])
        manifest["streams"][-1]["observed_rate_hz"] = (
            (stats["n"] - 1) * 1e9 / span if span else None
        )
        check(
            topic,
            "fail"
            if stats.get("decode_error")
            or stats.get("source_reversals")
            or stats["verdict"] not in ("ok", "EPOCH")
            else "pass",
            stats,
        )
    for relative, video in videos.items():
        name = file_id(relative)
        manifest["clocks"].append(
            {
                "id": name,
                "domain": "native segment video PTS",
                "unit": "ticks",
                "origin": "Segment rebased by FFmpeg; source timeline ranges in capture segments.csv",
                "synchronization": "Unknown mapping to exposure/host/LiDAR clocks",
                "segments": [],
            }
        )
        stream(
            name,
            "rgb",
            "iphone_optical",
            [relative],
            "video:0",
            video["count"],
            video["first"],
            video["last"],
            name,
            "native packet PTS",
            video["stream"],
            known=video["pts_known"],
        )
        manifest["streams"][-1]["timing_notes"] = (
            "Exact packet PTS/DTS and decoded indices in quality/video. Older ffprobe reports "
            "decoded best_effort_timestamp, not exposure time. Packet PTS can reorder for B frames."
        )
        manifest['streams'][-1]['sensor_identity'] = None
        if video['pts_known'] and int(video['last']) > int(video['first']):
            metadata = video['stream']
            span = ((int(video['last']) - int(video['first'])) * metadata['time_base_num']
                    / metadata['time_base_den'])
            manifest['streams'][-1]['observed_rate_hz'] = (video['count'] - 1) / span
        if not video["pts_known"]:
            manifest["streams"][-1].update(timestamp_status="unknown", summary_timestamp_field=None)
    recorded = {s["bindings"][0]["selector"] for s in manifest["streams"] if s["bindings"]}
    for name, kind, reason in [
        ("depth", "depth", "No depth camera in this capture profile"),
        (
            "phone_calibration",
            "other",
            "Intrinsics, distortion, stabilization/crop, rolling shutter and extrinsic unmeasured",
        ),
        ("independent_reference", "poses", "No survey or independent pose reference captured"),
        ("interaction_evidence", "events", "No mechanism/contact/physical-parameter measurements"),
    ]:
        manifest["streams"].append(
            {
                "id": name,
                "kind": kind,
                "status": "not_requested" if name == "depth" else "unavailable",
                "reason": reason,
                "timestamp_status": "unknown",
                "timing_notes": reason,
                "sensor_identity": None,
                "frame_id": None,
                "bindings": [],
                "payload_schema": {},
                "timestamp_fields": [],
                "summary_timestamp_field": None,
                "sample_count": None,
                "first_stamp": None,
                "last_stamp": None,
                "observed_rate_hz": None,
            }
        )
    for topic in inspection.get("requested_topics", []):
        if topic not in recorded:
            manifest["streams"].append(
                dict(
                    manifest["streams"][-1],
                    id="ros_" + topic.strip("/").replace("/", "."),
                    reason=f"Requested topic absent: {topic}",
                    sensor_identity=topic,
                )
            )
    manifest["calibrations"][0]["stream_ids"] = [
        s["id"] for s in manifest["streams"] if s["kind"] == "rgb"
    ]
    manifest["calibrations"][0]["bindings"] = bindings(
        ["meta/phone_camera.json"] if (run / "meta/phone_camera.json").is_file() else [], None)
    decisions = [(str(p.relative_to(run)), read(p))
                 for p in sorted((run / 'explore/vlm').glob('leg_*.json'))]
    if decisions:
        stamps = [row['wall'] for _, row in decisions]
        stream('agent_decisions', 'events', 'exploration_start', [p for p, _ in decisions],
               None, len(decisions), str(min(stamps)), str(max(stamps)), 'host',
               'wall', {'format': 'JSON decisions; context_manifest indexes prompts/images'},
               known=False)
        manifest['streams'][-1]['timestamp_fields'][0].update(
            unit='seconds', encoding='JSON float host epoch; not exposure time')
        missing_views = []
        for _, decision in decisions:
            context = read(run / 'explore/vlm' / decision.get('context_manifest', ''), {})
            missing_views.extend(name for name in context.get('images', [])
                                 if not (run / 'explore/vlm' / name).is_file())
            if not context:
                missing_views.append(decision.get('context_manifest', 'missing context manifest'))
        check('agent_context', 'fail' if missing_views else 'pass',
              {'decisions': len(decisions), 'missing_views': missing_views})
    if "/ws/laser_odometry_aligned" in recorded:
        manifest["transforms"].append(
            {
                "id": "T_ws_map_from_gravity",
                "source_frame": "gravity",
                "target_frame": "ws_map",
                "status": "estimated",
                "representation": "dynamic",
                "translation_m": None,
                "quaternion_xyzw": None,
                "bindings": bindings(chunks, "/ws/laser_odometry_aligned"),
                "validity": {"clock_id": "host", "epochs": inspection.get("epochs", [])},
                "uncertainty": None,
                "notes": ("Online odometry with covariance/epochs; not absolute ground truth. "
                       "Epoch boundaries here are host wall_s from bag inspection, not "
                       "MID-360 stamp_s or exporter epoch ordinals."),
            }
        )
    origins = [str(p.relative_to(run)) for p in sorted(
        (run / 'explore/views').glob('epoch_*/index.json'))
        if read(p).get('start_frame')]
    manifest['transforms'].append({
        "id": 'T_exploration_start_from_ws_map', "source_frame": 'ws_map',
        "target_frame": 'exploration_start', "status": 'unknown', "representation": 'unknown',
        "translation_m": None, "quaternion_xyzw": None, "bindings": bindings(origins, 'start_frame'),
        "validity": {}, "uncertainty": None,
        "notes": 'Planar zero and per-scan Z rule in bound indexes when recorded. No fixed 3D '
              'transform: each scan is re-grounded. View epoch IDs differ from SLAM IDs.'})
    check('exploration_frame_origin', 'pass' if origins else 'not_checked',
          {'indexes': origins, 'scope': 'Planar origin only; missing origins stay unknown.'})
    if not bag_matches or not inspection.get("passed") or missing:
        for key in ("geometry", "localization_evaluation"):
            manifest["readiness"][key]["status"] = "pending"
            manifest['readiness'][key]['gaps'].append(
                'Required metric streams are absent, failed, or lack matching inspection evidence.')
    manifest["splits"] = (
        [
            {
                "selection": "file_set",
                "role": "unassigned",
                "file_ids": [file_id(p) for p in chunks],
                "used_in_map": True,
                "notes": "SLAM used these scans. Reserve a separate repeat route before evaluation.",
            }
        ]
        if chunks
        else []
    )
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Capture contains symlink: {path}")
        if not path.is_file() or path.name == "capture_manifest.json" or path.name.endswith(".tmp"):
            continue
        relative = str(path.relative_to(run))
        if relative in ("launch.log", "quality/capture_health.json"):
            continue
        before = path.stat()
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if path.stat().st_mtime_ns != before.st_mtime_ns:
            raise ValueError(f"File changed during finalization: {relative}")
        role = (
            "native_bag"
            if relative in chunks
            else "native_video"
            if relative in videos
            else "derived"
            if relative.startswith(("video/", "quality/"))
            else "metadata_or_observation"
        )
        manifest["files"].append(
            {
                "id": file_id(relative),
                "path": relative,
                "role": role,
                "size_bytes": before.st_size,
                "sha256": digest,
                "finalized": True,
            }
        )
    hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
    matches = all(hashes.get(row["file"]) == row["sha256"] for row in archive.get("segments", []))
    check(
        "archive_hashes",
        "pass" if matches and videos else "fail",
        "Native archived bytes compared with archive.json hashes.",
    )
    check(
        "file_integrity",
        "pass",
        "All listed files hashed; symlinks refused. launch.log excluded: teardown can append.",
    )
    health = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- Python 3.10 recorder image
        "checks": checks,
        "readiness": manifest["readiness"],
        "collection_acceptance": "pending",
        "sensor_integrity": ("accepted" if manifest["collection_state"] == "finalized"
                          and all(c["outcome"] != "fail" for c in checks) else "rejected"),
        "gaps": ["Coverage review, clock mapping and frame/mount checks remain outstanding."],
    }
    write(quality / "capture_health.json", health)
    manifest["sensor_integrity"] = health["sensor_integrity"]
    data = (quality / "capture_health.json").read_bytes()
    manifest["files"].append(
        {
            "id": file_id("quality/capture_health.json"),
            "path": "quality/capture_health.json",
            "role": "capture_health",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "finalized": True,
        }
    )
    manifest["quality_report_file_id"] = file_id("quality/capture_health.json")
    manifest["scene_policy"]["authoritative_source_file_ids"] = [
        file_id(p) for p in chunks + list(videos)
    ]
    manifest["provenance"]["configuration_file_ids"] = [
        f["id"] for f in manifest["files"] if f["path"].startswith("meta/")
    ]
    for readiness in manifest["readiness"].values():
        readiness["evidence_file_ids"] = [manifest["quality_report_file_id"]]
    for path in ["explore/live_map.npz"] + [
        str(p.relative_to(run)) for p in (run / "video").glob("*") if p.is_file()
    ]:
        if (run / path).is_file():
            manifest["derived_products"].append(
                {
                    "id": file_id(path),
                    "output_file_ids": [file_id(path)],
                    "input_file_ids": [file_id(p) for p in chunks],
                    "producer": {"source": "meta/recorder_source"},
                    "parameters": {
                        "lineage": "See video/video_index.json; preview repeats RGB; PLY per epoch"
                    },
                    "calibration_ids": [],
                    "alignment_ids": [],
                    "frame_id": ('ws_map' if path.endswith('.ply') else
                              'exploration_start' if path.endswith('live_map.npz') else None),
                    "clock_id": None,
                }
            )
    write(run / "capture_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    if args.initialize:
        initialize(args.run.resolve())
    else:
        result = build(args.run)
        health = read(args.run / 'quality/capture_health.json')
        failed = [c['name'] for c in health['checks'] if c['outcome'] == 'fail']
        print("Capture manifest written; capability readiness:", json.dumps(result["readiness"]))
        print("Failed capture checks:", failed)
        raise SystemExit(1 if failed else 0)
