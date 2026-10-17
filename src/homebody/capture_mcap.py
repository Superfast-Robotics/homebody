"""Offline lossless channel filtering and verification of native MCAP sensor records."""
from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from contextlib import nullcontext

SENSOR_TOPICS = frozenset({
    "/utlidar/cloud_livox_mid360", "/utlidar/imu_livox_mid360",
    "/utlidar/cloud_livox_mid360_sec", "/utlidar/imu_livox_mid360_sec",
    "/ws/registered_scan_aligned", "/ws/laser_odometry_aligned",
    "/ws/laser_odometry", "/ws/state_estimation", "/ws/state_estimation_aligned",
    "/tf", "/tf_static", "/ws/tf", "/ws/tf_static", "/homebody/slam_epoch",
    "/homebody/clock_offset", "/homebody/iphone/color/image/compressed",
    "/lowstate", "/dex3/left/state", "/dex3/right/state",
})
REQUIRED_TOPICS = frozenset({
    "/utlidar/cloud_livox_mid360", "/utlidar/imu_livox_mid360",
    "/ws/registered_scan_aligned", "/ws/laser_odometry_aligned", "/homebody/slam_epoch",
})
STAMPED_TYPES = frozenset({
    "nav_msgs/msg/Odometry", "sensor_msgs/msg/PointCloud2", "sensor_msgs/msg/Imu",
    "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image", "sensor_msgs/msg/CameraInfo",
})


def scan(path, destination=None):
    """Read every message, optionally filtering to a new MCAP; retain exact native records."""
    try:
        from mcap.reader import NonSeekingReader, make_reader
        from mcap.writer import CompressionType, Writer
    except ImportError as error:
        raise RuntimeError("Offline capture export requires the exploration extra (mcap==1.5.0)") from error
    with path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary, header = reader.get_summary(), reader.get_header()
        if summary is None or summary.statistics is None:
            raise ValueError(f"MCAP is not closed with a summary: {path.name}")
        if header.profile != "ros2":
            raise ValueError(f"Unsupported MCAP profile: {header.profile!r}")
    counts, rows, digests, channel_ids = Counter(), {}, {}, {}
    output = destination.open("xb") if destination is not None else nullcontext(None)
    with path.open("rb") as stream, output as target:
        writer = Writer(target, compression=CompressionType.ZSTD, enable_data_crcs=True) if target else None
        if writer:
            writer.start(profile=header.profile)
        reader = NonSeekingReader(stream, validate_crcs=True)
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            if channel != summary.channels.get(channel.id):
                raise ValueError(f"Native channel differs from MCAP summary: {channel.topic}")
            if schema is not None and schema.id in summary.schemas and schema != summary.schemas[schema.id]:
                raise ValueError(f"Native schema differs from MCAP summary: {channel.topic}")
            counts[channel.id] += 1
            if channel.topic not in SENSOR_TOPICS:
                continue
            if schema is None or schema.encoding not in {"ros2msg", "ros2idl"} or channel.message_encoding != "cdr":
                raise ValueError(f"Missing native ROS schema/encoding for {channel.topic}")
            identity = {"schema_name": schema.name, "schema_encoding": schema.encoding,
                        "schema_sha256": hashlib.sha256(schema.data).hexdigest(),
                        "message_encoding": channel.message_encoding, "metadata": channel.metadata}
            row = rows.get(channel.topic)
            if row is None:
                rows[channel.topic] = row = {**identity, "count": 0, "first_log_time_ns": str(message.log_time),
                    "first_publish_time_ns": str(message.publish_time), "header_reversals": 0}
                digests[channel.topic] = hashlib.sha256()
            elif any(row[key] != value for key, value in identity.items()):
                raise ValueError(f"Native channel schema changed within a split: {channel.topic}")
            row["count"] += 1
            row["last_log_time_ns"] = str(message.log_time)
            row["last_publish_time_ns"] = str(message.publish_time)
            if schema.name in STAMPED_TYPES:
                if len(message.data) < 12 or message.data[:2] not in {b"\x00\x00", b"\x00\x01"}:
                    raise ValueError(f"Invalid stamped CDR payload: {channel.topic}")
                seconds, nanos = struct.unpack_from("<iI" if message.data[1] else ">iI", message.data, 4)
                if nanos >= 1_000_000_000:
                    raise ValueError(f"Invalid header timestamp: {channel.topic}")
                stamp = seconds * 1_000_000_000 + nanos
                row["header_reversals"] += stamp < int(row.get("last_header_stamp_ns", stamp))
                row.setdefault("first_header_stamp_ns", str(stamp))
                row["last_header_stamp_ns"] = str(stamp)
            if channel.topic == "/homebody/slam_epoch":
                length = struct.unpack_from("<I" if message.data[1] else ">I", message.data, 4)[0]
                epoch = json.loads(message.data[8:8 + length - 1])
                if not isinstance(epoch.get("epochs"), list):
                    raise ValueError("SLAM epoch record lacks an epoch list")
            digests[channel.topic].update(struct.pack("<QQIQ", message.log_time,
                message.publish_time, message.sequence, len(message.data)))
            digests[channel.topic].update(message.data)
            if writer:
                if channel.id not in channel_ids:
                    schema_id = writer.register_schema(schema.name, schema.encoding, schema.data)
                    channel_ids[channel.id] = writer.register_channel(channel.topic,
                        channel.message_encoding, schema_id, channel.metadata)
                writer.add_message(channel_ids[channel.id], message.log_time, message.data,
                                   message.publish_time, message.sequence)
        if writer:
            writer.finish()
    if sum(counts.values()) != summary.statistics.message_count or any(
            counts[identity] != count for identity, count in summary.statistics.channel_message_counts.items()):
        raise ValueError(f"MCAP summary does not match its messages: {path.name}")
    if set(counts) - summary.channels.keys():
        raise ValueError(f"MCAP summary omits message channels: {path.name}")
    for topic, row in rows.items():
        row["message_sha256"] = digests[topic].hexdigest()
    return {"topics": rows, "total_messages": sum(counts.values()),
            "retained_messages": sum(row["count"] for row in rows.values()),
            "actual_topics": sorted(channel.topic for channel in summary.channels.values())}


def filtered(source, destination):
    """Publish no file until source and rewritten native records agree exactly."""
    expected = scan(source, destination)
    actual = scan(destination)
    if expected["topics"] != actual["topics"] or set(actual["actual_topics"]) - SENSOR_TOPICS:
        raise ValueError("Filtered MCAP changed native sensor records or retained an excluded topic")
    if actual["total_messages"] != expected["retained_messages"]:
        raise ValueError("Filtered MCAP message counts do not match")
    return actual
