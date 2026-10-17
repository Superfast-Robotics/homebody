"""After a bag closes: every listed topic present, its rate over its own span, its biggest gap,
and for stamped messages the header clock's offset from the receive clock and its jumps.

postcheck.py BAG_DIR TOPICS_TXT      (inside the recorder: one pass over the bag via rosbag2_py)

A constant clock offset is reported; a jump in it is a failure."""

import json
import shutil
import statistics
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECT_HZ = {
    "/homebody/iphone/color/image/compressed": 5,
    "/utlidar/cloud_livox_mid360": 10,
    "/utlidar/imu_livox_mid360": 200,
    "/utlidar/cloud_livox_mid360_sec": 10,
    "/utlidar/imu_livox_mid360_sec": 200,
    "/ws/laser_odometry_aligned": 10,
    "/ws/registered_scan_aligned": 10,
    "/lowstate": 200,
    "/homebody/clock_offset": 1,
    "/exploration/state": 1,
}
MIN_FRACTION = 0.95
COVERAGE_MIN = 0.90  # share of the bag a required stream must span
# streams a SuperOdom restart silences
EPOCH_TOPICS = (
    "/ws/", "/tf", "/utlidar/cloud_livox_mid360_sec", "/utlidar/imu_livox_mid360_sec",
)
MAX_GAP_S = 1.0  # judged on streams expected at 5 Hz or more
JUMP_S = 1.0
# topics that may be absent; --optional adds more
OPTIONAL = {
    "/utlidar/cloud_livox_mid360_sec",
    "/utlidar/imu_livox_mid360_sec",
    "/ws/tf",
    "/lowcmd",
    "/dex3/left/state",
    "/dex3/right/state",
    "/ws/laser_odometry",
    "/ws/state_estimation",
    "/ws/state_estimation_health",
    "/ws/super_odometry_stats",
    "/ws/laser_cloud_map_aligned",
    "/ws/tf_static",
}
# types whose first field is a std_msgs/Header, stamped after the CDR encapsulation
STAMPED = {
    "nav_msgs/msg/Odometry",
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/msg/Imu",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CameraInfo",
    "geometry_msgs/msg/PoseStamped",
}


def header_stamp(data: bytes) -> float:
    sec, nanosec = struct.unpack_from("<iI" if data[1] == 1 else ">iI", data, 4)
    return sec + nanosec * 1e-9


def scan(bag: Path, details: Path | None = None) -> tuple[dict, list[float]]:
    """Per-topic statistics and the /homebody/slam_epoch wall times."""
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if details:
        from ament_index_python.packages import get_package_share_directory
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        for package in {t.split('/')[0] for t in types.values()} | {
                'builtin_interfaces', 'geometry_msgs', 'std_msgs'}:
            try:
                directory = Path(get_package_share_directory(package)) / 'msg'
                for source in directory.glob('*'):
                    if source.suffix in {'.msg', '.idl'}:
                        target = details.parent / 'ros_schemas' / package / source.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
            except LookupError:
                pass  # unavailable decoder types are reported per stream below
    stats: dict = {}
    epochs: list[float] = []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns * 1e-9
        s = stats.get(topic)
        if s is None:
            s = stats[topic] = {"n": 0, "first": t, "last": t, "gap": 0.0, "offsets": [], "jumps": 0, "jump": 0.0}
            if details:
                s.update(type=types[topic], first_ns=str(t_ns), source_reversals=0,
                         source_duplicates=0, sample={}, decode_error=None)
                try:
                    msg = deserialize_message(data, get_message(types[topic]))
                    s['sample'] = {"fields": msg.get_fields_and_field_types()}
                    if hasattr(msg, 'header'):
                        s['sample']['frame_id'] = msg.header.frame_id
                    for field in ('child_frame_id', 'encoding', 'height', 'width',
                                  'point_step', 'is_bigendian'):
                        if hasattr(msg, field):
                            s['sample'][field] = getattr(msg, field)
                    if hasattr(msg, 'fields'):
                        s['sample']['point_fields'] = [{"name": f.name, "offset": f.offset,
                            "datatype": f.datatype, "count": f.count} for f in msg.fields]
                except (ImportError, AttributeError, ValueError, TypeError, RuntimeError) as exc:
                    s['decode_error'] = str(exc)
        if details:
            s['last_ns'] = str(t_ns)
            if types[topic] in STAMPED:
                sec, ns = struct.unpack_from('<iI' if data[1] == 1 else '>iI', data, 4)
                stamp = sec * 1_000_000_000 + ns
                previous = int(s.get('source_last_ns', stamp))
                s['source_reversals'] += stamp < previous
                s['source_duplicates'] += s['n'] > 0 and stamp == previous
                s.setdefault('source_first_ns', str(stamp))
                s['source_last_ns'] = str(stamp)
        if t - s["last"] > s["gap"]:
            s["gap"], s["gap_end"] = t - s["last"], t
        if topic == "/homebody/slam_epoch":
            try:  # std_msgs/String CDR: 4-byte encapsulation, uint32 length, the JSON
                n = struct.unpack_from("<I", data, 4)[0]
                epochs = [e["wall_s"] for e in json.loads(data[8 : 8 + n - 1])["epochs"]]
            except (ValueError, KeyError, struct.error):
                pass
        s["n"], s["first"], s["last"] = s["n"] + 1, min(s["first"], t), max(s["last"], t)
        if types.get(topic) in STAMPED and len(data) >= 12:
            offset = t - header_stamp(data)
            if s["offsets"] and abs(offset - s["offsets"][-1]) > JUMP_S:
                s["jumps"] += 1
                s["jump"] = max(s["jump"], abs(offset - s["offsets"][-1]))
            s["offsets"].append(offset)
    return stats, epochs


def verdict(topic: str, s: dict | None, bag_span: tuple[float, float] | None,
            epochs: list[float]) -> str:
    if s is None:
        return "absent" if topic in OPTIONAL else "MISSING"
    span = s["last"] - s["first"]
    hz = (s["n"] - 1) / span if span > 0 else 0.0
    want = EXPECT_HZ.get(topic)
    if (want and want >= 5 and bag_span and bag_span[1] > bag_span[0]
            and span / (bag_span[1] - bag_span[0]) < COVERAGE_MIN):
        return "SHORT"  # the stream stopped early or started late
    end = s.get("gap_end", 0.0)
    # a gap around an epoch stamp is a SuperOdom restart
    epoch_gap = topic.startswith(EPOCH_TOPICS) and any(
        end - s["gap"] - 5.0 <= e <= end + 20.0 for e in epochs
    )
    if epoch_gap:  # the gap is a SuperOdom (re)start: the rate is judged without it
        hz = (s["n"] - 1) / (span - s["gap"]) if span - s["gap"] > 0 else 0.0
    if want and hz < MIN_FRACTION * want:
        return "LOW"
    if want and want >= 5 and s["gap"] > MAX_GAP_S:
        return "EPOCH" if epoch_gap else "GAP"
    return "JUMP" if s["jumps"] else "ok"


def main() -> int:
    if "--optional" in sys.argv:
        at = sys.argv.index("--optional")
        OPTIONAL.update(t for t in sys.argv[at + 1].split(",") if t)
        del sys.argv[at:at + 2]
    bag, topics_txt = Path(sys.argv[1]), Path(sys.argv[2])
    wanted = [
        line.strip()
        for line in topics_txt.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    details = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    stats, epochs = scan(bag, details)
    bag_span = None
    if not stats:  # an empty bag fails with every row MISSING
        print(f"\nbag {bag} · 0 messages")
    else:
        bag_span = (min(s["first"] for s in stats.values()), max(s["last"] for s in stats.values()))
    bad = 0
    total = sum(s["n"] for s in stats.values())
    print(f"\nbag {bag} · {total} messages · rates over each topic's own span")
    print(f"  {'':7s} {'topic':40s} {'msgs':>8s} {'Hz':>7s} {'max gap':>8s}  stamp offset")
    for topic in wanted:
        s = stats.get(topic)
        v = verdict(topic, s, bag_span, epochs)
        bad += v in ("MISSING", "LOW", "GAP", "JUMP", "SHORT") and topic not in OPTIONAL
        if s is None:
            print(f"  {v:7s} {topic:40s} {0:8d}")
            continue
        span = s["last"] - s["first"]
        hz = (s["n"] - 1) / span if span > 0 else 0.0
        want = EXPECT_HZ.get(topic)
        cover = span / max(bag_span[1] - bag_span[0], 1e-9)
        offset = (
            f"  receive - header median {statistics.median(s['offsets']):+.3f} s, min "
            f"{min(s['offsets']):+.3f} s"
            + (f", {s['jumps']} jump(s) up to {s['jump']:.1f} s" if s["jumps"] else "")
            if s["offsets"]
            else ""
        )
        print(
            f"  {v:7s} {topic:40s} {s['n']:8d} {hz:7.1f} {s['gap']:7.2f}s {cover:5.0%}"
            + (f"  (expect {want} Hz)" if want else "")
            + offset
        )
    failed = f"FAIL: {bad} required topic(s) missing, low, gapped or jumped"
    print(f"\n{failed if bad else 'PASS'}")
    if details:
        rows = {topic: {k: v for k, v in s.items() if k != 'offsets'} | {
            'verdict': verdict(topic, s, bag_span, epochs), 'expected_hz': EXPECT_HZ.get(topic),
            'offset_median_s': statistics.median(s['offsets']) if s['offsets'] else None,
        } for topic, s in stats.items()}
        details.write_text(json.dumps({"checked_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- Python 3.10 recorder image
            "reader": 'ROS 2 Humble rosbag2_py/CDR; whole bag read, first message per topic decoded',
            "bag_files": [{"path": p.name, "size": p.stat().st_size,
                            "mtime_ns": str(p.stat().st_mtime_ns)}
                       for p in sorted(bag.glob('*.mcap'))],
            "passed": not bad, "topics": rows, "requested_topics": wanted, "epochs": epochs},
            indent=2) + '\n')
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
