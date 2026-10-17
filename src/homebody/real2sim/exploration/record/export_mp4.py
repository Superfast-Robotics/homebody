"""Inside the recorder: the bag's camera streams as 30 fps .mp4 files, one per camera.

  python3 export_mp4.py <bag dir> <out dir> [--fps 30]

Also writes video_index.json (frame counts per file) and slam_map.ply, the last SuperOdom epoch's
registered scans on a 5 cm voxel grid (slam_map_epoch<k>.ply for each epoch)."""

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def stamp_s(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def decode(msg):
    return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)


def scan_voxels(msg, voxel_m: float, origin=None, self_clear_m: float = 1.0) -> np.ndarray:
    """Integer voxel indices of one PointCloud2's xyz (float32 fields x, y, z at their offsets),
    without the points within `self_clear_m` (xy) of `origin`, the lidar's pose for that scan."""
    offs = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"} <= set(offs):
        return np.zeros((0, 3), np.int64)
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
    xyz = np.stack([raw[:, offs[k] : offs[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
    xyz = xyz[np.isfinite(xyz).all(1)]
    if origin is not None:
        xyz = xyz[np.hypot(xyz[:, 0] - origin[0], xyz[:, 1] - origin[1]) >= self_clear_m]
    return np.floor(xyz / voxel_m).astype(np.int64)


def read_epochs(bag: str, types: dict) -> list[float]:
    """Start stamps of every SuperOdom epoch, from the latched /homebody/slam_epoch; [0.0] when the
    bag has none."""
    if "/homebody/slam_epoch" not in types:
        return [0.0]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/homebody/slam_epoch"]))
    last = None
    while reader.has_next():
        _, data, _ = reader.read_next()
        last = deserialize_message(data, get_message(types["/homebody/slam_epoch"])).data
    stamps = sorted(float(e["stamp_s"]) for e in json.loads(last)["epochs"]) if last else []
    return stamps or [0.0]


def write_ply(path: Path, pts: np.ndarray) -> None:
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(np.ascontiguousarray(pts, dtype="<f4").tobytes())


class Timeline:
    """Frames resampled onto an exact fps grid: each slot takes the frame captured nearest its
    centre."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.dt = path, fps, 1.0 / fps
        self.writer = None
        self.slot_t = None
        self.prev = None  # (stamp, frame, used)
        self.n = {"written": 0, "source": 0, "repeated": 0, "dropped": 0, "undecodable": 0}
        self.first = self.last_stamp = None

    def _write(self, entry):
        _t, frame, used = entry
        self.writer.write(frame)
        self.n["written"] += 1
        if used:
            self.n["repeated"] += 1
        entry[2] = True

    def push(self, t: float, frame):
        self.n["source"] += 1
        if frame is None:
            self.n["undecodable"] += 1
            return
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
            self.slot_t, self.first = t, t
        self.last_stamp = t
        cur = [t, frame, False]
        while self.slot_t <= t:  # every slot up to this stamp: the nearer of the two frames
            if self.prev is not None and abs(self.prev[0] - self.slot_t) <= abs(t - self.slot_t):
                self._write(self.prev)
            else:
                self._write(cur)
            self.slot_t += self.dt
        if self.prev is not None and not self.prev[2]:
            self.n["dropped"] += 1
        self.prev = cur

    def close(self):
        if self.writer is not None:
            if self.prev is not None and not self.prev[2]:
                self._write(self.prev)
            self.writer.release()
        return {
            **self.n,
            "first_stamp_s": self.first,
            "last_stamp_s": self.last_stamp,
            "fps": self.fps,
            "file": self.path.name,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--voxel-m", type=float, default=0.05)
    ap.add_argument("--self-clear-m", type=float, default=1.0,
                    help="lidar.self_clear_m: points this close (xy) to the lidar stay out of the cloud")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {}
    for topic in types:
        parts = topic.split("/")
        if len(parts) >= 4 and parts[1] == "homebody" and topic.endswith("/color/image/compressed"):
            wanted[topic] = Timeline(out / f"{parts[2]}_rgb_{args.fps:g}fps.mp4", args.fps)
    if not wanted:
        print("no /homebody/<name>/color image topics in the bag", file=sys.stderr)
    scan_topic = "/ws/registered_scan_aligned" if "/ws/registered_scan_aligned" in types else None
    odom_topic = "/ws/laser_odometry_aligned" if "/ws/laser_odometry_aligned" in types else None
    topics = list(wanted) + [t for t in (scan_topic, odom_topic) if t]
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    epochs = read_epochs(args.bag, types) if scan_topic else [0.0]
    voxels: dict[int, set] = {}
    n_scans: dict[int, int] = {}
    poses: list[tuple[float, tuple[float, float]]] = []  # (stamp, lidar xy)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == odom_topic:
            p = msg.pose.pose.position
            poses.append((stamp_s(msg), (float(p.x), float(p.y))))
            poses = poses[-50:]
            continue
        if topic == scan_topic:
            t = stamp_s(msg)
            k = max(bisect.bisect_right(epochs, t) - 1, 0)
            n_scans[k] = n_scans.get(k, 0) + 1
            near = min(poses, key=lambda o: abs(o[0] - t), default=None)
            origin = near[1] if near is not None and abs(near[0] - t) < 0.2 else None
            voxels.setdefault(k, set()).update(map(tuple, scan_voxels(msg, args.voxel_m, origin, args.self_clear_m)))
            continue
        wanted[topic].push(stamp_s(msg), decode(msg))
    index = {topic: tl.close() for topic, tl in wanted.items()}
    if scan_topic and voxels:
        last = max(voxels)
        for k in sorted(voxels):  # one cloud per epoch; slam_map.ply is the last epoch's
            pts = (np.array(sorted(voxels[k]), dtype=np.float32) + 0.5) * args.voxel_m
            names = [f"slam_map_epoch{k + 1}.ply"] + (["slam_map.ply"] if k == last else [])
            for name in names:
                write_ply(out / name, pts)
                index[f"{scan_topic}#{name}"] = {
                    "file": name,
                    "points": len(pts),
                    "scans": n_scans[k],
                    "epoch": k + 1,
                    "epochs_in_bag": len(epochs),
                    "epoch_stamp_s": epochs[k],
                    "voxel_m": args.voxel_m,
                    "frame": f"ws_map of epoch {k + 1} (aligned)",
                }
            print(
                f"{' + '.join(names)}: {len(pts)} points ({args.voxel_m:g} m voxels) from "
                f"{n_scans[k]} scans of epoch {k + 1}/{len(epochs)}"
            )
    (out / "video_index.json").write_text(json.dumps(index, indent=2))
    for topic, row in index.items():
        if "last_stamp_s" not in row:
            continue
        span = (row["last_stamp_s"] or 0) - (row["first_stamp_s"] or 0)
        print(
            f"{row['file']}: {row['written']} frames at {row['fps']:g} fps from {row['source']} "
            f"source frames over {span:.1f} s (repeated {row['repeated']}, dropped "
            f"{row['dropped']}, undecodable {row['undecodable']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
