"""Inside the recorder container: the phone stream as ROS topics beside the LiDAR, on one clock.

Topics: /homebody/iphone/color/image/compressed (HSCOLOR1 JPEG at --record-fps), /tf (ws_map ->
gravity, from /ws/laser_odometry_aligned), /tf_static (gravity -> mid360_link -> torso_link),
/homebody/cameras, /homebody/slam_epoch, /homebody/clock_offset, /exploration/state and
/bags/bridge_live.json. `gravity` is SuperOdom's odometry child, the MID-360 mounted upside down."""

import argparse
import array
import json
import os
import struct
import threading
import time

import numpy as np
import rclpy
import zmq
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

MAGIC, HEADER_LEN = b"HSCOLOR1", struct.Struct("<I")
STATE = "/explore/live_state.json"
LIDAR_IN_TORSO_XYZ = (0.0002835, 0.00003, 0.41618)  # the URDF's mid360_joint origin
LIDAR_IN_TORSO_PITCH = 0.04014257279586953  # the mid360_joint pitch


def quat_from_R(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return x, y, z, w


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def chain():
    """(parent, child, T_parent_child) for every static link, child points into parent."""
    c, s = np.cos(LIDAR_IN_TORSO_PITCH), np.sin(LIDAR_IN_TORSO_PITCH)
    pitch = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return [
        ("gravity", "mid360_link", rigid(np.diag([1.0, -1.0, -1.0]), np.zeros(3))),
        ("mid360_link", "torso_link", np.linalg.inv(rigid(pitch, LIDAR_IN_TORSO_XYZ))),
    ]


def stamp_ns(header, ns):
    header.stamp.sec, header.stamp.nanosec = ns // 1_000_000_000, ns % 1_000_000_000


class Bridge(Node):
    def __init__(self, streams, record_fps):
        super().__init__("color_bridge")
        self.record_fps = record_fps
        latched = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.static = self.create_publisher(TFMessage, "/tf_static", latched)
        self.state = self.create_publisher(String, "/exploration/state", 10)
        self.cameras = self.create_publisher(String, "/homebody/cameras", latched)
        self.clock = self.create_publisher(String, "/homebody/clock_offset", 10)
        # a deep queue: this callback runs in bursts while the zmq loop copies frames
        odom_qos = QoSProfile(
            depth=200,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Odometry, "/ws/laser_odometry_aligned", self.on_odom, odom_qos)
        self.poller = zmq.Poller()
        self.names = [name for name, _ in streams]
        self.streams = {}
        for name, endpoint in streams:
            sock = self.subscribe(endpoint)
            self.streams[sock] = (
                name,
                self.create_publisher(
                    CompressedImage, f"/homebody/{name}/color/image/compressed", qos_profile_sensor_data
                ),
            )
        self.geometry = {}  # name -> that stream's newest HSCOLOR1 geometry
        self.serials = {}  # name -> the serial its header names
        self.seen = {}  # what -> (header stamp s, receive wall s)
        self.kept_ns = {}  # colour stream -> capture stamp of the last frame recorded
        self.stamps = {}  # colour stream -> the last 60 capture stamps (its source rate)
        self.skipped = {}
        self.counts = {}  # what -> messages since the last bridge_live.json
        self.last_live = 0.0
        self.tf = self.create_publisher(TFMessage, "/tf", qos_profile_sensor_data)
        self.epoch_pub = self.create_publisher(String, "/homebody/slam_epoch", latched)
        self.epochs = []
        self.last_odom = None  # (stamp s, xy, yaw)
        self.bad = {}
        self.last_static = self.last_state = self.last_clock = 0.0

    def subscribe(self, endpoint):
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 4)
        sock.connect(endpoint)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.poller.register(sock, zmq.POLLIN)
        return sock

    def on_odom(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.seen["ws_odometry"] = (stamp, time.time())
        self.counts["ws_odometry"] = self.counts.get("ws_odometry", 0) + 1
        t = TransformStamped()
        t.header, t.child_frame_id = msg.header, "gravity"
        t.header.frame_id = "ws_map"
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        tr = t.transform.translation
        tr.x, tr.y, tr.z = p.x, p.y, p.z
        t.transform.rotation = q
        self.tf.publish(TFMessage(transforms=[t]))
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        near_origin = np.hypot(p.x, p.y) < 0.5 and abs(yaw) < 0.3
        last = self.last_odom
        # a restart re-origins: a silence over 2 s, a stamp going back or a jump to the origin
        jumped = last is not None and np.hypot(p.x - last[1][0], p.y - last[1][1]) > 1.0
        silent = last is not None and stamp - last[0] > 2.0
        went_back = last is not None and stamp < last[0] - 0.5
        if last is None or went_back or (near_origin and (silent or jumped)):
            self.epochs.append({"epoch": len(self.epochs), "wall_s": time.time(), "stamp_s": stamp})
            self.epoch_pub.publish(String(data=json.dumps({"epochs": self.epochs})))
        self.last_odom = (stamp, (p.x, p.y), yaw)

    def refuse(self, what, exc):
        self.bad[what] = self.bad.get(what, 0) + 1
        if self.bad[what] in (1, 100, 10000):
            print(f"[bridge] {what}: packet {self.bad[what]} refused: {exc}", flush=True)

    def publish_frame(self, name, image_pub, payload):
        if payload[: len(MAGIC)] != MAGIC:
            self.refuse(name, "not HSCOLOR1")
            return
        (n,) = HEADER_LEN.unpack_from(payload, len(MAGIC))
        start = len(MAGIC) + HEADER_LEN.size
        header = json.loads(payload[start : start + n])
        ns = int(header.get("capture_realtime_ns", time.time_ns()))
        hist = self.stamps.setdefault(name, [])
        hist.append(ns)
        del hist[:-60]
        rate = (len(hist) - 1) * 1e9 / max(hist[-1] - hist[0], 1) if len(hist) > 10 else 0.0
        every = round(rate / self.record_fps) if rate > 1.5 * self.record_fps else 1
        seq = header.get("sequence")
        last = self.kept_ns.get(name)
        if every > 1 and (
            (seq is not None and int(seq) % every)  # by sequence: exact, jitter-proof
            or (seq is None and last is not None and ns - last < 0.75e9 / self.record_fps)
        ):
            self.skipped[name] = self.skipped.get(name, 0) + 1
            return
        self.kept_ns[name] = ns
        self.counts[f"{name}_color"] = self.counts.get(f"{name}_color", 0) + 1
        msg = CompressedImage()
        stamp_ns(msg.header, ns)
        msg.header.frame_id = f"{name}_color"
        msg.format = "jpeg"
        msg.data = array.array("B", payload[start + n :])  # an array skips the per-byte check
        image_pub.publish(msg)
        self.seen[f"{name}_color"] = (ns * 1e-9, time.time())
        geometry = header.get("geometry") or {}
        if geometry:
            self.geometry[name] = geometry
        self.serials[name] = header.get("camera_serial")

    def publish_static(self) -> None:
        """/tf_static: the LiDAR mount chain from the odometry frame to the torso."""
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for parent, child, T in chain():
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
            tr, rot = t.transform.translation, t.transform.rotation
            tr.x, tr.y, tr.z = (float(v) for v in T[:3, 3])
            rot.x, rot.y, rot.z, rot.w = (float(v) for v in quat_from_R(T[:3, :3]))
            transforms.append(t)
        self.static.publish(TFMessage(transforms=transforms))

    def spin_once(self):
        for sock, _ in self.poller.poll(100):
            self.publish_frame(*self.streams[sock], sock.recv())
        now = time.time()
        if now - self.last_static > 10.0:
            self.publish_static()
            cameras = {
                name: {
                    "serial": self.serials.get(name),
                    "color": f"/homebody/{name}/color/image/compressed",
                    "depth": None,
                    "color_frame": f"{name}_color",
                    "depth_frame": None,
                    "timestamp_kind": "receiver_decode_time",
                    "tied_to_robot": False,
                    "geometry": self.geometry.get(name),
                }
                for name in self.names
            }
            self.cameras.publish(String(data=json.dumps(cameras)))
            self.last_static = now
        if now - self.last_live > 2.0:
            dt = now - self.last_live if self.last_live else 2.0
            counts, self.counts = self.counts, {}  # on_odom writes from the spin thread: swap first
            live = {k: round(v / dt, 1) for k, v in counts.items()}
            with open("/bags/.bridge_live.json", "w") as fh:
                json.dump({"wall_s": now, "fps": live, "skipped": self.skipped}, fh)
            os.replace("/bags/.bridge_live.json", "/bags/bridge_live.json")
            self.last_live = now
        if now - self.last_clock > 1.0:
            offsets = {
                what: {"stamp_s": s, "received_s": r, "wall_minus_stamp_s": round(r - s, 4)}
                for what, (s, r) in list(self.seen.items())
            }
            self.clock.publish(String(data=json.dumps({"wall_s": now, **offsets})))
            self.last_clock = now
        if now - self.last_state > 1.0 and os.path.exists(STATE):
            with open(STATE) as state:
                self.state.publish(String(data=state.read()))
            self.last_state = now


def spin(node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:  # the container is stopping; main() sees rclpy.ok() drop
        return


def main():
    ap = argparse.ArgumentParser(description="The phone stream and frame/clock provenance as ROS topics")
    ap.add_argument("--iphone", required=True, help="the phone receiver's zmq endpoint")
    ap.add_argument("--record-fps", type=float, required=True, help="the recorded phone frame rate")
    args = ap.parse_args()
    rclpy.init()
    node = Bridge([("iphone", args.iphone)], args.record_fps)
    threading.Thread(target=spin, args=(node,), daemon=True).start()
    while rclpy.ok():
        node.spin_once()


if __name__ == "__main__":
    main()
