"""Time-field rescaler for the G1's MID-360 PointCloud2.

The driver's per-point `time` is float32 relative nanoseconds (0..1.0e8 over a 100 ms scan);
SuperOdom's velodyne parser expects relative seconds. This node republishes the identical cloud
with time *= 1e-9 (fields, layout, stamps and frame_id unchanged; a RELIABLE publisher so both
best-effort and reliable subscribers can connect), and the IMU with its acceleration scaled from
g to m/s^2 (the driver reports |acc| = 1 at rest; the engine preintegrates against g_norm):

    /utlidar/cloud_livox_mid360  ->  /utlidar/cloud_livox_mid360_sec
    /utlidar/imu_livox_mid360    ->  /utlidar/imu_livox_mid360_sec   (acc *= 9.80511)

It works for the live robot and for bag replay, and publishes only the _sec topics.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2

G_M_S2 = 9.80511   # the engine's own g_norm (superodom/g1_mid360.yaml)


def scale_imu(msg, scale):
    """The IMU sample with linear acceleration multiplied by `scale`; stamps, frame and
    angular velocity untouched (the gyro already reports rad/s)."""
    msg.linear_acceleration.x = float(msg.linear_acceleration.x) * scale
    msg.linear_acceleration.y = float(msg.linear_acceleration.y) * scale
    msg.linear_acceleration.z = float(msg.linear_acceleration.z) * scale
    return msg


class _Cdr:
    """A reader over a CDR (XCDR1, little-endian) payload: the 4-byte encapsulation header,
    then primitives aligned to their size relative to the body."""

    def __init__(self, payload: bytes):
        if payload[:2] != b"\x00\x01":
            raise ValueError("not little-endian CDR")
        self.b = payload
        self.i = 4

    def align(self, n: int):
        body = self.i - 4
        self.i += (-body) % n

    def u8(self) -> int:
        v = self.b[self.i]
        self.i += 1
        return v

    def u32(self) -> int:
        self.align(4)
        v = int.from_bytes(self.b[self.i:self.i + 4], "little")
        self.i += 4
        return v

    def string(self) -> str:
        n = self.u32()                      # length including the NUL
        s = self.b[self.i:self.i + n - 1].decode()
        self.i += n
        return s


def cloud_layout(payload: bytes) -> tuple[int, int, int, int]:
    """(data offset, data length, point_step, time-field offset) of a serialized PointCloud2,
    read from its CDR bytes; ValueError when the message has no float32 'time' field."""
    c = _Cdr(payload)
    c.u32(); c.u32(); c.string()            # header: stamp.sec, stamp.nanosec, frame_id
    c.u32(); c.u32()                        # height, width
    time_off = None
    for _ in range(c.u32()):                # fields
        name = c.string()
        off, datatype, _count = c.u32(), c.u8(), c.u32()
        if name == "time":
            if datatype != 7:               # FLOAT32
                raise ValueError("'time' is not float32")
            time_off = off
    if time_off is None:
        raise ValueError("no 'time' field")
    c.u8()                                  # is_bigendian
    point_step, _row_step = c.u32(), c.u32()
    data_len = c.u32()
    return c.i, data_len, point_step, time_off


def rescale_cdr(payload: bytes, scale: float) -> tuple[bytes, float]:
    """The serialized cloud with every point's 'time' multiplied by `scale`, and the last
    point's time, without deserializing the cloud."""
    data_off, data_len, point_step, time_off = cloud_layout(payload)
    out = bytearray(payload)
    n = data_len // point_step
    times = np.ndarray(shape=(n,), dtype="<f4", buffer=out, offset=data_off + time_off,
                       strides=(point_step,))
    times *= np.float32(scale)
    return bytes(out), float(times.max()) if n else 0.0


class TimeRescale(Node):
    def __init__(self, topic_in, topic_out, scale, clouds: bool = True):
        super().__init__("mid360_time_rescale" if clouds else "mid360_imu_rescale")
        self.scale = scale
        self.n = 0
        self._cost_ms = []
        self._warned = False
        self._imu_pub = None
        self._imu_n = 0
        if not clouds:
            return
        pub_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(PointCloud2, topic_out, pub_qos)
        self._topic_in = topic_in
        self._sub = self._subscribe()
        self.get_logger().info(f"{topic_in} -> {topic_out} (time *= {scale:g}, raw CDR, "
                               "RELIABLE reader)")
        # The driver's publisher comes and goes (its stream stops, its stamps step) and an
        # existing subscription can stay silent afterwards. Silent past SILENT_S: re-subscribe;
        # past RESUBSCRIBES of that: exit for the respawn loop.
        self._last_rx = time.monotonic()
        self._resubscribes = 0
        self.create_timer(1.0, self._check_silence)

    def add_imu(self, topic_in, topic_out, acc_scale):
        self._acc_scale = float(acc_scale)
        self._imu_pub = self.create_publisher(Imu, topic_out, qos_profile_sensor_data)
        self._imu_sub = self.create_subscription(Imu, topic_in, self._imu_cb,
                                                 qos_profile_sensor_data)
        self.get_logger().info(f"{topic_in} -> {topic_out} (acc *= {self._acc_scale})")

    def _imu_cb(self, msg):
        self._imu_pub.publish(scale_imu(msg, self._acc_scale))
        self._imu_n += 1
        if self._imu_n % 2000 == 1:
            a = msg.linear_acceleration
            self.get_logger().info(f"republished {self._imu_n} imu samples (last |acc| "
                                   f"{(a.x * a.x + a.y * a.y + a.z * a.z) ** 0.5:.2f} m/s^2)")

    # A re-subscription in the same process does not recover; a fresh process does. Exit on
    # the first silence, for the respawn.
    SILENT_S = 3.0
    RESUBSCRIBES = 0

    def _check_silence(self):
        quiet = time.monotonic() - self._last_rx
        if quiet < self.SILENT_S:
            return
        if self._resubscribes >= self.RESUBSCRIBES:
            self.get_logger().error(
                f"no cloud for {quiet:.0f}s after {self._resubscribes} re-subscriptions — "
                "exiting for the respawn loop")
            raise SystemExit(3)
        self._resubscribes += 1
        self.get_logger().warn(f"no cloud for {quiet:.0f}s — re-subscribing "
                               f"({self._resubscribes}/{self.RESUBSCRIBES})")
        self.destroy_subscription(self._sub)
        self._sub = self._subscribe()
        self._last_rx = time.monotonic()

    # RELIABLE, like the driver's writer: a lost fragment of a 441 KB cloud is re-sent instead
    # of poisoning Cyclone's reassembly.
    CLOUD_QOS = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_LAST)

    def _subscribe(self):
        # Raw serialized bytes, never a Python PointCloud2: deserializing costs most of a
        # core at 10 Hz.
        return self.create_subscription(PointCloud2, self._topic_in, self._cb_raw,
                                        self.CLOUD_QOS, raw=True)

    def _cb_raw(self, payload: bytes):
        t0 = time.monotonic()
        self._last_rx = t0
        self._resubscribes = 0
        try:
            out, t_max = rescale_cdr(payload, self.scale)
        except (ValueError, IndexError) as exc:
            if not self._warned:
                self.get_logger().warn(f"raw rescale failed ({exc}) — deserializing instead")
                self._warned = True
            self._cb(deserialize_message(payload, PointCloud2))
            return
        self.pub.publish(out)
        self._account(t0, t_max)

    def _account(self, t0: float, t_max: float) -> None:
        self.n += 1
        self._cost_ms.append(1000.0 * (time.monotonic() - t0))
        if self.n % 100 == 1:
            cost = self._cost_ms[-100:]
            self.get_logger().info(
                f"republished {self.n} clouds (last time-max {t_max:.4f}s; "
                f"callback mean {sum(cost) / len(cost):.1f} ms, max {max(cost):.1f} ms)")
            self._cost_ms = self._cost_ms[-100:]

    def _cb(self, msg: PointCloud2):
        t0 = time.monotonic()
        self._last_rx = t0
        self._resubscribes = 0
        tf = [f for f in msg.fields if f.name == "time"]
        if not tf or tf[0].datatype != 7:          # 7 = FLOAT32
            if not self._warned:
                self.get_logger().warn("no float32 'time' field — passing through")
                self._warned = True
            self.pub.publish(msg)
            return
        off = tf[0].offset
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).copy()
        pts = buf.reshape(-1, msg.point_step)
        t = pts[:, off:off + 4].copy().view(np.float32) * np.float32(self.scale)
        pts[:, off:off + 4] = t.view(np.uint8).reshape(-1, 4)
        msg.data = buf.tobytes()
        self.pub.publish(msg)
        self._account(t0, float(t.max()))


def main():
    ap = argparse.ArgumentParser(description="PointCloud2 time-field rescaler")
    ap.add_argument("--in", dest="tin", default="/utlidar/cloud_livox_mid360")
    ap.add_argument("--out", dest="tout", default="/utlidar/cloud_livox_mid360_sec")
    ap.add_argument("--scale", type=float, default=1e-9)
    ap.add_argument("--imu-in", dest="imu_in", default="/utlidar/imu_livox_mid360")
    ap.add_argument("--imu-out", dest="imu_out", default="/utlidar/imu_livox_mid360_sec")
    ap.add_argument("--acc-scale", dest="acc_scale", type=float, default=G_M_S2,
                    help="g -> m/s^2; 0 disables the IMU republish")
    ap.add_argument("--imu-only", dest="imu_only", action="store_true",
                    help="republish the IMU only: its own process, so 200 Hz of samples "
                         "never queue behind a 10 Hz cloud")
    args = ap.parse_args()
    rclpy.init()
    node = TimeRescale(args.tin, args.tout, args.scale, clouds=not args.imu_only)
    if args.acc_scale:
        node.add_imu(args.imu_in, args.imu_out, args.acc_scale)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
