"""The G1's MID-360 topics, made SuperOdom-ready on the workstation.

Runs inside the SuperOdom image next to the engine (rclpy and numpy are in the image), reading
the robot's raw driver topics over the LAN and republishing them under /ws so nothing collides
with the robot-resident chain:

    /utlidar/cloud_livox_mid360  (PointCloud2, time f32 ns, robot driver)
        -> /ws/utlidar/cloud_livox_mid360_sec   time *= 1e-9, bytes otherwise identical
    /utlidar/imu_livox_mid360    (Imu, |acc| = 1 g)
        -> /ws/utlidar/imu_livox_mid360_sec     acc *= 9.80511 (m/s^2, the engine's g_norm)

That is time_rescale_node.py's job with different topic names, so the node is imported, not
copied: mount the two files side by side. The engine reads the _sec pair through its
PointCloud2 ("velodyne") parser.

    python3 superodom_ws_bridge.py --hold-margin-s 0.01 --hold-max-s 0.08   # the clouds
    python3 superodom_ws_bridge.py --imu-only                               # the IMU process
"""
from __future__ import annotations

import argparse
import collections
import os
import struct
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_rescale_node import G_M_S2, TimeRescale, rescale_cdr

PREFIX = "/ws"


def _stamp(payload: bytes) -> float:
    """header.stamp of a serialized message: the first two u32 after the CDR encapsulation."""
    sec, nanosec = struct.unpack_from("<II", payload, 4)
    return sec + nanosec * 1e-9


class ImuCoveredRescale(TimeRescale):
    """The cloud lane, each cloud held until the IMU stream has passed its last point.

    The engine deskews a scan the moment it arrives and never checks that its IMU buffer
    reaches the scan's end; a point newer than the newest IMU sample takes a stale orientation,
    which a turning robot shows as the scan's tail thrown across the map. The driver's clouds
    usually land a few tens of milliseconds before the IMU sample that covers their end."""

    def __init__(self, topic_in, topic_out, scale, imu_in, margin_s, max_s):
        super().__init__(topic_in, topic_out, scale)
        self.margin_s, self.max_s = margin_s, max_s  # the IMU lane's own hop; a silent IMU never stalls
        self._held: collections.deque = collections.deque()  # (end stamp, receive time, payload)
        self._imu_stamp = 0.0
        self._hold_ms: list[float] = []
        self._timeouts = 0
        self._imu_watch = self.create_subscription(Imu, imu_in, self._on_imu,
                                                   qos_profile_sensor_data, raw=True)
        self.create_timer(0.005, self._flush)
        self.get_logger().info(
            f"clouds held until {imu_in} passes their last point "
            f"(+{1000 * margin_s:.0f} ms, at most {1000 * max_s:.0f} ms)")

    def _on_imu(self, payload: bytes):
        self._imu_stamp = max(self._imu_stamp, _stamp(payload))
        self._flush()

    def _cb_raw(self, payload: bytes):
        t0 = time.monotonic()
        self._last_rx = t0
        self._resubscribes = 0
        out, t_max = rescale_cdr(payload, self.scale)
        self._held.append((_stamp(payload) + t_max, t0, out))
        self._account(t0, t_max)
        self._flush()

    def _flush(self):
        now = time.monotonic()
        while self._held:
            end, t_rx, out = self._held[0]
            late = now - t_rx > self.max_s
            if self._imu_stamp < end + self.margin_s and not late:
                return
            self._held.popleft()
            self.pub.publish(out)
            self._timeouts += late
            self._hold_ms.append(1000.0 * (now - t_rx))
            if len(self._hold_ms) >= 100:
                self.get_logger().info(
                    f"held 100 clouds: mean {sum(self._hold_ms) / 100:.1f} ms, max "
                    f"{max(self._hold_ms):.1f} ms, {self._timeouts} released on the timeout so far")
                self._hold_ms = []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="tin", default="/utlidar/cloud_livox_mid360")
    ap.add_argument("--imu-in", dest="imu_in", default="/utlidar/imu_livox_mid360")
    ap.add_argument("--prefix", default=PREFIX, help="output namespace (never the robot's)")
    ap.add_argument("--imu-only", action="store_true",
                    help="the IMU republish alone, in its own process, so 200 Hz never queues "
                         "behind a 440 KB cloud")
    ap.add_argument("--acc-scale", type=float, default=G_M_S2)
    ap.add_argument("--hold-margin-s", type=float,
                    help="hold each cloud until the IMU passes its last point by this much")
    ap.add_argument("--hold-max-s", type=float, help="release a held cloud after this long regardless")
    args = ap.parse_args()
    if not args.prefix.startswith("/") or args.prefix.rstrip("/") == "":
        raise SystemExit("--prefix must be a non-root namespace such as /ws")
    if (args.hold_margin_s is None) != (args.hold_max_s is None):
        raise SystemExit("--hold-margin-s and --hold-max-s go together")
    prefix = args.prefix.rstrip("/")
    rclpy.init()
    cloud_out = f"{prefix}/utlidar/cloud_livox_mid360_sec"
    imu_out = f"{prefix}/utlidar/imu_livox_mid360_sec"
    node = (ImuCoveredRescale(args.tin, cloud_out, 1e-9, args.imu_in, args.hold_margin_s, args.hold_max_s)
            if args.hold_max_s is not None and not args.imu_only
            else TimeRescale(args.tin, cloud_out, 1e-9, clouds=not args.imu_only))
    if args.imu_only:
        node.add_imu(args.imu_in, imu_out, args.acc_scale)
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
