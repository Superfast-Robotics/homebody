"""Run inside an isolated container: check binaries and synthetic scan/odom flow.

No robot devices or live inputs are needed. The README supplies the isolated
Docker invocation; this is build validation, not localization qualification.
"""

import argparse
import json
import math
import os
import signal
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster


def check_binaries():
    folder = Path("/opt/superodom_ws/install/super_odometry/lib/super_odometry")
    names = ("feature_extraction_node", "laser_mapping_node",
             "imu_preintegration_node", "frame_normalizer_node")
    for name in names:
        path = folder / name
        assert os.access(path, os.X_OK), path
        result = subprocess.run(["ldd", str(path)], check=True, text=True, capture_output=True)
        assert "not found" not in result.stdout, result.stdout
    source = Path("/opt/superodom_ws/src/SuperOdom")
    lock = json.loads(Path("/checks/source-lock.json").read_text())
    repositories = {"superodom": source, "sophus": Path("/opt/sources/Sophus"),
                    "gtsam": Path("/opt/sources/gtsam")}
    if Path("/opt/sources/unitree_ros2").exists():
        repositories["unitree_ros2"] = Path("/opt/sources/unitree_ros2")
    for name, path in repositories.items():
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           text=True).strip()
        assert revision == lock[name]["revision"], (name, revision)
    diff = subprocess.check_output(["git", "-C", str(source), "diff", "--numstat"], text=True)
    assert diff.strip() == "1\t1\tsuper_odometry/src/imuPreintegration_node.cpp", diff
    assert "SingleThreadedExecutor executor;" in (
        source / "super_odometry/src/imuPreintegration_node.cpp").read_text()
    return names


def check_normalizer():
    rclpy.init()
    node = Node("homebody_build_check")
    received = {}
    subscriptions = [node.create_subscription(kind, "/ws/" + topic + "_aligned",
                     lambda msg, key=topic: received.__setitem__(key, msg), 10)
                     for kind, topic in ((Odometry, "laser_odometry"),
                                         (PointCloud2, "registered_scan"))]
    odom_pub = node.create_publisher(Odometry, "/ws/laser_odometry", 10)
    scan_pub = node.create_publisher(PointCloud2, "/ws/registered_scan", 10)
    broadcaster = StaticTransformBroadcaster(node)
    transform = TransformStamped()
    transform.header.frame_id, transform.child_frame_id = "ws_sensor", "gravity"
    transform.transform.rotation.w = 1.0
    transform.header.stamp = node.get_clock().now().to_msg()
    broadcaster.sendTransform(transform)
    odom = Odometry()
    odom.header.frame_id, odom.child_frame_id = "ws_map", "ws_sensor"
    odom.header.stamp = node.get_clock().now().to_msg()
    odom.pose.pose.orientation.w = 1.0
    odom.pose.pose.position.x, odom.pose.pose.position.y = 1.25, -0.5
    scan = PointCloud2()
    scan.header = odom.header
    scan.height, scan.width, scan.point_step, scan.row_step = 1, 2, 16, 32
    scan.is_dense = True
    scan.fields = [PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
                   for index, name in enumerate(("x", "y", "z", "intensity"))]
    scan.data = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 8.0)
    binary = "/opt/superodom_ws/install/super_odometry/lib/super_odometry/frame_normalizer_node"
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([binary, "--ros-args", "--params-file",
                                    "/checks/g1_frame_normalizer_ws.yaml"], stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and len(received) < 2:
                assert process.poll() is None, "frame normalizer exited"
                odom_pub.publish(odom)
                scan_pub.publish(scan)
                rclpy.spin_once(node, timeout_sec=0.1)
            assert len(received) == 2, f"Missing aligned outputs: {sorted(received)}"
            output = received["laser_odometry"]
            assert output.header == odom.header
            assert output.child_frame_id == "gravity"
            assert math.isclose(output.pose.pose.position.x, 1.25, abs_tol=1e-6)
            assert math.isclose(output.pose.pose.position.y, -0.5, abs_tol=1e-6)
            output = received["registered_scan"]
            assert output.header == scan.header and output.width == 2
            offsets = {field.name: field.offset for field in output.fields}
            input_offsets = {field.name: field.offset for field in scan.fields}
            for point in range(2):
                for component in ("x", "y", "z", "intensity"):
                    value = struct.unpack_from("<f", output.data,
                                               point * output.point_step + offsets[component])[0]
                    expected = struct.unpack_from("<f", scan.data,
                                                  point * 16 + input_offsets[component])[0]
                    assert math.isclose(value, expected, abs_tol=1e-6), (component, value, expected)
            return {"scan_points": output.width, "timestamps_preserved": True}
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.seek(0)
            print(log.read(), end="")
            del subscriptions
            node.destroy_node()
            rclpy.shutdown()


def check_recorder():
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from unitree_api.msg import Request
    from unitree_go.msg import LowState as GoState
    from unitree_hg.msg import LowState

    assert Request() is not None and GoState() is not None
    with tempfile.TemporaryDirectory() as folder:
        options = rosbag2_py.StorageOptions(uri=str(Path(folder) / "bag"), storage_id="mcap")
        convert = rosbag2_py.ConverterOptions("cdr", "cdr")
        writer = rosbag2_py.SequentialWriter()
        writer.open(options, convert)
        writer.create_topic(rosbag2_py.TopicMetadata(
            name="/test/lowstate", type="unitree_hg/msg/LowState", serialization_format="cdr"))
        payload = serialize_message(LowState())
        writer.write("/test/lowstate", payload, 1_000_000_000)
        del writer
        reader = rosbag2_py.SequentialReader()
        reader.open(options, convert)
        name, actual, stamp = reader.read_next()
        assert (name, actual, stamp) == ("/test/lowstate", payload, 1_000_000_000)
        assert deserialize_message(actual, LowState) == LowState()
        assert not reader.has_next()
    return {"unitree_message_packages": 3, "mcap_cdr_roundtrip": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", action="store_true")
    args = parser.parse_args()
    result = {"executables": check_binaries(), "normalizer": check_normalizer()}
    if args.recorder:
        result["recorder"] = check_recorder()
    print(json.dumps(result, sort_keys=True))
