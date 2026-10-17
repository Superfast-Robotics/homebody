"""Read-only health check of the workstation engine's inputs and outputs under one prefix."""
import sys
import time


def inputs(prefix):
    """The rescaled cloud and IMU topics the engine reads."""
    return prefix + "/utlidar/cloud_livox_mid360_sec", prefix + "/utlidar/imu_livox_mid360_sec"


def healthy(counts, last, now):
    return (all(counts[k] >= floor for k, floor in
                {"cloud": 15, "imu": 150, "odom": 15, "scan": 15, "state": 15}.items())
            and all(now - last.get(k, -1e9) < 1 for k in counts)
            and counts["health"] > 0)


def main():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, PointCloud2
    from std_msgs.msg import Bool

    prefix = sys.argv[1]
    cloud, imu = inputs(prefix)
    rclpy.init()
    node = Node("superodom_ws_status")
    counts = dict.fromkeys(("cloud", "imu", "odom", "scan", "state", "health"), 0)
    last = {}
    health = [False]

    def receive(key, message):
        counts[key] += 1
        last[key] = time.monotonic()
        if key == "health":
            health[0] = message.data

    for key, typ, topic in (
        ("cloud", PointCloud2, cloud), ("imu", Imu, imu),
        ("odom", Odometry, prefix + "/laser_odometry_aligned"),
        ("scan", PointCloud2, prefix + "/registered_scan_aligned"),
        ("state", Odometry, prefix + "/state_estimation_aligned"),
        ("health", Bool, prefix + "/state_estimation_health"),
    ):
        node.create_subscription(typ, topic, lambda msg, k=key: receive(k, msg),
                                 qos_profile_sensor_data)
    start = time.monotonic()
    while time.monotonic() - start < 3:
        rclpy.spin_once(node, timeout_sec=0.1)
    ok = healthy(counts, last, time.monotonic()) and health[0]
    rates = ", ".join(f"{k} {v/3:.0f} Hz" for k, v in counts.items())
    print("superodom_ws: " + rates + (" — LIVE" if ok else " — STALLED or unhealthy"))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
