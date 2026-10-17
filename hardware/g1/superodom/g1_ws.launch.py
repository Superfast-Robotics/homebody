"""SuperOdom on the workstation, fed by the G1's MID-360 over the robot LAN.

Runs inside the headless SuperOdom image (amd64), mounted read-only under /run/superodom, on the
robot's own DDS domain (0). Every output is isolated under a prefix so it can run beside a
robot-resident engine without either seeing the other:
    engine topics   <prefix>/laser_odometry, <prefix>/registered_scan, ... (PROJECT_NAME)
    normalizer      <prefix>/laser_odometry_aligned, <prefix>/registered_scan_aligned, ...
    TF              /tf -> <prefix>/tf, /tf_static -> <prefix>/tf_static (remapped on
                    every node: the normalizer's sensor->gravity lookup must never read
                    the robot engine's tree)
    frames          ws_map / ws_sensor (yaml), the child "gravity" is hard-coded in SuperOdom
The engine parameters are hardware/g1/superodom/g1_mid360.yaml itself, mounted read-only; this
file overrides only what isolation needs (PROJECT_NAME, the frames, the input topics).
run_superodom_ws.sh passes laser_topic/imu_topic for the workstation's own rescale
(superodom_ws_bridge.py).
    ros2 launch /run/superodom/g1_ws.launch.py config_file:=/run/superodom/config.yaml \
        normalizer_file:=/run/superodom/g1_frame_normalizer_ws.yaml
"""
import os

import launch_ros
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "/ws"


def generate_launch_description():
    calib_default = os.path.join(get_package_share_directory("super_odometry"),
                                 "config/livox/livox_mid360_calibration.yaml")
    args = [
        DeclareLaunchArgument("config_file", default_value="/run/superodom/config.yaml",
                              description="hardware/g1/superodom/g1_mid360.yaml, mounted"),
        DeclareLaunchArgument("calibration_file", default_value=calib_default),
        DeclareLaunchArgument("normalizer_file",
                              default_value=os.path.join(_HERE, "g1_frame_normalizer_ws.yaml")),
        DeclareLaunchArgument("laser_topic", default_value="/utlidar/cloud_livox_mid360_sec"),
        DeclareLaunchArgument("imu_topic", default_value="/utlidar/imu_livox_mid360_sec"),
    ]
    common = [LaunchConfiguration("config_file"),
              {"calibration_file": LaunchConfiguration("calibration_file"),
               "laser_topic": LaunchConfiguration("laser_topic"),
               "imu_topic": LaunchConfiguration("imu_topic"),
               "PROJECT_NAME": PREFIX,
               # the frames: g1_frame_normalizer_ws.yaml names the same two
               "world_frame": "ws_map", "world_frame_rot": "ws_map_rot",
               "sensor_frame": "ws_sensor", "sensor_frame_rot": "ws_sensor_rot"}]
    tf_remap = [("/tf", PREFIX + "/tf"), ("/tf_static", PREFIX + "/tf_static")]
    engine = [
        Node(package="super_odometry", executable=exe, output="screen",
             parameters=common, remappings=tf_remap)
        for exe in ("feature_extraction_node", "laser_mapping_node", "imu_preintegration_node")
    ]
    normalizer = Node(
        package="super_odometry", executable="frame_normalizer_node", output="screen",
        parameters=[LaunchConfiguration("normalizer_file")],
        remappings=tf_remap + [
            # the two path topics the normalizer names outright
            ("/state_estimation_aligned_path", PREFIX + "/state_estimation_aligned_path"),
            ("/laser_odometry_aligned_path", PREFIX + "/laser_odometry_aligned_path"),
        ])
    return LaunchDescription([launch_ros.actions.SetParameter(name="use_sim_time", value="false"),
                              *args, *engine, normalizer])
