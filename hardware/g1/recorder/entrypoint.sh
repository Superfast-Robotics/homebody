#!/usr/bin/env bash
# Inside the recorder container: the colour bridge (given this script's arguments) in the
# background, the recorder as the process docker stops with SIGINT (so the mcap summary and
# metadata.yaml are written).
set -eo pipefail          # no -u: ROS setup files read unset variables
source /opt/ros/humble/setup.bash
[ -f /opt/unitree_msgs/setup.bash ] && source /opt/unitree_msgs/setup.bash
[ -f /opt/superodom_ws/install/setup.bash ] && source /opt/superodom_ws/install/setup.bash
python3 /rec/color_bridge.py "$@" &
mapfile -t TOPICS < <(grep -v '^\s*#' /conf/topics.txt | sed '/^\s*$/d')
exec ros2 bag record -o /bags/ws -s mcap --storage-config-file /conf/mcap.yaml \
    -d 300 --max-cache-size 268435456 -p 200 \
    --qos-profile-overrides-path /conf/qos.yaml "${TOPICS[@]}"
