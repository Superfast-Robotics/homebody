#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /opt/superodom_ws/install/setup.bash
exec "$@"
