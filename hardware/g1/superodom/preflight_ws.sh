#!/usr/bin/env bash
# What the workstation engine needs before `docker run` (see README.md here).
# Usage: preflight_ws.sh <iface> <image> <container> <prefix>; exit 1 names the first failure.
set -uo pipefail
IFACE="$1"; IMAGE="$2"; NAME="$3"; PREFIX="$4"; DDS_XML="${5:-}"; DOMAIN="${6:-0}"
fail() { echo "refusing superodom_ws: $1" >&2; echo "  fix: $2" >&2; exit 1; }
ip -o -4 addr show dev "$IFACE" 2>/dev/null | grep -q '192\.168\.123\.' \
    || fail "$IFACE has no 192.168.123.x address" "connect the robot ethernet or pass --iface"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || fail "image $IMAGE is not built" "run hardware/g1/superodom/build.sh odometry"
# A 440 KB cloud is 7 datagrams of 64 KB from a rescale node on this host (loopback bursts, no
# pacing): the kernel default of 212992 B drops most of them, so the socket buffer is raised.
RMEM=$(sysctl -n net.core.rmem_max)
[ "$RMEM" -ge 33554432 ] \
    || fail "net.core.rmem_max is $RMEM (< 32 MB)" "sudo install -m 644 hardware/g1/superodom/99-homebody-udp.conf /etc/sysctl.d/ && sudo sysctl --system"
docker ps -a --format '{{.Names}}' | grep -qx "$NAME" \
    && fail "container $NAME already exists (running or stopped)" "run_superodom_ws.sh down first"
# Two engines under one prefix would interleave two world frames on the bridge's topic. Query on
# the robot's NIC with the runtime xml (the bare image xml binds the first multicast NIC).
[ -n "$DDS_XML" ] && [ -f "$DDS_XML" ] \
    || fail "no runtime cyclonedds xml handed to the preflight ($DDS_XML)" "run through run_superodom_ws.sh"
# Another live engine under this prefix, not a name in discovery: a stopped engine's endpoints
# linger for the participant lease (~10 s), its data does not.
LIVE=$(docker run --rm -i --net=host -e "ROS_DOMAIN_ID=$DOMAIN" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        -e CYCLONEDDS_URI=file:///run/superodom/cyclonedds.xml -v "$DDS_XML:/run/superodom/cyclonedds.xml:ro" \
        "$IMAGE" python3 - "$PREFIX" <<'PY'
import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
p = sys.argv[1]; rclpy.init(); n = Node("superodom_ws_preflight"); c = [0]
n.create_subscription(Odometry, p + "/laser_odometry_aligned", lambda m: c.__setitem__(0, c[0] + 1), qos_profile_sensor_data)
t0 = time.time()
while time.time() - t0 < 2.0 and c[0] == 0:
    rclpy.spin_once(n, timeout_sec=0.1)
print(f"live={c[0]}"); rclpy.shutdown()
PY
)
case "$LIVE" in live=0) ;; live=*) fail "another engine is publishing $PREFIX/laser_odometry_aligned" \
        "find it (docker ps here; ros2 topic info -v $PREFIX/laser_odometry_aligned) and stop it" ;;
    *) fail "the domain probe did not run ($LIVE)" "docker run of $IMAGE failed; see above" ;; esac
echo "superodom_ws preflight: iface $IFACE, image $IMAGE, rmem_max $RMEM, $PREFIX free"
