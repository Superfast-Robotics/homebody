#!/usr/bin/env bash
# SuperOdom on the workstation for the G1 (README.md here). Two containers from the
# homebody-superodom image, everything under /ws:
#   superodom_ws          the engine + frame normalizer (g1_ws.launch.py)
#   superodom_ws_bridge   the raw-topic rescale (superodom_ws_bridge.py, one process per stream)
#
#   run_superodom_ws.sh --iface IF --image IMAGE up [--hold-margin-s S --hold-max-s S]
#                                          start; the hold options pass to the cloud bridge
#   run_superodom_ws.sh down               stop both (stop-timeout 20 s, then rm)
#   run_superodom_ws.sh down-owned         stop only this HOMEBODY_WORKSPACE's containers
#   run_superodom_ws.sh --iface IF --image IMAGE status   one line, exit 0 only when /ws odometry is fresh
#   run_superodom_ws.sh logs               follow the engine
#   run_superodom_ws.sh --iface IF --image IMAGE probe <topic> [s]   read-only `ros2 topic hz`
# Nothing here reaches the robot's own chain: the containers publish /ws/* only, and the
# preflight refuses to start beside another /ws publisher.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
conf() { sed -n "s/^$1=//p" "$HERE/runtime.conf"; }
IFACE=""; IMAGE=""
while [ "${1:-}" = --iface ] || [ "${1:-}" = --image ]; do
    case "$1" in --iface) IFACE="${2:-}" ;; --image) IMAGE="${2:-}" ;; esac
    shift 2 || { echo "$1 needs a value" >&2; exit 2; }
done
NAME="$(conf container_name)"; DOMAIN="$(conf ros_domain_id)"
PREFIX="$(conf topic_prefix)"; STOP_S="$(conf stop_timeout_s)"
BRIDGE="${NAME}_bridge"
WORKSPACE="$(realpath -m -- "${HOMEBODY_WORKSPACE:-$REPO/runs}")"
LOGS="$WORKSPACE/superodom_ws"
OWNER_LABEL="homebody.exploration.workspace"

engine() {
    [[ "$IFACE" =~ ^[a-zA-Z0-9_.:-]+$ ]] || { echo "pass --iface: the robot network interface" >&2; return 1; }
    [ -n "$IMAGE" ] || { echo "pass --image: the SuperOdom image" >&2; return 1; }
}

dds_xml() {
    # Written atomically and only when it changes: it is bind-mounted into the running engine
    # and every probe, and a concurrent rewrite would hand Cyclone an empty file.
    local out="$LOGS/cyclonedds.runtime.xml" tmp
    mkdir -p "$LOGS"
    tmp="$(mktemp "$LOGS/.cyclonedds.XXXXXX")"
    sed "s/@ROBOT_INTERFACE@/$IFACE/" "$HERE/cyclonedds.xml" > "$tmp"
    if [ -f "$out" ] && cmp -s "$tmp" "$out"; then rm -f "$tmp"; else mv -f "$tmp" "$out"; fi
    echo "$out"
}

# Keep the configured DDS domain independent of other local ROS applications.
common_env=(-e "ROS_DOMAIN_ID=$DOMAIN" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            -e CYCLONEDDS_URI=file:///run/superodom/cyclonedds.xml)

up() {
    engine
    "$HERE/preflight_ws.sh" "$IFACE" "$IMAGE" "$NAME" "$PREFIX" "$(dds_xml)" "$DOMAIN"
    local xml; xml="$(dds_xml)"
    # Host ownership makes minimal-mode logs removable when its temporary workspace closes.
    mkdir -p "$LOGS/home" "$LOGS/ros"
    [[ -w "$LOGS/home" && -w "$LOGS/ros" ]] || {
        echo "SuperOdometry home/log directories must be writable by the launching user: $LOGS" >&2
        return 1
    }
    local runtime=(--user "$(id -u):$(id -g)" --label "$OWNER_LABEL=$WORKSPACE"
                   -e HOME=/run/homebody/home -e ROS_LOG_DIR=/run/homebody/ros
                   -v "$LOGS/home:/run/homebody/home" -v "$LOGS/ros:/run/homebody/ros")
    local laser="$PREFIX/utlidar/cloud_livox_mid360_sec" imu="$PREFIX/utlidar/imu_livox_mid360_sec"
    # Two processes: the 10 Hz clouds and the 200 Hz IMU never share one Python executor;
    # each respawns after its own silence exit.
    docker run -d --name "$BRIDGE" --net=host --stop-timeout "$STOP_S" --restart unless-stopped \
        --log-opt max-size=50m --log-opt max-file=3 "${common_env[@]}" "${runtime[@]}" \
        -v "$xml:/run/superodom/cyclonedds.xml:ro" \
        -v "$HERE/superodom_ws_bridge.py:/run/superodom/superodom_ws_bridge.py:ro" \
        -v "$HERE/time_rescale_node.py:/run/superodom/time_rescale_node.py:ro" \
        "$IMAGE" bash -c '
            respawn() { while true; do python3 /run/superodom/superodom_ws_bridge.py "$@" || true; sleep 1; done; }
            respawn --imu-only & respawn "$@"; wait' bridge "$@" >/dev/null
    docker run -d --name "$NAME" --net=host --stop-timeout "$STOP_S" --restart unless-stopped \
        --log-opt max-size=50m --log-opt max-file=3 "${common_env[@]}" "${runtime[@]}" \
        -v "$xml:/run/superodom/cyclonedds.xml:ro" \
        -v "$HERE/g1_ws.launch.py:/run/superodom/g1_ws.launch.py:ro" \
        -v "$HERE/g1_mid360.yaml:/run/superodom/config.yaml:ro" \
        -v "$HERE/g1_frame_normalizer_ws.yaml:/run/superodom/g1_frame_normalizer_ws.yaml:ro" \
        "$IMAGE" ros2 launch /run/superodom/g1_ws.launch.py \
            config_file:=/run/superodom/config.yaml \
            normalizer_file:=/run/superodom/g1_frame_normalizer_ws.yaml \
            "laser_topic:=$laser" "imu_topic:=$imu" >/dev/null
    echo "superodom_ws up (domain $DOMAIN on $IFACE, outputs under $PREFIX); " \
         "keep the robot still ~2 s (IMU gravity init)"
}

down() {
    local owned="${1:-}" names c metadata container_id owner failed=0
    if [ "$owned" = owned ] && [ -z "${HOMEBODY_WORKSPACE:-}" ]; then
        echo "down-owned requires HOMEBODY_WORKSPACE" >&2; return 1
    fi
    names="$(docker ps -a --format '{{.Names}}')" || return 1
    for c in "$NAME" "$BRIDGE"; do
        grep -qx "$c" <<< "$names" || continue
        metadata="$(docker inspect -f '{{.Id}}|{{index .Config.Labels "homebody.exploration.workspace"}}' "$c")" || {
            failed=1; continue;
        }
        container_id="${metadata%%|*}"; owner="${metadata#*|}"
        [ "$owned" != owned ] || [ "$owner" = "$WORKSPACE" ] || continue
        [[ "$container_id" =~ ^[a-f0-9]{64}$ ]] || {
            echo "invalid container identity for $c" >&2; failed=1; continue;
        }
        # Use the inspected identity: a replacement using the fixed name may have a new owner.
        if ! docker stop -t "$STOP_S" "$container_id" >/dev/null; then
            echo "failed to stop $c ($container_id)" >&2; failed=1; continue
        fi
        if ! docker rm "$container_id" >/dev/null; then
            echo "failed to remove $c ($container_id)" >&2; failed=1
        fi
    done
    [ "$failed" = 0 ] || return 1
    echo "superodom_ws ${owned:+workspace containers }down"
}

# Read-only: a 3 s subscriber inside the image.
status() {
    engine
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || {
        echo "superodom_ws: container down"; return 1; }
    docker run --rm --net=host "${common_env[@]}" \
        -v "$(dds_xml):/run/superodom/cyclonedds.xml:ro" \
        -v "$HERE/status_probe.py:/run/superodom/status_probe.py:ro" \
        "$IMAGE" python3 /run/superodom/status_probe.py "$PREFIX"
}

probe() {
    engine
    docker run --rm --net=host "${common_env[@]}" -v "$(dds_xml):/run/superodom/cyclonedds.xml:ro" \
        "$IMAGE" timeout -s INT "${2:-10}" ros2 topic hz "$1" --window 30
}

case "${1:-}" in
    up) shift; up "$@" ;;
    down) down ;;
    down-owned) down owned ;;
    status) status ;;
    logs) docker logs -f "$NAME" ;;
    probe) shift; probe "$@" ;;
    *) sed -n 2,14p "$0"; exit 2 ;;
esac
