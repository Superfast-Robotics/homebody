#!/usr/bin/env bash
# The exploration bag: raw LiDAR + SuperOdom + nav + Unitree state + iPhone RGB and frame/clock
# provenance + explorer state, one mcap per 300 s, in a detached container (outside the user
# session scope). Values come from RUN_DIR/meta/settings.json; build the image with
# hardware/g1/superodom/build.sh recorder.
#
#   record.sh start RUN_DIR      preflight, then record into RUN_DIR/ws
#   record.sh stop  RUN_DIR      SIGINT (the mcap summary + metadata.yaml are written), then check
#   record.sh check RUN_DIR      counts, rates, max gaps and stamp offsets per topic
#   record.sh capture RUN_DIR    closed iPhone capture manifest and health report
#   record.sh status             one line
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
PY="${HOMEBODY_PYTHON:-python3}"
NAME=homebody_bag_ws
# setting RUN GROUP.KEY: one value of the run's resolved settings (integral numbers as integers)
setting() {
    "$PY" -c 'import json, sys
group, key = sys.argv[2].split(".")
value = json.load(open(sys.argv[1]))[group][key]
print(int(value) if isinstance(value, float) and value.is_integer() else value)' "$1/meta/settings.json" "$2"
}

dds_xml() {
    local out="$1/cyclonedds.xml"
    [[ "$IFACE" =~ ^[a-zA-Z0-9_.:-]+$ ]] || { echo "robot network interface required" >&2; return 1; }
    sed "s/@ROBOT_INTERFACE@/$IFACE/" "$REPO/hardware/g1/superodom/cyclonedds.xml" > "$out"
    echo "$out"
}

case "${1:-}" in
start)
    RUN="${2:?record.sh start RUN_DIR}"
    mkdir -p "$RUN/explore" "$RUN/meta"
    IMAGE="$(setting "$RUN" recording.image)"; MIN_FREE_GB="$(setting "$RUN" recording.min_free_gb)"
    IFACE="$(setting "$RUN" robot.iface)"; ENDPOINT="$(setting "$RUN" iphone.endpoint)"
    FPS="$(setting "$RUN" recording.color_fps)"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
        echo "refusing to record: image $IMAGE is missing — fix: hardware/g1/superodom/build.sh recorder" >&2; exit 1; }
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "refusing to record: container $NAME exists (a bag may still be open) — fix: $0 stop <its RUN_DIR>" >&2
        exit 1
    fi
    free_gb=$(( $(df --output=avail -k "$RUN" | tail -1) / 1024 / 1024 ))
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "refusing to record: $free_gb GB free under $RUN (need $MIN_FREE_GB; LiDAR + iPhone; full-run capacity depends on duration and scene)" >&2
        exit 1
    fi
    [ -e "$RUN/ws" ] && { echo "refusing to record: $RUN/ws already exists" >&2; exit 1; }
    [ -n "$ENDPOINT" ] || { echo "iphone endpoint missing" >&2; exit 1; }
    XML="$(dds_xml "$RUN")"
    cp "$HERE/topics.txt" "$RUN/meta/record_topics.txt"
    python3 - "$RUN/meta/record_interval.json" <<'PY_START'
import json, sys, time
from datetime import datetime, timezone
ns = time.time_ns()
utc = datetime.fromtimestamp(ns // 1_000_000_000, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
json.dump(dict(start_s=ns / 1e9, start_ns=str(ns), started_at_utc=f'{utc}.{ns % 1_000_000_000:09d}Z',
               end_s=None), open(sys.argv[1], 'w'))
PY_START
    "$PY" "$HERE/capture_contract.py" --initialize "$RUN"
    docker run -d --name "$NAME" --net=host --stop-signal SIGINT --stop-timeout 90 \
        --log-opt max-size=20m --user "$(id -u):$(id -g)" \
        -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        -e CYCLONEDDS_URI=file:///conf/cyclonedds.xml -e HOME=/tmp \
        -v "$HERE/color_bridge.py:/rec/color_bridge.py:ro" -v "$HERE/entry.sh:/rec/entry.sh:ro" \
        -v "$XML:/conf/cyclonedds.xml:ro" -v "$HERE/qos.yaml:/conf/qos.yaml:ro" \
        -v "$HERE/mcap.yaml:/conf/mcap.yaml:ro" -v "$RUN/meta/record_topics.txt:/conf/topics.txt:ro" \
        -v "$RUN/explore:/explore:ro" -v "$RUN:/bags" \
        "$IMAGE" --iphone "$ENDPOINT" --record-fps "$FPS" >/dev/null
    sleep 5
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
        echo "refusing: the recorder exited at once:" >&2; docker logs "$NAME" 2>&1 | tail -20 >&2
        docker rm "$NAME" >/dev/null 2>&1 || true; exit 1; }
    echo "[record] recording into $RUN/ws ($free_gb GB free)" ;;
stop)
    RUN="${2:?record.sh stop RUN_DIR}"
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "[record] stopping (SIGINT, up to 90 s for the mcap summary)"
        docker stop -t 90 "$NAME" >/dev/null || true
        # A docker stop whose client exited still ends the container daemon-side: wait past the
        # 90 s, then kill whatever still writes.
        timeout 100 docker wait "$NAME" >/dev/null 2>&1 || true
        if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
            echo "[record] $NAME is still running after the stop: killing it (mcap recover salvages the bag)" >&2
            docker kill "$NAME" >/dev/null 2>&1 || true
        fi
        docker logs "$NAME" > "$RUN/recorder.log" 2>&1 || true
        docker rm "$NAME" >/dev/null 2>&1 || true
    fi
    if [ -f "$RUN/meta/record_interval.json" ]; then
        python3 - "$RUN/meta/record_interval.json" <<'PY_INTERVAL'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1]); data = json.loads(p.read_text())
if data.get("end_s") is None:
    data["end_s"] = time.time(); p.write_text(json.dumps(data) + "\n")
PY_INTERVAL
    fi
    "$0" check "$RUN" ;;
check)
    RUN="${2:?record.sh check RUN_DIR}"
    [ -d "$RUN/ws" ] || { echo "no bag at $RUN/ws" >&2; exit 1; }
    "$0" reindex "$RUN"
    mkdir -p "$RUN/quality"
    IMAGE="$(setting "$RUN" recording.image)"; OPTIONAL="$(setting "$RUN" recording.optional_topics)"
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e "OPTIONAL=$OPTIONAL" \
        -v "$RUN:/bags:ro" -v "$HERE:/rec:ro" -v "$RUN/quality:/quality" \
        --entrypoint /bin/bash "$IMAGE" -c \
        'source /opt/ros/humble/setup.bash; [ ! -f /opt/superodom_ws/install/setup.bash ] || source /opt/superodom_ws/install/setup.bash; [ -f /opt/unitree_msgs/setup.bash ] && source /opt/unitree_msgs/setup.bash; ros2 bag info /bags/ws; topics=/bags/meta/record_topics.txt; [ -f "$topics" ] || topics=/rec/topics.txt; python3 /rec/postcheck.py --optional "$OPTIONAL" /bags/ws "$topics" /quality/bag_inspection.json' \
        | tee "$RUN/bag_check.txt" ;;
capture)
    "$PY" "$HERE/capture_contract.py" "${2:?record.sh capture RUN_DIR}" ;;
reindex)  # a bag whose recorder died has no metadata.yaml: rebuild it so the tools can read it
    RUN="${2:?record.sh reindex RUN_DIR}"
    [ -d "$RUN/ws" ] && [ ! -f "$RUN/ws/metadata.yaml" ] || exit 0
    IMAGE="$(setting "$RUN" recording.image)"
    echo "[record] $RUN/ws has no metadata.yaml (its recorder died): reindexing"
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$RUN:/bags" --entrypoint /bin/bash "$IMAGE" -c \
        'source /opt/ros/humble/setup.bash; ros2 bag reindex /bags/ws mcap' ;;
export)
    RUN="${2:?record.sh export RUN_DIR}"
    [ -d "$RUN/ws" ] || { echo "no bag at $RUN/ws" >&2; exit 1; }
    "$0" reindex "$RUN"
    mkdir -p "$RUN/video"
    IMAGE="$(setting "$RUN" recording.image)"; EXPORT_FPS="$(setting "$RUN" recording.export_fps)"
    SELF_CLEAR_M="$(setting "$RUN" lidar.self_clear_m)"
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$RUN:/bags" -v "$HERE:/rec:ro" \
        --entrypoint /bin/bash "$IMAGE" -c \
        "source /opt/ros/humble/setup.bash; exec python3 /rec/export_mp4.py /bags/ws /bags/video --fps $EXPORT_FPS --self-clear-m $SELF_CLEAR_M" \
        | tee "$RUN/video/export.txt" ;;
status)
    docker ps --filter "name=^${NAME}$" --format '{{.Names}} {{.Status}}' | grep . || echo "not recording" ;;
*)
    sed -n '2,12p' "$0"; exit 2 ;;
esac
