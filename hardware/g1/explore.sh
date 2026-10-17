#!/usr/bin/env bash
# G1 exploration process owner. Full capture is on by default; STOP and Ctrl-C close this console.
#
#   homebody real2sim explore [--no-record] [--adopt]      then open the console
#
# Invoked as `explore.sh run|setup` with HOMEBODY_ROOT, HOMEBODY_PYTHON, HOMEBODY_WORKSPACE and
# HOMEBODY_RUN_DIR set; every value comes from $HOMEBODY_WORKSPACE/meta/settings.json.
#
# Normal launch starts the body host and opens hands. Supported robot: remote Y initializes,
# A starts AMO standing. Start exploration surveys and runs the requested task.
# STOP and Ctrl-C close the console and owned resources while AMO keeps balancing.
set -euo pipefail

REPO="${HOMEBODY_ROOT:?invoke homebody explore}"
cd "$REPO"
SETUP=0; [ "${1:-run}" = setup ] && SETUP=1
RUN_DIR="${HOMEBODY_WORKSPACE:?}"
SETTINGS="$RUN_DIR/meta/settings.json"
# NAME=group.key ...: shell assignments from the run's resolved settings (booleans as 1/0)
settings() {
    "${HOMEBODY_PYTHON:?}" -c 'import json, shlex, sys
groups = json.load(open(sys.argv[1]))
for pair in sys.argv[2:]:
    name, path = pair.split("=")
    group, key = path.split(".")
    value = groups[group][key]
    value = int(value) if isinstance(value, bool) or isinstance(value, float) and value.is_integer() else value
    print(f"{name}={shlex.quote(str(value))}")' "$SETTINGS" "$@"
}
eval "$(settings RECORD=exploration.record OBSERVE=exploration.observe_only ADOPT=slam.adopt \
    HOST=exploration.host PORT=exploration.port BUDGET_S=exploration.budget_s BAG_ROOT=exploration.output \
    IFACE=robot.iface SSH_TARGET=robot.ssh_target BODY_ADDRESS=body.address BODY_CORES=body.cores \
    LOCK=body.writer_lock_path IPHONE_INPUT=iphone.input_url IPHONE_ENDPOINT=iphone.endpoint \
    IPHONE_RELAY=iphone.relay_service IPHONE_MAX_AGE_S=iphone.max_age_s IPHONE_HWACCEL=iphone.hwaccel \
    SLAM_IMAGE=slam.image RECORDER_IMAGE=recording.image ZONE_SIZE_M=exploration.zone_size_m)"
# The password comes from the environment the CLI resolved, else configs/exploration.toml; never meta/.
SSHPASS_VALUE="$("${HOMEBODY_PYTHON:?}" -c 'import sys
from homebody.real2sim.exploration.settings import install, load, ssh_password
install(load(sys.argv[1])); print(ssh_password())' "$REPO")"
echo "[launch] observe-only=$OBSERVE SLAM=SuperOdometry on this workstation"
# ssh + sshpass: the cleanup thaws the robot's LiDAR watchdog over ssh, and meta/clocks.json
# samples the Orin's clock; without them the thaw fails silently and the watchdog stays frozen.
COMMANDS=(ip ssh sshpass)
[ "$SETUP" = 1 ] || COMMANDS+=(docker)
for command in "${COMMANDS[@]}"; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "refusing launch: required command '$command' is not installed" >&2
        echo "  fix: install it (ip: iproute2; docker: docker.io; ssh/sshpass: openssh-client, sshpass)" >&2
        exit 1
    }
done
mkdir -p "$RUN_DIR/meta" "$RUN_DIR/explore" "$RUN_DIR/phone"

# The persistent balancing owner has a separate operator-controlled lifecycle.
BODY=(env PYTHONPATH="$REPO/src:$REPO/vendor" "${HOMEBODY_PYTHON:?}" -m homebody.backends.g1.body_host
      --settings-json "$SETTINGS")

PHONE_PID=""
PHONE_MODE=disabled
stop_phone_receiver() {
    [ -n "$PHONE_PID" ] || return 0
    local requested_stop=0 receiver_status=0
    # Signal only this script's still-running job; other receivers are never hunted by name.
    if jobs -pr | grep -qx "$PHONE_PID"; then
        echo "[phone] closing launch-owned receiver and its final video segment"
        if kill -TERM "$PHONE_PID" 2>/dev/null; then requested_stop=1; fi
    fi
    wait "$PHONE_PID" 2>/dev/null || receiver_status=$?
    # The receiver reports a finalized interruption as 130. Do not excuse an earlier
    # crash or a failed encoder shutdown merely because cleanup has now been requested.
    if [ "$receiver_status" != 0 ] && ! { [ "$requested_stop" = 1 ] && [ "$receiver_status" = 130 ]; }; then
        echo "[phone] receiver exited with an error; recordings kept" >&2
        [ "$RECORD" = 0 ] || FINALIZE_FAILED=1
    fi
    PHONE_PID=""
}

# 0. One launcher at a time (pidfile, not process name): the body writer has a single owner. A second
#    launch is refused because the first one's cleanup may still be closing a bag.
LAUNCH_PID=/tmp/homebody-launch-explore.pid
for pidfile in "$LAUNCH_PID"; do
    OLD="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$OLD" ] && [ "$OLD" != "$$" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "refusing launch: another launcher ($pidfile, pid $OLD) is running — stop it first" >&2
        exit 1
    fi
done
echo "$$" > "$LAUNCH_PID"
FINALIZE_FAILED=0
# 0a. The bag. One cleanup on every exit path: it stops the console (which stops the explorer and
#    closes the body), then closes the bag with SIGINT so the mcap summary is written.
APP_PID=""
# Ctrl+\ is the one key here that takes the legs down: policy stopped at once, joints limp, host
# ended. Ctrl-C, a crash, a closed pane or a lost link leave the robot standing.
halt_now() {
    [ "$OBSERVE" != "1" ] || { echo "[launch] observation-only: no body halt sent"; return; }
    set +e
    printf '{"cmd":"halt","claim":"%s:launcher"}' "$(date +%s.%N)" \
        > "/dev/udp/${BODY_ADDRESS%:*}/${BODY_ADDRESS##*:}" 2>/dev/null || true
    echo "[launch] Ctrl+\\: HALT sent — policy stopped at once, every joint limp, the body host ends"
    "${BODY[@]}" halt || true
}
cleanup() {
    # The CLI and foreground shell can both receive the same interrupt. Mask duplicates
    # before any cleanup command: a second exit must not bypass detached-recorder closure.
    trap '' INT TERM HUP
    trap halt_now QUIT   # emergency halt still works during teardown; the bag still closes
    local code="$1"   # original status from the EXIT trap (130 = Ctrl-C, 131 = Ctrl+\)
    trap - EXIT
    # Ctrl-C is ignored during teardown so the bag is always closed; Ctrl+\ still halts the legs.
    set +e   # teardown runs to the end whatever a step returns
    echo "[launch] exiting with code $code at $(date '+%H:%M:%S'); observe-only=$OBSERVE"
    if [ "$OBSERVE" = 1 ]; then
        echo "[launch] observation-only teardown: closing recording; no body commands"
    else
        echo "[launch] teardown: Ctrl-C is ignored until the bag is closed; Ctrl+\\ halts the legs"
    fi
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        echo "[launch] stopping the console and closing its resources"
        kill -TERM "$APP_PID" 2>/dev/null || true
        for _ in $(seq 1 40); do kill -0 "$APP_PID" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$APP_PID" 2>/dev/null && { echo "[launch] the console did not exit: killing it"; kill -KILL "$APP_PID" 2>/dev/null || true; }
    fi
    # the console freezes the robot's LiDAR watchdog for the run; a killed console cannot thaw it
    if [ "${WATCHDOG_FROZEN:-0}" = 1 ]; then
    SSHPASS="$SSHPASS_VALUE" sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$SSH_TARGET" \
        'p=$(cat /tmp/homebody-lidar-watchdog.pid 2>/dev/null) && kill -CONT $p 2>/dev/null && echo "[launch] robot watchdog thawed"' 2>/dev/null || true
    fi
    stop_phone_receiver
    if [ "$RECORD" = 1 ]; then
    echo "[launch] closing the bag and checking it"
    # a Ctrl+\ here kills the stop (its docker CLI too) before the halt trap runs: the retry closes the bag
    if [ -d "$RUN_DIR/ws" ]; then
        src/homebody/real2sim/exploration/record/record.sh stop "$RUN_DIR" || FINALIZE_FAILED=1
    fi
    fi
    # A restarted instance is ours even after --adopt. Matching the workspace label
    # avoids stopping an external SLAM owner and lets temporary logs close cleanly.
    bash hardware/g1/superodom/run_superodom_ws.sh down-owned || FINALIZE_FAILED=1
    if [ "$RECORD" = 1 ]; then
    # The bag is closed and checked above; from here only derived files are made, re-runnable.
    if [ -d "$RUN_DIR/ws" ]; then
        echo "[launch] exporting the 30 fps phone video and the SLAM clouds (video/) — Ctrl-C skips it"
        src/homebody/real2sim/exploration/record/record.sh export "$RUN_DIR" \
            || { echo "[launch] export incomplete; recordings kept" >&2; FINALIZE_FAILED=1; }
    else
        echo "[launch] no bag was recorded (the recorder never started): nothing to check or export"
    fi
    if [ -n "${PHONE_RECORDING_ROOT:-}" ] && [ -d "$RUN_DIR/ws" ]; then
        "${HOMEBODY_PYTHON:?}" src/homebody/real2sim/exploration/phone_archive.py "$RUN_DIR" \
            --record-root "$PHONE_RECORDING_ROOT" \
            || { echo "[launch] native phone archive incomplete; recordings kept" >&2; FINALIZE_FAILED=1; }
    fi
    cat > "$RUN_DIR/README.txt" <<EOF_README
Exploration run $(basename "$RUN_DIR")
  capture_manifest.json  native file/stream/timing bindings and separate capability readiness
  capture_manifest.schema.json  structural contract (not a hardware-readiness certificate)
  quality/            capture_health.json, exact video PTS/time bases, bag inspection and ROS schemas
  ws/                 the rosbag (mcap, 300 s splits): MID-360 raw cloud + IMU, SuperOdom odometry,
                      registered scans + map + tf, nav, /lowstate joints, Dex3 state,
                      active camera topics, /homebody/cameras, /tf_static, /homebody/clock_offset,
                      /exploration/state (selected contract: meta/record_topics.txt)
  phone/native/       finalized native phone segments and archive.json; open segments may be pending
  meta/cameras.json   camera names, known serials (null when unknown), endpoints and topics
  meta/phone_camera.json  unknown phone identity/lens/mount and unverified reference assumptions;
                      these are not operator declarations; guides/phone-camera.md
  meta/phone_context.json  configured live RGB endpoint and receiver lifecycle;
                      phone/native/archive.json reports actual included segments
  video/              <name>_rgb_30fps.mp4 per camera, on an exact 30 fps
                      timeline from the capture stamps (video_index.json counts frames/repeats/drops);
                      slam_map.ply (the last SuperOdom epoch) and slam_map_epoch<k>.ply per epoch, 5 cm voxels
  bag_check.txt       per-topic counts, rates, max gaps, stamp offsets and verdicts
  meta/frames.txt     the TF chain, the distortion and clock notes
  explore/run.json    stops, routes with their intermediate points, drift side-steps, stop reason
                      + every model decision (click, snap, facing, why), the brief
  explore/vlm/        the model's pictures as sent per decision, its click drawn, the prompt
  explore/live_map.npz  the lidar map in the start frame (raster, T_map_px, occupancy)
  explore/boundary.json the run range, start-frame metres
  console.log launch.log recorder.log   the three processes' output
  meta/               settings.json (every resolved setting), calibration and configs used
EOF_README
    fi
    if [ "$RECORD" = 1 ] && [ -d "$RUN_DIR/ws" ]; then
        src/homebody/real2sim/exploration/record/record.sh capture "$RUN_DIR" \
            || { echo "[launch] capture contract incomplete; recordings kept" >&2; FINALIZE_FAILED=1; }
    fi
    rm -f "$LAUNCH_PID"
    if [ "$FINALIZE_FAILED" = 1 ]; then
        echo "[launch] resource cleanup or output finalization failed; recordings are kept when enabled" >&2
        code=1
    fi
    echo "[launch] console closed; persistent body host left balancing"
    exit "$code"
}
trap 'cleanup "$?"' EXIT
trap 'trap "" INT TERM HUP; exit 130' INT TERM
trap 'trap "" INT TERM HUP; exit 129' HUP   # a closed pane: the cleanup still closes the bag
halt_body() {
    trap - QUIT
    halt_now
    exit 131   # the EXIT trap runs the cleanup: console stopped, bag closed, video exported
}
trap halt_body QUIT


# 0b. Block system sleep while this launcher lives (`tail --pid` ends the lock with the launcher, even a killed one).
inhibit_sleep() {
    command -v systemd-inhibit >/dev/null 2>&1 || { echo "[launch] systemd-inhibit missing: sleep is not blocked"; return 0; }
    systemd-inhibit --what=sleep:handle-lid-switch --mode=block --who=homebody-launch-explore \
        --why="exploration run: a sleep kills the GPU, the body host and the bag" tail --pid=$$ -f /dev/null &
    echo "[launch] sleep and the lid switch are blocked while this launcher runs"
}
inhibit_sleep

# 1. The robot interface: resolved and checked by homebody real2sim explore.
# A segfault prints every Python thread's stack into launch.log instead of just "core dumped".
export PYTHONFAULTHANDLER=1
echo "[network] robot interface: $IFACE"

# 2. Stale-writer hunt by file descriptor: a leaked writer is an orphan renamed to pt_main_thread, so the
#    lease file is the handle. The .client lease belongs to a body-host client.
hunt() { "${HOMEBODY_PYTHON:?}" -c \
    "from homebody.backends.g1.writer import hunt; import os; \
     pids, blind = hunt('$LOCK'); more, _ = hunt('$LOCK.client'); pids = sorted({*pids, *more}); \
     print(' '.join(str(p) for p in pids if p != os.getpid())); print(blind)"; }
# A failing check must refuse, not abort with a bare interpreter error: `set -e` would kill the
# script on the assignment and print nothing an operator can act on.
if [ "$OBSERVE" != "1" ]; then
if ! HUNT_OUT="$(hunt)"; then
    echo "refusing launch: the by-fd stale-writer hunt could not run — fix: check" >&2
    echo "  "${HOMEBODY_PYTHON:?}" works, then re-run." >&2
    exit 1
fi
PIDS="$(printf '%s\n' "$HUNT_OUT" | sed -n 1p)"
BLIND="$(printf '%s\n' "$HUNT_OUT" | sed -n 2p)"
[ "${BLIND:-0}" = "0" ] || echo "[preflight] NOTE: $BLIND process(es) unreadable from /proc" \
    "(another user, likely sudo) — a stale writer among them is invisible here; re-run" \
    "with sudo -E to be sure"
if [ -n "$PIDS" ]; then
    if [ "$SETUP" = 1 ]; then
        echo "refusing setup: setup will not signal another writer or console client ($PIDS)" >&2
        echo "  close that workflow normally before retrying; the body host owns its separate lifecycle" >&2
        exit 1
    fi
    echo "[preflight] a stale G1 body writer or homebody client still holds the lease: $PIDS"
    echo "[preflight] SIGTERM first, so its controller can run its own stop tick"
    # shellcheck disable=SC2086
    kill -TERM $PIDS 2>/dev/null || true
    for _ in $(seq 1 10); do
        sleep 0.5
        PIDS="$(hunt | sed -n 1p)" || { echo "refusing launch: the stale-writer hunt failed mid-wait" >&2; exit 1; }
        [ -n "$PIDS" ] || break
    done
    if [ -n "$PIDS" ]; then
        echo "[preflight] refusing to launch: $PIDS still hold $LOCK." >&2
        echo "  the same motors is not a state this script will start into." >&2
        echo "  fix: kill -KILL $PIDS   # then re-run" >&2
        exit 1
    fi
    echo "[preflight] released"
fi

fi
# Same provisioned video service for setup and per-run receiver reset. Setup uses
# start so an already connected publisher is not interrupted.
phone_relay() {
    "${HOMEBODY_PYTHON:?}" - "$SETTINGS" "$1" "$2" <<'PY_PHONE_RELAY'
import re, shlex, sys
from pathlib import Path
from homebody.real2sim.exploration.settings import from_json, install
install(from_json(Path(sys.argv.pop(1)).read_text()))
from homebody.real2sim.exploration.transport import SshTransport
action, service = sys.argv[1:]
if action not in ("start", "restart") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*\.service", service):
    raise SystemExit("iphone.relay_service must name a systemd service")
transport = SshTransport(timeout_s=15)
unit = shlex.quote(service)
print(transport.run_sudo(f"systemctl {action} {unit} && systemctl is-active {unit}",
                         "initialize phone video relay"), flush=True)
PY_PHONE_RELAY
}

if [ "$SETUP" = 1 ]; then
    SERVICE="$IPHONE_RELAY"
    if [ -n "$SERVICE" ]; then
        echo "[setup] starting/reusing Jetson video service: $SERVICE"
        phone_relay start "$SERVICE" || {
            echo "refusing setup: phone video service could not start; check SSH and the provisioned service" >&2
            exit 1
        }
    else
        echo "[setup] no relay service selected; leaving externally managed video alone"
    fi
    "${BODY[@]}" start --iface "$IFACE" --bind "$BODY_ADDRESS" ${BODY_CORES:+--cores "$BODY_CORES"} || {
        echo "refusing setup: AMO body host did not start; read /tmp/homebody-body-host.log" >&2
        exit 1
    }
    echo "[setup] setup complete; body host and any selected video service remain running"
    echo "[setup] remote Y initializes the pose; after the ramp, A enables AMO standing"
    echo "[setup] start your phone broadcast, then launch with the same video settings; see guides/phone-camera.md"
    echo "[setup] no camera frames verified or recording started; log: /tmp/homebody-body-host.log"
    exit 0
fi
# Fresh frames are required before any body/perception startup. The longer wait gives
# the broadcaster time to retry its connection after the listener is created.
phone_ready() {
    "${HOMEBODY_PYTHON:?}" - "$1" "${PHONE_PID:-}" "$IPHONE_ENDPOINT" "$IPHONE_MAX_AGE_S" <<'PY_PHONE_CHECK'
import os, sys, time
from homebody.primitives.cameras.context import ContextCamera
camera = ContextCamera(sys.argv[3], float(sys.argv[4]))
try:
    deadline = time.monotonic() + float(sys.argv[1])
    while time.monotonic() < deadline:
        if sys.argv[2]:
            try:
                os.kill(int(sys.argv[2]), 0)
            except ProcessLookupError:
                raise SystemExit(1)
        image, meta = camera.snapshot()
        if image is not None:
            print("[preflight] fresh iPhone RGB:", image.shape)
            break
        time.sleep(0.1)
    else:
        raise SystemExit(1)
finally:
    camera.close()
PY_PHONE_CHECK
}
if [ -z "$IPHONE_INPUT" ]; then
    if ! phone_ready 3; then
        echo "refusing launch: no fresh RGB. Set iphone.input_url (--iphone-input) for automatic receiver startup." >&2
        echo "  Keep the phone streaming and its network/relay connected." >&2
        exit 1
    fi
    PHONE_MODE=reused
    echo "[phone] using external RGB; the existing receiver will be left running"
else
    if phone_ready 1; then
        echo "refusing managed phone launch: another receiver is publishing at $IPHONE_ENDPOINT." >&2
        echo "  Stop that receiver, or launch without --iphone-input to use external RGB." >&2
        exit 1
    fi
    command -v ffmpeg >/dev/null || { echo "iphone receiver requires ffmpeg" >&2; exit 1; }
    mkdir -p "$BAG_ROOT"
    # A rapid retry must never append captures or overwrite metadata in an older run.
    mkdir -p "$RUN_DIR"
    CAMERA_ARGS=(--reconnect)
    if [ "$RECORD" = 1 ]; then
        PHONE_RECORDING_ROOT="$RUN_DIR/phone/live"
        CAMERA_ARGS+=(--record-root "$PHONE_RECORDING_ROOT" --segment-seconds 30)
        echo "[phone] native video: $PHONE_RECORDING_ROOT"
    else
        PHONE_RECORDING_ROOT=""
        echo "[phone] live receiver; native recording off"
    fi
    PYTHONPATH="$REPO/src" "${HOMEBODY_PYTHON:?}" src/homebody/real2sim/exploration/context_camera.py \
        --input "$IPHONE_INPUT" --publish "$IPHONE_ENDPOINT" \
        "${CAMERA_ARGS[@]}" \
        --hwaccel "$IPHONE_HWACCEL" \
        >> "$RUN_DIR/phone/receiver.log" 2>&1 &
    PHONE_PID=$!
    PHONE_MODE=started
    if [ -n "$IPHONE_RELAY" ]; then
        echo "[phone] resetting Jetson video relay for this run"
        if ! phone_relay restart "$IPHONE_RELAY" > "$RUN_DIR/phone/relay.log" 2>&1; then
            echo "refusing launch: Jetson video relay could not start; check Ethernet, USB and the installed service." >&2
            tail -n 12 "$RUN_DIR/phone/relay.log" >&2
            exit 1
        fi
    fi
    if ! phone_ready 30 || ! jobs -pr | grep -qx "$PHONE_PID"; then
        echo "refusing launch: receiver exited or did not deliver fresh RGB within 30 seconds." >&2
        echo "  Check $RUN_DIR/phone/receiver.log, the broadcasting app and its network/relay." >&2
        tail -n 12 "$RUN_DIR/phone/receiver.log" >&2
        exit 1
    fi
fi
# 2b. The body host owns the legs outside this launcher.
if [ "$OBSERVE" != "1" ]; then
    "${BODY[@]}" start --iface "$IFACE" --bind "$BODY_ADDRESS" ${BODY_CORES:+--cores "$BODY_CORES"} || {
        echo "refusing launch: the body host did not start — read /tmp/homebody-body-host.log" >&2; exit 1; }
fi

if [ "$RECORD" = 1 ]; then
# 3. Disk and the recorder image, before anything on the robot is cycled.
mkdir -p "$BAG_ROOT"
FREE_GB=$(( $(df --output=avail -k "$BAG_ROOT" | tail -1) / 1024 / 1024 ))
# Recording runs at roughly 2.5 GB/min; the bag runs from Start exploration, so recommend
# 3.5 GB/min over the budget plus 5 min.
GB_PER_MIN_HALVES=7
NEED_GB="$(( (${BUDGET_S%.*} / 60 + 5) * GB_PER_MIN_HALVES / 2 ))"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "[warning] low recording space: $FREE_GB GB free under $BAG_ROOT ($NEED_GB GB recommended for a $BUDGET_S s budget); continuing launch" >&2
    echo "  Recording may fail if the disk fills. Free space or pass --output <another disk>." >&2
fi
docker image inspect "$RECORDER_IMAGE" >/dev/null 2>&1 || {
    echo "refusing launch: the recorder image is missing — fix: hardware/g1/superodom/build.sh recorder" >&2
    exit 1; }
if docker ps -a --format '{{.Names}}' | grep -qx homebody_bag_ws; then
    OLD_RUN="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/bags"}}{{.Source}}{{end}}{{end}}' homebody_bag_ws 2>/dev/null || true)"
    if docker ps --format '{{.Names}}' | grep -qx homebody_bag_ws; then
        echo "refusing launch: an earlier recorder (homebody_bag_ws) is still running on ${OLD_RUN:-?}: its bag is open" >&2
        echo "  fix: src/homebody/real2sim/exploration/record/record.sh stop ${OLD_RUN:-<its RUN_DIR>}   # closes and checks that bag" >&2
        exit 1
    fi
    # A dead recorder (killed with its launcher, exit 137) cannot be writing: its container
    # record only blocks the next launch. Cleared here; that bag's check/export is a command away.
    # its log is the only record of why it died: saved beside its bag before the container goes
    { [ -d "${OLD_RUN:-}" ] && docker logs homebody_bag_ws > "$OLD_RUN/recorder.log" 2>&1; } || true
    docker rm homebody_bag_ws >/dev/null 2>&1 || true
    echo "[preflight] cleared a dead recorder (homebody_bag_ws) from ${OLD_RUN:-?}; check and export that bag with:"
    echo "  src/homebody/real2sim/exploration/record/record.sh check ${OLD_RUN:-<RUN_DIR>}; src/homebody/real2sim/exploration/record/record.sh export ${OLD_RUN:-<RUN_DIR>}"
fi
fi
# the console outlives a launcher that dies (a closed pane): one console per robot. Only a
# python running app.py counts (an editor or a linter naming the file is not a console).
for p in $(pgrep -f 'python[^ ]* -m homebody.real2sim.exploration.app' || true); do
    echo "refusing launch: an explore console is already running on the robot (pid $p) — stop it first" >&2
    exit 1
done
[ "$RECORD" = 0 ] || echo "[preflight] $FREE_GB GB free under $BAG_ROOT; recorder image present"

# 3b. One folder for the whole run: this launcher's own output, the bag, the explorer's map and
#     route log, the console log, the recorder log and check, and what the run was made of.
mkdir -p "$RUN_DIR/explore" "$RUN_DIR/meta" "$RUN_DIR/phone"
# phone/: the operator's own recording of the same walk, copied here afterwards: the phone's full
# rate on the phone's clock; the bag is the robot's cameras on the robot's clock.
cat > "$RUN_DIR/phone/README.txt" <<'EOF_PHONE'
The bag records sampled live phone RGB. The receiver keeps native full-rate
video. With INPUT configured, the launcher starts a fresh receiver and closes its receiver
on exit. Cleanup archives finalized segments in phone/native/ when a receiver root is known;
inspect native/archive.json for copied files and any open segments left behind.
After those segments close, rerun src/homebody/real2sim/exploration/phone_archive.py with this run and receiver root.
The archive uses approximate receiver time; it does not prove continuous or synchronized capture.
Optional original recordings made on the phone can also be placed here, preserving originals.
Reference mount: landscape across the upper chest below the head,
  with rear cameras facing forward and the lens cluster offset to one side.
Reference lens: rear ultra-wide camera at 0.5x zoom.
  Neither is verified or supplied by the operator for this capture. Actual device,
  serial, selected lens and mounting are unknown; see guides/phone-camera.md.
The RGB timestamps are receiver decode times, not phone exposure times. Keep the bag's
clock/epoch metadata and a shared visible alignment event. Phone intrinsics, its mount transform
and exposure-clock alignment must be measured before metric fusion with the LiDAR map.
EOF_PHONE
exec > >(trap '' INT QUIT HUP; tee -p -a "$RUN_DIR/launch.log") 2>&1   # keep logging if the terminal closes during cleanup
echo "[launch] run folder: $RUN_DIR"
if [ "$RECORD" = 1 ]; then
for f in hardware/g1/superodom/g1_mid360.yaml hardware/g1/superodom/g1_frame_normalizer_ws.yaml \
         src/homebody/real2sim/exploration/record/topics.txt src/homebody/real2sim/exploration/record/qos.yaml; do
    cp "$f" "$RUN_DIR/meta/" || echo "[launch] NOTE: $f not copied into meta/"
done
{ docker image inspect "$RECORDER_IMAGE" -f "$RECORDER_IMAGE {{.Id}}";
  docker image inspect "$SLAM_IMAGE" -f "$SLAM_IMAGE {{.Id}}"; } > "$RUN_DIR/meta/digests.txt"
URDF="$("${HOMEBODY_PYTHON:?}" -c 'from homebody.backends.g1.assets import ROBOT_URDF; print(ROBOT_URDF)')"
cp "$URDF" "$RUN_DIR/meta/" || { echo "refusing launch: could not copy the robot URDF $URDF" >&2; exit 1; }
# Preserve the reference assumptions without reporting them as observed camera identity.
cat > "$RUN_DIR/meta/phone_camera.json" <<EOF_PHONECAM
{
  "device": null,
  "serial": null,
  "identity_status": "unknown",
  "role": "operator recording and optional uncalibrated live RGB context; see phone_context.json",
  "mount": {
    "where": null,
    "reference_body": "torso (unverified reference assumption)",
    "reference_assumption": "landscape across upper chest below head; rear cameras forward, lens cluster laterally offset",
    "evidence": "reference profile only; actual mount not verified or operator-supplied",
    "extrinsic_measured": false,
    "register_by": "calibrated lens and shared known-size target observations; measure camera-to-LiDAR transform"
  },
  "lens": {
    "zoom": null,
    "lens": null,
    "reference_assumption": "rear ultra-wide camera at 0.5x; unverified for this capture",
    "intrinsics_measured": false,
    "note": "Identify and calibrate the actual selected lens; no per-frame intrinsics are recorded"
  },
  "clock": "sampled RGB uses workstation decode time; native archive retains remuxed segment PTS/time base and rebasing logs, not calibrated phone exposure time",
  "align": "unknown; estimate and validate lens, mount and clock offset/drift using shared scene and motion evidence across the route",
  "metric_frame": "PLY uses ws_map within a SLAM epoch; live_map.npz uses the separate exploration start frame; no ARKit pose is recorded",
  "doc": "guides/phone-camera.md",
  "stated_not_measured": false,
  "reference_profile_unverified": true
}
EOF_PHONECAM
"${HOMEBODY_PYTHON:?}" - "$RUN_DIR/meta/phone_context.json" "$PHONE_MODE" "$PHONE_PID" \
    "$IPHONE_ENDPOINT" "$IPHONE_RELAY" <<'PY_PHONE_CONTEXT'
import json
import sys
from pathlib import Path

endpoint = sys.argv[4].strip()
Path(sys.argv[1]).write_text(json.dumps({
    "schema": 1,
    "enabled": bool(endpoint),
    "endpoint": endpoint or None,
    "receiver_lifecycle": sys.argv[2],
    "relay_service": (sys.argv[5] or None) if sys.argv[2] == "started" else None,
    "launch_owned_receiver_pid": int(sys.argv[3]) if sys.argv[3] else None,
    "recording_included": False,
    "capture_verified": False,
    "timestamp_kind": "receiver_decode_time",
    "metric_calibration": False,
    "note": "Capture begins with incoming video. Cleanup closes a launch-owned receiver; reused receivers remain running. See phone/native/archive.json for inclusion, gaps and pending files.",
}, indent=2) + "\n")
PY_PHONE_CONTEXT
cat > "$RUN_DIR/meta/frames.txt" <<'EOF_IPHONE_FRAMES'
LiDAR + iPhone profile. Aligned ws_map -> gravity comes from registered odometry.
Static links are gravity -> mid360_link (existing Rx180 convention), then mid360_link ->
torso_link from the robot URDF. Verify these against the actual mount before movement.
No metric phone camera transform is active.
Phone CompressedImage stamps are workstation receiver decode time, not phone exposure time.
LiDAR/SLAM headers use the MID-360 clock. Keep epochs and clock-offset evidence with the bag.
Native phone videos are independent full-rate segments; the bag records sampled phone RGB.
EOF_IPHONE_FRAMES

fi

# 3c. A manual zone overrides the region the model infers before it walks.
if [ "$ZONE_SIZE_M" != 0 ]; then
    echo "[launch] manual zone: a $ZONE_SIZE_M m square around the start"
else
    echo "[launch] no manual zone: the model infers the task region before walking"
fi
if [ "$OBSERVE" = 1 ]; then
    echo "[launch] Start preview records sensors and proposes viewpoints; no movement"
else
    echo "[launch] Start exploration runs autonomously; STOP ends exploration while the walking policy balances"
fi

# 4. Fresh localization: restarting SuperOdom puts its origin where the robot stands now. Then the hands.
# Exploration consumes raw LiDAR directly; robot-side map localization and cameras are not inputs.
"${HOMEBODY_PYTHON:?}" - "$ADOPT" "$SETTINGS" <<'PY_PHONE_SLAM'
import sys
import time
from pathlib import Path
from homebody.real2sim.exploration import superodom_ws
from homebody.real2sim.exploration.settings import from_json, install

install(from_json(Path(sys.argv[2]).read_text()))
adopt = sys.argv[1] == "1"
if not adopt:
    print(superodom_ws.down(), flush=True)
    print(superodom_ws.up(), flush=True)
deadline = time.monotonic() + (0 if adopt else 40)
while True:
    healthy, status = superodom_ws.status()
    print(status, flush=True)
    if healthy:
        break
    if time.monotonic() >= deadline:
        raise SystemExit("refusing launch: workstation LiDAR/IMU/SLAM is not healthy")
    time.sleep(0.5)
PY_PHONE_SLAM
WATCHDOG_FROZEN=1
# Freeze the robot's LiDAR watchdog before the console starts: it cycles the whole stack on a
# `_sec` gap.
SSHPASS="$SSHPASS_VALUE" sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
    "$SSH_TARGET" \
    'p=$(cat /tmp/homebody-lidar-watchdog.pid 2>/dev/null) && grep -q lidar_watchdog /proc/$p/cmdline \
     2>/dev/null && kill -STOP $p && echo FROZEN-$p || echo NOT-FROZEN' \
    | sed 's/^/[launch] robot watchdog: /' || echo "[launch] NOTE: the robot watchdog was not frozen here; the console freezes it at Start exploration"

# the backend's gripper_state term (every drive NEEDS it) raises RealHardwareRequired when the
# Dex3 does not answer: open them, or refuse now
if [ "$OBSERVE" != "1" ]; then
"${HOMEBODY_PYTHON:?}" hardware/g1/open_hands.py --settings-json "$SETTINGS" || {
    echo "refusing launch: the Dex3 hands did not open — check they are powered and empty" >&2
    exit 1
}
fi
if [ "$RECORD" = 1 ]; then
# both clocks against this workstation's, for the bag's consumer (meta/clocks.json)
"${HOMEBODY_PYTHON:?}" - "$RUN_DIR/meta/clocks.json" "$SETTINGS" <<'PY' || echo "[launch] NOTE: clocks.json not written"
import json, subprocess, sys, time
from pathlib import Path
from homebody.real2sim.exploration.settings import from_json, install
from homebody.real2sim.exploration.transport import SshTransport
install(from_json(Path(sys.argv[2]).read_text()))
t = SshTransport(); rows = []
for _ in range(5):
    t0 = time.time(); orin = float(t.run("date +%s.%N", "orin clock").strip().splitlines()[-1]); t1 = time.time()
    rows.append({"workstation_send_s": t0, "orin_s": orin, "workstation_recv_s": t1, "round_trip_s": round(t1 - t0, 4)})
best = min(rows, key=lambda r: r["round_trip_s"])
json.dump({"samples": rows, "orin_minus_workstation_s": round(best["orin_s"] - (best["workstation_send_s"] + best["workstation_recv_s"]) / 2, 4),
           "note": "camera stamps are Orin realtime; lidar/SuperOdom stamps are the MID-360 clock (see bag_check min offsets)"},
          open(sys.argv[1], "w"), indent=2)
print(f"[launch] clocks: Orin - workstation = {best['orin_s'] - (best['workstation_send_s'] + best['workstation_recv_s']) / 2:+.4f} s (round trip {best['round_trip_s']} s)")
PY
"${HOMEBODY_PYTHON:?}" - "$RUN_DIR/meta/cameras.json" "$IPHONE_ENDPOINT" <<'PY' || echo "[launch] NOTE: cameras.json not written"
import json, sys
cams = {"iphone": {"device": None, "serial": None, "lens": None,
        "identity_status": "unknown", "reference_profile_unverified": True,
        "color_endpoint": sys.argv[2], "metric_calibration": False,
        "timestamp_kind": "receiver_decode_time", "topics": ["/homebody/iphone/color/image/compressed"]}}
json.dump(cams, open(sys.argv[1], "w"), indent=2)
print("[launch] cameras: iphone unknown serial")
PY
fi


# The bag starts when Start exploration is pressed (the console runs record.sh start); the cleanup closes it.
if [ "$RECORD" = 1 ]; then
    echo "[launch] recording starts at Start exploration; bag: $RUN_DIR/ws"
else
    echo "[launch] recording off; temporary files are removed when the console closes"
fi

# 5. The console. Y/A can initialize and balance before Start exploration, which starts route execution.
if [ "$OBSERVE" = 1 ]; then
    echo "[launch] observation-only console: http://localhost:$PORT; Start preview starts sensor capture"
else
    echo "[launch] gantry rigged? feet clear? E-stop in hand? open http://localhost:$PORT"
fi
APP_ARGS=(--settings-json "$SETTINGS" --out "$RUN_DIR/explore" --run-dir "${HOMEBODY_RUN_DIR:-$RUN_DIR}")
[ "$RECORD" = 0 ] || APP_ARGS+=(--bag-dir "$RUN_DIR")
"${HOMEBODY_PYTHON:?}" -m homebody.real2sim.exploration.app "${APP_ARGS[@]}" \
    > >(trap '' INT QUIT HUP; tee -p "$RUN_DIR/console.log") 2>&1 &   # preserve console shutdown output if the terminal closes
APP_PID=$!
set +e   # the console's own exit code is the launcher's; a signal lands in the traps above
wait "$APP_PID"
CODE=$?
set -e
echo "[launch] the console exited with code $CODE"
exit "$CODE"
