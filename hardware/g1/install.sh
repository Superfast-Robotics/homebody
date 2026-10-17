#!/usr/bin/env bash
# Turnkey G1 install for REAL2SIM exploration. It builds and copies; it never starts robot motion or the body host.
#   workstation (robot.iface): SuperOdometry and recorder containers, body host, console and phone receiver
#   G1 onboard computer (robot.ssh_target): vendor MID-360 driver (pre-installed, untouched), phone RTSP relay
#   hardware/g1/phone/ is the only G1-side piece; copy and phone install it when iphone.relay_service names its unit
#
#   bash hardware/g1/install.sh [--settings my.toml] [--robot user@ip] [--password P] [--iface eth]
#                              [--plan] [--only prereqs|odometry|recorder|copy|phone|verify]
# Values come from configs/exploration.toml and --settings (robot.ssh_target, robot.ssh_password,
# robot.iface, slam.image, recording.image, iphone.*); --robot, --password and --iface override them
# for this run only. --plan prints every command and runs none.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${HOMEBODY_PYTHON:-$(command -v python || true)}"
STEPS=(prereqs odometry recorder copy phone verify)
PLAN=0; ONLY=""; SETTINGS_FILE=""; ROBOT=""; PASSWORD=""; IFACE_OPT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --plan) PLAN=1 ;;
        --only) ONLY="${2:?--only needs a step}"; shift ;;
        --settings) SETTINGS_FILE="${2:?--settings needs a file}"; shift ;;
        --robot) ROBOT="${2:?--robot needs user@ip}"; shift ;;
        --password) PASSWORD="${2:?--password needs a value}"; shift ;;
        --iface) IFACE_OPT="${2:?--iface needs an interface}"; shift ;;
        -h|--help) sed -n 2,11p "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
case " ${STEPS[*]} " in *" ${ONLY:-prereqs} "*) ;; *) echo "--only takes one of: ${STEPS[*]}" >&2; exit 2 ;; esac
[ -n "$PY" ] && [ -x "$PY" ] || { echo "refusing install: no python on PATH; activate the environment from the README" >&2; exit 1; }

# One resolution of the settings; the password lands in a shell variable and is never printed.
RESOLVED="$("$PY" -c 'import json, shlex, sys
from homebody.real2sim.exploration.settings import install, load, ssh_password, to_json
overrides = {"ssh_target": sys.argv[3] or None, "ssh_password": sys.argv[4] or None, "iface": sys.argv[5] or None}
groups = json.loads(to_json(install(load(sys.argv[1], sys.argv[2] or None, options={"robot": overrides}))))
lock = json.load(open("hardware/g1/superodom/source-lock.json"))
robot, iphone = groups["robot"], groups["iphone"]
values = {"SSH_TARGET": robot["ssh_target"], "IFACE": robot["iface"],
          "SSHPASS_VALUE": sys.argv[4] or ssh_password(),
          "SLAM_IMAGE": groups["slam"]["image"], "RECORDER_IMAGE": groups["recording"]["image"],
          "RECORD": int(groups["exploration"]["record"]),
          "IPHONE_INPUT": iphone["input_url"], "IPHONE_RELAY": iphone["relay_service"],
          "LOCK_SLAM": lock["superodom"]["image"], "LOCK_RECORDER": lock["unitree_ros2"]["image"]}
for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")' "$REPO" "$SETTINGS_FILE" "$ROBOT" "$PASSWORD" "$IFACE_OPT")" \
    || { echo "refusing install: the exploration settings did not resolve (above)" >&2; exit 1; }
eval "$RESOLVED"

# The relay pieces, named by the shipped unit; the MediaMTX pin is guides/phone-camera.md's.
PHONE_DIR=hardware/g1/phone
UNIT=homebody-phone-rtsp.service
RELAY_BIN="$(sed -n 's/^ExecStart=\([^ ]*\) .*/\1/p' "$PHONE_DIR/$UNIT")"
RELAY_CONF="$(sed -n 's/^ExecStart=[^ ]* //p' "$PHONE_DIR/$UNIT")"
REMOTE_DIR=homebody-phone   # staging copy of hardware/g1/phone/ in the robot user's home
MEDIAMTX_URL=https://github.com/bluenviron/mediamtx/releases/download/v1.21.1/mediamtx_v1.21.1_linux_arm64.tar.gz
MEDIAMTX_SHA256=6a3aa635fb60ea9b8d566ec306f0a42ff1b6b52a3942bc2baffbe55880d4c3dd
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=5)

STEP=""
say() { echo "[install] $*"; }
refuse() { echo "refusing install ($STEP): $1" >&2; echo "  fix: $2" >&2; exit 1; }
sq() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
# show ARG...: the command as a shell line, quoting what needs it
show() {
    local out="" arg
    for arg in "$@"; do
        if [[ "$arg" =~ ^[A-Za-z0-9_./:=,@+%-]+$ ]]; then out+=" $arg"; else out+=" $(sq "$arg")"; fi
    done
    printf '%s' "${out# }"
}
# x CMD...: printed under --plan, printed and run otherwise
x() { if [ "$PLAN" = 1 ]; then printf '    $ %s\n' "$(show "$@")"; else say "$STEP: $(show "$@")"; "$@"; fi; }
# robot CMD: one shell command on the G1 as robot.ssh_target; the password travels in SSHPASS
robot() {
    if [ "$PLAN" = 1 ]; then printf '    $ SSHPASS=*** sshpass -e ssh %s %s %s\n' "${SSH_OPTS[*]}" "$SSH_TARGET" "$(sq "$1")"; return 0; fi
    say "$STEP: ssh $SSH_TARGET $1" >&2
    SSHPASS="$SSHPASS_VALUE" sshpass -e ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$1"
}
# robot_root CMD: the same as root; sudo -S reads the password on stdin, as transport.run_sudo does
robot_root() {
    local remote; remote="sudo -S -p '' bash -c $(sq "$1")"
    if [ "$PLAN" = 1 ]; then printf '    $ SSHPASS=*** sshpass -e ssh %s %s sudo bash -c %s   # sudo -S reads the password on stdin\n' "${SSH_OPTS[*]}" "$SSH_TARGET" "$(sq "$1")"; return 0; fi
    say "$STEP: ssh $SSH_TARGET sudo $1" >&2
    printf '%s\n' "$SSHPASS_VALUE" | SSHPASS="$SSHPASS_VALUE" sshpass -e ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$remote"
}
robot_copy() {
    if [ "$PLAN" = 1 ]; then printf '    $ SSHPASS=*** sshpass -e scp %s %s %s:%s/\n' "${SSH_OPTS[*]}" "$*" "$SSH_TARGET" "$REMOTE_DIR"; return 0; fi
    say "$STEP: scp $* $SSH_TARGET:$REMOTE_DIR/"
    SSHPASS="$SSHPASS_VALUE" sshpass -e scp "${SSH_OPTS[@]}" "$@" "$SSH_TARGET:$REMOTE_DIR/"
}
# skip_if WHY PROBE...: a real run skips the work when PROBE succeeds; --plan prints the condition
skip_if() {
    local why="$1"; shift
    if [ "$PLAN" = 1 ]; then echo "    skipped when $why"; return 1; fi
    if "$@" >/dev/null 2>&1; then say "$STEP: already done ($why)"; return 0; fi
    return 1
}
# off WHY: a step these settings do not need; --plan still prints its commands
off() {
    if [ "$PLAN" = 1 ]; then echo "    not needed with these settings: $1; commands shown for when it is"; return 1; fi
    say "$STEP: skipped, $1"
}
relay_off() {
    if [ -z "$IPHONE_RELAY" ]; then
        off "iphone.relay_service is empty, so the phone video is managed outside HomeBody and nothing goes on the robot"
    elif [ "$IPHONE_RELAY" != "$UNIT" ]; then
        off "iphone.relay_service names a unit other than the shipped $UNIT; provision it as guides/phone-camera.md describes"
    else
        return 1
    fi
}
# The workstation interface on the robot network: robot.iface, or the one 192.168.123.x link (cli.robot_interface).
robot_iface() {
    if [ "$PLAN" = 1 ]; then echo "${IFACE:-<the only 192.168.123.x link>}"; return 0; fi
    "$PY" -c 'import sys
from homebody.real2sim.exploration.cli import robot_interface
try:
    print(robot_interface(sys.argv[1]))
except ValueError as error:
    sys.exit(f"refusing install: {error}")' "$IFACE"
}
workstation_ip() {
    [ "$PLAN" = 0 ] || { echo "<robot.iface 192.168.123.x address>"; return 0; }
    local iface; iface="$(robot_iface)" || exit 1
    ip -o -4 addr show dev "$iface" | awk '{split($4, a, "/"); print a[1]}' | grep -m1 '^192\.168\.123\.' \
        || refuse "$iface has no 192.168.123.x address" "connect the robot Ethernet or set iface in [robot]"
}
rmem_ok() { [ "$(sysctl -n net.core.rmem_max)" -ge 33554432 ]; }
copy_current() {
    [ "$(cd "$PHONE_DIR" && sha256sum mediamtx.yml "$UNIT")" = "$(robot "cd $REMOTE_DIR && sha256sum mediamtx.yml $UNIT")" ]
}

step_prereqs() {
    local missing=() command
    if [ "$PLAN" = 1 ]; then
        echo "    requires docker ip ssh sshpass ffmpeg ffprobe on PATH and docker usable by this user"
    else
        for command in docker ip ssh sshpass ffmpeg ffprobe; do
            command -v "$command" >/dev/null 2>&1 || missing+=("$command")
        done
        [ "${#missing[@]}" = 0 ] || refuse "missing commands: ${missing[*]}" \
            "install them (ip: iproute2; docker: docker.io; ssh/sshpass: openssh-client, sshpass; ffmpeg, ffprobe: ffmpeg)"
        docker info >/dev/null 2>&1 || refuse "docker is not usable by $(id -un)" "start the daemon or add the user to the docker group"
    fi
    # DDS cloud receive buffers; preflight_ws.sh refuses to start SuperOdometry below 32 MB
    if ! skip_if "net.core.rmem_max is at least 32 MB" rmem_ok; then
        x sudo install -m 644 hardware/g1/superodom/99-homebody-udp.conf /etc/sysctl.d/
        x sudo sysctl --system
    fi
    # CycloneDDS 0.10.2 under the G1 SDK bindings of the exploration extra
    if ! skip_if "the venv imports cyclonedds" "$PY" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("cyclonedds") is None)'; then
        if [ "$PLAN" = 0 ]; then
            for command in git cmake c++; do
                command -v "$command" >/dev/null 2>&1 || refuse "building CycloneDDS needs $command" "install a C/C++ compiler, CMake and Git"
            done
        fi
        export CYCLONEDDS_HOME="$REPO/.cache/cyclonedds-0.10.2"
        if [ "$PLAN" = 1 ] || [ ! -d .cache/build/cyclonedds-src ]; then
            x mkdir -p .cache/build
            x git clone --depth 1 --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git .cache/build/cyclonedds-src
        fi
        if [ "$PLAN" = 1 ] || [ ! -x "$CYCLONEDDS_HOME/bin/idlc" ]; then
            x cmake -S .cache/build/cyclonedds-src -B .cache/build/cyclonedds \
                -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_IDLC=ON -DENABLE_SSL=NO -DENABLE_SECURITY=NO -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
            x cmake --build .cache/build/cyclonedds --parallel 2
            x cmake --install .cache/build/cyclonedds
        fi
        x env CYCLONEDDS_HOME="$CYCLONEDDS_HOME" "$PY" -m pip install -e '.[exploration]'
        [ "$PLAN" = 1 ] || say "keep CYCLONEDDS_HOME=$CYCLONEDDS_HOME set for later installs and launches"
    fi
}

step_odometry() {
    skip_if "docker already has $SLAM_IMAGE" docker image inspect "$SLAM_IMAGE" && return 0
    if [ "$SLAM_IMAGE" = "$LOCK_SLAM" ]; then
        x bash hardware/g1/superodom/build.sh odometry
    else
        x docker pull "$SLAM_IMAGE"
    fi
}

step_recorder() {
    if [ "$RECORD" != 1 ] && off "exploration.record is off and only recording uses $RECORDER_IMAGE"; then return 0; fi
    skip_if "docker already has $RECORDER_IMAGE" docker image inspect "$RECORDER_IMAGE" && return 0
    if [ "$RECORDER_IMAGE" = "$LOCK_RECORDER" ]; then
        # hardware/g1/recorder/Dockerfile builds FROM the pinned SuperOdometry tag
        [ "$PLAN" = 1 ] || docker image inspect "$LOCK_SLAM" >/dev/null 2>&1 \
            || refuse "the recorder builds on $LOCK_SLAM, which is not built" "bash hardware/g1/superodom/build.sh odometry"
        x bash hardware/g1/superodom/build.sh recorder
    else
        x docker pull "$RECORDER_IMAGE"
    fi
}

step_copy() {
    relay_off && return 0
    skip_if "the robot user's $REMOTE_DIR/ matches $PHONE_DIR/" copy_current && return 0
    robot "mkdir -p $REMOTE_DIR"
    robot_copy "$PHONE_DIR/mediamtx.yml" "$PHONE_DIR/$UNIT"
}

step_phone() {
    relay_off && return 0
    [ "$PLAN" = 1 ] || robot "test -f $REMOTE_DIR/$UNIT" >/dev/null 2>&1 \
        || refuse "the relay files are not on the robot" "bash hardware/g1/install.sh --only copy"
    if ! skip_if "the robot has $RELAY_BIN" robot "test -x $RELAY_BIN"; then
        # shellcheck disable=SC2016  # expands on the robot
        robot 'cd '"$REMOTE_DIR"' || exit 1; if [ "$(uname -m)" != aarch64 ]; then echo "the pinned relay is linux_arm64; this computer is $(uname -m)" >&2; exit 1; fi; mkdir -p download && curl --fail --location --output download/mediamtx.tar.gz '"$MEDIAMTX_URL"' && printf "%s  %s\n" '"$MEDIAMTX_SHA256"' download/mediamtx.tar.gz | sha256sum --check - && tar -xzf download/mediamtx.tar.gz -C download mediamtx LICENSE'
        robot_root "install -d -m 755 ${RELAY_BIN%/*} ${RELAY_CONF%/*} && install -m 755 $REMOTE_DIR/download/mediamtx $RELAY_BIN && install -m 644 $REMOTE_DIR/download/LICENSE ${RELAY_BIN%/*}/LICENSE"
    fi
    # The allowlists: reader = this workstation's robot address; publisher = the phone's USB address,
    # which no setting holds. An installed config is the operator's and is kept.
    if ! skip_if "the robot has $RELAY_CONF" robot "test -f $RELAY_CONF"; then
        local ws_ip; ws_ip="$(workstation_ip)"
        robot "cd $REMOTE_DIR && { [ -f phone.local.yml ] || awk -v ip=$ws_ip '/127\.0\.0\.1/ && ++n == 2 { sub(/127\.0\.0\.1/, ip) } 1' mediamtx.yml > phone.local.yml; }"
        if [ "$PLAN" = 1 ]; then
            echo "    stops until 127.0.0.1 in ~/$REMOTE_DIR/phone.local.yml on the robot is the phone's USB source address"
        elif robot "grep -q '127\.0\.0\.1' $REMOTE_DIR/phone.local.yml" >/dev/null 2>&1; then
            refuse "the relay's publisher allowlist still holds the 127.0.0.1 placeholder" \
                "on the robot, put the phone's USB source address (PHONE_USB_IP in guides/phone-camera.md) in ~/$REMOTE_DIR/phone.local.yml, then bash hardware/g1/install.sh --only phone"
        fi
        robot "$RELAY_BIN --validate-conf=$REMOTE_DIR/phone.local.yml"
        robot_root "install -m 644 $REMOTE_DIR/phone.local.yml $RELAY_CONF"
    fi
    # Enabled for boot, not started: explore setup starts it and each launch restarts it.
    robot_root "cmp -s $REMOTE_DIR/$UNIT /etc/systemd/system/$UNIT || { install -m 644 $REMOTE_DIR/$UNIT /etc/systemd/system/ && systemctl daemon-reload; }; systemctl enable $UNIT"
}

step_verify() {
    # shellcheck disable=SC2054  # the commas are --tmpfs options
    local failed=0 iface checks=(docker run --rm --network none --read-only --cap-drop ALL
        --security-opt no-new-privileges --tmpfs /tmp:rw,size=256m -e HOME=/tmp -e ROS_DOMAIN_ID=71
        -e ROS_LOCALHOST_ONLY=1 -v "$REPO/hardware/g1/superodom:/checks:ro")
    x "$PY" -m homebody real2sim explore check ${SETTINGS_FILE:+--settings "$SETTINGS_FILE"} || failed=1
    # the device-free image checks of hardware/g1/superodom/README.md
    x "${checks[@]}" "$SLAM_IMAGE" python3 /checks/check_image.py || failed=1
    if [ "$RECORD" = 1 ]; then
        x "${checks[@]}" --entrypoint /bin/bash "$RECORDER_IMAGE" -lc 'source /opt/ros/humble/setup.bash; source /opt/superodom_ws/install/setup.bash; source /opt/unitree_msgs/setup.bash; python3 /checks/check_image.py --recorder' || failed=1
    fi
    robot "date +%s.%N" || failed=1
    [ "$IPHONE_RELAY" != "$UNIT" ] || robot "systemctl is-enabled $UNIT" || failed=1
    # read-only status probes; they report rates and state and start nothing
    iface="$(robot_iface)" || exit 1
    x bash hardware/g1/superodom/run_superodom_ws.sh --iface "$iface" --image "$SLAM_IMAGE" probe /utlidar/cloud_livox_mid360 5 || true
    x bash hardware/g1/superodom/run_superodom_ws.sh --iface "$iface" --image "$SLAM_IMAGE" probe /utlidar/imu_livox_mid360 5 || true
    x bash hardware/g1/superodom/run_superodom_ws.sh --iface "$iface" --image "$SLAM_IMAGE" status || true
    x bash src/homebody/real2sim/exploration/record/record.sh status || true
    [ "$failed" = 0 ] || refuse "a check above failed" "fix it, then bash hardware/g1/install.sh --only <step> and --only verify"
}

title() {
    case "$1" in
        prereqs) echo "workstation commands, UDP buffers, CycloneDDS 0.10.2 and the exploration extra" ;;
        odometry) echo "SuperOdometry image $SLAM_IMAGE on this workstation" ;;
        recorder) echo "recorder image $RECORDER_IMAGE on this workstation" ;;
        copy) echo "hardware/g1/phone/ to ~/$REMOTE_DIR on $SSH_TARGET" ;;
        phone) echo "MediaMTX relay and $UNIT on $SSH_TARGET (installed and enabled, not started)" ;;
        verify) echo "explore check, image checks, robot SSH and the read-only status probes" ;;
    esac
}

if [ "$PLAN" = 1 ]; then echo "G1 install plan: nothing below runs under --plan"; else echo "G1 install"; fi
echo "  settings     configs/exploration.toml${SETTINGS_FILE:+ + $SETTINGS_FILE}"
echo "  robot        $SSH_TARGET, password *** ($([ -n "$PASSWORD" ] && echo "--password" || echo "robot.ssh_password or HOMEBODY_G1_SSH_PASSWORD"))"
echo "  workstation  robot.iface ${IFACE:-<empty: the only 192.168.123.x link>}"
echo "  images       slam.image $SLAM_IMAGE; recording.image $RECORDER_IMAGE (recording $([ "$RECORD" = 1 ] && echo on || echo off))"
echo "  phone        iphone.relay_service ${IPHONE_RELAY:-<empty>}; iphone.input_url $([ -n "$IPHONE_INPUT" ] && echo set || echo empty)"
n=0
for STEP in "${STEPS[@]}"; do
    n=$((n + 1))
    [ -z "$ONLY" ] || [ "$ONLY" = "$STEP" ] || continue
    echo "[$n/${#STEPS[@]}] $STEP: $(title "$STEP")"
    "step_$STEP"
done
if [ "$PLAN" = 1 ]; then
    echo "plan only: nothing was run, built, copied or started"
else
    say "done; next: homebody real2sim explore check, then the supervised explore setup (guides/exploration.md)"
fi
