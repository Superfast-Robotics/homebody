"""`homebody real2sim explore`: resolve the settings once, then check, plan, set up or run.

Help, `check` and `--plan` start nothing. `setup` and a run write the resolved settings to
`<run>/meta/settings.json` and hand that run to hardware/g1/explore.sh, which owns every
robot-side process."""
import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

from homebody.vlm.catalog import requirement

from .settings import Settings, load, ssh_password, to_dict, to_json

ROBOT_SUBNET = "192.168.123."
DEFAULTS = Settings()
# flag: (group, key, help); the default shown is the settings default
OPTIONS = {
    "task": ("exploration", "task", "Exploration instruction; the operator still presses Start exploration"),
    "budget-s": ("exploration", "budget_s", "Active exploration budget in seconds"),
    "host": ("exploration", "host", "Console bind address"),
    "port": ("exploration", "port", "Console port"),
    "output": ("exploration", "output", "Run directory root (default: <release>/runs/explore)"),
    "model": ("agent", "model", "Decision model, by its name in the model catalog"),
    "reasoning-effort": ("agent", "reasoning_effort", "Model reasoning effort"),
    "service-tier": ("agent", "service_tier", "Agent service tier"),
    "decision-timeout-s": ("agent", "timeout", "Decision deadline in seconds"),
    "iface": ("robot", "iface", "Workstation network interface connected to the G1"),
    "iphone-input": ("iphone", "input_url", "FFmpeg input URL, e.g. rtsp://<relay>:8554/iphone"),
    "iphone-endpoint": ("iphone", "endpoint", "Local RGB receiver endpoint"),
    "iphone-relay-service": ("iphone", "relay_service",
                             "Jetson systemd relay unit to manage; otherwise the video service is left alone"),
}
SWITCHES = {
    "observe-only": ("exploration", "observe_only",
                     "Run live sensing without deploying the body controller or exploring"),
    "enable-ask-operator": ("exploration", "ask_operator",
                            "Allow the model to pause exploration and ask the operator a question"),
    "adopt": ("slam", "adopt", "Keep an already healthy workstation SLAM instance"),
}
OMITTED_FROM_PLAN = {("exploration", "task"), ("iphone", "input_url"), ("iphone", "relay_service")}


def release_root(explicit=None):
    """The release tree holding configs/exploration.toml: EXPLICIT, else the source tree or the working directory."""
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "configs/exploration.toml").is_file():
            raise ValueError("release root needs configs/exploration.toml; pass --root PATH")
        return root
    package_root = Path(__file__).resolve().parents[4]
    cwd = Path.cwd().resolve()
    for root in (package_root, cwd, *cwd.parents):
        if (root / "configs/exploration.toml").is_file():
            return root
    raise ValueError("run from the release checkout or pass --root PATH containing configs/exploration.toml")


def parser():
    value = argparse.ArgumentParser(prog="homebody real2sim explore",
                                    description="G1 + LiDAR + iPhone exploration")
    value.add_argument("operation", choices=("run", "check", "setup"), nargs="?", default="run")
    value.add_argument("--root", type=Path, help="Release root containing configs, hardware and assets")
    value.add_argument("--settings", type=Path, help="Exploration TOML overrides")
    for flag, (group, key, help_text) in OPTIONS.items():
        default = getattr(getattr(DEFAULTS, group), key)
        shown = f" (default: {default})" if default not in ("", None) and "default:" not in help_text else ""
        value.add_argument("--" + flag, type=type(default), help=help_text + shown)
    value.add_argument("--record", action=argparse.BooleanOptionalAction, default=None,
                       help="Keep full local sensor, video and model evidence (default: "
                       + ("on)" if DEFAULTS.exploration.record else "off)"))
    for flag, (_group, _key, help_text) in SWITCHES.items():
        value.add_argument("--" + flag, action="store_true", default=None, help=help_text)
    value.add_argument("--plan", action="store_true", help="Show resolved settings without starting services")
    return value


def options(args) -> dict:
    """The command-line values as settings tables; unset options are None and leave the files' values."""
    tables = {"exploration": {"record": args.record}}
    for flag, (group, key, _help) in {**OPTIONS, **SWITCHES}.items():
        tables.setdefault(group, {})[key] = getattr(args, flag.replace("-", "_"))
    return tables


def plan(settings) -> dict:
    """The resolved settings without the task text, the phone URL or the relay unit."""
    groups = to_dict(settings)
    for group, key in OMITTED_FROM_PLAN:
        groups[group].pop(key)
    return groups


def check(root, settings):
    missing = []
    for package in ("numpy", "scipy", "PIL", "cv2", "zmq", "matplotlib", "viser", "yourdfpy",
                    "torch", "cyclonedds", "holosoma_inference", "skimage", "unitree_sdk2py", "unitree_interface"):
        found = find_spec(package) is not None
        print(f"{'OK' if found else 'MISSING'} package {package}")
        if not found:
            missing.append(package)
    executable, key = requirement(settings.agent.provider)
    commands = ("docker", "ip", "ssh", "sshpass", "ffmpeg", "ffprobe") + ((executable,) if executable else ())
    for command in commands:
        found = shutil.which(command) is not None
        print(f"{'OK' if found else 'MISSING'} command {command}")
        if not found:
            missing.append(command)
    if key:
        found = bool(os.environ.get(key))
        print(f"{'OK' if found else 'MISSING'} provider key {key}")
        if not found:
            missing.append(key)
    for name in ("amo_jit.pt", "adapter_jit.pt", "adapter_norm_stats.pt"):
        found = (root / "assets/models" / name).is_file()
        print(f"{'OK' if found else 'MISSING'} walking asset {name}")
        if not found:
            missing.append(name)
    print("Local file/import checks only. No services, robot connection or model request started.")
    print("Build the SLAM image explicitly with hardware/g1/superodom/build.sh.")
    if settings.exploration.record:
        print("Recording also requires the recorder image.")
    return int(bool(missing))


def robot_interface(name):
    """NAME when it carries a robot-network address, else the one interface that does (ValueError otherwise)."""
    if shutil.which("ip") is None:
        raise ValueError("required command 'ip' is not installed (iproute2)")
    command = ["ip", "-o", "-4", "addr", "show"] + (["dev", name] if name else ["up"])
    lines = subprocess.run(command, capture_output=True, text=True, check=False).stdout.splitlines()
    links = {f[1] for f in map(str.split, lines) if len(f) > 3 and f[3].startswith(ROBOT_SUBNET)}
    if name:
        if name not in links:
            raise ValueError(f"--iface {name} has no {ROBOT_SUBNET}x address; name the link on the robot network")
        return name
    if len(links) != 1:
        raise ValueError(f"expected one {ROBOT_SUBNET}x robot interface, found {len(links)}; pass --iface")
    return links.pop()


def minimal_summary(workspace, settings, *, code, elapsed, usage):
    """The diagnostics summary: named numeric fields and closed lifecycle values only."""
    state = {}
    try:
        state = json.loads((workspace / "explore/live_state.json").read_text())
    except (OSError, ValueError):
        pass
    allowed_phases = {"idle", "starting", "explore", "preview", "pausing", "paused",
                      "stopped", "done", "aborted", "error", "arming", "restarting SLAM"}
    timing = {key: state[key] for key in ("elapsed_s", "active_elapsed_s", "paused_elapsed_s")
              if type(state.get(key)) in (int, float) and 0 <= state[key] < 1e12}
    agent, exploration = settings.agent, settings.exploration
    return {"schema": 1, "recording": exploration.record, "exit_code": code,
            "result": "process_closed" if code == 0 else "process_failed",
            "last_phase": state.get("phase") if state.get("phase") in allowed_phases else None,
            "wall_elapsed_s": elapsed, "timing": timing,
            "resources": {"child_user_cpu_s": usage.ru_utime,
                          "child_system_cpu_s": usage.ru_stime,
                          "child_peak_rss_kib": usage.ru_maxrss},
            "settings": {"model": agent.model, "reasoning_effort": agent.reasoning_effort,
                         "service_tier": agent.service_tier, "decision_timeout_s": agent.timeout,
                         "budget_s": exploration.budget_s, "record": exploration.record,
                         "observe_only": exploration.observe_only,
                         "ask_operator": exploration.ask_operator}}


def launch(root, settings, *, setup=False):
    """Run explore.sh on a new run directory; the recorded settings carry the resolved output root."""
    output = Path(settings.exploration.output or root / "runs/explore").expanduser().resolve()
    settings = replace(settings, exploration=replace(settings.exploration, output=str(output)))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = output / f"{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True)
    started, code = time.monotonic(), 1
    record = settings.exploration.record
    workspace_context = nullcontext(str(run)) if record else tempfile.TemporaryDirectory(
        prefix="homebody-explore-")
    with workspace_context as directory:
        workspace = Path(directory)
        # the resolved SSH password reaches the launcher's processes here, never through a file
        child_env = dict(os.environ, HOMEBODY_G1_SSH_PASSWORD=ssh_password(settings),
                         HOMEBODY_ROOT=str(root), HOMEBODY_PYTHON=sys.executable,
                         HOMEBODY_WORKSPACE=str(workspace), HOMEBODY_RUN_DIR=str(run),
                         PYTHONPATH=os.pathsep.join((str(root / "src"), str(root / "vendor"))))
        (workspace / "meta").mkdir(exist_ok=True)
        (workspace / "meta/settings.json").write_text(to_json(settings))
        print(f"Exploration output: {run}\nRecording: {'on' if record else 'off (no sensor bag or phone video)'}", flush=True)
        process = None
        finished = False
        previous = {}
        requested = []
        def terminate(signum, _frame):
            # signal only the launch-owned shell; its Ctrl-\ handler is the emergency halt
            if requested and signum != signal.SIGQUIT:
                return
            requested.append(signum)
            if process is not None and process.poll() is None:
                process.send_signal(signum)
        try:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
                previous[signum] = signal.signal(signum, terminate)
            process = subprocess.Popen(["bash", str(root / "hardware/g1/explore.sh"),
                                        "setup" if setup else "run"], env=child_env)
            if requested:
                process.send_signal(requested[-1])
            code = process.wait()
            finished = True
            return code
        finally:
            # stop the launcher before its workspace is removed
            if process is not None and not finished:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                process.wait()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            summary = minimal_summary(workspace, settings, code=code,
                                      elapsed=time.monotonic() - started,
                                      usage=resource.getrusage(resource.RUSAGE_CHILDREN))
            (run / "diagnostics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))


def main(argv=None):
    value = parser()
    args = value.parse_args(argv)
    try:
        root = release_root(args.root)
        settings = load(root, args.settings, options(args))
    except (OSError, ValueError, TypeError) as error:
        value.error(str(error))
    if args.plan:
        print(json.dumps({"operation": args.operation, "settings": plan(settings)}, indent=2))
        print("No services, robot connection or provider call started.")
        return 0
    if args.operation == "check":
        return check(root, settings)
    try:
        iface = robot_interface(settings.robot.iface)
    except ValueError as error:
        print(f"refusing launch: {error}", file=sys.stderr)
        return 1
    return launch(root, replace(settings, robot=replace(settings.robot, iface=iface)),
                  setup=args.operation == "setup")


if __name__ == "__main__":
    raise SystemExit(main())
