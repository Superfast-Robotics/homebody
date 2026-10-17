"""Exercise the launcher's actual exit traps with local fake resources, never hardware."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + text)
    path.chmod(0o755)


@pytest.fixture
def launcher(tmp_path):
    source = (ROOT / "hardware/g1/explore.sh").read_text()
    # Only definitions and trap installation; none of the hardware startup is evaluated.
    start = source.index("cleanup() {")
    end = source.index("\n# 0b.", start)
    lifecycle = source[start:end]
    launch_tee = next(line for line in source.splitlines()
                      if line.startswith("exec > >(") and "launch.log" in line)
    console_tee = next(line for line in source.splitlines()
                       if line.lstrip().startswith("> >(") and "console.log" in line)
    # Deliver the duplicate at the trap window without relying on scheduler timing.
    lifecycle = lifecycle.replace("    trap - EXIT\n", "    trap - EXIT\n    inject_repeat\n", 1)
    executable(tmp_path / "src/homebody/real2sim/exploration/record/record.sh", '''
printf 'record:%s\n' "$1" >> "$RUN_DIR/actions"
if [ "$1" = stop ]; then
    rm -f "$RUN_DIR/recorder-running"
    sleep "${STOP_DELAY:-0}"
fi
if [ "$1" = stop ] && [ "${FAIL_RECORD_STOP:-0}" = 1 ]; then exit 1; fi
''')
    executable(tmp_path / "hardware/g1/superodom/run_superodom_ws.sh", '''
printf 'slam:%s\n' "$1" >> "$RUN_DIR/actions"
''')
    (tmp_path / "ws").mkdir()
    (tmp_path / "recorder-running").touch()
    script = tmp_path / "launcher.sh"
    script.write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$RUN_DIR"
RECORD=1
OBSERVE=1
WATCHDOG_FROZEN=0
PHONE_RECORDING_ROOT=""
APP_PID=""
FINALIZE_FAILED=0
LAUNCH_PID="$RUN_DIR/launcher.pid"
echo "$$" > "$LAUNCH_PID"
stop_phone_receiver() { echo phone:closed >> "$RUN_DIR/actions"; }
halt_now() { echo emergency:halt >> "$RUN_DIR/actions"; }
inject_repeat() {
    if [ -n "${REPEAT_SIGNAL:-}" ]; then
        echo repeat:sent >> "$RUN_DIR/actions"
        kill -s "$REPEAT_SIGNAL" "$$"
    fi
}
''' + lifecycle + '''
if [ "$EXIT_MODE" = lost-output ]; then
''' + launch_tee + '''
fi
if [ "$EXIT_MODE" = normal ]; then
    bash -c 'exit 0' &
elif [ "$EXIT_MODE" = lost-output ]; then
    bash -c 'while [ ! -f "$RUN_DIR/release-app" ]; do sleep .01; done; echo "[fake console] closed"' \\
''' + console_tee + '''
else
    sleep 30 >/dev/null 2>&1 &
fi
APP_PID=$!
echo ready > "$RUN_DIR/ready"
set +e
wait "$APP_PID"
code=$?
set -e
echo "[launch] the console exited with code $code"
exit "$code"
''')

    def run(mode="normal", repeat="", first=signal.SIGINT, fail_stop=False):
        env = dict(os.environ, RUN_DIR=str(tmp_path), EXIT_MODE=mode,
                   REPEAT_SIGNAL=repeat, FAIL_RECORD_STOP=str(int(fail_stop)))
        process = subprocess.Popen(["bash", str(script)], cwd=tmp_path, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True, text=True)
        try:
            deadline = time.monotonic() + 3.
            while not (tmp_path / "ready").exists():
                assert time.monotonic() < deadline, "fake launcher did not reach its wait"
                assert process.poll() is None, "fake launcher exited before readiness"
                time.sleep(.005)
            if mode == "signal":
                process.send_signal(first)
            if mode == "lost-output":
                process.stdout.close()
                (tmp_path / "release-app").touch()
                process.wait(timeout=5.)
                deadline = time.monotonic() + 1.
                while True:
                    output = (tmp_path / "launch.log").read_text()
                    if "persistent body host left balancing" in output or time.monotonic() >= deadline:
                        break
                    time.sleep(.005)
            else:
                output, _ = process.communicate(timeout=5.)
        finally:
            # Only the isolated test process group; no production process is selected.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return process.returncode, output

    return tmp_path, run


def assert_closed(path, output):
    actions = (path / "actions").read_text().splitlines()
    assert actions.count("phone:closed") == 1
    assert actions.count("record:stop") == 1
    assert actions.count("slam:down-owned") == 1
    assert actions.count("record:export") == 1
    assert actions.count("record:capture") == 1
    assert not (path / "recorder-running").exists()
    assert not (path / "launcher.pid").exists()
    assert output.count("[launch] exiting with code") == 1
    assert "persistent body host left balancing" in output
    return actions


def test_normal_console_exit_closes_every_owned_resource_once(launcher):
    path, run = launcher
    code, output = run()
    assert code == 0
    assert "emergency:halt" not in assert_closed(path, output)


@pytest.mark.skipif(sys.platform != "linux", reason="the launcher's log tee uses GNU tail --pid")
def test_lost_output_reader_preserves_logs_and_closes_owned_resources(launcher):
    path, run = launcher
    code, output = run("lost-output")
    assert code == 0
    assert "emergency:halt" not in assert_closed(path, output)
    assert "[fake console] closed" in output
    assert (path / "console.log").read_text() == "[fake console] closed\n"


@pytest.mark.parametrize("first,expected", [(signal.SIGINT, 130), (signal.SIGTERM, 130),
                                          (signal.SIGHUP, 129)])
@pytest.mark.parametrize("repeat", ["INT", "TERM", "HUP"])
def test_repeated_interrupt_cannot_skip_recorder_closure(launcher, first, expected, repeat):
    path, run = launcher
    code, output = run("signal", repeat, first)
    assert code == expected
    actions = assert_closed(path, output)
    assert actions.count("repeat:sent") == 1
    assert "emergency:halt" not in actions


def test_interrupt_during_normal_exit_still_closes_recorder(launcher):
    path, run = launcher
    code, output = run(repeat="INT")
    assert code == 0
    assert_closed(path, output)


def test_failed_recorder_close_reports_failure_but_finishes_other_cleanup(launcher):
    path, run = launcher
    code, output = run(fail_stop=True)
    assert code == 1
    assert_closed(path, output)
    assert "output finalization failed" in output


def test_emergency_quit_still_halts_and_closes_recording(launcher):
    path, run = launcher
    code, output = run("signal", first=signal.SIGQUIT)
    assert code == 131
    actions = assert_closed(path, output)
    assert actions.count("emergency:halt") == 1


def test_emergency_quit_during_cleanup_halts_without_skipping_bag_close(launcher):
    path, run = launcher
    code, output = run("signal", repeat="QUIT")
    assert code == 130
    actions = assert_closed(path, output)
    assert actions.count("emergency:halt") == 1


def test_cli_forwards_foreground_interrupt_and_waits_for_cleanup(launcher):
    path, _ = launcher
    script = path / "hardware/g1/explore.sh"
    script.write_text((path / "launcher.sh").read_text())
    wrapper = '''
import json, sys
from dataclasses import replace
from pathlib import Path
from homebody.real2sim.exploration import cli, settings
root = Path(sys.argv[1])
original = cli.subprocess.Popen
class TracedProcess(original):
    def send_signal(self, signum):
        with (root / "forwarded.jsonl").open("a") as stream:
            stream.write(json.dumps(int(signum)) + "\\n")
        return super().send_signal(signum)
cli.subprocess.Popen = TracedProcess
configured = settings.Settings()
configured = replace(configured, exploration=replace(
    configured.exploration, output=str(root / "cli-output"), record=True))
raise SystemExit(cli.launch(root, configured))
'''
    env = dict(os.environ, RUN_DIR=str(path), EXIT_MODE="signal", REPEAT_SIGNAL="INT",
               STOP_DELAY=".15", HOMEBODY_G1_SSH_PASSWORD="synthetic-test-secret",
               PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "vendor"))))
    process = subprocess.Popen([sys.executable, "-c", wrapper, str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                               start_new_session=True, text=True)
    try:
        deadline = time.monotonic() + 5.
        while not (path / "ready").exists():
            assert time.monotonic() < deadline, "CLI did not reach the fake launcher"
            assert process.poll() is None, "CLI exited before readiness"
            time.sleep(.005)
        os.killpg(process.pid, signal.SIGINT)
        output, _ = process.communicate(timeout=5.)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    assert process.returncode == 130, output
    assert_closed(path, output)
    assert (path / "forwarded.jsonl").read_text().splitlines() == [str(signal.SIGINT.value)]
    reports = list((path / "cli-output").glob("*/diagnostics.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["exit_code"] == 130
