"""The workstation SuperOdometry engine: start, stop and health, through local Docker only.

Each verb calls hardware/g1/superodom/run_superodom_ws.sh with the robot interface, image and
cloud delay from the installed exploration settings."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .settings import current

SCRIPT = Path(__file__).resolve().parents[4] / "hardware/g1/superodom/run_superodom_ws.sh"


class SuperodomWsError(RuntimeError):
    """The workstation engine did not complete the requested lifecycle operation."""


def _run(*verb: str, timeout: float = 30.0) -> tuple[int, str]:
    proc = subprocess.run([str(SCRIPT), *verb], capture_output=True, text=True, timeout=timeout,
                          check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _engine() -> tuple[str, ...]:
    settings = current()
    return "--iface", settings.robot.iface, "--image", settings.slam.image


def _checked_run(*verb: str) -> str:
    try:
        code, out = _run(*verb, timeout=60.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SuperodomWsError(f"superodom_ws {' '.join(verb)} failed: {exc}") from exc
    if code != 0:
        raise SuperodomWsError(out or f"superodom_ws {' '.join(verb)} exited with status {code}")
    return out


def status() -> tuple[bool, str]:
    """(healthy, line): clouds reaching the engine, /ws odometry rate, the engine's own health."""
    try:
        code, out = _run(*_engine(), "status", timeout=40.0)
    except subprocess.TimeoutExpired:
        return False, "superodom_ws: the status probe did not answer within 40 s"
    return code == 0, out.splitlines()[-1] if out else "superodom_ws: no answer"


def up() -> str:
    slam = current().slam
    hold = (("--hold-margin-s", f"{slam.hold_margin_s:g}", "--hold-max-s", f"{slam.hold_max_s:g}")
            if slam.hold else ())
    return _checked_run(*_engine(), "up", *hold)


def down() -> str:
    return _checked_run("down")
