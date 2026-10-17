"""The only module that talks to the robot.

The password is passed through the ``SSHPASS`` environment variable and never appears
in an argv or an emitted string.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .settings import current, ssh_password

# one password-authenticated master connection shared by every command
SSH_MUX = ("-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=auto",
           "-o", "ControlPath=/tmp/homebody-ssh-%C", "-o", "ControlPersist=120")

SSH_TIMEOUT_S = 20
_MISSING = "MISSING"


class RobotUnreachable(RuntimeError):
    """The robot did not answer. The message names host, operation and the fix."""


def target() -> str:
    return current().robot.ssh_target


def password() -> str:
    return ssh_password()


def _digest_probe(paths: tuple[str, ...]) -> str:
    """One shell command that answers 'digest or MISSING' for every path at once."""
    quoted = " ".join(shlex.quote(p) for p in paths)
    return (f'for p in {quoted}; do if [ -f "$p" ]; then sha256sum "$p"; '
            f'else echo "{_MISSING}  $p"; fi; done')


def _write_command(remote: str, mode: int) -> str:
    """Make the parent, take the bytes on stdin, set the mode — one round trip."""
    quoted = shlex.quote(remote)
    return (f"mkdir -p {shlex.quote(str(Path(remote).parent))} && cat > {quoted}"
            f" && chmod {mode:o} {quoted}")


def _said(exc: BaseException) -> str:
    """What the remote command printed when it failed: a refusal that names the row."""
    lines = []
    for stream in (getattr(exc, "stdout", None), getattr(exc, "stderr", None)):
        if stream:
            text = stream.decode(errors="replace") if isinstance(stream, bytes) else str(stream)
            lines += [line for line in text.splitlines() if line.strip()]
    return f"; it said: {' | '.join(lines[-8:])}" if lines else ""


def parse_digests(stdout: str, paths: tuple[str, ...]) -> dict[str, str | None]:
    """``{path: digest or None}`` from the probe's output; unanswered paths stay None."""
    answers: dict[str, str | None] = {p: None for p in paths}
    for line in stdout.splitlines():
        head, _, path = line.strip().partition("  ")
        if path in answers and head != _MISSING:
            answers[path] = head
    return answers


@dataclass
class SshTransport:
    """Real I/O over ssh/scp to the robot; construction opens no connection."""

    host: str = field(default_factory=target)
    timeout_s: int = SSH_TIMEOUT_S

    def _env(self) -> dict[str, str]:
        return {**os.environ, "SSHPASS": password()}

    def _run(self, argv: list[str], operation: str, stdin: bytes | None = None):
        try:
            return subprocess.run(argv, env=self._env(), input=stdin, timeout=self.timeout_s,
                                  capture_output=True, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            raise RobotUnreachable(
                f"{self.host}: {operation} failed ({type(exc).__name__}){_said(exc)}. The "
                f"robot LAN is 192.168.123.0/24 — check the ethernet link and "
                f"robot.ssh_target in the exploration settings."
            ) from exc

    def _ssh(self, command: str, operation: str) -> str:
        argv = ["sshpass", "-e", "ssh", "-o", "BatchMode=no", *SSH_MUX,
                self.host, command]
        return self._run(argv, operation).stdout.decode()

    def run(self, command: str, operation: str = "remote command") -> str:
        return self._ssh(command, operation)

    def run_sudo(self, command: str, operation: str = "remote root command") -> str:
        """The same, as root on the robot; the password reaches ``sudo -S`` over stdin."""
        argv = ["sshpass", "-e", "ssh", "-o", "BatchMode=no", *SSH_MUX,
                self.host, f"sudo -S -p '' bash -c {shlex.quote(command)}"]
        return self._run(argv, operation, stdin=(password() + "\n").encode()).stdout.decode()

    def digests(self, paths: tuple[str, ...]) -> dict[str, str | None]:
        if not paths:
            return {}
        return parse_digests(self._ssh(_digest_probe(paths), "sha256 probe"), paths)

    def put(self, data: bytes, remote: str, mode: int) -> None:
        argv = ["sshpass", "-e", "ssh", *SSH_MUX, self.host,
                _write_command(remote, mode)]
        self._run(argv, f"write {remote}", stdin=data)
