"""The sole writer: a flock-enforced singleton on the arm/gripper command path.

A second writer on the same motors is refused at start-up. The lock owner is found by file
descriptor, never by process name; the lease file carries the owner's pid.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import socket
from pathlib import Path

from homebody.real2sim.exploration.settings import current


def lease_path(explicit: str | Path | None = None) -> Path:
    """The writer lease: the explicit path, else the body settings' `writer_lock_path`."""
    return Path(explicit or current().body.writer_lock_path)


class WriterHeld(RuntimeError):
    """The writer flock is already taken; the message names the pid to kill."""


def hunt(lock_path: str | Path) -> tuple[tuple[int, ...], int]:
    """``(pids with the lease open, unreadable processes)``: an in-process `fuser` over /proc,
    matching the lease by `(st_dev, st_ino)`."""
    target = Path(lock_path)
    try:
        info = target.stat()
        node = (info.st_dev, info.st_ino)
    except OSError:
        return (), 0
    found, unreadable = [], 0
    for entry in Path("/proc").glob("[0-9]*/fd"):
        try:
            pid = int(entry.parent.name)
        except ValueError:
            continue
        try:
            links = list(entry.iterdir())
        except PermissionError:
            unreadable += 1               # another user's process
            continue
        except OSError:
            continue                      # it exited mid-walk
        for link in links:
            try:
                stat = link.resolve().stat()
                if (stat.st_dev, stat.st_ino) == node:
                    found.append(pid)
                    break
            except OSError:
                continue
    return tuple(sorted(set(found))), unreadable


class SoleWriter:
    """The exclusive command lease on one robot's arm/gripper motors: raises WriterHeld when
    another process owns them; the OS releases it when the process exits."""

    def __init__(self, lock_path: str | Path):
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._namespace = None
        self._fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            # The by-fd hunt first, the recorded pid as a fallback.
            live, unreadable = hunt(self.lock_path)
            live = tuple(p for p in live if p != os.getpid())
            recorded = self._holder_pid()
            os.close(self._fd)
            self._fd = -1
            if live:
                who = f" by pid {', '.join(map(str, live))} (found by fd)"
            elif recorded:
                who = (f" by pid {recorded} (the pid in the lease file; no open descriptor was found, "
                       f"so it may already be dead)")
            else:
                who = " by a process that could not be identified"
            raise WriterHeld(
                f"the sole-writer lock {self.lock_path} is held{who}"
                + (f"; {unreadable} process(es) were not readable from /proc, re-run with "
                   f"sudo -E to see them" if unreadable else "")
                + ". Stop that pid with kill -TERM <pid>; do not pkill by name, since torch "
                  "renames its spawn child to pt_main_thread"
            ) from exc
        self._claim_namespace()
        os.truncate(self._fd, 0)
        os.write(self._fd, str(os.getpid()).encode())

    def _claim_namespace(self) -> None:
        """Bind an abstract unix socket named after the lease path: a singleton with no
        filesystem entry, released by the kernel when the process exits."""
        key = hashlib.sha256(str(self.lock_path.resolve()).encode()).hexdigest()[:16]
        self._namespace = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self._namespace.bind("\0homebody-sole-writer-" + key)
        except OSError as exc:
            self._namespace.close()
            self._namespace = None
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1
            live, _blind = hunt(self.lock_path)
            raise WriterHeld(
                f"another process already owns the motors for {self.lock_path}: it holds the "
                f"abstract-namespace claim"
                + (f" (pid {', '.join(map(str, live))})" if live else
                   ", and the lease file is gone or unheld; removing the lease does not free the motors")
                + ". Stop that pid, then re-run"
            ) from exc

    def _holder_pid(self) -> int | None:
        """The pid written into the lease file at acquisition, or None."""
        try:
            text = self.lock_path.read_text().strip()
            return int(text) if text else None
        except (OSError, ValueError):
            return None

    @property
    def held(self) -> bool:
        return self._fd >= 0

    def release(self) -> None:
        if self._namespace is not None:
            self._namespace.close()
            self._namespace = None
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1
