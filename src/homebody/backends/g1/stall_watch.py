"""The owner of the process's `faulthandler` timer, the 250 Hz control loop's stall watch.

The loop re-arms the timer every ARM_EVERY_TICKS; a loop that misses STALL_S writes every
thread's stack to <directory>/stalls/<pid>.log, or to a temporary directory.
"""
from __future__ import annotations

import faulthandler
import os
import tempfile
import threading
import time
from pathlib import Path

STALL_S = 0.35            # a loop gap this long is a stall (a tick is 4 ms)
ARM_EVERY_TICKS = 25      # re-arm period in ticks, 100 ms at 250 Hz
ARM_WINDOW_S = 0.1  # timer slack past STALL_S between re-arms
STALL_FILE_MAX_BYTES = 5_000_000   # dumps stop past this size; the stamps continue
STALL_DIR = Path(tempfile.gettempdir()) / f"homebody-g1-stalls-{os.getuid()}"

class StallWatch:
    """The control loop's heartbeat: a gap of STALL_S after the last `arm` dumps every thread's
    stack to `<directory>/stalls/<pid>.log` (the system temp directory when none is given)."""

    def __init__(self, directory=None):
        self.directory = Path(directory) / "stalls" if directory else STALL_DIR
        self._file = None
        self._lock = threading.Lock()

    def _open(self):
        if self._file is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._file = (self.directory / f"{os.getpid()}.log").open("a")
        return self._file

    def arm(self) -> None:
        fh = self._open()
        if fh.tell() < STALL_FILE_MAX_BYTES:
            faulthandler.dump_traceback_later(STALL_S + ARM_WINDOW_S, repeat=False, file=fh)

    def disarm(self) -> None:
        """Cancel the timer and remove a stall file that stayed empty."""
        faulthandler.cancel_dump_traceback_later()
        with self._lock:
            if self._file is not None:
                path = self._file.name
                self._file.close()
                self._file = None
                if os.path.getsize(path) == 0:
                    os.remove(path)

    def note_stall(self, gap_s: float, where: str) -> None:
        """Stamp the stall that the dump above belongs to, once the loop ticks again."""
        with self._lock:
            fh = self._open()
            fh.write(f"== {time.strftime('%H:%M:%S')} {where}: no tick for {1000.0 * gap_s:.0f} ms ==\n")
            fh.flush()


# A forked child starts with the timer disarmed.
os.register_at_fork(before=faulthandler.cancel_dump_traceback_later)


