"""Drive events buffered in memory; the console loop owns disk writes."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from homebody.primitives.navigation.contracts import jsonable


class DriveLog:
    def __init__(self, path: Path, capacity: int = 8192):
        self.path, self.capacity = Path(path), capacity
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self.dropped = self._reported_drops = 0
        self.error = ""

    def emit(self, context: int, kind: str, payload=None) -> None:
        if self.error or kind == "observe":  # poses are tapped; never copy images or maps
            return
        checker = None
        if kind == "drive.collision_refused":
            payload = dict(payload or {})
            checker = payload.pop("_checker", None)
        grid = None
        if checker is not None:
            # copy the refused collision grid; the console flush writes it
            grid = {"free": np.asarray(checker.grid, dtype=bool).copy()}
            if getattr(checker, "exact_footprint", False):
                payload["collision_geometry"] = "exact_circle_cells_v1"
                for name in ("occupied", "obstacles", "center_allowed"):
                    value = getattr(checker, name, None)
                    if value is not None:
                        grid[name] = np.asarray(value, dtype=bool).copy()
            payload["grid_metadata"] = {k: getattr(checker, k) for k in
                                        ("xmin", "ymin", "cell", "nx", "ny", "radius_baked")}
            payload["collision_grid"] = f"drive_collision_{context}_{time.monotonic_ns()}.npz"
        row = {"wall_s": time.time(), "monotonic_s": time.monotonic(),
               "context": context, "event": kind, "data": jsonable(payload or {})}
        if grid is not None:
            row["_collision_grid"] = grid
        with self._lock:
            if len(self._rows) < self.capacity:
                self._rows.append(row)
            else:
                self.dropped += 1

    def flush(self) -> None:
        """Append the buffered rows to the log; called by the console, without fsync."""
        with self._lock:
            rows, self._rows = self._rows, []
            dropped = self.dropped
        if self.error or (not rows and dropped == self._reported_drops):
            return
        if dropped != self._reported_drops:
            rows.append({"wall_s": time.time(), "monotonic_s": time.monotonic(),
                         "event": "logging.gap", "data": {"dropped_total": dropped}})
        try:
            for row in rows:
                grid = row.pop("_collision_grid", None)
                if grid is not None:
                    np.savez_compressed(self.path.parent / row["data"]["collision_grid"], **grid)
            text = "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                           for row in rows)
            with self.path.open("a") as stream:
                stream.write(text)
            self._reported_drops = dropped
        except (OSError, TypeError, ValueError) as exc:
            # a logging failure is recorded, never raised
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                print(f"[drive log] recording incomplete: {self.path}: {self.error}", flush=True)
            except (OSError, ValueError):  # Ctrl-C can close the launcher's output pipe first
                pass
