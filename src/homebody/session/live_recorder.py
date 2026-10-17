"""Persistent live task records; the bag recorder owns sensor acquisition."""
import json
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

from .recorder import json_value, revision, write_json


class LiveRecorder:
    """Record Session events and retain provider attempts without rendering sensor frames."""

    def __init__(self, path: Path, *, settings, root: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self._lock = threading.RLock()
        self._task_count = 0
        self._task_path = None
        self._failure = None
        self._closed = False
        try:
            self._finalization("open")
            write_json(self.path / "provenance.json", {
                "schema_version": 1, "backend": "g1",
                "created_utc": datetime.now(UTC).isoformat(), "revision": revision(root),
                "settings": asdict(settings) if is_dataclass(settings) else dict(settings),
                "sensor_evidence": "The enclosing capture retains native bags, video and clock metadata.",
            })
            self._events = (self.path / "events.jsonl").open("a", buffering=1)
        except Exception as error:
            self._failed(error)
            raise

    @property
    def failure(self) -> str | None:
        return None if self._failure is None else f"{type(self._failure).__name__}: {self._failure}"

    def _finalization(self, status):
        temporary = self.path / "finalization.json.tmp"
        write_json(temporary, {"status": status, "failure": self.failure})
        temporary.replace(self.path / "finalization.json")

    def _failed(self, error):
        if self._failure is None:
            self._failure = error
        try:
            self._finalization("failed")
        except OSError:
            return  # A full or unavailable disk cannot persist its own failure receipt.

    @property
    def recording(self) -> bool:
        return self._task_path is not None

    def event(self, kind: str, **fields):
        with self._lock:
            try:
                row = json.dumps({"kind": kind, "wall_time": time.time(), **fields},
                                 default=json_value, allow_nan=False) + "\n"
                self._events.write(row)
                self._events.flush()
            except Exception as error:
                self._failed(error)
                raise

    def start_task(self, task: str, epoch: int | None) -> Path:
        with self._lock:
            if self._events.closed:
                raise RuntimeError("Live recorder is closed")
            if self._task_path is not None:
                raise RuntimeError("Another task is already recording")
            try:
                path = self.path / "tasks" / f"task-{self._task_count:03d}"
                path.mkdir(parents=True, exist_ok=False)
                self._task_count += 1
                self._task_path = path
                write_json(path / "task.json", {
                    "task": task, "epoch": epoch, "started_unix": time.time(),
                })
                self.event("task_started", task=task, epoch=epoch, path=path.relative_to(self.path))
                return path
            except Exception as error:
                self._failed(error)
                raise

    def end_task(self, outcome: dict):
        with self._lock:
            if self._task_path is None:
                return
            path = self._task_path
            try:
                temporary = path / "outcome.json.tmp"
                write_json(temporary, {**outcome, "finished_unix": time.time(),
                                       "independently_scored": False})
                temporary.replace(path / "outcome.json")
                self._task_path = None
                self.event("task_finished", **outcome)
            except Exception as error:
                self._failed(error)
                raise
            finally:
                try:
                    self._events.flush()
                except Exception as error:
                    self._failed(error)
                    raise

    def close(self):
        """Finalize an open task and close the event log even if finalization raises."""
        with self._lock:
            if self._closed:
                if self._failure is not None:
                    raise self._failure
                return
            try:
                self.end_task({"status": "interrupted", "reason": "Session closed during task"})
            except Exception as error:  # noqa: BLE001 - the retained failure is re-raised below
                self._failed(error)
            finally:
                try:
                    self._events.close()
                except Exception as error:  # noqa: BLE001 - the retained failure is re-raised below
                    self._failed(error)
                self._closed = True
            try:
                self._finalization("closed" if self._failure is None else "failed")
            except Exception as error:  # noqa: BLE001 - the retained failure is re-raised below
                self._failed(error)
            if self._failure is not None:
                raise self._failure
