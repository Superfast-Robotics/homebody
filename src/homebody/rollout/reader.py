"""A recorded rollout read back for replay (rollout/replay.py): one task's session events, its frames, and the
monitor's readings and verdicts: the live run's `monitor/`, or the offline evaluator's `monitor-offline/` (`source`).
An imported rollout's rows are checked, not trusted: an unreadable row is counted and skipped.
"""
import json
from pathlib import Path

from homebody.monitor.offline import FOLDER as OFFLINE
from homebody.monitor.offline import chosen, read_events
from homebody.monitor.targets import Claim, Label, Vocabulary, reading_from, verdict_from
from homebody.rollout.paths import inside


def vocabulary_from(event: dict) -> Vocabulary:
    return Vocabulary(event.get("goal_id") or "task",
                      tuple(Label(**{**label, "stages": tuple(label.get("stages", ()))})
                            for label in event.get("labels", ())),
                      tuple(Claim(**claim) for claim in event.get("standing", ())), source=event.get("source", ""))


def parsed(read, rows) -> list:
    """READ of each of ROWS that reads (an imported rollout's monitor events are checked, not trusted)."""
    kept = []
    for row in rows:
        try:
            kept.append(read(row))
        except (TypeError, KeyError, ValueError, AttributeError):
            continue
    return kept


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


class Rollout:
    def __init__(self, root: Path, task: str | None = None, source: str = "auto"):
        records = read_events(root)
        found = next(chosen(records, task), None)
        if found is None:
            raise ValueError(f"{root.name} has no task {task}" if task else f"{root.name} has no recorded task")
        self.root, (self.task_name, started, rest) = root, found
        self.records = rest  # the task's session events, in order
        self.instruction = started.get("task", "")
        folder = root / "tasks" / self.task_name
        self.has_live, self.has_offline = (folder / "monitor/readings.jsonl").is_file(), \
            (folder / OFFLINE / "readings.jsonl").is_file()
        self.source = (("offline" if self.has_offline and not self.has_live else "live") if source == "auto"
                       else source)
        finished = next((r for r in rest if r.get("kind") == "task_finished"), {})
        self.status, self.claim = finished.get("status"), finished.get("model_claim") or finished.get("reason")
        self.frames = [(r["simulation_time"], frame) for r in rest
                       if r.get("kind") == "frame" and isinstance(r.get("simulation_time"), (int, float))
                       and (frame := inside(root, r.get("path"))) is not None]
        monitor_events = jsonl(folder / OFFLINE / "events.jsonl") if self.source == "offline" else \
            [r for r in rest if r.get("kind") == "monitor"]
        readings_path = folder / (OFFLINE if self.source == "offline" else "monitor") / "readings.jsonl"
        self.readings, self.unreadable = [], 0  # an imported rollout's rows are checked, not trusted
        for row in jsonl(readings_path):
            try:
                self.readings.append((reading_from(row), row.get("action")))
            except (TypeError, KeyError, ValueError):
                self.unreadable += 1
        self.readings.sort(key=lambda item: item[0].sim_time)
        self.vocabularies = parsed(vocabulary_from, [e for e in monitor_events if e.get("type") == "vocabulary"])
        self.by_goal = {vocabulary.goal_id: vocabulary for vocabulary in self.vocabularies}  # the newest of each
        self.verdicts = sorted(parsed(verdict_from, [e.get("verdict") for e in monitor_events
                                                     if e.get("type") == "fired"]),
                               key=lambda verdict: verdict.sim_time)
        times = [t for t, _ in self.frames] + [r[0].sim_time for r in self.readings]
        self.t0, self.t1 = (min(times), max(times)) if times else (0.0, 0.0)
