"""The offline evaluator: the live monitor run over a recorded rollout, for rollouts that ran without it (or to try
other settings on one that did).

It replays the rollout's events in order through an inline Monitor: each task start, each turn's goal request (the
plan, the hands, the head and top-down views), each skill call and stage, and each recorded camera frame. A decision
call is made every `rollouts.offline_period_s` simulation seconds, on the newest frames, with the same vocabulary
model, decision model and rules as a live run. Nothing can be stopped after the fact, so its verdicts are what the
monitor *would* have done.

Rollouts recorded before turns were recorded have no plan or hands: the task is the goal throughout. Recorded frames
come at `recording.frame_period` (about 2 Hz), sparser than live capture; the report states the rate it had.

Writes, per task, `tasks/task-NNN/monitor-offline/`: readings.jsonl, vocab-NNN/, decisions/, events.jsonl (the
monitor's own events) and report.json. A live run's `monitor/` is never touched.
"""
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.monitor.monitor import Monitor
from homebody.monitor.targets import NodeInfo, Phase, VocabularyRequest, request_from
from homebody.rollout.paths import inside

FOLDER = "monitor-offline"
TOKENS_PER_IMAGE = 450  # estimate for a dry run when the rollout has no live readings to measure from
TEXT_TOKENS = 700


def read_events(rollout: Path) -> list[dict]:
    return [json.loads(line) for line in (rollout / "events.jsonl").read_text().splitlines() if line.strip()]


def tasks(records: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Each task_started record with the records up to the next task's start."""
    starts = [n for n, record in enumerate(records) if record.get("kind") == "task_started"]
    return [(records[n], records[n + 1:(starts[i + 1] if i + 1 < len(starts) else len(records))])
            for i, n in enumerate(starts)]


def chosen(records, task: str | None):
    for started, rest in tasks(records):
        name = Path(str(started.get("path", ""))).name
        if name in ("", ".", "..") or not name.startswith("task-"):
            continue
        if task is None or name == task:
            yield name, started, rest


def frame_image(rollout: Path, record: dict) -> np.ndarray | None:
    folder = inside(rollout, record.get("path"))
    path = folder / "rgb.png" if folder is not None else None
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def task_goal(started: dict) -> VocabularyRequest:
    """The goal of a rollout recorded without turns: the task itself."""
    task = started.get("task", "")
    return VocabularyRequest(task, "", NodeInfo("task", task, None, "active"), (), tuple(started.get("skills", ())),
                             "unknown")


def frame_times(rest: list[dict]) -> list[float]:
    return [r["simulation_time"] for r in rest if r.get("kind") == "frame" and "simulation_time" in r]


def estimated_calls(times: list[float], period: float) -> int:
    """The decision calls an inline monitor makes over frames at TIMES, one at most every PERIOD seconds."""
    calls, last = 0, float("-inf")
    for t in times:
        if t - last >= period:
            calls, last = calls + 1, t
    return calls


def tokens_per_call(rollout: Path, name: str, window: int) -> float:
    live = rollout / "tasks" / name / "monitor/readings.jsonl"
    if live.is_file():
        counts = [json.loads(line).get("input_tokens", 0) for line in live.read_text().splitlines() if line.strip()]
        counts = [count for count in counts if count]
        if counts:
            return float(np.mean(counts))
    return window * TOKENS_PER_IMAGE + TEXT_TOKENS


def estimate(rollout: Path, settings, task: str | None = None) -> dict:
    """What evaluating ROLLOUT would cost: per task the frames, the decision calls and their price."""
    period, monitor = settings.rollouts.offline_period_s, settings.monitor
    rows = []
    for name, _, rest in chosen(read_events(rollout), task):
        first = next((n for n, r in enumerate(rest) if r.get("kind") == "turn"), 0)  # no goal, no call before it
        calls = estimated_calls(frame_times(rest[first:]), period)
        times = frame_times(rest)
        usd = calls * tokens_per_call(rollout, name, monitor.window_frames) * monitor.decision_usd_per_mtok / 1e6
        rows.append({"task": name, "frames": len(times), "calls": calls, "usd": round(usd, 4)})
    return {"tasks": rows, "calls": sum(r["calls"] for r in rows), "usd": round(sum(r["usd"] for r in rows), 4),
            "cap_usd": settings.rollouts.offline_max_cost_usd}


class Replay:
    """One task's events driven through an inline monitor."""

    def __init__(self, rollout: Path, started: dict, monitor: Monitor):
        self.rollout, self.started, self.monitor = rollout, started, monitor
        self.goal_request: VocabularyRequest = task_goal(started)
        self.calls_announced = False  # the rollout records each skill's request (else skill_started events stand in)

    def play(self, rest: list[dict], progress: Callable[[float], None]):
        self.calls_announced = any(r.get("kind") == "skill_request" for r in rest)
        if not any(r.get("kind") == "turn" for r in rest):  # recorded before turns were: the task is the goal
            self.monitor.goal(self.goal_request, self.goal_request.hands)
        frames = max(1, sum(1 for r in rest if r.get("kind") == "frame"))
        seen = 0
        for record in rest:
            if self.monitor.halted:
                break
            seen += self.handle(record)
            progress(seen / frames)

    def handle(self, record: dict) -> int:
        """Apply RECORD to the monitor; 1 when it was a camera frame."""
        kind, t = record.get("kind"), record.get("simulation_time")
        if kind == "turn":
            self.goal_request = request_from(record["request"], self.rollout)
            self.monitor.goal(self.goal_request, record.get("hands", ""), plan=record.get("plan"))
            self.monitor.set_phase(Phase("planning"))
            if t is not None:
                self.monitor.decision(t)
        elif kind == "skill_request":
            self.monitor.skill_started(request_from(record["request"], self.rollout), t)
        elif kind == "skill":
            self.skill(record["event"], t)
        elif kind == "frame":
            rgb = frame_image(self.rollout, record)
            if rgb is not None and t is not None:
                self.monitor.capture(rgb, t)
                self.monitor.poll()
            return 1
        return 0

    def skill(self, event: dict, t):
        kind = event.get("type")
        if kind == "skill_started" and not self.calls_announced:
            self.monitor.skill_started(replace(self.goal_request, skill=event.get("skill"), images=()), t)
        elif kind == "skill_stage":
            self.monitor.set_phase(Phase("tool_call", event.get("skill"), event.get("stage")))
        elif kind == "skill_finished":
            self.monitor.skill_finished((event.get("result") or {}).get("code"), t)


def evaluate(rollout: Path, settings, vocabulary_model, decision_model, *, task: str | None = None,
             progress: Callable[[str, float], None] = lambda task, fraction: None) -> dict:
    """Run the monitor over ROLLOUT's recorded tasks (or only TASK); the report of each, also written to its
    monitor-offline/report.json."""
    monitor_settings = replace(settings.monitor, max_cost_usd=settings.rollouts.offline_max_cost_usd,
                               keep_frames=False)
    reports = []
    for name, started, rest in chosen(read_events(rollout), task):
        folder = rollout / "tasks" / name / FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        events: list[dict] = []
        with (folder / "events.jsonl").open("w") as log:
            def sink(kind, events=events, log=log, **fields):
                events.append({"kind": kind, **fields})
                log.write(json.dumps({"kind": kind, **fields}, default=str) + "\n")
            monitor = Monitor(monitor_settings, vocabulary_model, decision_model, sink=sink, inline=True,
                              period_s=settings.rollouts.offline_period_s, folder=FOLDER)
            monitor.start_task(started.get("task", ""), rollout / "tasks" / name)
            try:
                Replay(rollout, started, monitor).play(rest, lambda fraction, n=name: progress(n, fraction))
            finally:
                summary = monitor.end_task()
        report = task_report(name, started, rest, events, summary, monitor, rollout)
        (folder / "report.json").write_text(json.dumps(report, indent=1))
        reports.append(report)
    return {"rollout": rollout.name, "tasks": reports}


def task_report(name, started, rest, events, summary, monitor, rollout) -> dict:
    times = frame_times(rest)
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    fired = [e["verdict"] for e in events if e.get("type") == "fired"]
    live = [r["verdict"] for r in rest if r.get("kind") == "monitor" and r.get("type") == "fired"]
    finished = next((r for r in rest if r.get("kind") == "task_finished"), {})
    return {"task": name, "instruction": started.get("task", ""), "status": finished.get("status"),
            "frames": len(times), "simulated_s": round(span, 2),
            "frame_hz": round((len(times) - 1) / span, 2) if span > 0 else None,
            "period_s": monitor.period(), "recorded_turns": any(r.get("kind") == "turn" for r in rest),
            "vocabulary_sources": sorted({e.get("source", "") for e in events if e.get("type") == "vocabulary"}),
            "would": [{"action": f"would_{v['action']}", "goal_id": v.get("goal_id"), "label": v.get("label"),
                       "p": v.get("p"), "reason": v.get("reason"), "sim_time": v.get("sim_time")} for v in fired],
            "live_verdicts": len(live), "errors": [e.get("error") for e in events if e.get("type") == "error"],
            "halted": monitor.halted, "has_physics": (rollout / "tasks" / name / "physics.jsonl").is_file(),
            **summary}
