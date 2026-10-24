"""A recorded rollout read back for replay: what the console showed at any simulation time of one task.

`Rollout.view(t)` is the monitor View the console's drawers take (plan, phase, timeline lanes, readings, verdicts)
as it stood at time t, built from the session's events (turns, skill events, decisions) and the monitor's readings:
the live run's `monitor/`, or the offline evaluator's `monitor-offline/` (`source`).
"""
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from homebody.monitor.monitor import View, history_entry
from homebody.monitor.offline import FOLDER as OFFLINE
from homebody.monitor.offline import chosen, read_events
from homebody.monitor.targets import (
    Claim,
    Label,
    Phase,
    Vocabulary,
    reading_from,
    verdict_from,
)
from homebody.rollout.paths import inside


@dataclass(frozen=True)
class Turn:
    t: float
    plan: dict | None
    goal_id: str
    goal: str


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
        self.turns = self._turns(rest)
        self.thoughts = [(r["simulation_time"], (r.get("decision") or {}).get("text", ""),
                          (r.get("decision") or {}).get("skill")) for r in rest
                         if r.get("kind") == "decision" and "simulation_time" in r]
        self.stages, self.spans = self._skills(rest)
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

    def _turns(self, rest) -> list[Turn]:
        turns = []
        for record in rest:
            if record.get("kind") != "turn":
                continue
            node = (record.get("request") or {}).get("node") or {}
            turns.append(Turn(record.get("simulation_time", 0.0), record.get("plan"), node.get("goal_id", "task"),
                              node.get("goal", self.instruction)))
        return turns

    @staticmethod
    def _skills(rest):
        """Stage changes (t, skill, stage) and skill calls (start, end, skill, code)."""
        stages, spans, open_span = [], [], None
        for record in rest:
            event, t = record.get("event") or {}, record.get("simulation_time")
            if record.get("kind") != "skill" or t is None:
                continue
            if event.get("type") == "skill_started":
                open_span = (t, event.get("skill"))
                stages.append((t, event.get("skill"), None))
            elif event.get("type") == "skill_stage":
                stages.append((t, event.get("skill"), event.get("stage")))
            elif event.get("type") == "skill_finished" and open_span is not None:
                spans.append((open_span[0], t, open_span[1], (event.get("result") or {}).get("code")))
                stages.append((t, None, None))
                open_span = None
        if open_span is not None:
            spans.append((open_span[0], None, open_span[1], None))
        return stages, spans

    # --- state at a time ---
    def turn_at(self, t: float) -> Turn | None:
        index = bisect_right([turn.t for turn in self.turns], t)
        return self.turns[index - 1] if index else None

    def frame_at(self, t: float) -> Path | None:
        index = bisect_right([time for time, _ in self.frames], t)
        if not index:
            return None
        path = self.frames[index - 1][1] / "rgb.png"
        return path if path.is_file() else None

    def thought_at(self, t: float) -> str:
        index = bisect_right([time for time, _, _ in self.thoughts], t)
        return self.thoughts[index - 1][1] if index else ""

    def phase_at(self, t: float) -> Phase:
        index = bisect_right([time for time, _, _ in self.stages], t)
        if not index:
            return Phase("planning") if self.turns else Phase()
        _, skill, stage = self.stages[index - 1]
        calls = sum(1 for start, *_ in self.spans if start <= t)
        return Phase("tool_call", skill, stage, calls) if skill else Phase("planning", call=calls)

    def doing_at(self, t: float) -> str:
        phase = self.phase_at(t)
        if phase.skill:
            return f"{phase.skill}: {phase.stage}" if phase.stage else phase.skill
        return "the agent is choosing the next step" if phase.name == "planning" else ""

    def goal_at(self, t: float) -> str:
        turn = self.turn_at(t)
        if turn is not None or self.turns:
            return turn.goal_id if turn is not None else "task"
        earlier = [reading.goal_id for reading, _ in self.readings if reading.sim_time <= t]
        return earlier[-1] if earlier else "task"

    @staticmethod
    def call_at(t, spans, shown) -> int:
        """The skill call at T in the live monitor's own numbering (it counts only the calls it was told of): the
        newest reading's, or the next when that reading was taken before the call running now began."""
        if not shown:
            return len(spans)
        call = shown[-1][0].phase.call
        running = spans and spans[-1][1] is None
        return call + 1 if running and shown[-1][0].sim_time < spans[-1][0] else call

    def view(self, t: float) -> View:
        """The monitor View as it stood at simulation time T."""
        turn = self.turn_at(t)
        goal_id = turn.goal_id if turn is not None else "task"
        vocabulary = self.by_goal.get(goal_id)
        shown = [(r, a) for r, a in self.readings if r.sim_time <= t]
        history = [history_entry(reading, self.by_goal.get(reading.goal_id), action) for reading, action in shown]
        goals, last = [], None
        # the plan's goals by turn; a rollout recorded before turns has them only on the monitor's readings
        changes = [(item.t, item.goal_id) for item in self.turns] if self.turns else \
            [(reading.sim_time, reading.goal_id) for reading, _ in self.readings]
        for when, goal in changes:
            if when <= t and goal != last:
                goals.append((when, goal))
                last = goal
        spans = tuple((start, end if end is not None and end <= t else None, skill, self.goal_at(start),
                       code if end is not None and end <= t else None)
                      for start, end, skill, code in self.spans if start <= t)
        verdicts = tuple(v for v in self.verdicts if v.sim_time <= t)
        return View(status=f"replay ({self.source} monitor)" if self.readings else "replay (no monitor logs)",
                    goal_id=goal_id, plan=turn.plan if turn is not None else None, task=self.instruction,
                    phase=self.phase_at(t), vocabulary=vocabulary, reading=shown[-1][0] if shown else None,
                    history=tuple(history), spans=spans,
                    decisions=tuple(when for when, _, _ in self.thoughts if when <= t),
                    skill_call=self.call_at(t, spans, shown), sim_start=self.t0, sim_now=t, goals=tuple(goals),
                    verdicts=verdicts[-8:], calls=len(shown),
                    cost_usd=sum(reading.cost_usd for reading, _ in shown))
