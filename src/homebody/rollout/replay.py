"""A recorded rollout on the dashboard: its task as the dashboard would have shown it live.

The session's events (turns with the plan, decisions, skill events, frames, the task's end) and the monitor's
readings (the live run's or the offline evaluator's, `Rollout.source`) are handed to a fresh dashboard in
simulation-time order, each stamped at its own time: the run ends now, and the dashboard's own history inspection
(the timeline, the stored camera frames, the incidents) is the replay.
"""
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from homebody.rollout.paths import inside
from homebody.skills.contract import Result
from homebody.ui.dashboard.monitor_feed import value
from homebody.ui.dashboard.session_state import goal_texts


class Clock:
    """The replay's time: simulation time T, placed so that the rollout's last moment is NOW."""

    def __init__(self, t0: float, t1: float, now: float):
        self.base, self.t = now - (t1 - t0) - t0, t0

    def __call__(self) -> float:
        return self.base + self.t


def plan_state(plan: dict | None) -> dict | None:
    """A recorded plan snapshot (session/monitored.plan_snapshot) as the dashboard's store takes it."""
    if not plan or not isinstance(plan.get("tree"), dict):
        return None
    return {**plan, "goals": goal_texts(plan["tree"])}


def result(data) -> Result:
    data = data if isinstance(data, dict) else {}
    return Result(str(data.get("skill", "")), str(data.get("code", "")), str(data.get("message", "")),
                  data.get("details") or {})


def events(rollout) -> list[tuple[float, int, str, Any]]:
    """(simulation time, order, kind, payload) of everything the dashboard is told, in order."""
    found, t = [], rollout.t0
    for index, record in enumerate(rollout.records):
        t = record.get("simulation_time", t) if isinstance(record.get("simulation_time"), (int, float)) else t
        kind = record.get("kind")
        if kind in ("turn", "decision", "skill", "task_finished"):
            found.append((float(t), index, kind, record))
        elif kind == "frame" and (folder := inside(rollout.root, record.get("path"))) is not None:
            found.append((float(t), index, "frame", folder / "rgb.png"))
    verdicts = {v.sim_time: v for v in rollout.verdicts}
    for reading, _ in rollout.readings:
        found.append((reading.sim_time, len(rollout.records), "reading", (reading, verdicts.get(reading.sim_time))))
    return sorted(found, key=lambda item: (item[0], item[1]))


def replay(dashboard, rollout, tau: float, now=None):
    """Show ROLLOUT on DASHBOARD (reset first). TAU is the monitor's threshold its readings are weighed against."""
    clock = Clock(rollout.t0, rollout.t1, time.time() if now is None else now)
    dashboard.reset(scene_name=f"Replay: {rollout.root.name}", initial_task=rollout.instruction, clock=clock)
    state = {"task": rollout.instruction, "plan": None, "robot": {}}
    dashboard.update("Thinking", state=state)
    for t, _, kind, payload in events(rollout):
        clock.t = t
        if kind == "turn":
            state = {**state, "plan": plan_state(payload.get("plan"))}
            dashboard.update("Thinking", state=state)
        elif kind == "decision":
            decision = payload.get("decision") or {}
            arguments = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
            dashboard.select(SimpleNamespace(text=str(decision.get("text", "")), skill=str(decision.get("skill", "")),
                                             arguments=arguments), None)
            dashboard.update("Executing", str(decision.get("text", "")), state=state)
        elif kind == "skill":
            event = dict(payload.get("event") or {})
            if event.get("type") == "skill_finished":
                event["result"] = result(event.get("result"))
            if event.get("type") in ("skill_started", "skill_stage", "skill_finished", "intervention"):
                dashboard.skill_event(event)
        elif kind == "frame" and dashboard.history.due(clock()) and payload.is_file():
            dashboard.remote_frame("robot", np.asarray(Image.open(payload).convert("RGB")), {"sim": t})
        elif kind == "reading":
            reading, verdict = payload
            try:
                dashboard.monitor(value(reading, rollout.by_goal.get(reading.goal_id), verdict, tau))
            except ValueError:  # an imported rollout's reading out of range: skipped, like an unreadable row
                continue
        elif kind == "task_finished":
            dashboard.update(str(payload.get("status") or "Ready"),
                             str(payload.get("model_claim") or payload.get("reason") or ""), state=state)
    if rollout.status is None:  # an unfinished rollout: it ends where its record does
        dashboard.update("interrupted", "The recording ends here.", state=state)
    frames = dashboard.history.index()
    if frames:  # the camera shows where the recording ends
        frame_id, t, _ = frames[-1]
        dashboard.hub.frame("robot", {"t": t, "label": "Last recorded frame"}, dashboard.history.get(frame_id))
    dashboard.clock = dashboard.store.clock = time.time  # what the operator does next happens now
    dashboard.hub.changed()
