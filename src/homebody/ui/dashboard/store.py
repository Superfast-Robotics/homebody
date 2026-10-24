"""What the dashboard knows about the session, built only from the session's own callbacks.

Every write takes the lock and replaces records whole; `snapshot` reads one consistent copy for the browser.
Times are wall-clock seconds from one clock, so every panel shares one time axis.
"""
import time
from collections import OrderedDict
from dataclasses import asdict, replace
from itertools import count
from threading import RLock

from homebody.ui.dashboard.outcome import evaluate, lifecycle
from homebody.ui.dashboard.planning import PlanTracking
from homebody.ui.dashboard.records import (
    Decision,
    Incident,
    Instruction,
    Interval,
    Log,
    ToolCall,
    arguments,
    clip,
    reading,
)
from homebody.ui.dashboard.tree import plan_nodes

BUSY = ("Thinking", "Executing")
CODER = "call_coder"  # session.coder.CALL: the coding assistant, a recovery tool
PLANNER = ("plan", "goal_done", "goal_failed", "done")  # decisions the session answers itself, not tools
# a task's end status: its tone and, for a failure, the incident it raises
ENDINGS = {
    "model_done": ("success", None),
    "step_limit": ("warning", "Step limit reached"),
    "cancelled": ("neutral", None),
    "provider_error": ("critical", "Provider error"),
    "error": ("critical", "Session error"),
    "interrupted": ("warning", "Session interrupted"),
    "physics_invalid": ("critical", "Physics became invalid"),
    "Reset required": ("critical", "Physics became invalid"),
}
LOG_LIMIT, TOOL_LIMIT, INSTRUCTION_LIMIT, INCIDENT_LIMIT, TASK_LIMIT = 500, 200, 50, 100, 20
ANOMALY_GAP = 3.  # s between anomalous readings of one source that still belong to one episode
KEEP_SECONDS = 600.  # timeline history kept for the expanded view; the main view shows the last 60 s


def bounded(records: OrderedDict, limit: int) -> OrderedDict:
    while len(records) > limit:
        records.popitem(last=False)
    return records


class Store(PlanTracking):
    def __init__(self, *, scene_name="Simulation", model_name="", initial_task="", clock=time.time):
        self.clock, self.lock, self._ids = clock, RLock(), count(1)
        self.scene_name, self.model_name, self.initial_task = scene_name, model_name, initial_task
        self.status, self.text, self.agent_text, self.thinking = "Loading", "", "", False
        self.task = None  # {"text", "start", "end", "outcome"}
        self.stop_pending = self.reset_pending = False
        self.plan = None  # {"tree", "status", "reasons", "active"} from the session's planner
        self.robot = {}
        self.frame_id = None  # the newest stored camera frame, attached to incidents
        self.version = 0
        self.intervals, self.decisions, self.tools = OrderedDict(), OrderedDict(), OrderedDict()
        self.incidents, self.logs, self.instructions = OrderedDict(), OrderedDict(), OrderedDict()
        self.readings, self.tasks = [], []  # tasks: every recent task's span, for the monitoring tracks
        self._open = {}  # track -> id of its running interval
        self._recoveries = {}  # goal id -> {"incident", "failed"} while a recovery goal runs
        self._superseded = []  # trees the agent replaced, newest last
        self._anomalies = {}  # source -> (incident id, time of its latest anomalous reading)

    @property
    def busy(self):
        return self.status in BUSY

    @property
    def accepting(self):
        """Whether a new instruction can start now; otherwise it is refused, never queued behind a task."""
        return self.status not in (*BUSY, "Reset required", "Resetting", "Loading")

    def note_robot(self, **values):
        with self.lock:
            self.robot = {**self.robot, **values}
            self._touch()

    def _id(self, prefix):
        return f"{prefix}{next(self._ids)}"

    def _touch(self):
        self.version += 1

    def log(self, severity, source, message, node=None):
        with self.lock:
            entry = Log(self._id("l"), self.clock(), severity, source, clip(message, 600), node)
            bounded(self.logs, LOG_LIMIT)[entry.id] = entry
            self._touch()

    # --- intervals ---
    def _begin(self, track, label, detail=None):
        self._end(track, "ended")
        interval = Interval(self._id("i"), track, clip(label, 120), self.clock(), detail=detail or {})
        self.intervals[interval.id] = interval
        self._open[track] = interval.id

    def _end(self, track, status):
        interval_id = self._open.pop(track, None)
        if interval_id is not None and interval_id in self.intervals:
            self.intervals[interval_id] = replace(self.intervals[interval_id], end=self.clock(), status=status)

    # --- session callbacks ---
    def update(self, status, text="", *, task=None, plan=None, robot=None):
        """A session status; TASK, PLAN and ROBOT are read on the session thread alongside it."""
        with self.lock:
            busy, was_busy = status in BUSY, self.status in BUSY
            if robot:
                self.robot = {**self.robot, **robot}
            if busy and not was_busy:
                self._task_started(task or self.text)
            self._plan(plan if plan is None or busy else {**plan, "active": None})  # idle: no goal runs
            self._thinking(status == "Thinking")
            if status != "Executing":
                self._close_tools("ended")
            if not busy:
                self.stop_pending = False
            if status in ENDINGS or (was_busy and not busy):
                self._task_ended(status, text)
            if status == "Ready":
                self.reset_pending = False
            self.status, self.text = status, clip(text)
            if status == "Executing" and text:
                self.agent_text = clip(text)
            self.log("error" if ENDINGS.get(status, ("",))[0] == "critical" else "info", "session",
                     f"{status}{': ' + text if text else ''}")

    def _thinking(self, thinking):
        if thinking and not self.thinking:
            self._begin("thinking", "Thinking")
        elif not thinking and self.thinking:
            self._end("thinking", "success")
        self.thinking = thinking

    def _task_started(self, text):
        """A new task. Alerts from earlier tasks stay as they were: only an operator acknowledges or resolves one."""
        now = self.clock()
        self.task = {"id": self._id("k"), "text": clip(text), "start": now, "end": None, "outcome": None,
                     "result": None, "frame": None}
        self.tasks = [*self.tasks, self.task][-TASK_LIMIT:]
        self._recoveries, self._superseded = {}, []
        self.agent_text, self.robot = "", {key: value for key, value in self.robot.items() if key != "target"}
        waiting = [i for i in self.instructions.values() if i.state == "pending"]
        pending = next((i for i in waiting if i.text == text), waiting[0] if waiting else None)
        if pending is not None:
            self.instructions[pending.id] = replace(pending, state="running", reason="The agent started it.")
        self.plan = None

    def _task_ended(self, status, text):
        if self.task is None or self.task["end"] is not None:
            return
        tone, title = ENDINGS.get(status, ("neutral", None))
        self.task = {**self.task, "end": self.clock(), "outcome": status, "frame": self.frame_id}
        self._end("goal", "ended")
        self._end("skill", "cancelled" if status == "cancelled" else "ended")
        result = evaluate(self.task, self.plan, self._recovery_counts())
        self.task = {**self.task, "result": result}
        self.tasks = [*self.tasks[:-1], self.task]
        running = next((i for i in self.instructions.values() if i.state == "running"), None)
        if running is not None:
            state = "done" if tone == "success" else "failed" if tone in ("critical", "warning") else "stopped"
            self.instructions[running.id] = replace(running, state=state, reason=clip(text, 300) or status)
        if title is not None:
            self.incident("task", tone if tone != "neutral" else "warning", title, text, alert=True,
                          detection="Session")
        if result["state"] in ("failed", "partial"):
            self.incident("task", "warning", f"Task outcome: {result['label']}", result["reason"], alert=True,
                          detection="Task evaluation (plan goals and their checks)")
        self.log("info", "session", f"Outcome: {result['label']}. {result['reason']}")

    def _recovery_counts(self):
        task = self.task["id"] if self.task else None
        mine = [i for i in self.incidents.values() if i.kind == "recovery" and i.task == task]
        return {"started": sum(i.tag == "recovery_start" for i in mine),
                "succeeded": sum(i.tag == "recovery_ok" for i in mine),
                "failed": sum(i.tag == "recovery_fail" for i in mine), "replans": sum(i.tag == "replan" for i in mine)}

    def decision(self, decision, goal=None):
        """An executed decision: an agent-decision mark and, for a skill or coder call, a running tool call."""
        with self.lock:
            now, skill = self.clock(), str(decision.skill)
            args = arguments(getattr(decision, "arguments", {}))
            tool = None
            if skill not in PLANNER:
                start = self.task["start"] if self.task else float("-inf")
                attempt = 1 + sum(t.goal == goal and t.skill == skill and t.start >= start for t in self.tools.values())
                tool = ToolCall(self._id("t"), skill, args, now, goal, attempt=attempt,
                                recovery=skill == CODER or goal in self._recoveries)
                bounded(self.tools, TOOL_LIMIT)[tool.id] = tool
            mark = Decision(self._id("d"), now, skill, clip(decision.text), args, goal,
                            None if tool is None else tool.id)
            self.decisions[mark.id] = mark
            self.log("info", "agent", f"Decision {skill}: {decision.text}", goal)

    def skill(self, event):
        """A skill event: stages and results update the running tool call and the skill track."""
        with self.lock:
            kind = event.get("type")
            tool = self._running_tool()
            if kind == "skill_started":
                self._begin("skill", event.get("skill", "skill"), {} if tool is None else {
                    "tool": tool.id, "goal": tool.goal, "attempt": tool.attempt, "recovery": tool.recovery})
                self.log("info", "skill", f"{event.get('skill')} started", tool and tool.goal)
            elif kind == "skill_stage":
                if tool is not None:
                    self.tools[tool.id] = replace(tool, stages=(*tool.stages, (self.clock(), event.get("stage"))))
                self.log("info", "skill", f"{event.get('skill')}: {event.get('stage')}", tool and tool.goal)
            elif kind == "skill_finished":
                self._finished(event.get("result"), tool)
            self._touch()

    def _running_tool(self):
        return next((t for t in reversed(self.tools.values()) if t.state == "running"), None)

    def _finished(self, result, tool):
        code, message = str(getattr(result, "code", "")), clip(getattr(result, "message", ""), 600)
        skill = str(getattr(result, "skill", ""))
        ok = code == "OK"
        self._end("skill", "success" if ok else "failure")
        if tool is not None and tool.skill == skill:
            self.tools[tool.id] = replace(tool, end=self.clock(), state="succeeded" if ok else "failed",
                                          code=code, message=message)
        self.log("info" if ok else "warning", "skill", f"{skill} {code}: {message}", tool and tool.goal)
        if tool is not None and tool.attempt > 1:
            self._retried(tool, code)
        if not ok:
            self.incident("skill", "warning", f"{skill} failed: {code}", message,
                          node=tool and tool.goal, tool=tool and tool.id, skill=skill,
                          detection=f"Robot skill executor ({skill} result)",
                          action="Result returned to the agent, which chooses the next step.")

    def _retried(self, tool, code):
        """A later attempt of a skill is the outcome of its last failure in the same goal."""
        for incident in reversed(self.incidents.values()):
            if incident.kind == "skill" and incident.skill == tool.skill and incident.node == tool.goal:
                if not incident.outcome:
                    self._outcome(incident.id, f"Retried (attempt {tool.attempt}): {code}")
                return

    def _close_tools(self, state):
        for tool in [t for t in self.tools.values() if t.state == "running"]:
            self.tools[tool.id] = replace(tool, end=self.clock(), state=state,
                                          message=tool.message or "No result reported to the dashboard.")

    def incident(self, kind, severity, title, reason="", **fields):
        with self.lock:
            running = self._running_tool()
            fields.setdefault("skill", running and running.skill)
            fields.setdefault("task", self.task["id"] if self.task else None)
            incident = Incident(self._id("x"), self.clock(), kind, severity, clip(title, 160),
                                clip(reason, 600), frame=self.frame_id, **fields)
            bounded(self.incidents, INCIDENT_LIMIT)[incident.id] = incident
            self.log("error" if severity == "critical" else "warning" if severity == "warning" else "info",
                     fields.get("source", "session"), f"{title}{': ' + reason if reason else ''}",
                     fields.get("node"))
            return incident

    def monitor(self, value):
        """A monitor reading from an adapter, stamped with what ran; an observed anomaly raises an incident."""
        with self.lock:
            running = self._running_tool()
            entry = reading(value, self.clock(), during_task=self.busy, node=(self.plan or {}).get("active"),
                            skill=running and running.skill)
            self.readings.append(entry)
            if entry.level == "anomaly" and entry.kind == "observation" and not self._continues(entry):
                title = entry.reason or "Anomaly detected"
                self.incident("monitor", "critical", f"Synthetic: {title}" if entry.synthetic else title,
                              entry.reason, source=entry.source, node=entry.node, skill=entry.skill, alert=True,
                              detection=f"Monitoring model ({entry.source})",
                              observation=asdict(entry), evidence={key: value for key, value in (
                                  ("risk", entry.risk), ("threshold", entry.threshold),
                                  ("confidence", entry.confidence), ("synthetic", entry.synthetic or None))
                                  if value is not None})
                self._anomalies[entry.source] = (next(reversed(self.incidents)), entry.t)
            self._touch()

    def _continues(self, entry):
        """Whether ENTRY extends its source's open anomaly (one incident per episode, not one per reading); the
        incident then keeps the newest observation and the peak risk."""
        incident_id, last = self._anomalies.get(entry.source, (None, None))
        incident = self.incidents.get(incident_id)
        if incident is None or incident.state == "resolved" or entry.t - last > ANOMALY_GAP:
            return False
        risks = [v for v in (incident.evidence.get("risk"), entry.risk) if v is not None]
        evidence = {**incident.evidence, "readings": incident.evidence.get("readings", 1) + 1,
                    **({"risk": max(risks)} if risks else {})}
        self.incidents[incident_id] = replace(incident, observation=asdict(entry), evidence=evidence)
        self._anomalies[entry.source] = (incident_id, entry.t)
        return True

    # --- operator ---
    def instruction(self, text, state, reason=""):
        with self.lock:
            entry = Instruction(self._id("n"), self.clock(), clip(text), state, reason)
            bounded(self.instructions, INSTRUCTION_LIMIT)[entry.id] = entry
            self.log("info" if state == "pending" else "warning", "operator",
                     f"Instruction {state}: {text}{' (' + reason + ')' if reason else ''}")
            return entry

    def intervention(self, kind, title, reason=""):
        with self.lock:
            if kind == "stop":
                self.stop_pending = True
            elif kind == "reset":
                self.reset_pending = True
            return self.incident("intervention", "info", title, reason, source="operator",
                                 node=(self.plan or {}).get("active"), tag=kind, detection="Human operator",
                                 initiator="Human operator", executor="Session controls")

    def acknowledge(self, incident_id):
        with self.lock:
            incident = self.incidents.get(incident_id)
            if incident is None or not incident.alert or incident.state != "active":
                return False
            self.incidents[incident_id] = replace(incident, state="acknowledged", acknowledged_at=self.clock())
            self.log("info", "operator", f"Acknowledged: {incident.title}", incident.node)
            return True

    def resolve(self, incident_id):
        """The operator closes an alert; a recovery succeeding never does that by itself."""
        with self.lock:
            incident = self.incidents.get(incident_id)
            if incident is None or not incident.alert or incident.state == "resolved":
                return False
            self.incidents[incident_id] = replace(incident, state="resolved")
            self.log("info", "operator", f"Resolved: {incident.title}", incident.node)
            return True

    def outcome(self):
        """The newest task's outcome: fixed when it ended, else its progress so far."""
        if self.task is None:
            return None
        result = self.task["result"] or evaluate(self.task, self.plan, self._recovery_counts())
        unresolved = sum(1 for i in self.incidents.values() if i.alert and i.state != "resolved")
        return {**result, "unresolved_alerts": unresolved, "frame": self.task.get("frame")}

    def cleared(self):
        """A scene reset: the plan and running work are gone, the history stays."""
        with self.lock:
            self._close_tools("cancelled")
            for track in tuple(self._open):
                self._end(track, "cancelled")
            self.plan, self.robot = None, {**self.robot, "pose": None}
            self.status, self.text = "Resetting", "Resetting the scene"
            self._touch()

    def snapshot(self, controls):
        """One consistent JSON-safe copy of everything the browser renders."""
        with self.lock:
            now = self.clock()
            keep = now - KEEP_SECONDS

            def recent(records, stamp):
                return [asdict(r) for r in records if stamp(r) is None or stamp(r) >= keep]
            self.readings = [r for r in self.readings if r.t >= keep]
            tree = plan_nodes(self.plan, self.tools.values(), self.task)
            return {
                "now": now, "version": self.version, "scene": self.scene_name, "model": self.model_name,
                "initial_task": self.initial_task, "status": self.status, "text": self.text,
                "agent_text": self.agent_text, "task": self.task, "robot": self.robot,
                "controls": {**controls, "busy": self.busy, "stop_pending": self.stop_pending,
                             "reset_pending": self.reset_pending, "accepting": self.accepting},
                "tree": tree,
                "lifecycle": lifecycle(self.status, {**controls, "stop_pending": self.stop_pending},
                                       tree and tree["mode"]),
                "outcome": self.outcome(),
                "superseded": self._superseded,
                "tasks": [t for t in self.tasks if t["end"] is None or t["end"] >= keep],
                "intervals": recent(self.intervals.values(), lambda r: r.end),
                "decisions": recent(self.decisions.values(), lambda r: r.t),
                "tools": [asdict(t) for t in self.tools.values()],
                "incidents": [asdict(i) for i in self.incidents.values()],
                "readings": [asdict(r) for r in self.readings],
                "logs": [asdict(entry) for entry in self.logs.values()],
                "instructions": [asdict(i) for i in self.instructions.values()],
            }
