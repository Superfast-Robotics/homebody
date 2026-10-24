"""What the harness server's session talks to in place of a monitor in its own process: `MonitorLink` takes the
session's monitor calls and sends them to the commanding client, whose monitor (homebody/monitor) watches the frames
it is sent (or a simulation server's stream); the client's stop and verdicts come back here, where the session reads
them as it reads a local monitor's. So neither the session nor any skill knows the monitor is remote."""
import json
import re
import threading
from collections import deque
from dataclasses import asdict
from pathlib import Path

from homebody.monitor.targets import Phase, Verdict, verdict_from
from homebody.server import protocol

FINDINGS = 6  # recent verdicts the vocabulary model reads, as a local monitor keeps
MAX_EVENT = 64 * 1024  # bytes of a client monitor event kept in the evidence


class MonitorLink:
    """The session's monitor, run by the commanding client. Without one connected it watches nothing: no frames are
    due, no skill is stopped, there are no verdicts."""

    def __init__(self, hub, settings, *, quality: int, root: Path, sink=lambda kind, **fields: None):
        self.hub, self.settings, self.quality, self.root, self.sink = hub, settings, quality, root, sink
        self.lock = threading.Lock()
        self.connected = False
        self.send_frames = True  # False when the monitor reads a simulation server's camera stream itself
        self._reset()

    def _reset(self):
        self.last_shot = float("-inf")
        self.skill_call = 0
        self.stop_reason = None
        self.verdicts: list[Verdict] = []
        self.findings = deque(maxlen=FINDINGS)
        self.stages: dict[str, list[str]] = {}
        self.alert_goal = None

    def send(self, call: str, **fields):
        if self.connected:
            self.hub.publish(protocol.text("monitor", call=call, **fields))

    def request(self, request) -> dict:
        """REQUEST for the client: its images travel inside it, since the client cannot read this machine's files."""
        data = asdict(request)
        data["images"] = [protocol.embedded(Path(path), self.quality) for path in request.images
                          if Path(path).is_file()]
        return data

    # --- the client's answers ---
    def client_connected(self, connected: bool):
        with self.lock:
            self.connected = connected
            if not connected:
                self.stop_reason = None

    def client_stop(self, reason: str, call: int):
        """The client's monitor stopped skill call CALL; a stop for an earlier call is stale: refused, and the
        refusal sent back as an intervention event."""
        with self.lock:
            current = self.skill_call
            if call == current:
                self.stop_reason = reason
                return
        event = {"type": "intervention", "stage": "rejected", "initiator": "monitor", "action": "stop_skill",
                 "reason": reason, "call": call if call > 0 else None,
                 "note": f"The request named skill call {call}; call {current} runs now"}
        self.sink("skill", event=event)
        self.hub.publish(protocol.text("skill", event=event))

    def client_verdicts(self, verdicts: list[dict]):
        with self.lock:
            for data in verdicts:
                verdict = verdict_from(data)
                self.verdicts.append(verdict)
                self.findings.append(f"{verdict.action} {verdict.text()}")
                if verdict.action in ("abort_skill", "reopen_goal"):
                    self.alert_goal = verdict.goal_id

    def client_event(self, fields: dict):
        """The client monitor's own event, recorded in this run's evidence under a kind of its own
        ("remote_monitor"), so it can never pass for the harness's records; oversized events are cut."""
        kind = re.sub(r"[^a-z_]", "", str(fields.get("kind", "monitor")).lower())[:40] or "monitor"
        body = {re.sub(r"[^A-Za-z0-9_]", "", str(key))[:40] or "field": value for key, value in fields.items()
                if key not in ("kind", "remote", "wall_time")}
        if len(json.dumps(body, default=str)) > MAX_EVENT:
            body = {"type": str(body.get("type", ""))[:40], "truncated": True}
        self.sink(f"remote_{kind}", **body)

    # --- the session's calls (homebody/monitor/monitor.Monitor's session-facing methods) ---
    def start_task(self, task: str, task_dir: Path):
        with self.lock:
            self._reset()
        self.send("start_task", task=task, folder=Path(task_dir).name)

    def end_task(self) -> dict:
        self.send("end_task")
        return {"remote": True, "connected": self.connected}

    def capture_due(self, sim_time: float) -> bool:
        return self.connected and self.send_frames and (sim_time - self.last_shot >= 1 / self.settings.capture_hz
                                   or sim_time < self.last_shot)

    def capture(self, rgb, sim_time: float, view: str = "head"):
        if view == "head":
            self.last_shot = sim_time
        if self.connected:
            self.hub.publish(protocol.image("shot", rgb, self.quality, t=sim_time, view=view), monitor=True)

    def goal(self, request, hands: str, plan: dict | None = None):
        self.send("goal", request=self.request(request), hands=hands, plan=plan)

    def set_phase(self, phase: Phase):
        with self.lock:
            if phase.skill and phase.stage and phase.stage not in self.stages.setdefault(phase.skill, []):
                self.stages[phase.skill].append(phase.stage)
        self.send("set_phase", phase=asdict(phase))

    def skill_started(self, request, sim_time: float):
        with self.lock:
            self.skill_call += 1
            self.stop_reason = None
        self.send("skill_started", request=self.request(request), t=sim_time, number=self.skill_call)

    def skill_finished(self, code: str | None = None, sim_time: float | None = None):
        with self.lock:
            self.stop_reason = None
        self.send("skill_finished", code=code, t=sim_time)

    def operator_resumed(self):
        self.send("operator_resumed")

    def dismiss(self) -> str | None:
        with self.lock:
            goal, self.alert_goal = self.alert_goal, None
        self.send("dismiss")
        return goal

    def mark(self, kind: str, text: str, sim_time: float):
        self.send("mark", label=kind, text=text, t=sim_time)  # "kind" is the wire message's own type

    def decision(self, sim_time: float):
        self.send("decision", t=sim_time)

    def stop_requested(self) -> str | None:
        return self.stop_reason

    def take_verdicts(self) -> list[Verdict]:
        with self.lock:
            taken, self.verdicts = self.verdicts, []
        return taken

    def known_stages(self) -> dict:
        with self.lock:
            return {name: tuple(stages) for name, stages in self.stages.items()}

    def recent_findings(self) -> tuple[str, ...]:
        with self.lock:
            return tuple(self.findings)
