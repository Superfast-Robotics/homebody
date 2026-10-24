"""A dashboard and its monitor connected to a harness server (`homebody connect`).

The server's messages drive the dashboard (status with the session's task, plan and hands, decisions, camera
frames, skill events) and this machine's monitor (the session's monitor calls and the head camera frames). The monitor's stop and verdicts, its events and the operator's commands
go back to the server. The monitor's files are written here, under FOLDER/<session>/<task>/monitor.
"""
import re
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from homebody.monitor.targets import NodeInfo, Phase, VocabularyRequest
from homebody.server import protocol
from homebody.skills.contract import Result

CONSOLE_SKILL_EVENTS = ("skill_started", "skill_stage", "skill_finished", "intervention")
VIEWS = ("robot", "observer", "hand")  # the cameras a harness renders for a dashboard elsewhere
CAMERA_POLL_S = 0.5  # how often the dashboard's watched view is checked, to ask the harness for that camera


class RemoteControls:
    """What the console's buttons call, sent to the server as commands."""

    def __init__(self, client: "HarnessClient"):
        self.client = client

    def _send(self, kind: str, **fields):
        self.client.send(kind, **fields)

    @property
    def paused(self):
        """The harness's operator pause (an Event the dashboard reads), None before a harness is connected."""
        return getattr(self.client, "paused", None)

    def submit(self, text: str):
        self._send("submit", text=text)

    def tell(self, text: str):
        self._send("tell", text=text)

    def stop(self):
        self._send("stop")

    def pause(self):
        self._send("pause")

    def unpause(self):
        self._send("unpause")

    def resume(self):
        """The dashboard's Resume: from the operator's pause and from one the monitor asked for."""
        self._send("unpause")
        self._send("resume_task")

    def stop_skill(self):
        self._send("stop_skill")

    def dismiss(self):
        self._send("dismiss")

    def resume_task(self):
        self._send("resume_task")

    def camera(self, name: str):
        self._send("camera", camera=name)

    def preset(self, name: str, text: str = ""):
        self._send("preset", name=name, text=text)


def safe_name(value, default: str) -> str:
    """VALUE (a name the server sent) when it is one plain folder name, else DEFAULT."""
    text = str(value or "")
    return text if re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", text) and text not in (".", "..") else default


def request_from_wire(data: dict, folder: Path, name: str) -> VocabularyRequest:
    """A request the server sent, its embedded images written under FOLDER as NAME-N.jpg."""
    images = tuple(protocol.unembedded(image, folder / f"{name}-{index}.jpg")
                   for index, image in enumerate(data.get("images", ())))
    return VocabularyRequest(**{**data, "node": NodeInfo(**data["node"]),
                                "done": tuple(NodeInfo(**node) for node in data.get("done", ())),
                                "skills": tuple(data.get("skills", ())), "images": images,
                                "findings": tuple(data.get("findings", ()))})


class HarnessClient:
    def __init__(self, url: str, token: str, *, console=None, monitor=None, folder: Path = Path("runs/remote"),
                 role: str = "commander"):
        self.url, self.token, self.role = url.rstrip("/"), token, role
        self.console, self.monitor, self.folder = console, monitor, folder
        self.connection = None
        self.epoch, self.session, self.status = 0, "", ("Connecting", "")
        self.task_folder: Path | None = None
        self.requests = 0
        self.sent_stop = None
        self.calls: dict[int, int] = {}  # the local monitor's skill call -> the server's number for it
        self.offer_presets = True  # the harness's presets are the dashboard's scenarios (not in the monitor app)
        self.presets: list[dict] = []
        self.camera_sent = "robot"
        self.closed = threading.Event()
        self.received_state = threading.Event()
        self.paused = threading.Event()  # the harness's operator pause, as its state reports it
        self.listener: threading.Thread | None = None

    # --- connection ---
    def connect(self, timeout: float = 10.0):
        self.connection = connect(f"{self.url}/stream", max_size=protocol.MAX_MESSAGE, open_timeout=timeout)
        self.connection.send(protocol.text("hello", token=self.token, role=self.role))
        self.listener = threading.Thread(target=self.listen, name="harness-client", daemon=True)
        self.listener.start()
        threading.Thread(target=self.follow_camera, name="harness-camera", daemon=True).start()
        if not self.received_state.wait(timeout):
            raise ConnectionError("the harness server sent no state (a bad token closes the connection)")
        return self

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.closed.set()

    def send(self, kind: str, **fields):
        if self.connection is None:
            return
        try:
            self.connection.send(protocol.text(kind, epoch=self.epoch, **fields))
        except ConnectionClosed:
            self.closed.set()

    def listen(self):
        try:
            for message in self.connection:
                try:
                    fields, rgb = protocol.parse(message)
                    self.handle(fields, rgb)
                except Exception as error:  # noqa: BLE001 -- one bad message must not end the connection
                    self.show(f"A message from the harness failed: {type(error).__name__}: {error}")
        except ConnectionClosed:
            pass
        finally:
            self.closed.set()
            if self.monitor is not None:
                self.monitor.end_task()
            self.show("Disconnected from the harness.")

    # --- the server's messages ---
    def handle(self, fields: dict, rgb):
        kind = fields["type"]
        if kind == "state":
            self.epoch, self.session = fields.get("epoch", self.epoch), fields.get("session", self.session)
            self.role = fields.get("role", self.role)
            if "paused" in fields:
                (self.paused.set if fields["paused"] else self.paused.clear)()
            self.received_state.set()
            if self.offer_presets and isinstance(fields.get("presets"), list):
                self.presets = [p for p in fields["presets"] if isinstance(p, dict) and p.get("name")]
            if "status" in fields:
                self.update(fields["status"], fields.get("notice") or fields.get("text", ""))
        elif kind == "status":
            if "paused" in fields:  # the session lifts the operator's pause itself on "continue"
                (self.paused.set if fields["paused"] else self.paused.clear)()
            state = fields.get("state")
            self.update(fields.get("status", ""), fields.get("text", ""), state if isinstance(state, dict) else None)
        elif kind in ("notice", "refused"):
            self.show(fields.get("text") or f"Refused: {fields.get('reason', '')}")
        elif kind == "view" and rgb is not None and self.console is not None:
            view = fields.get("camera") if fields.get("camera") in VIEWS else "robot"
            meta = {key: fields[key] for key in ("sim", "seq", "epoch", "base_pose", "label") if key in fields}
            self.console.remote_frame(view, rgb, meta)
        elif kind == "decision" and self.console is not None:
            decision = SimpleNamespace(text=str(fields.get("text", "")), skill=str(fields.get("skill", "")),
                                       arguments=fields.get("arguments") if isinstance(fields.get("arguments"), dict) else {})
            self.console.select(decision, None)
        elif kind == "skill" and self.console is not None:
            event = dict(fields.get("event") or {})
            if event.get("type") not in CONSOLE_SKILL_EVENTS:  # the others draw on frames only the harness has
                return
            if isinstance(event.get("result"), dict):
                result = event["result"]
                event["result"] = Result(result.get("skill", ""), result.get("code", ""),
                                         result.get("message", ""), result.get("details") or {})
            self.console.skill_event(event)
        elif kind == "clear_view" and self.console is not None:
            self.console.clear_view()
        elif kind == "shot" and rgb is not None and self.monitor is not None:
            self.monitor.capture(rgb, float(fields["t"]), fields.get("view", "head"))
        elif kind == "monitor" and self.monitor is not None:
            self.monitor_call(fields.get("call"), fields)

    def update(self, status: str, text: str = "", state: dict | None = None):
        self.status = (status, text)
        if self.console is not None:
            self.console.update(status, text, state)

    def show(self, text: str):
        if self.console is not None:
            self.console.update(self.status[0], text)

    def follow_camera(self):
        """Ask the harness for the camera the dashboard's page watches (its observer or hand view)."""
        while not self.closed.wait(CAMERA_POLL_S):
            camera = getattr(self.console, "camera", "robot")
            if camera != self.camera_sent:
                self.camera_sent = camera
                self.send("camera", camera=camera)

    def request(self, data: dict) -> VocabularyRequest:
        self.requests += 1
        folder = (self.task_folder or self.folder) / "requests"
        return request_from_wire(data, folder, f"{self.requests:04d}")

    def monitor_call(self, call: str | None, fields: dict):
        monitor = self.monitor
        if call == "start_task":
            self.task_folder = self.folder / safe_name(self.session, "session") / safe_name(fields.get("folder"), "task")
            self.sent_stop = None
            monitor.start_task(fields.get("task", ""), self.task_folder)
        elif call == "end_task":
            monitor.end_task()
        elif call == "goal":
            monitor.goal(self.request(fields["request"]), fields.get("hands", ""), plan=fields.get("plan"))
        elif call == "set_phase":
            monitor.set_phase(Phase(**fields.get("phase", {})))
        elif call == "skill_started":
            monitor.skill_started(self.request(fields["request"]), float(fields.get("t") or 0.0))
            self.calls[monitor.skill_call] = int(fields.get("number", -1))
        elif call == "skill_finished":
            monitor.skill_finished(fields.get("code"), fields.get("t"))
        elif call == "operator_resumed":
            monitor.operator_resumed()
        elif call == "dismiss":
            monitor.dismiss()
        elif call == "mark":
            monitor.mark(fields.get("label", ""), fields.get("text", ""), float(fields.get("t") or 0.0))
        elif call == "decision":
            monitor.decision(float(fields.get("t") or 0.0))

    # --- the monitor's answers ---
    def monitor_updated(self, view):
        """The local monitor's view changed: draw it, and send the server any stop and verdicts."""
        if self.console is not None:
            self.console.monitor_update(view)
        monitor = self.monitor
        reason = monitor.stop_requested()
        call = self.calls.get(monitor.skill_call, -1)  # -1: a call the server never named, never stopped
        if reason is not None and (reason, call) != self.sent_stop:
            self.sent_stop = (reason, call)
            self.send("monitor_stop", reason=reason, call=call)
        verdicts = monitor.take_verdicts()
        if verdicts:
            self.send("monitor_verdicts", verdicts=[asdict(verdict) for verdict in verdicts])

    def monitor_event(self, kind, **fields):
        """The local monitor's event: kept in the server's evidence."""
        self.send("monitor_event", event={"kind": kind, **fields})
