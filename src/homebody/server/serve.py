"""The harness server (`homebody serve`): a sim (and, with the REAL release, a real) harness that consoles and their
monitors connect to over one port.

HTTP GET /health answers without a token; /state and /rollouts/<id>.tar.gz (the running session, or one of its
tasks, as a bundle) need it as `Authorization: Bearer <token>`. The WebSocket is /stream: the client's first message
is `hello` with the token and a role. One commander at a time sends the operator's commands and runs the monitor;
any other client is a viewer. A stop starts a new epoch, and a command of an older epoch is refused. When the
commander goes away mid-task the run pauses (server.on_disconnect = "pause") until a commander resumes or stops it.
"""
import asyncio
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from homebody.server import protocol

HELLO_WAIT = 10.0  # s a client has to say hello
QUEUE = 256  # messages waiting for one client; a client this far behind is dropped
DROPPABLE_BACKLOG = 8  # camera images are skipped for a client this far behind
RUNNING = ("Thinking", "Executing")
BUNDLE_NAME = re.compile(r"^/rollouts/(?P<id>[A-Za-z0-9_.-]+)\.tar\.gz$")


def load_token(path: str) -> str:
    """The server's token from PATH, made (owner-only) when the file does not exist yet. Never printed."""
    file = Path(path).expanduser()
    if not file.is_file():
        file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_urlsafe(32))
    if file.stat().st_mode & 0o077:
        raise ValueError(f"the server token {file} is readable by others; chmod 600 it")
    token = file.read_text().strip()
    if len(token) < 16:
        raise ValueError(f"the server token in {file} is too short")
    return token


class Client:
    def __init__(self, connection: ServerConnection, role: str):
        self.connection, self.role = connection, role
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE)


class Hub:
    """The connected clients, and publishing to them from any thread."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[Client] = set()

    def publish(self, message, *, droppable=False, monitor=False):
        """Send MESSAGE to every client (MONITOR: the commander only). DROPPABLE ones are skipped for a client
        that is behind."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fan_out, message, droppable, monitor)

    def _fan_out(self, message, droppable, monitor):
        for client in list(self.clients):
            if monitor and client.role != "commander":
                continue
            if droppable and client.queue.qsize() > DROPPABLE_BACKLOG:
                continue
            try:
                client.queue.put_nowait(message)
            except asyncio.QueueFull:
                self.clients.discard(client)  # its handler ends with the close and frees the commander
                asyncio.ensure_future(client.connection.close(1013, "too far behind"))


class HarnessServer:
    def __init__(self, settings, controls, ui, monitor, *, session_dir: Path | None, token: str, hub: Hub,
                 sink=lambda kind, **fields: None, presets=(), scene: Path | None = None):
        """SINK records each operator command and client connection in the run's evidence; PRESETS are the
        scenarios a console may run (server/presets.py), SCENE the one simulated now."""
        self.settings, self.controls, self.ui, self.monitor, self.sink = settings, controls, ui, monitor, sink
        self.session_dir, self.token, self.hub = session_dir, token, hub
        self.presets, self.scene = {preset.name: preset for preset in presets}, scene
        self.switch: tuple | None = None  # the next session's target and task, waiting for this session's end
        self.sim = None  # the simulation server the session runs on (backends/warehouse.WarehouseRobot), if any
        self.epoch, self.commander = 1, None
        self.thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.stopping: asyncio.Event | None = None
        self.port = None
        self.exporting: asyncio.Semaphore | None = None  # one bundle built at a time (each is held in memory)

    # --- lifecycle ---
    def start(self):
        self.thread = threading.Thread(target=lambda: asyncio.run(self._serve()), name="harness-server",
                                       daemon=True)
        self.thread.start()
        if not self.ready.wait(10):
            raise RuntimeError("the harness server did not start")

    async def _serve(self):
        self.hub.loop, self.stopping, self.exporting = asyncio.get_running_loop(), asyncio.Event(), asyncio.Semaphore()
        server = self.settings.server
        async with serve(self.connected, server.host, server.port, process_request=self.http,
                         max_size=protocol.MAX_MESSAGE) as listening:
            self.port = next(iter(listening.sockets)).getsockname()[1]
            self.ready.set()
            await self.stopping.wait()

    def attach(self, session_dir: Path, scene: Path, sink, sim=None):
        """A session is ready: its evidence folder, scene, recorder and (backends/warehouse) simulation server;
        consoles hear of it."""
        self.session_dir, self.scene, self.sink, self.sim = session_dir, scene, sink, sim
        self.hub.publish(protocol.text("state", **self.state()))

    def sim_status(self) -> dict | None:
        """The simulation server link, as the monitor app shows it (None: physics in this process)."""
        if self.sim is None:
            return None
        try:
            return {"connected": True, **self.sim.status()}
        except Exception as error:  # noqa: BLE001 -- a lost link is a status, not a failure of the server
            return {"connected": False, "endpoint": self.sim.target.endpoint, "error": f"{type(error).__name__}: {error}"[:200]}

    def take_switch(self) -> tuple[Path | None, str | None]:
        """The scene and task a preset asked for once this scene's session ended; (None, None): none, stop."""
        switch, self.switch = self.switch, None
        return switch if switch is not None else (None, None)

    def stop(self):
        if self.hub.loop is not None and self.stopping is not None:
            self.hub.loop.call_soon_threadsafe(self.stopping.set)
        if self.thread is not None:
            self.thread.join(timeout=5)

    # --- HTTP ---
    def authorized(self, headers: Headers) -> bool:
        given = headers.get("Authorization", "")
        return given.startswith("Bearer ") and hmac.compare_digest(given[7:].encode(), self.token.encode())

    async def http(self, connection: ServerConnection, request: Request):
        path = request.path.split("?")[0]
        if path == "/stream":
            return None  # the WebSocket handshake goes on
        if path == "/health":
            return connection.respond(200, "ok\n")
        if not self.authorized(request.headers):
            return connection.respond(401, "a token is required\n")
        if path == "/state":
            body = json.dumps(self.state()).encode()
            return Response(200, "OK", Headers({"Content-Type": "application/json",
                                                "Content-Length": str(len(body))}), body)
        match = BUNDLE_NAME.match(path)
        if match is None or self.session_dir is None:
            return connection.respond(404, "not found\n")
        rollout_id = match["id"]
        task = rollout_id.removeprefix(self.session_dir.name + "-") if rollout_id != self.session_dir.name else None
        if task is not None and not re.fullmatch(r"task-\d{3}", task):
            return connection.respond(404, "no such rollout on this server\n")
        try:
            from homebody.rollout import bundle
        except ImportError:
            return connection.respond(404, "this harness has no rollout export\n")
        try:
            async with self.exporting:
                body = await asyncio.to_thread(self.bundle, task)
        except bundle.BundleError as error:
            return connection.respond(404, f"{error}\n")
        return Response(200, "OK", Headers({"Content-Type": "application/gzip", "Content-Length": str(len(body)),
                                            "Content-Disposition": f'attachment; filename="{rollout_id}.tar.gz"'}),
                        body)

    def bundle(self, task: str | None) -> bytes:
        from homebody.rollout import bundle
        with tempfile.TemporaryDirectory() as scratch:
            return bundle.export(self.session_dir, Path(scratch), task=task,
                                 light=self.settings.rollouts.light).read_bytes()

    def state(self) -> dict:
        status, text = self.ui.status
        paused = getattr(self.controls, "paused", None)  # the operator's pause holds a skill mid-motion too
        monitor = getattr(self.settings, "monitor", None)  # a branch without the live monitor has no [monitor]
        return {"session": self.session_dir.name if self.session_dir else "", "epoch": self.epoch,
                "sim": self.sim_status(),
                "status": status, "text": text, "scene": self.scene.name if self.scene else "",
                "paused": bool(paused is not None and paused.is_set()),
                "presets": [preset.public() for preset in self.presets.values()],
                "camera": self.ui.camera, "commander": self.commander is not None,
                "clients": len(self.hub.clients), "monitor_capture_hz": monitor.capture_hz if monitor is not None else None}

    # --- the WebSocket ---
    async def connected(self, connection: ServerConnection):
        try:
            message = await asyncio.wait_for(connection.recv(), HELLO_WAIT)
            if not isinstance(message, str):
                raise ValueError("hello is text")  # noqa: TRY004 -- every bad hello is one refusal
            hello, _ = protocol.parse(message)
        except (TimeoutError, ValueError, ConnectionClosed):
            await connection.close(1008, "say hello with the token first")
            return
        token = str(hello.get("token", ""))
        if hello.get("type") != "hello" or not hmac.compare_digest(token.encode(), self.token.encode()):
            await connection.close(1008, "refused: bad token")
            return
        role = hello.get("role") if hello.get("role") in protocol.ROLES else "viewer"
        notice = ""
        if role == "commander" and self.commander is not None:
            role, notice = "viewer", "another client commands this harness; you are watching"
        client = Client(connection, role)
        self.sink("client_connected", role=role, remote=str(connection.remote_address)[:60])
        sender = None
        try:
            if role == "commander":
                self.commander = client
                if self.monitor is not None:
                    self.monitor.client_connected(True)
            self.hub.clients.add(client)
            await connection.send(protocol.text("state", role=role, notice=notice, **self.state()))
            sender = asyncio.create_task(self.send_queued(client))
            async for message in connection:
                await self.received(client, message)
        except ConnectionClosed:
            pass
        finally:
            if sender is not None:
                sender.cancel()
            self.hub.clients.discard(client)
            self.sink("client_disconnected", role=client.role)
            if client is self.commander:
                self.commander = None
                if self.monitor is not None:
                    self.monitor.client_connected(False)
                self.commander_left()

    async def send_queued(self, client: Client):
        try:
            while True:
                await client.connection.send(await client.queue.get())
        except ConnectionClosed:
            pass

    def commander_left(self):
        if self.settings.server.on_disconnect == "pause" and self.ui.status[0] in RUNNING:
            if not hasattr(self.controls, "pause"):  # an older console's Controls cannot pause: stop instead
                self.controls.stop()
                return
            self.controls.pause()
            self.hub.publish(protocol.text("notice", text="The commanding console went away; the run is paused "
                                                          "until a commander resumes or stops it."))

    async def received(self, client: Client, message):
        try:
            if not isinstance(message, str):  # clients send no images; nothing of theirs is decoded
                raise ValueError("clients send text messages only")  # noqa: TRY004 -- every bad message is one refusal
            fields, _ = protocol.parse(message)
        except ValueError as error:
            await client.connection.send(protocol.text("refused", reason=f"malformed message: {error}"))
            return
        kind = fields["type"]
        if client.role != "commander":
            await client.connection.send(protocol.text("refused", reason="viewers cannot command", command=kind))
            return
        if kind in protocol.MONITOR:
            if self.monitor is None:
                await client.connection.send(protocol.text("refused", command=kind, reason="this harness runs no monitor"))
                return
            try:
                self.monitor_message(kind, fields)
            except (TypeError, KeyError, ValueError, AttributeError) as error:
                await client.connection.send(protocol.text("refused", command=kind,
                                                           reason=f"malformed {kind}: {type(error).__name__}"))
            return
        if kind not in protocol.COMMANDS:
            await client.connection.send(protocol.text("refused", reason="unknown command", command=kind))
            return
        if fields.get("epoch") != self.epoch:
            await client.connection.send(protocol.text("refused", command=kind, epoch=self.epoch,
                                                       reason=f"epoch {fields.get('epoch')} is over"))
            return
        refusal = self.command(kind, fields)
        if refusal:
            await client.connection.send(protocol.text("refused", command=kind, reason=refusal))

    def monitor_message(self, kind: str, fields: dict):
        if kind == "monitor_stop":
            call = fields.get("call")
            self.monitor.client_stop(str(fields.get("reason", ""))[:500], call if type(call) is int else -1)
        elif kind == "monitor_verdicts":
            self.monitor.client_verdicts([v for v in fields.get("verdicts", ()) if isinstance(v, dict)])
        else:
            event = fields.get("event")
            if isinstance(event, dict):
                self.monitor.client_event(dict(event))

    def command(self, kind: str, fields: dict) -> str | None:
        """Carry out a commander's command; why it was refused, or None."""
        controls, value = self.controls, fields.get("text", "")
        self.sink("operator_command", command=kind, epoch=self.epoch, status=self.ui.status[0],
                  **({"camera": str(fields.get("camera"))[:20]} if kind == "camera" else {}))
        if kind == "submit" and isinstance(value, str) and value.strip():
            controls.submit(value[:4000])
        elif kind == "tell" and isinstance(value, str) and value.strip():
            controls.tell(value[:2000])
        elif kind == "camera" and fields.get("camera") in ("robot", "observer", "hand"):
            self.ui.camera = fields["camera"]
        elif kind == "stop":
            controls.stop()
            self.epoch += 1
            self.hub.publish(protocol.text("state", **self.state()))
        elif kind in ("pause", "unpause", "stop_skill", "dismiss", "resume_task"):
            if not hasattr(controls, kind):  # an older console's Controls (submit, stop, reset) has none of these
                return f"this harness cannot {kind.replace('_', ' ')}"
            getattr(controls, kind)()
            if kind in ("pause", "unpause"):  # held at once, even mid-skill: consoles see it without a status change
                self.hub.publish(protocol.text("state", **self.state()))
        elif kind == "preset":
            return self.run_preset(fields)
        elif kind == "attach_sim":
            return self.attach_sim(fields.get("target"))
        return None

    def attach_sim(self, target) -> str | None:
        """Move this harness to another simulation server (backends/warehouse.SimTarget fields) when idle."""
        from homebody.backends.warehouse import SimTarget
        try:
            sim = SimTarget.from_json(target if isinstance(target, dict) else {})
        except (ValueError, TypeError) as error:
            return f"bad simulation target: {error}"
        if self.ui.status[0] in (*RUNNING, "Paused", "Loading") or self.switch is not None:
            return "stop the running task before attaching another simulation server"
        self.switch = (sim, None)
        self.ui.update("Loading", f"Attaching to {sim.endpoint}…")
        self.controls.closing.set()  # this session ends; serve builds the next on the new server
        return None

    def run_preset(self, fields: dict) -> str | None:
        """Run a preset: its task (or the console's edited TEXT) here, or after loading its scene."""
        preset = self.presets.get(fields.get("name"))
        if preset is None:
            return "no such preset on this harness"
        text = fields.get("text") if isinstance(fields.get("text"), str) and fields["text"].strip() else preset.task
        text = text.strip()[:4000]
        if preset.scene == self.scene:
            self.controls.submit(text)
            return None
        if self.ui.status[0] in (*RUNNING, "Paused", "Loading") or self.switch is not None:
            return "stop the running task before loading another scene"
        self.switch = (preset.scene, text)
        self.ui.update("Loading", f"Loading {preset.name}…")
        self.controls.closing.set()  # this scene's session ends; serve builds the preset's
        return None
