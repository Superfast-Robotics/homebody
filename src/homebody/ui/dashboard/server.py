"""One port for the dashboard: its static page, stored camera frames, and a WebSocket at /ws.

The asyncio loop runs on its own thread. Other threads only call the thread-safe methods: `changed` (state to
resend), `frame` (a camera JPEG), `telemetry` (small live readings) and `close`. Browser commands go to
ON_COMMAND on the loop thread, which must not block.
"""
import asyncio
import json
import re
import struct
from http import HTTPStatus
from pathlib import Path
from threading import Event, Thread
from urllib.parse import urlsplit

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

STATIC = Path(__file__).with_name("static")
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
POLICY = ("default-src 'self'; img-src 'self' blob: data:; connect-src 'self' ws: wss:; "
          "frame-src http: https:; style-src 'self'; script-src 'self'; base-uri 'none'; form-action 'none'")
HISTORY = re.compile(r"^/history/(\d{1,12})\.jpg$")
STATE_PERIOD = 0.2  # s between state sends while it keeps changing
MESSAGE_LIMIT = 64 * 1024  # bytes of one browser message


def static_files(root=STATIC):
    """URL path -> (content type, bytes) for every page file, read once; nothing else is served."""
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in TYPES:
            files["/" + path.relative_to(root).as_posix()] = (TYPES[path.suffix], path.read_bytes())
    files["/"] = files["/index.html"]
    return files


def response(status, content_type, body):
    headers = Headers([("Content-Type", content_type), ("Content-Length", str(len(body))),
                       ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                       ("Referrer-Policy", "no-referrer")])
    if content_type.startswith("text/html"):
        headers["Content-Security-Policy"] = POLICY
    return Response(status.value, status.phrase, headers, body)


def same_origin(request) -> bool:
    """A WebSocket opened by this page: no Origin (a non-browser client) or one naming the requested host."""
    origin = request.headers.get("Origin")
    return origin is None or urlsplit(origin).netloc == request.headers.get("Host")


def frame_message(meta: dict, data: bytes) -> bytes:
    """A binary frame message: 4-byte big-endian header length, the JSON header, then the JPEG."""
    header = json.dumps(meta).encode()
    return struct.pack(">I", len(header)) + header + data


class Client:
    def __init__(self, connection):
        self.connection, self.view, self.sending = connection, "robot", False


class Hub:
    def __init__(self, host, port, *, snapshot, on_command, history=None, files=None):
        """SNAPSHOT() returns the state dict; ON_COMMAND(message) returns the reply dict."""
        self.host, self.port = host, port
        self.snapshot, self.on_command, self.history = snapshot, on_command, history
        self.files = static_files() if files is None else files
        self.clients, self.latest, self._tasks = set(), {}, set()
        self._dirty, self._stop = asyncio.Event(), asyncio.Event()  # bound to the loop on first use
        self._started, self._loop, self._error = Event(), None, None
        self._thread = Thread(target=self._run, name="dashboard-server", daemon=True)

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        self._thread.start()
        self._started.wait()
        if self._error is not None:
            raise self._error
        return self

    # --- thread-safe entry points ---
    def changed(self):
        self._call(self._dirty.set)

    def frame(self, view, meta, data):
        message = frame_message({**meta, "view": view}, data)
        self._call(lambda: self._frame(view, message))

    def telemetry(self, values):
        message = json.dumps({"type": "telemetry", **values})
        self._call(lambda: self._send_all(message))

    def forget(self, view):
        """Drop VIEW's last frame so a new viewer does not see a stale still."""
        self._call(lambda: self.latest.pop(view, None))

    def wanted(self):
        return {client.view for client in tuple(self.clients)}

    def close(self):
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=5)

    def _call(self, callback):
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(callback)

    # --- loop thread ---
    def _run(self):
        try:
            asyncio.run(self._serve())
        except Exception as error:  # noqa: BLE001 -- start() re-raises a bind failure in the caller
            self._error = error
            self._started.set()

    async def _serve(self):
        self._loop = asyncio.get_running_loop()
        async with serve(self._connect, self.host, self.port, process_request=self._http,
                         max_size=MESSAGE_LIMIT, compression=None) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._started.set()
            sender = asyncio.create_task(self._send_states())
            await self._stop.wait()
            sender.cancel()

    def _http(self, connection, request):
        path = urlsplit(request.path).path
        if path == "/ws":
            return None if same_origin(request) else response(HTTPStatus.FORBIDDEN, "text/plain", b"Forbidden")
        match = HISTORY.match(path)
        if match and self.history is not None:
            data = self.history.get(int(match.group(1)))
            if data is not None:
                return response(HTTPStatus.OK, "image/jpeg", data)
            return response(HTTPStatus.NOT_FOUND, "text/plain", b"That frame is no longer kept")
        if path in self.files:
            return response(HTTPStatus.OK, *self.files[path])
        return response(HTTPStatus.NOT_FOUND, "text/plain", b"Not found")

    async def _connect(self, connection):
        client = Client(connection)
        self.clients.add(client)
        try:
            await connection.send(self._state())
            await self._subscribe(client, client.view)
            async for message in connection:
                await self._message(client, message)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(client)

    async def _message(self, client, message):
        try:
            request = json.loads(message) if isinstance(message, str) else None
        except json.JSONDecodeError:
            request = None
        if not isinstance(request, dict):
            await client.connection.send(json.dumps({"type": "reply", "ok": False, "message": "Malformed message"}))
            return
        if request.get("type") == "view":
            await self._subscribe(client, request.get("view"))
            return
        reply = self.on_command(request)
        await client.connection.send(json.dumps({"type": "reply", "id": request.get("id"), **reply}))
        self._dirty.set()

    async def _subscribe(self, client, view):
        if not isinstance(view, str) or len(view) > 32:
            return
        client.view = view
        if view in self.latest:
            await client.connection.send(self.latest[view])

    def _state(self):
        return json.dumps({"type": "state", "state": self.snapshot()})

    async def _send_states(self):
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            self._send_all(self._state())
            await asyncio.sleep(STATE_PERIOD)

    def _send_all(self, message):
        for client in tuple(self.clients):
            self._spawn(self._send(client, message))

    def _frame(self, view, message):
        """Send a camera frame to the clients watching VIEW; a client still sending its last frame skips it."""
        self.latest[view] = message
        for client in tuple(self.clients):
            if client.view == view and not client.sending:
                client.sending = True
                self._spawn(self._send(client, message, frame=True))

    def _spawn(self, coroutine):
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self, client, message, frame=False):
        try:
            await client.connection.send(message)
        except ConnectionClosed:
            self.clients.discard(client)
        finally:
            if frame:
                client.sending = False
