"""What the monitor app keeps watching: each harness (its /state), each simulation server (warehouse-sim's describe,
read-only: the monitor never takes command ownership) and the active simulation server's camera stream."""
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

POLL_S = 2.0  # how often a server's status is refreshed
TIMEOUT_S = 2.0  # a status request slower than this counts as a lost link


@dataclass(frozen=True)
class Status:
    """One server's last known state, as the Servers panel draws it."""
    name: str
    kind: str  # "harness" or "sim"
    connected: bool = False
    latency_ms: float | None = None
    detail: dict = field(default_factory=dict)
    error: str = ""
    checked: float = 0.0  # wall time of the last check


def harness_status(harness, timeout=TIMEOUT_S) -> Status:
    """HARNESS's /state (config.Harness), through its token."""
    url = harness.url.replace("ws://", "http://", 1) + "/state"
    start = time.monotonic()
    try:
        token = Path(harness.token_file).expanduser().read_text().strip()
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            state = json.loads(reply.read())
    except (OSError, urllib.error.URLError, ValueError) as error:
        return Status(harness.name, "harness", error=f"{type(error).__name__}: {error}"[:200], checked=time.time())
    return Status(harness.name, "harness", True, round((time.monotonic() - start) * 1e3, 1), state, checked=time.time())


def sim_status(sim, timeout=TIMEOUT_S) -> Status:
    """SIM's describe (config.Sim) at the address this machine reaches it by; read-only."""
    from warehouse_sim.client import SimulatorClient
    start = time.monotonic()
    try:
        description = SimulatorClient(sim.local_endpoint, timeout_s=timeout).describe()
    except Exception as error:  # noqa: BLE001 -- any failure to answer is a lost link, shown as such
        return Status(sim.name, "sim", error=f"{type(error).__name__}: {error}"[:200], checked=time.time())
    detail = {"profile": description["profile"]["id"], "mode": description["mode"],
              "controller": description.get("controller"), "stream": description.get("camera_stream"),
              "overshoot": description.get("motor_target_overshoot", {})}
    return Status(sim.name, "sim", True, round((time.monotonic() - start) * 1e3, 1), detail, checked=time.time())


class Watcher:
    """Refreshes the status of every server in CONFIG every POLL_S on a thread; ON_CHANGE(statuses) after each
    round. `use(config)` swaps the servers watched (Refresh config)."""

    def __init__(self, config, on_change=lambda statuses: None, poll_s=POLL_S, check=None):
        self.config, self.on_change, self.poll_s = config, on_change, poll_s
        self._check = check or (lambda kind, server: harness_status(server) if kind == "harness" else sim_status(server))
        self.statuses: dict[tuple[str, str], Status] = {}
        self.last_ok: dict[tuple[str, str], float] = {}  # wall time each server last answered
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="server-watch", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def use(self, config):
        with self._lock:
            self.config = config
            names = {("harness", n) for n in config.harnesses} | {("sim", n) for n in config.sims}
            self.statuses = {key: value for key, value in self.statuses.items() if key in names}

    def refresh(self):
        """One round now (also what the thread does every poll_s): every server at once, so one that does not
        answer costs a round its timeout, not the sum of them."""
        with self._lock:
            servers = [("harness", h) for h in self.config.harnesses.values()] + \
                      [("sim", s) for s in self.config.sims.values()]
        if not servers:
            return {}
        with ThreadPoolExecutor(max_workers=len(servers), thread_name_prefix="server-check") as pool:
            checks = {(kind, server.name): pool.submit(self._check, kind, server) for kind, server in servers}
            results = {key: check.result() for key, check in checks.items()}
        with self._lock:
            self.statuses.update(results)
            self.last_ok.update((key, status.checked) for key, status in results.items() if status.connected)
            statuses = dict(self.statuses)
        self.on_change(statuses)
        return statuses

    def status(self, kind: str, name: str) -> Status | None:
        with self._lock:
            return self.statuses.get((kind, name))

    def _run(self):
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception as error:  # noqa: BLE001 -- a bad round must not end the watch
                print(f"server watch: {type(error).__name__}: {error}")
            self._stop.wait(self.poll_s)

    def stop(self):
        self._stop.set()


class SimVideo:
    """The active simulation server's camera stream: ON_FRAME({camera, stamp, rgb}) for each frame received.
    `watch(endpoint)` switches servers (None: none)."""

    def __init__(self, on_frame, on_error=lambda error: None):
        """ON_ERROR(text) for a frame that arrived but could not be decoded."""
        self.on_frame, self.on_error, self.endpoint = on_frame, on_error, None
        self.frames, self.error = 0, ""
        self._lock, self._stop, self._changed = threading.Lock(), threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="sim-video", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def watch(self, endpoint: str | None):
        with self._lock:
            self.endpoint, self.frames, self.error = endpoint, 0, ""
        self._changed.set()

    def _run(self):
        import zmq
        from warehouse_sim.stream import decode
        context = zmq.Context.instance()
        while not self._stop.is_set():
            self._changed.clear()
            with self._lock:
                endpoint = self.endpoint
            if endpoint is None:
                self._changed.wait(0.5)
                continue
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.setsockopt(zmq.RCVHWM, 4)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(endpoint)
            try:
                while not self._stop.is_set() and not self._changed.is_set():
                    if not socket.poll(250):
                        continue
                    _, payload = socket.recv_multipart()
                    try:
                        frame = decode(payload)
                    except ValueError as error:
                        self.error = str(error)[:200]
                        self.on_error(self.error)
                        continue
                    self.frames += 1
                    self.on_frame(frame)
            except Exception as error:  # noqa: BLE001 -- a broken stream is shown, then retried
                self.error = f"{type(error).__name__}: {error}"[:200]
                self._changed.wait(1.0)
            finally:
                socket.close(0)

    def stop(self):
        self._stop.set()
        self._changed.set()


def stale(status: Status | None, now: float | None = None, poll_s: float = POLL_S) -> Status | None:
    """STATUS marked disconnected when its last check is older than three polls (the watcher stalled)."""
    now = time.time() if now is None else now
    if status is not None and status.connected and now - status.checked > 3 * poll_s:
        return replace(status, connected=False, error="no status for %.0f s" % (now - status.checked))
    return status
