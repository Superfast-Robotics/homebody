"""The monitor app (homebody monitor --config FILE): a console and the monitor on this machine, over the harnesses and
simulation servers a config names. It watches every server's status, takes camera frames straight from the active
simulation server's stream (the live view and the monitor's frames), and runs a scenario by attaching its harness to
its simulation server and submitting its task."""
import threading
import time
from pathlib import Path

from homebody.server.client import RemoteControls
from homebody.server.config import load
from homebody.server.watch import SimVideo, Watcher, harness_status

ATTACH_WAIT_S = 240.0  # a harness rebuilding its session on another server: settings, models, the first frames
CAMERAS = {"robot": "ego", "observer": "observer", "hand": "right_wrist"}  # the console's view -> the stream's camera
VIEW_PERIOD_S = 0.1  # the live view is redrawn at most this often


class MonitorControls(RemoteControls):
    """The console's buttons, sent to whichever harness the app is connected to; a scenario runs through the app."""

    def __init__(self, app):
        self.app = app

    @property
    def client(self):
        return self.app.client

    def preset(self, name: str, text: str = ""):
        self.app.run_scenario_soon(name, text)


class MonitorApp:
    def __init__(self, config_path: Path, *, console, monitor=None, make_client=None, harness_state=None,
                 watcher=None, video=None):
        """MAKE_CLIENT(harness) -> a connected HarnessClient; HARNESS_STATE(harness) -> its /state (dict, or None
        when it does not answer). Both default to the real ones; tests pass fakes."""
        self.config_path, self.console, self.monitor = Path(config_path), console, monitor
        self.config = load(self.config_path)
        self.make_client = make_client or self._connect
        self.harness_state = harness_state or (lambda harness: harness_status(harness).detail or None)
        self.watcher = watcher or Watcher(self.config, on_change=self._statuses)
        self.video = video or SimVideo(self.on_frame)
        self.client, self.harness, self.sim = None, None, None
        self.running = threading.Lock()  # one scenario at a time
        self.message = ""
        self._last_view = 0.0
        self._last_shot = float("-inf")

    # --- the active harness ---
    def _connect(self, harness):
        from homebody.server.client import HarnessClient
        token = Path(harness.token_file).expanduser().read_text().strip()
        client = HarnessClient(harness.url, token, console=self.console, monitor=self.monitor,
                               folder=Path("runs/remote"))
        client.offer_presets = False  # scenarios come from the config, not the harness
        if self.monitor is not None:
            self.monitor.on_update, self.monitor.sink = client.monitor_updated, client.monitor_event
        return client.connect()

    def use_harness(self, name: str):
        """Connect the console and the monitor to the harness NAME (closing any other)."""
        if self.harness == name and self.client is not None and not self.client.closed.is_set():
            return self.client
        if self.client is not None:
            self.client.close()
        self.client, self.harness = None, None
        self.client = self.make_client(self.config.harnesses[name])
        self.harness = name
        return self.client

    # --- scenarios ---
    def scenarios(self) -> list[dict]:
        """The config's scenarios as the console's Scenario list shows them."""
        return [{"name": s.name, "scene": s.sim, "task": s.task} for s in self.config.scenarios.values()]

    def run_scenario_soon(self, name: str, text: str = ""):
        threading.Thread(target=self.run_scenario, args=(name, text), name="scenario", daemon=True).start()

    def say(self, status: str, text: str):
        self.message = text
        self.console.update(status, text)

    def run_scenario(self, name: str, text: str = "") -> bool:
        """Its simulation server answering with a controller; its harness connected and attached to that server;
        the video switched to it; then its task (or TEXT, the instruction as edited) submitted."""
        if not self.running.acquire(blocking=False):
            self.say("Ready", "A scenario is already starting.")
            return False
        try:
            scenario = self.config.scenarios.get(name)
            if scenario is None:
                raise RuntimeError(f"no scenario {name!r} in {self.config_path.name}")
            sim = self.config.sims[scenario.sim]
            status = self.watcher.status("sim", sim.name)
            if status is None or not status.connected:
                raise RuntimeError(f"simulation server {sim.name} does not answer ({status.error if status else 'not checked yet'})")
            controller = status.detail.get("controller") or {}
            if not controller.get("running"):
                raise RuntimeError(f"simulation server {sim.name} has no running controller plugin "
                                   f"({controller.get('error') or 'serve it with --controller'})")
            self.say("Loading", f"{name}: connecting to the harness {scenario.harness}…")
            client = self.use_harness(scenario.harness)
            harness = self.config.harnesses[scenario.harness]
            state = self.harness_state(harness) or {}
            attached = (state.get("sim") or {}).get("endpoint") == sim.endpoint
            if not attached:
                self.say("Loading", f"{name}: attaching {scenario.harness} to {sim.name}…")
                client.send("attach_sim", target=sim.target())
                state = self.wait_attached(harness, sim)
            self.sim = sim
            self.video.watch(sim.local_stream)
            if state.get("status") in ("Thinking", "Executing", "Paused"):
                raise RuntimeError(f"{scenario.harness} is running a task; stop it first")
            client.send("submit", text=(text or scenario.task).strip())
            self.say("Ready", f"{name}: task sent to {scenario.harness} on {sim.name}.")
            return True
        except Exception as error:  # noqa: BLE001 -- every failed step is shown in the console with its reason
            self.say("Ready", f"Scenario failed: {error}")
            return False
        finally:
            self.running.release()

    def wait_attached(self, harness, sim, timeout: float = ATTACH_WAIT_S, poll: float = 1.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.harness_state(harness) or {}
            link = state.get("sim") or {}
            if link.get("endpoint") == sim.endpoint and link.get("connected") and state.get("status") != "Loading":
                return state
            time.sleep(poll)
        raise RuntimeError(f"{harness.name} did not attach to {sim.name} within {timeout:.0f} s")

    # --- frames from the simulation server ---
    def on_frame(self, frame: dict):
        """A camera frame from the active server's stream: the live view (the console's camera choice) and, for the
        robot's own camera, the monitor's frames."""
        camera, rgb, sim_time = frame["camera"], frame["rgb"], float(frame["stamp"]["sim_time_s"])
        if camera == "ego" and self.monitor is not None and self.monitor.capture_due(sim_time):
            self.monitor.capture(rgb, sim_time)
        console = self.console
        if getattr(console, "replaying", False) or camera != CAMERAS.get(console.camera, "ego"):
            return
        now = time.monotonic()
        if now - self._last_view < VIEW_PERIOD_S:
            return
        self._last_view = now
        console.ego.image = rgb
        console.ego.label = f"{self.sim.name if self.sim else 'Simulation'}: {camera}"

    # --- the Servers panel ---
    def _statuses(self, statuses):
        panel = getattr(self, "panel", None)
        if panel is not None:
            panel.draw(statuses)

    def refresh_config(self) -> str:
        """Read the config again: new servers are watched, removed ones dropped; a bad config changes nothing."""
        try:
            config = load(self.config_path)
        except ValueError as error:
            return f"Config not reloaded: {error}"
        self.config = config
        self.watcher.use(config)
        self.console.set_presets(self.scenarios(), self.sim.name if self.sim else "")
        if self.harness not in config.harnesses and self.client is not None:
            self.client.close()
            self.client, self.harness = None, None
        if self.sim is not None and self.sim.name not in config.sims:
            self.sim = None
            self.video.watch(None)
        return f"Config reloaded: {len(config.harnesses)} harnesses, {len(config.sims)} simulation servers, " \
               f"{len(config.scenarios)} scenarios."
