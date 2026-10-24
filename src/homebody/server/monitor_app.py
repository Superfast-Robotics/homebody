"""The monitor app (homebody monitor --config FILE): a dashboard and the monitor on this machine, over the harnesses
and simulation servers a config names. It watches every server's status (the dashboard's Servers list), takes camera
frames straight from the active simulation server's stream (the dashboard's views and the monitor's frames), and runs
a scenario by attaching its harness to its simulation server and submitting its task."""
import threading
import time
from pathlib import Path

from homebody.server.client import RemoteControls
from homebody.server.config import load
from homebody.server.watch import SimVideo, Watcher, harness_status
from homebody.ui.servers import server_rows

ATTACH_WAIT_S = 240.0  # a harness rebuilding its session on another server: settings, models, the first frames
VIEWS = {"ego": "robot", "observer": "observer", "right_wrist": "hand"}  # the stream's camera -> the dashboard's view


class MonitorControls(RemoteControls):
    """The dashboard's buttons, sent to whichever harness the app is connected to."""

    def __init__(self, app):
        self.app = app

    @property
    def client(self):
        return self.app.client

    def _send(self, kind, **fields):
        if self.client is None:
            raise RuntimeError("no harness connected; run a scenario first")
        self.client.send(kind, **fields)


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
        self.video = video or SimVideo(self.on_frame, on_error=lambda error: self.console.frame_undecodable(error))
        self.client, self.harness, self.sim = None, None, None
        self.running = threading.Lock()  # one scenario at a time
        self.message = ""
        self.statuses: dict = {}
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
        """A camera frame from the active server's stream: the dashboard's view of that camera and, for the robot's
        own camera, the monitor's frames."""
        camera, rgb, sim_time = frame["camera"], frame["rgb"], float(frame["stamp"]["sim_time_s"])
        if camera == "ego" and self.monitor is not None and self.monitor.capture_due(sim_time):
            self.monitor.capture(rgb, sim_time)
        view = VIEWS.get(camera)
        if view is not None:
            label = f"{self.sim.name if self.sim else 'Simulation'}: {camera}"
            self.console.remote_frame(view, rgb, {"sim": sim_time, "label": label})

    # --- the dashboard's Servers list and scenarios ---
    def attach(self, dashboard):
        """DASHBOARD shows this app's servers and scenarios and runs its commands."""
        dashboard.extras.update(servers=self.server_rows, scenarios=self.scenarios, robot_link=self.robot_link,
                                controller=self.controller_state)
        dashboard.handlers.update(run_scenario=self._run_command, refresh_config=self._refresh_command)

    def robot_link(self) -> dict:
        """The dashboard's Robot row: the harness in use and its own link to its simulation server."""
        if self.client is None:
            return {"state": "unknown", "detail": "No harness in use: run a scenario"}
        if self.client.closed.is_set():
            return {"state": "lost", "detail": f"Harness {self.harness}: the connection closed"}
        status = self.statuses.get(("harness", self.harness))
        if status is None or not status.connected:
            return {"state": "unknown", "detail": f"Harness {self.harness}: no status yet"}
        link = status.detail.get("sim")
        if link is None:
            return {"state": "connected", "detail": f"Harness {self.harness} (physics in the harness)"}
        where = f"Harness {self.harness} on {link.get('endpoint', '?')}"
        return {"state": "connected", "detail": where} if link.get("connected") else \
            {"state": "lost", "detail": f"{where}: {link.get('error', 'the simulation server does not answer')}"}

    def server_rows(self) -> list[dict]:
        return server_rows(self.config, self.statuses, self.harness, self.sim.name if self.sim else None,
                           last_ok=getattr(self.watcher, "last_ok", {}))

    def controller_state(self) -> dict:
        """The controller plugin as the simulation server in use reports it (warehouse-sim describe): running,
        stopped or failed, with when that was checked; unknown without a server in use or an answer."""
        if self.sim is None:
            return {"state": "unknown", "detail": "No simulation server in use", "checked": None}
        status = self.statuses.get(("sim", self.sim.name))
        if status is None or not status.connected:
            return {"state": "unknown", "detail": f"{self.sim.name} does not answer", "checked": status and status.checked}
        controller = status.detail.get("controller") or {}
        if not controller:
            return {"state": "unknown", "detail": f"{self.sim.name} runs no controller plugin", "checked": status.checked}
        state = "running" if controller.get("running") else "error" if controller.get("error") else "stopped"
        return {"state": state, "checked": status.checked, "ticks": controller.get("ticks"),
                "detail": f"{self.sim.name} controller plugin{': ' + controller['error'] if controller.get('error') else ''}"}

    def _run_command(self, request: dict) -> dict:
        name, text = request.get("name"), request.get("text", "")
        if not isinstance(name, str) or name not in self.config.scenarios:
            return {"ok": False, "message": "Choose a scenario from the list."}
        self.run_scenario_soon(name, text if isinstance(text, str) else "")
        return {"ok": True, "message": f"Starting {name}…"}

    def _refresh_command(self, request: dict) -> dict:
        message = self.refresh_config()
        self.statuses = self.watcher.refresh()
        return {"ok": not message.startswith("Config not"), "message": message}

    def _statuses(self, statuses):
        self.statuses = statuses
        changed = getattr(getattr(self.console, "hub", None), "changed", None)
        if changed is not None:
            changed()

    def refresh_config(self) -> str:
        """Read the config again: new servers are watched, removed ones dropped; a bad config changes nothing."""
        try:
            config = load(self.config_path)
        except ValueError as error:
            return f"Config not reloaded: {error}"
        self.config = config
        self.watcher.use(config)
        if self.harness not in config.harnesses and self.client is not None:
            self.client.close()
            self.client, self.harness = None, None
        if self.sim is not None and self.sim.name not in config.sims:
            self.sim = None
            self.video.watch(None)
        return f"Config reloaded: {len(config.harnesses)} harnesses, {len(config.sims)} simulation servers, " \
               f"{len(config.scenarios)} scenarios."
