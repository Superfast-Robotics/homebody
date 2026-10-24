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
FAULTED = ("Error", "Reset required")  # a harness session that needs a new attach before it runs again
BUSY = ("Thinking", "Executing", "Paused", "Stopping")  # a harness with a task: never switched under it
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
        # the Session Configuration: an Apply in progress, its last failure, the last config refresh
        self.applying, self.apply_error, self.refreshed, self.refresh_message = None, "", None, ""
        self._lifecycle = (None, "")  # (harness, its last known status)

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
        client = self.make_client(self.config.harnesses[name])  # connected before the old one closes: a failure keeps it
        if self.client is not None:
            self.client.close()
        self.client, self.harness = client, name
        return client

    # --- scenarios ---
    def scenarios(self) -> list[dict]:
        """The config's scenarios as the console's Scenario list shows them."""
        return [{"name": s.name, "scene": s.sim, "task": s.task} for s in self.config.scenarios.values()]

    def run_scenario_soon(self, name: str, text: str = ""):
        threading.Thread(target=self.run_scenario, args=(name, text), name="scenario", daemon=True).start()

    def say(self, status: str, text: str):
        self.message = text
        self.console.update(status, text)

    def sim_answer(self, sim):
        """SIM's status for a bind or an Apply: the watcher's, rechecked when it did not answer last time."""
        answering = getattr(self.watcher, "answering", None)
        return answering("sim", sim) if answering else self.watcher.status("sim", sim.name)

    def bind(self, harness_name: str, sim_name: str, label: str) -> dict:
        """The simulation server SIM_NAME answering with a controller; the harness HARNESS_NAME connected and attached
        to it (confirmed by the harness's own state); the video switched to it. Returns the harness's state; raises
        with the failed step's reason."""
        sim = self.config.sims[sim_name]
        status = self.sim_answer(sim)
        if status is None or not status.connected:
            raise RuntimeError(f"simulation server {sim.name} does not answer ({status.error if status else 'not checked yet'})")
        controller = status.detail.get("controller") or {}
        if not controller.get("running"):
            raise RuntimeError(f"simulation server {sim.name} has no running controller plugin "
                               f"({controller.get('error') or 'serve it with --controller'})")
        self.say("Loading", f"{label}: connecting to the harness {harness_name}…")
        client = self.use_harness(harness_name)
        harness = self.config.harnesses[harness_name]
        state = self.harness_state(harness) or {}
        attached = (state.get("sim") or {}).get("endpoint") == sim.endpoint and state.get("status") not in FAULTED
        if not attached:  # a harness whose session faulted on this server attaches again: that builds a new session
            if state.get("status") in BUSY:
                raise RuntimeError(f"{harness_name} is running a task on another server; stop it first")
            self.say("Loading", f"{label}: attaching {harness_name} to {sim.name}…")
            client.send("attach_sim", target=sim.target())
            state = self.wait_attached(harness, sim)
        self.sim = sim
        self.video.watch(sim.local_stream)
        return state

    def run_scenario(self, name: str, text: str = "") -> bool:
        """Its servers bound (bind); then its task (or TEXT, the instruction as edited) submitted."""
        if not self.running.acquire(blocking=False):
            self.say("Ready", "A scenario is already starting.")
            return False
        try:
            scenario = self.config.scenarios.get(name)
            if scenario is None:
                raise RuntimeError(f"no scenario {name!r} in {self.config_path.name}")
            state = self.bind(scenario.harness, scenario.sim, name)
            if state.get("status") in BUSY:
                raise RuntimeError(f"{scenario.harness} is running a task; stop it first")
            client = self.client
            client.send("submit", text=(text or scenario.task).strip())
            self.say("Ready", f"{name}: task sent to {scenario.harness} on {scenario.sim}.")
            return True
        except Exception as error:  # noqa: BLE001 -- every failed step is shown in the console with its reason
            self.say("Ready", f"Scenario failed: {error}")
            return False
        finally:
            self.running.release()

    # --- the Session Configuration ---
    def apply_refusal(self, harness_name, sim_name) -> str:
        """Why the pair cannot be applied now ("" when it can): unknown names, a server that does not answer or has
        no controller, the harness in use running a task (a session never switches under a task)."""
        if harness_name not in self.config.harnesses:
            return f"{harness_name!r} is not a configured harness"
        if sim_name not in self.config.sims:
            return f"{sim_name!r} is not a configured simulation server"
        harness = self.watcher.status("harness", harness_name)
        if harness is not None and not harness.connected:
            return f"{harness_name} does not answer ({harness.error or 'no answer'})"
        sim = self.sim_answer(self.config.sims[sim_name])
        if sim is None or not sim.connected:
            return f"{sim_name} does not answer ({sim.error if sim else 'not checked yet'})"
        if not (sim.detail.get("controller") or {}).get("running"):
            return f"{sim_name} has no running controller plugin"
        if self.harness is not None and self.lifecycle() in BUSY:
            return f"{self.harness} is running a task; stop it or let it finish before switching"
        detail = (harness.detail or {}) if harness is not None else {}
        if (harness_name != self.harness and detail.get("status") in BUSY
                and (detail.get("sim") or {}).get("endpoint") != self.config.sims[sim_name].endpoint):
            return f"{harness_name} is running a task on another server; stop it first"
        return ""

    def apply_session(self, harness_name, sim_name) -> bool:
        """Bind the pair (bind); the active pair changes only when the harness confirms the attachment."""
        if not self.start_apply(harness_name, sim_name):
            return False
        return self._apply_held(harness_name, sim_name)

    def start_apply(self, harness_name, sim_name) -> bool:
        """Take the one-change lock and mark the Apply in progress (False, with apply_error, when another runs)."""
        if not self.running.acquire(blocking=False):
            self.apply_error = "Another session change or scenario is starting"
            return False
        self.applying, self.apply_error = (harness_name, sim_name), ""
        return True

    def _apply_held(self, harness_name, sim_name) -> bool:
        previous = self.harness
        try:
            refusal = self.apply_refusal(harness_name, sim_name)
            if refusal:
                raise RuntimeError(refusal)
            self.bind(harness_name, sim_name, "Session")
            self.say("Ready", f"Session: {harness_name} → {sim_name}.")
            return True
        except Exception as error:  # noqa: BLE001 -- shown in the Session Configuration with its reason
            self.apply_error = f"Unable to activate the selected session: {error}"
            self._restore(previous)
            self.say("Ready", self.apply_error)
            return False
        finally:
            self.applying = None
            self.running.release()
            self._statuses(self.statuses)

    def _restore(self, previous):
        """After a failed Apply, the console and monitor go back to the harness they were on (the video never moved)."""
        if previous is None or previous == self.harness or previous not in self.config.harnesses:
            return
        try:
            self.use_harness(previous)
        except Exception as error:  # noqa: BLE001 -- the failed Apply is already reported; this adds why
            self.apply_error += f" (and {previous} could not be reconnected: {error})"

    def lifecycle(self) -> str:
        """The harness in use's own status (Ready, Executing, …), as its last answer says; "" without one."""
        if not self.harness:
            return ""
        status = self.statuses.get(("harness", self.harness))
        if status is None:  # dropped from the config but still in use: its last known state still guards a switch
            return self._lifecycle[1] if self._lifecycle[0] == self.harness else ""
        value = (status.detail or {}).get("status", "") if status.connected else ""
        self._lifecycle = (self.harness, value)
        return value

    def session(self) -> dict:
        """The Session Configuration's view: the active pair as confirmed (the harness this app drives, and the
        simulation server that harness reports being attached to), the harness's lifecycle, an Apply in progress or
        its failure, the last config refresh, and warnings for an active server the config no longer names."""
        harness = self.harness if self.client is not None and not self.client.closed.is_set() else None
        status = self.statuses.get(("harness", harness)) if harness else None
        link = ((status.detail or {}).get("sim") or {}) if status is not None and status.connected else {}
        confirmed = next((name for name, sim in self.config.sims.items() if sim.endpoint == link.get("endpoint")), None)
        sim = confirmed or (self.sim.name if self.sim is not None else None)
        warnings = [f"The active {kind} {name} is no longer in the configuration; it stays active until you switch"
                    for kind, name, names in (("harness", harness, self.config.harnesses), ("simulation server", sim, self.config.sims))
                    if name is not None and name not in names]
        return {"harness": harness, "sim": sim, "sim_confirmed": confirmed is not None and confirmed == sim,
                "lifecycle": self.lifecycle(), "applying": list(self.applying) if self.applying else None,
                "error": self.apply_error, "refreshed": self.refreshed, "refresh_message": self.refresh_message,
                "warnings": warnings}

    def wait_attached(self, harness, sim, timeout: float = ATTACH_WAIT_S, poll: float = 1.0) -> dict:
        """HARNESS's state once it reports SIM attached; raises at once when, after Loading, it reports the attach
        failed (its Error text says why: a server that does not answer or serves another model), else at TIMEOUT."""
        deadline, loading = time.monotonic() + timeout, False
        while time.monotonic() < deadline:
            state = self.harness_state(harness) or {}
            link = state.get("sim") or {}
            if link.get("endpoint") == sim.endpoint and link.get("connected") and state.get("status") not in ("Loading", *FAULTED):
                return state
            loading = loading or state.get("status") == "Loading"
            if loading and state.get("status") == "Error":  # an Error from before this attach does not count
                raise RuntimeError(state.get("text") or f"{harness.name} could not attach to {sim.name}")
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
                                controller=self.controller_state, session=self.session)
        dashboard.handlers.update(run_scenario=self._run_command, refresh_config=self._refresh_command,
                                  apply_session=self._apply_command)

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
        self.refreshed, self.refresh_message = time.time(), message
        return {"ok": not message.startswith("Config not"), "message": message}

    def _apply_command(self, request: dict) -> dict:
        harness, sim = request.get("harness"), request.get("sim")
        if not isinstance(harness, str) or not isinstance(sim, str):
            return {"ok": False, "message": "Choose a harness and a simulation server."}
        if not self.start_apply(harness, sim):  # taken here, so a second click is refused before any thread starts
            return {"ok": False, "message": self.apply_error}
        refusal = self.apply_refusal(harness, sim)
        if refusal:
            self.applying = None
            self.running.release()
            return {"ok": False, "message": refusal}
        threading.Thread(target=self._apply_held, args=(harness, sim), name="session", daemon=True).start()
        return {"ok": True, "message": f"Applying {harness} → {sim}…"}

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
        # an active server the new config no longer names stays active (session() warns); it is never disconnected
        # or switched by a refresh, only by the operator's Apply
        return f"Config reloaded: {len(config.harnesses)} harnesses, {len(config.sims)} simulation servers, " \
               f"{len(config.scenarios)} scenarios."
