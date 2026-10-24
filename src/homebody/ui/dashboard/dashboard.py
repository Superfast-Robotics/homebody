"""The operator dashboard: the session's UI callbacks feed the store and the camera views; browser commands only
enqueue session commands. Session callbacks (show, select, skill_event, update, clear_view) run on the session
thread, which is also the only thread that reads the session's planner, task and hands.
"""
import sys
import time
import traceback
from threading import Condition, Thread

from homebody.session.settings import Topdown
from homebody.skills import navigate, pick, place
from homebody.ui.controls import capabilities
from homebody.ui.dashboard.server import Hub
from homebody.ui.dashboard.store import PLANNER, Store
from homebody.ui.dashboard.views import (
    AUTO_CAMERAS,
    HISTORY_QUALITY,
    HISTORY_WIDTH,
    LIVE,
    VIEWS,
    History,
    jpeg,
    render,
)
from homebody.ui.overlays import Overlays

FRAME_PERIOD = 1 / 15  # s between frames of one live view
TELEMETRY_PERIOD = 0.25  # s between pose and freshness updates
INSTRUCTION_LIMIT = 4000  # characters of one operator instruction
TARGET_KEYS = ("point_normalized_1000", "goal_map_1000", "facing_map_1000", "goal_xy_m", "facing_xy_m",
               "release_point_1000", "x_ahead_m", "y_left_m", "hand")


def goal_texts(tree):
    texts = {}

    def walk(node):
        if node.get("type") == "goal":
            texts[node["id"]] = node.get("goal", node["id"])
        for child in node.get("children", ()):
            walk(child)
    walk(tree)
    return texts


def target(decision):
    """What a decision aims at, as its own arguments state it: pixel clicks and map points, never object names."""
    args = decision.arguments if isinstance(decision.arguments, dict) else {}
    named = {key: args[key] for key in TARGET_KEYS if key in args}
    return {"skill": decision.skill, **named} if named else {"skill": decision.skill}


class Dashboard:
    def __init__(self, controls, *, host="127.0.0.1", port=8080, initial_task="", scene_name="Simulation",
                 model_name="", topdown_settings: Topdown | None = None, render_in_background=True,
                 scene_port=None, hub=Hub, clock=time.time, places=None):
        """The page is served at HOST:PORT; the 3D scene, when a SceneView asks for it, at SCENE_PORT (PORT + 1)."""
        self.controls, self.clock = controls, clock
        self.host, self.scene_port = host, port + 1 if scene_port is None else scene_port
        self.store = Store(scene_name=scene_name, model_name=model_name, initial_task=initial_task, clock=clock)
        self.overlays = Overlays(topdown_settings if topdown_settings is not None else Topdown())
        self.history = History()
        self.places = places  # the scene's fixtures, for a location label grounded in its own metadata
        self.scene_view = self.session = self._scene_server = None
        self._stills, self._sent, self._telemetry_at, self._force = {}, {}, float("-inf"), False
        self.hub = hub(host, port, snapshot=self._snapshot, on_command=self.command, history=self.history).start()
        self._pending, self._latest, self._closing, self._render_failed = Condition(), None, False, False
        self._renderer = Thread(target=self._render_newest, name="dashboard-render", daemon=True) \
            if render_in_background else None
        if self._renderer is not None:
            self._renderer.start()

    @property
    def url(self):
        return self.hub.url

    @property
    def server(self):
        """The viser server the 3D scene view draws into, started on first use and shown in the page's iframe."""
        if self._scene_server is None:
            import viser

            from homebody.ui.theme import BRAND
            server = viser.ViserServer(host=self.host, port=self.scene_port)
            server.gui.configure_theme(control_layout="collapsible", dark_mode=True, brand_color=BRAND,
                                       show_logo=False, show_share_button=False)
            server.gui.main_panel.minimize()  # the page's iframe shows the scene; viser's panel stays out of the way
            self._scene_server, self.scene_port = server, server.get_port()
        return self._scene_server

    @property
    def arms(self):
        return self.overlays.arms

    @arms.setter
    def arms(self, arms):
        self.overlays.arms = arms

    # --- frames ---
    def show(self, frame):
        if self.overlays.observe(frame):
            self.clear_selection()
        if self._renderer is None:
            self.render(frame)
            return
        with self._pending:
            self._latest = frame
            self._pending.notify()

    def _render_newest(self):
        """Draw the newest frame shown since the last drawing, skipping any shown meanwhile."""
        while True:
            with self._pending:
                self._pending.wait_for(lambda: self._latest is not None or self._closing)
                if self._closing:
                    return
                frame, self._latest = self._latest, None
            try:
                self.render(frame)
            except Exception:  # noqa: BLE001 -- one bad frame must not freeze the live view
                if not self._render_failed:
                    self._render_failed = True
                    print("dashboard: rendering a frame failed; later failures are not printed", file=sys.stderr)
                    traceback.print_exc()

    def render(self, frame, force=False):
        """Encode the live views someone watches, store a history frame when due, and send telemetry. FORCE (or
        a redraw request) skips the frame-rate limit, for overlays that just changed."""
        force, self._force, now = force or self._force, False, self.clock()
        meta = {"t": now, "sim": float(frame.time), "seq": int(frame.sequence), "epoch": int(frame.epoch)}
        wanted = self.hub.wanted() & set(LIVE)
        for view in wanted:
            if force or now - self._sent.get(view, float("-inf")) >= FRAME_PERIOD:
                rgb = render(view, self.overlays, frame)
                self._sent[view] = now
                self.hub.frame(view, meta, jpeg(rgb))
        if self.history.due(now):
            rgb = self.overlays.ego_rgb(frame)
            self.store.frame_id = self.history.add(now, jpeg(rgb, HISTORY_QUALITY, HISTORY_WIDTH))
        if force or now - self._telemetry_at >= TELEMETRY_PERIOD:
            self._telemetry_at = now
            x, y, yaw = (float(v) for v in frame.base_pose)
            place = None if self.places is None else self.places.near(x, y)
            self.hub.telemetry({**meta, "pose": [x, y, yaw], "frame": "map", "place": place})

    @property
    def camera(self):
        """The extra camera the session should render (session.show_camera): "observer", "hand" or "robot"."""
        wanted = self.hub.wanted()
        if "observer" in wanted:
            return "observer"
        return "hand" if "hand" in wanted else "robot"

    def show_camera(self, rgb, label):
        """An observer or hand camera image the session rendered because someone watches that view."""
        view = "observer" if label.lower().startswith("observer") else "hand"
        now = self.clock()
        if now - self._sent.get(view, float("-inf")) >= FRAME_PERIOD:
            self._sent[view] = now
            self._stills[view] = True
            self.hub.frame(view, {"t": now, "label": label}, jpeg(rgb))

    def _redraw(self, frame):
        if frame is not None:
            self._force = True
            self.show(frame)

    def _still(self, view, rgb):
        self._stills[view] = rgb is not None
        if rgb is None:
            self.hub.forget(view)
        else:
            self.hub.frame(view, {"t": self.clock()}, jpeg(rgb))

    def clear_selection(self):
        self.overlays.clear()
        for view in ("decision", "topdown"):
            self._still(view, None)
        if self.scene_view is not None:
            self.scene_view.clear_selection()

    def clear_view(self):
        """A scene reset: running work and overlays are gone, the timeline keeps its history."""
        self.clear_selection()
        self.store.cleared()
        self.hub.changed()

    # --- session callbacks ---
    def select(self, decision, frame):
        """Show an executed decision on its captured frame; the click view stays until the next decision."""
        self.clear_selection()
        self.overlays.select(decision, frame)
        self.store.decision(decision, goal=self._active_goal())
        if decision.skill not in PLANNER:
            self.store.note_robot(target=target(decision))
        selection = self.overlays.selection
        if selection is not None:
            self._still("decision", selection.overlay(frame, frame.rgb))
        if decision.skill == "place":
            self._still("topdown", self.overlays.topdown())
        if self.scene_view is not None and selection is not None:
            self.scene_view.show_selection(selection.point)
        self._redraw(frame)
        self.hub.changed()

    def skill_event(self, event):
        kind = event.get("type")
        self.store.skill(event)
        if kind == "skill_prepared":
            self.prepared(event["plan"])
        elif kind == "arm_return_prepared":
            self.overlays.returning(event)
            if self.scene_view is not None:
                self.scene_view.clear_arm_path()
            self._redraw(self.overlays.latest)
        self.hub.changed()

    def prepared(self, plan):
        """Draw a prepared plan: route and stance, and the palm's reference path, also in 3D."""
        overlays = self.overlays
        frame = overlays.prepared(plan)
        self._redraw(frame)
        if isinstance(plan, place.Plan) and self._stills.get("topdown"):
            self._still("topdown", overlays.topdown())
        if self.scene_view is None:
            return
        if isinstance(plan, (navigate.Plan, pick.ApproachPlan)):
            self.scene_view.show_route(overlays.marks["route"], overlays.marks["stance"][2])
        if isinstance(plan, (pick.Plan, place.Plan)) and overlays.palm_path is not None:
            self.scene_view.show_arm_path(overlays.palm_path)

    def update(self, status, text=""):
        self.store.update(status, text, task=self._task(), plan=self._plan(), robot=self._robot())
        if status == "Ready":
            self.clear_selection()
        self.hub.changed()

    def monitor(self, reading: dict):
        """The adapter a monitoring service feeds: one reading (t, level, kind observation or heartbeat, source,
        risk, threshold, confidence, reason, synthetic). Raises ValueError for a malformed reading."""
        self.store.monitor(reading)
        self.hub.changed()

    # --- what the session thread reads for the store ---
    def _task(self):
        history = getattr(self.session, "history", None) or ()
        return next((entry["operator_task"] for entry in reversed(history) if "operator_task" in entry), None)

    def _plan(self):
        planner = getattr(self.session, "planner", None)
        if planner is None:
            return None
        plan = planner.plan
        if plan is None:
            return {"tree": None, "abandoned": planner.abandoned}
        goal = planner.goal
        return {"tree": plan.tree, "goals": goal_texts(plan.tree), "status": dict(plan.status),
                "reasons": dict(plan.reasons), "active": None if goal is None else goal["id"],
                "replans": planner.replans, "replans_allowed": planner.replans_allowed, "feedback": planner.feedback}

    def _active_goal(self):
        goal = getattr(getattr(self.session, "planner", None), "goal", None)
        return None if goal is None else goal["id"]

    def _robot(self):
        context = getattr(self.session, "context", None)
        if context is None:
            return {}
        carried = dict(context.carried)
        return {"hands": {side: "free" if side not in carried else carried[side].get("state", "held")
                          for side in ("left", "right")}}

    # --- browser commands (server loop thread) ---
    def command(self, request: dict) -> dict:
        kind = request.get("type")
        handler = {"submit": self._submit, "stop_task": self._stop_task, "reset": self._reset,
                   "acknowledge": self._acknowledge, "resolve": self._resolve, "pause": self._optional("pause", "Pause requested"),
                   "resume": self._optional("resume", "Resume requested"),
                   "stop_skill": self._optional("stop_skill", "Stop requested for the current skill")}.get(kind)
        if handler is None:
            return {"ok": False, "message": f"Unknown command {str(kind)[:40]!r}"}
        return handler(request)

    def _submit(self, request):
        text = request.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "message": "Write an instruction first."}
        text = text.strip()
        if len(text) > INSTRUCTION_LIMIT:
            return {"ok": False, "message": f"Instructions are limited to {INSTRUCTION_LIMIT} characters."}
        if self.store.busy and capabilities(self.controls)["tell"]:
            self.controls.tell(text)
            self.store.instruction(text, "sent", "The agent reads it before its next decision.")
            return {"ok": True, "message": "Sent. The agent reads it before its next decision."}
        if not self.store.accepting:
            reason = self._busy_reason()
            self.store.instruction(text, "rejected", reason)
            return {"ok": False, "message": reason}
        self.controls.submit(text)
        self.store.instruction(text, "pending")
        return {"ok": True, "message": "Queued. The agent starts it next."}

    def monitor_feed(self, settings):
        """The live monitor's on_update for this dashboard ([monitor] SETTINGS)."""
        from homebody.ui.dashboard.monitor_feed import MonitorFeed
        return MonitorFeed(self, settings)

    def _busy_reason(self):
        status = self.store.status
        if status == "Reset required":
            return "The physics needs a scene reset before another task (More → Reset scene)."
        if status in ("Resetting", "Loading"):
            return "The simulation is not ready yet."
        return "The agent is working on a task. Stop it first, or wait for it to finish."

    def _stop_task(self, request):
        if not self.store.busy:
            return {"ok": False, "message": "No task is running."}
        self.controls.stop()
        self.store.intervention("stop", "Operator stopped the task",
                                "Cancels the running skill and the task. This is not an emergency stop.")
        return {"ok": True, "message": "Stop requested; the skill ends at its next cancellation check."}

    def _reset(self, request):
        self.controls.reset()
        self.store.intervention("reset", "Operator reset the scene", "Cancels any task and restores the scene.")
        return {"ok": True, "message": "Reset requested."}

    def _acknowledge(self, request):
        incident = request.get("incident")
        if not isinstance(incident, str) or not self.store.acknowledge(incident):
            return {"ok": False, "message": "That alert is not active."}
        return {"ok": True, "message": "Acknowledged."}

    def _resolve(self, request):
        incident = request.get("incident")
        found = self.store.incidents.get(incident) if isinstance(incident, str) else None
        if found is None or not self.store.resolve(incident):
            return {"ok": False, "message": "That alert is already resolved or does not exist."}
        if found.kind == "monitor" and capabilities(self.controls)["dismiss"]:
            self.controls.dismiss()  # the live monitor clears its alert and its mark on the plan
        return {"ok": True, "message": "Resolved."}

    def _optional(self, name, done):
        def handle(request):
            if not capabilities(self.controls)["pause" if name in ("pause", "resume") else name]:
                return {"ok": False, "message": "This session's controller does not support that control."}
            getattr(self.controls, name)()
            self.store.intervention(name, f"Operator: {done.lower()}")
            return {"ok": True, "message": done}
        return handle

    def _snapshot(self):
        controls = capabilities(self.controls)
        paused = getattr(self.controls, "paused", None)
        # paused: the operator's pause, or one the monitor asked for (status Paused until Resume)
        held = (paused is not None and paused.is_set()) or self.store.status == "Paused"
        state = self.store.snapshot({**controls, "paused": held})
        views = {name: {"label": label, "available": live or self._stills.get(name, False)}
                 for name, (label, live) in VIEWS.items()}
        return {**state, "views": views, "history": self.history.index(), "auto_cameras": AUTO_CAMERAS,
                "scene_port": self.scene_port if self._scene_server is not None else None}

    def close(self):
        if self._renderer is not None:
            with self._pending:
                self._closing = True
                self._pending.notify()
            self._renderer.join()
        self.hub.close()
        if self._scene_server is not None:
            self._scene_server.stop()
