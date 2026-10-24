"""Viser displays supplied frames; callbacks only enqueue session commands."""
import socket
import sys
import time
import traceback
from dataclasses import dataclass, replace
from html import escape
from itertools import pairwise
from threading import Condition, Lock, Thread

import numpy as np

from homebody.monitor.monitor import View
from homebody.session.settings import Topdown
from homebody.skills import navigate, pick, place
from homebody.ui import monitor_panel, plan_view
from homebody.ui.controls import Controls
from homebody.ui.overlays import Overlays
from homebody.ui.theme import (
    note,
    open_server,
    section,
)

AWAITING = "Top-down: awaiting a decision"
CAMERAS = {"Robot view": "robot", "Observer view": "observer", "Hand camera": "hand"}  # the live view's choices
CUSTOM = "Custom instruction"  # the scenario list's first entry: the instruction as typed
STATUS = {"model_done": "Model reports done", "step_limit": "Step limit reached",
          "provider_error": "Provider error", "cancelled": "Cancelled", "error": "Error",
          "interrupted": "Interrupted"}


def claim(host, port):
    """Raise OSError while another server listens on HOST:PORT. TIME_WAIT connections do not count."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))


def blank(height, width):
    return np.full((height, width, 3), 235, dtype=np.uint8)


def thought_bubble(message="", *, thinking=False, model=""):
    """The agent bubble's HTML with the model's decision text."""
    title = "Agent" + (f": {model}" if model else "") + (" Thinking about the next step…" if thinking else "")
    text = message or ("" if thinking else "Ready for your task.")
    return (f'<div class="hb-bubble" role="status" aria-live="polite">'
            f'<div class="hb-title">{escape(title)}</div>'
            f'<div class="hb-message">{escape(text)}</div></div>')


def stage_progress(skill, stages, result=None, *, durations=None, simulated=None):
    """The stage panel's HTML for the reached stages and, once finished, the result."""
    count = len(stages)
    title = f"{skill.capitalize()}: {stages[-1]}" if stages else skill.capitalize()
    refused = result is not None and result.code != "OK" and count <= 1  # it never moved
    if refused:
        title = f"{skill.capitalize()} refused: {result.message}"
    label = f"Stage {count}" if result is None else f"Result: {result.code}"
    if durations is not None:
        label += f" · {sum(durations):.1f}s elapsed"
        if simulated is not None:
            label += f" ({simulated:.1f}s sim)"
    recent = list(zip(stages, [None] * count if durations is None else durations, strict=True))[-5:]
    last_class = "is-current" if result is None else "is-done" if result.code == "OK" else "is-failed"
    steps = []
    for index, (name, duration) in enumerate(recent):
        css_class = last_class if index == len(recent) - 1 else "is-done"
        clock = f'<span class="hb-step-time">{duration:.1f}s</span>' if duration is not None else ""
        steps.append(f'<li class="{css_class}">{escape(name)}{clock}</li>')
    steps = "".join(steps)
    history = f'<ol class="hb-steps">{steps}</ol>' if steps else ""
    message = f'<div class="hb-stage-count">{escape(result.message)}</div>' if result is not None and not refused else ""
    css = " hb-refused" if refused else " hb-result" if result is not None else ""
    return (f'<div class="hb-stage{css}" role="status" aria-live="polite">'
            f'<div class="hb-stage-head"><strong>{escape(title)}</strong>'
            f'<span class="hb-stage-count">{escape(label)}</span></div>{history}{message}</div>')


@dataclass(frozen=True)
class Progress:
    """The stage panel's state, replaced whole so the render thread reads a consistent snapshot."""
    skill: str = ""
    stages: tuple = ()
    times: tuple = ()
    result: object = None
    finished_at: float | None = None
    sim_start: float | None = None
    sim_now: float | None = None


class Console:
    replaying = False  # a recorded rollout fills the views (ui/rollouts.py); live drawing waits

    def __init__(self, controls: Controls, *, host="127.0.0.1", port=8080, initial_task="",
                 scene_name="Simulation", model_name="", topdown_settings: Topdown | None = None,
                 render_in_background=False, monitor=None):
        """MONITOR (the [monitor] settings, when on) adds the monitor panel and a Resume button."""
        """With RENDER_IN_BACKGROUND the live view and map are drawn on a UI thread from the newest frame."""
        claim(host, port)
        self.server = open_server(host, port, scene_name, "sim")
        self.scene_view = None
        self.model_name = model_name
        self.overlays = Overlays(topdown_settings if topdown_settings is not None else Topdown())
        self._detail_skill = None
        self.pause_button = self.unpause_button = self.stop_skill_button = None
        if monitor is not None:  # the operator's ways in, first in the panel, weighted by how much they end
            self.run_heading = self.server.gui.add_html(section("Run"))
            self.pause_button = self.server.gui.add_button("Pause", color="gray", icon="player-pause",
                                                           hint="Freeze the simulation and the agent")
            self.unpause_button = self.server.gui.add_button("Resume", color="blue", icon="player-play",
                                                             visible=False, hint="Continue from where it froze")
            self.stop_skill_button = self.server.gui.add_button(
                "Stop skill", color="gray", icon="hand-stop", disabled=True,
                hint="End the running skill; the agent chooses what to do next")
            self.stop_button = self.server.gui.add_button("Stop task", color="red", icon="player-stop",
                                                          hint="End the task and stop the robot")
            self.dismiss_button = self.server.gui.add_button("Dismiss alert", color="gray", visible=False,
                                                             hint="Clear the monitor's alert and its mark on the plan")
            self.note = self.server.gui.add_text("Tell the agent", initial_value="", multiline=True,
                                                 hint="The agent reads it before its next decision")
            self.send_note = self.server.gui.add_button("Send with the next step", color="gray", icon="send")
            self.pause_button.on_click(lambda _: self._set_paused(True))
            self.unpause_button.on_click(lambda _: self._set_paused(False))
            self.stop_skill_button.on_click(lambda _: controls.stop_skill())
            self.dismiss_button.on_click(lambda _: controls.dismiss())
            self.send_note.on_click(lambda _: self._send_note())
        self.server.gui.add_html(section("Live view"))
        self.camera_choice = self.server.gui.add_dropdown("Camera", tuple(CAMERAS), initial_value="Robot view")
        if hasattr(controls, "camera"):  # a console on a harness server tells it which camera to send
            self.camera_choice.on_update(lambda _: controls.camera(self.camera))
        self.ego = self.server.gui.add_image(blank(480, 640), label="Live camera", format="jpeg")
        self.scenario = self.run_scenario = None
        self._presets: dict[str, dict] = {}
        if hasattr(controls, "preset"):  # a harness server offers scenarios: a scene and its instruction
            self.scenario = self.server.gui.add_dropdown("Scenario", (CUSTOM,), initial_value=CUSTOM,
                                                         hint="Fills the instruction; Run scenario loads its scene")
            self.scenario.on_update(lambda _: self._chose_scenario())
        self.task = self.server.gui.add_text("Instruction", initial_value=initial_task, multiline=True)
        self.submit_button = self.server.gui.add_button("Start task", icon="player-play", disabled=True,
                                                        **({"color": "dark"} if monitor is not None else {}))
        if self.scenario is not None:
            self.run_scenario = self.server.gui.add_button("Run scenario", icon="player-play", color="blue",
                                                           disabled=True, visible=False,
                                                           hint="Load the scenario's scene if needed, then start it")
            self.run_scenario.on_click(lambda _: self._run_scenario())
        if monitor is None:
            self.stop_button = self.server.gui.add_button("Stop", color="gray", icon="player-stop")
        self.status = self.server.gui.add_markdown("Loading the simulation. You can edit the instruction.")
        if monitor is not None:
            self.status.visible = False  # the strip over the main view says it
        self.server.gui.add_html(section("Agent thoughts"))
        self.thought = self.server.gui.add_html(thought_bubble("Waiting for the simulation to load.",
                                                              model=self.model_name))
        self._bubble_message = ""
        self.stage = self.server.gui.add_html("")
        self._progress = Progress()
        self.explanation = self.server.gui.add_html("")
        self.monitor_settings, self.monitor_panel, self.resume_button = monitor, None, None
        if monitor is not None:
            # every style once, here: live updates below send plain HTML, so the fonts load once
            self.server.gui.add_html(section("Monitor") + plan_view.TOKENS + plan_view.STYLE + monitor_panel.STYLE
                                     + monitor_panel.BAR_STYLE)
            self.resume_button = self.server.gui.add_button("Resume", icon="player-play", visible=False,
                                                            hint="The monitor asked for a person; continue")
            self.resume_button.on_click(lambda _: controls.resume_task())
            self.monitor_panel = self.server.gui.add_html(monitor_panel.monitor_html(View()))
            # the main view: the behaviour tree over the 3D view, and a bar along its bottom with the task's timeline
            self.strip = self.server.gui.add_html(plan_view.status_strip("Ready"))
            self.plan_view = self.server.gui.add_html(plan_view.plan_view(None))
            self._view, self._decisions = None, 0
            self.timeline = self.server.gui.add_html(monitor_panel.timeline_bar(View()))
        with self.server.gui.add_folder("Details", expand_by_default=False):
            self.click = self.server.gui.add_image(blank(480, 640), label="Decision click", format="jpeg",
                                                   visible=False)
            self.map = self.server.gui.add_image(blank(240, 960), label="Navigation map", visible=False)
            self.topdown = self.server.gui.add_image(blank(240, 960), label=AWAITING, visible=False)
        self.controls, self._starting = controls, Lock()
        self.submit_button.on_click(self._start)
        self.stop_button.on_click(lambda _: controls.stop())
        self._pending, self._latest, self._closing = Condition(), None, False
        self._render_failed = False
        self._renderer = Thread(target=self._render_newest, name="console-render", daemon=True) \
            if render_in_background else None
        if self._renderer is not None:
            self._renderer.start()

    def set_presets(self, presets: list[dict], scene: str = ""):
        """The harness's scenarios (name, scene, task), and the scene it simulates now."""
        if self.scenario is None:
            return
        self._presets = {item["name"]: item for item in presets if isinstance(item, dict) and item.get("name")}
        self._scene = scene
        self.scenario.options = (CUSTOM, *self._presets)
        if self.scenario.value not in self.scenario.options:
            self.scenario.value = CUSTOM
        self._chose_scenario()

    def _chose_scenario(self):
        preset = self._presets.get(self.scenario.value)
        if preset is not None:
            self.task.value = preset["task"]
        if self.run_scenario is not None:
            self.run_scenario.visible = preset is not None
            other = preset is not None and preset.get("scene") != getattr(self, "_scene", "")
            self.run_scenario.label = f"Load {preset['scene']} and run" if other else "Run scenario"

    def _run_scenario(self):
        preset = self._presets.get(self.scenario.value)
        if preset is None or self.run_scenario.disabled:
            return
        self.run_scenario.disabled = self.submit_button.disabled = True
        self.controls.preset(preset["name"], self.task.value)

    def _start(self, _):
        """One task per click: a second click before the session reports back is ignored."""
        with self._starting:
            if self.submit_button.disabled or not self.task.value.strip():
                return
            self.submit_button.disabled = True
        self.controls.submit(self.task.value)

    @property
    def arms(self):
        return self.overlays.arms

    @arms.setter
    def arms(self, arms):
        self.overlays.arms = arms

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
                    print("console: rendering a frame failed; later failures are not printed",
                          file=sys.stderr)
                    traceback.print_exc()

    @property
    def camera(self):
        """The live view's camera: "robot" (the head camera with the plan drawn on it), "observer" or "hand"."""
        return CAMERAS.get(getattr(getattr(self, "camera_choice", None), "value", "Robot view"), "robot")

    def show_camera(self, rgb, label):
        """An observer or hand camera image the session rendered for the live view."""
        if self.camera != "robot" and not getattr(self, "replaying", False):
            self.ego.image = rgb
            self.ego.label = label

    def _set_paused(self, paused):
        (self.controls.pause if paused else self.controls.unpause)()
        self.pause_button.visible, self.unpause_button.visible = not paused, paused

    def _send_note(self):
        text = self.note.value
        if text.strip():
            self.controls.tell(text)
            self.note.value = ""

    def render(self, frame):
        self._refresh_stage(simulation_time=frame.time)
        if self.camera != "robot" or getattr(self, "replaying", False):
            return
        self.ego.image = self.overlays.ego_rgb(frame)
        self.ego.label = "Live camera"
        if self._detail_skill == "navigate":
            self.map.image = self.overlays.map(frame)
            self.map.label = f"Navigation map: {frame.time:.1f} s"

    def clear_selection(self):
        self.overlays.clear()
        self._detail_skill = None
        self.click.visible = False
        self.map.visible = False
        self.topdown.visible = False
        if self.scene_view is not None:
            self.scene_view.clear_selection()

    def clear_view(self):
        """Blank every view for a scene reset."""
        self.clear_selection()
        self.ego.image = np.zeros_like(self.ego.image)
        self.ego.label = "Live view: waiting for fresh observations"
        self.map.image = np.zeros_like(self.map.image)
        self.map.label = "Map: waiting for fresh observations"
        self.topdown.image = np.zeros_like(self.topdown.image)
        self.topdown.label = AWAITING
        self.stage.content = self.explanation.content = ""
        self._progress = Progress()
        self.status.content = "Resetting the scene"
        self.submit_button.disabled = True

    def select(self, decision, frame):
        """Show a decision on its captured frame. The click view stays until the next decision."""
        self.clear_selection()
        self.overlays.select(decision, frame)
        self._detail_skill = decision.skill
        self.map.visible = decision.skill == "navigate"
        self.topdown.visible = decision.skill == "place"
        self._progress = Progress()
        self.stage.content = ""
        self.render(frame)
        if self.topdown.visible:
            self.topdown.image = self.overlays.topdown()
            self.topdown.label = f"Place target: {frame.time:.1f} s"
        selection = self.overlays.selection
        if decision.skill == "pick" and selection is not None:
            self.click.image = selection.overlay(frame, frame.rgb)
            self.click.label = f"Selected segment: {frame.time:.1f} s"
            self.click.visible = True
        if self.scene_view is not None and selection is not None:
            self.scene_view.show_selection(selection.point)

    def skill_event(self, event):
        kind = event["type"]
        if getattr(self, "stop_skill_button", None) is not None and kind in ("skill_started", "skill_finished"):
            self.stop_skill_button.disabled = kind == "skill_finished"
        if kind == "skill_stage":
            now, sim = time.monotonic(), event.get("simulation_time")
            progress = self._progress
            self._progress = replace(
                progress, skill=event["skill"], stages=(*progress.stages, event["stage"]),
                times=(*progress.times, now),
                sim_start=sim if progress.sim_start is None else progress.sim_start,
                sim_now=progress.sim_now if sim is None else max(progress.sim_now or sim, sim))
            self._refresh_stage(now=now)
        elif kind == "skill_prepared":
            self.prepared(event["plan"])
        elif kind == "arm_return_prepared":
            self.overlays.returning(event)
            if self.overlays.latest is not None:
                self.ego.image = self.overlays.ego_rgb(self.overlays.latest)
            if self.scene_view is not None:
                self.scene_view.clear_arm_path()
        elif kind == "skill_finished":
            progress, now = self._progress, time.monotonic()
            self._progress = replace(progress, result=event["result"], finished_at=now,
                                     sim_now=event.get("simulation_time", progress.sim_now))
            self._refresh_stage(now=now)

    def _refresh_stage(self, *, now=None, simulation_time=None):
        """Redraw the stage panel from one snapshot of the progress."""
        progress = self._progress
        if not progress.stages:
            return
        sim_now = progress.sim_now
        if simulation_time is not None and progress.result is None:
            sim_now = max(sim_now or simulation_time, simulation_time)
        now = time.monotonic() if now is None else now
        if progress.finished_at is not None:
            now = progress.finished_at
        durations = [max(0., end - start) for start, end in pairwise(progress.times)]
        durations.append(max(0., now - progress.times[-1]))
        simulated = (None if progress.sim_start is None or sim_now is None else
                     max(0., sim_now - progress.sim_start))
        skill = progress.result.skill if progress.result is not None else progress.skill
        if progress is not self._progress:  # a reset or new decision replaced it meanwhile
            return
        self.stage.content = stage_progress(skill, progress.stages, progress.result,
                                            durations=durations, simulated=simulated)

    def prepared(self, plan):
        """Draw a prepared plan: route and stance, and the palm's reference path, also in 3D."""
        overlays = self.overlays
        self.render(overlays.prepared(plan))
        if self.topdown.visible and isinstance(plan, place.Plan):
            self.topdown.image = overlays.topdown()
        if self.scene_view is None:
            return
        if isinstance(plan, (navigate.Plan, pick.ApproachPlan)):
            self.scene_view.show_route(overlays.marks["route"], overlays.marks["stance"][2])
        if isinstance(plan, (pick.Plan, place.Plan)) and overlays.palm_path is not None:
            self.scene_view.show_arm_path(overlays.palm_path)

    def update(self, status: str, text: str = ""):
        if getattr(self, "replaying", False):  # the run's state is kept for when the replay ends
            self._state, self._note = status, text
            return
        self.status.content = STATUS.get(status, status)
        if status == "Thinking":
            self.ego.label = "Last camera frame (agent thinking)"
        elif status == "Executing":
            self.ego.label = "Last camera frame"
        elif status == "Ready":
            self.ego.label = "Live camera"
        if status == "Executing":
            self._bubble_message = text
        elif status == "Ready":
            self._bubble_message = ""
        if status == "Ready":
            self.stage.content = ""
            self.clear_selection()
        self.thought.content = thought_bubble(self._bubble_message, thinking=status == "Thinking",
                                              model=self.model_name)
        self.explanation.content = "" if status == "Executing" else note(text, done=status == "model_done")
        self.submit_button.disabled = status in ("Thinking", "Executing", "Reset required", "Paused", "Loading")
        if getattr(self, "run_scenario", None) is not None:
            self.run_scenario.disabled = self.submit_button.disabled
        if getattr(self, "resume_button", None) is not None:  # only a console with the monitor has it
            self.resume_button.visible = status == "Paused" and "monitor" in text.lower()
            running = status in ("Thinking", "Executing", "Paused")
            self.task.visible = self.submit_button.visible = not running  # the run's controls lead while it runs
            if getattr(self, "scenario", None) is not None:
                self.scenario.visible = not running
                self.run_scenario.visible = not running and self.scenario.value in self._presets
            if status == "Thinking":
                self._decisions += 1
            self._state, self._note = status, text
            self.strip.content = plan_view.status_strip(status, self._view, self._decisions, text)

    def monitor_update(self, view):
        """Redraw the monitor panel from VIEW (called from the monitor's threads)."""
        if self.monitor_panel is not None:
            self._view = view
            if not self.replaying:
                self.draw_view(view, getattr(self, "_state", "Ready"), self._decisions, getattr(self, "_note", ""))

    def draw_view(self, view, state, decisions, note=""):
        """The strip, plan, timeline and monitor panel from VIEW (live, or a replay's at its time)."""
        s = self.monitor_settings
        self.monitor_panel.content = monitor_panel.monitor_html(view, s.tau, s.tau_unsafe)
        self.dismiss_button.visible = view.alert is not None and not self.replaying
        self.timeline.content = monitor_panel.timeline_bar(view, s.tau, s.ema)
        self.plan_view.content = plan_view.plan_view(view.plan, view.phase, view.task)
        self.strip.content = plan_view.status_strip(state, view, decisions, note)

    def end_replay(self):
        """Back to the live run: its newest view and state."""
        self.replaying = False
        self.ego.label = "Live camera"
        self.draw_view(self._view or View(), getattr(self, "_state", "Ready"), self._decisions,
                       getattr(self, "_note", ""))
        self.thought.content = thought_bubble(self._bubble_message, model=self.model_name)

    def close(self):
        if self._renderer is not None:
            with self._pending:
                self._closing = True
                self._pending.notify()
            self._renderer.join()
        self.server.stop()
