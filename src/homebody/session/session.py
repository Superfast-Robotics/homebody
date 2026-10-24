"""One physics owner, one decision, one skill result; operator tasks can continue."""
import json
import math
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from homebody import harness
from homebody.monitor.targets import Phase, request_json
from homebody.primitives.actions import PhysicsInvalid
from homebody.session import monitored
from homebody.session.coder import CALL, Coder
from homebody.session.planned import OWN as PLAN_DECISIONS
from homebody.session.planned import PlanRunner
from homebody.session.recovery import Recovery, request, stuck_key
from homebody.session.trace import Trace
from homebody.skills.contract import (
    COMMON_FAILURES,
    MONITOR_STOPPED,
    OPERATOR_STOPPED,
    Context,
    Runner,
)
from homebody.skills.registry import Prompts, registry
from homebody.vlm.observations import annotations, model_observation, rounded
from homebody.vlm.provider import Cancelled, ProviderBusy, ProviderError

# what the operator says to go on with the plan after a pause
CONTINUE = re.compile(r"(continue|resume|go on|carry on|proceed)( (with )?(the )?(plan|task))?[.!]?")


def prompted(result: dict) -> dict:
    """A skill result without its traceback detail, as the prompted history keeps it."""
    details = {key: value for key, value in result.get("details", {}).items() if key != "traceback"}
    return {**result, "details": details}


def hand_status(side: str, carry: dict | None) -> str:
    """The prompt's account of one hand, from the carry record its last pick left."""
    if carry is None:
        return f"the {side} hand is free"
    if carry["state"] == "verified":
        return (f"the {side} hand HOLDS the object from its last pick, not placed yet; "
                "only a place by that hand opens it")
    return f"the {side} hand's grasp is uncertain; repeat pick on its target to recheck it"


REAL_TIME_SLACK = 0.5  # s behind the wall clock before the pace restarts
HISTORY_STEPS = 24  # recent steps shown each turn
DETAILED_STEPS = 4  # the newest of them keep their results' details; older ones keep code and message


def history_line(entry, full):
    """One recent step as the prompt shows it; an older step's result drops its details."""
    line = {key: entry[key] for key in ("operator_task", "base_pose", "decision", "result") if key in entry}
    if not full and "result" in line:
        line["result"] = {key: value for key, value in line["result"].items() if key != "details"}
    return line



PLACEHOLDER = re.compile(r"\{(\w+(?:\.\w+)*)\}")


def prompt_line(text, settings):
    """A skill's PROMPT with its {group.value} settings filled in. A coder-written prompt may also hold literal braces
    (a JSON example): then only the placeholders that name a setting are filled and the rest stays as written."""
    try:
        return text.format_map(settings)
    except (KeyError, ValueError, IndexError, AttributeError):
        def fill(match):
            value = settings.get(match.group(1).split(".")[0], None)
            for name in match.group(1).split(".")[1:]:
                value = getattr(value, name, None)
            return match.group(0) if value is None else str(value)
        return PLACEHOLDER.sub(fill, text)

class ObservedBackend:
    """Expose only the declared observation and action methods to skills. With `real_time` the physics is paced to the wall clock."""
    def __init__(self, backend, publish: Callable, publish_period: float, scene_update=None,
                 real_time=False, capture: Callable | None = None, paused: Callable[[], bool] = lambda: False):
        self.__backend = backend
        self._publish = publish
        self._capture = capture
        self._paused = paused
        self._publish_period = publish_period
        self._scene_update = scene_update
        self._last = float("-inf")
        self._real_time, self._anchor = real_time, None

    @property
    def epoch(self):
        return self.__backend.epoch

    @property
    def time(self):
        return self.__backend.time

    def update_scene(self):
        if self._scene_update is not None:
            self._scene_update()

    def snapshot(self):
        frame = self.__backend.snapshot()
        self.update_scene()
        self._publish(frame)
        self._last = frame.time
        return frame

    def measure(self):
        return self.__backend.measure()

    def wrist_view(self, side):
        return self.__backend.wrist_view(side)

    def wrist_views(self):
        """The wrist cameras' RGB, for skills whose policy takes them (not part of Robot)."""
        return self.__backend.wrist_views()

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.__backend.base_velocity(epoch, forward, lateral, yaw_rate)

    def arm_target(self, epoch, side, joints):
        self.__backend.arm_target(epoch, side, joints)

    def grip(self, epoch, side, closure):
        self.__backend.grip(epoch, side, closure)

    def stop(self, epoch):
        self.__backend.stop(epoch)

    def advance(self, epoch, seconds):
        if self._paused():  # the operator froze the world: no physics until Resume
            while self._paused():
                time.sleep(.05)
            self._anchor = None
        try:
            self.__backend.advance(epoch, seconds)
        finally:
            self.update_scene()
        if self._real_time:
            self._pace(seconds)
        if self._capture is not None:
            self._capture()
        if self.time - self._last >= self._publish_period or self.time < self._last:
            self.snapshot()


    def _pace(self, seconds):
        """Hold the step of SECONDS to the wall clock. Behind by more than REAL_TIME_SLACK, or ahead by more than the
        step itself could explain (the simulation's clock moved on its own, as a free-running server's does), the
        pace restarts from now: a step never waits longer than its own length."""
        wall, sim = time.monotonic(), self.__backend.time
        ahead = None if self._anchor is None else (sim - self._anchor[1]) - (wall - self._anchor[0])
        if ahead is None or ahead < -REAL_TIME_SLACK or ahead > seconds + REAL_TIME_SLACK:
            self._anchor = (wall, sim)
            return
        if ahead > 0:
            time.sleep(min(ahead, seconds))


class Session:
    def __init__(self, backend, provider, settings, recorder, controls, *, arms, jaw_samples,
                 ui=None, semantic_context=None, observer_frame=None, shoulder_frame=None,
                 scene_update=None, real_time=False, monitor=None):
        self.backend, self.provider = backend, provider
        self.monitor = monitor  # the live monitor (homebody/monitor.Monitor), or None
        self.settings, self.recorder = settings, recorder
        if hasattr(provider, "cache_key"):  # one prompt-cache key per session (vlm.api.LiteLlm)
            provider.cache_key = f"homebody-{Path(getattr(recorder, 'path', 'session')).name}"
        self.controls, self.ui = controls, ui
        self.semantic_context = semantic_context or {}
        self.observer_frame, self.shoulder_frame, self.view_side = observer_frame, shoulder_frame, "right"
        self._running_skill = self._stop_taken = None  # the skill running now, and who stopped it (interruption)
        self.skill_calls, self._call = 0, None  # this task's skill calls; the running one's number (Phase.call)
        self._observer_epoch = self._frame_epoch = None
        self._observer_time = self._frame_time = self._view_time = float("-inf")
        self.observed = ObservedBackend(backend, self.publish, settings.recording.publish_period,
                                        scene_update, real_time,
                                        capture=self.capture if monitor is not None else None,
                                        paused=lambda: self.frozen() and not self.commanded())
        self.context = Context(observations=self.observed, actions=self.observed, arms=arms,
                               settings=settings, jaw_samples=jaw_samples,
                               cancelled=controls.cancel.is_set, emit=self.skill_event,
                               interrupted=self.interruption)
        self.runner = Runner(self.context)
        self.coder = Coder(settings.coder, registry(settings.robot.hands))
        # [coder] calls = 0 turns the coding assistant off: not offered, never automatic
        self.coder_on = settings.coder.calls > 0
        self.auto = self.coder_on and settings.coder.auto
        self.recovery = Recovery(settings.coder.tries, settings.coder.rescues)
        self.planner = None
        self.trace = Trace(backend, lambda: self.context.carried)
        self.history = []
        self.monitor_note = self.operator_note = ""
        self.hold_text = ""  # what the paused robot waits for (hold_for_operator)
        self.paused_seconds = 0.0
        self.status = "Ready"
        self._physics_fault = None

    @property
    def requires_reset(self):
        return self._physics_fault is not None

    def physics_failed(self, error):
        fault = self.backend.epoch, str(error)
        if fault != self._physics_fault:
            self._physics_fault = fault
            self.recorder.event("physics_invalid", epoch=fault[0], reason=fault[1],
                                simulation_seconds=self.backend.time)
            self.backend.stop(self.backend.epoch)
        self.update("Reset required", f"{fault[1]} Restart the console to continue.")

    def update(self, status, text=""):
        self.status = status
        if self.ui is not None:
            self.ui.update(status, text)

    def publish(self, frame):
        """Show every frame; record it and the operator views only while a task records."""
        if self.ui is not None:
            self.ui.show(frame)
            self.show_camera()
        if not self.recorder.recording:
            return
        r = self.settings.recording
        if frame.epoch != self._frame_epoch or frame.time - self._frame_time >= r.frame_period:
            self.recorder.frame(frame)
            self._frame_epoch, self._frame_time = frame.epoch, frame.time
        if self.observer_frame is not None and (frame.epoch != self._observer_epoch or
                                               frame.time - self._observer_time >= r.room_period):
            self.recorder.observer(self.observer_frame(), epoch=frame.epoch,
                                   simulation_time=frame.time, fps=1 / r.room_period)
            self._observer_epoch, self._observer_time = frame.epoch, frame.time
        if self.shoulder_frame is not None and frame.time - self._view_time >= r.shoulder_period:
            self.recorder.observer(self.shoulder_frame(self.view_side), epoch=frame.epoch,
                                   simulation_time=frame.time, stream="shoulder",
                                   fps=1 / r.shoulder_period)
            self._view_time = frame.time

    def show_camera(self):
        """Render the console's chosen observer or hand camera (here, where the renderer lives)."""
        camera = getattr(self.ui, "camera", "robot")
        if camera == "observer" and hasattr(self.backend, "overview_frame"):
            self.ui.show_camera(self.backend.overview_frame(), "Observer view")
        elif camera == "hand" and hasattr(self.backend, "wrist_view"):
            self.ui.show_camera(self.backend.wrist_view(self.view_side).rgb, f"Hand camera: {self.view_side} hand")

    def skill_event(self, event):
        if event["type"] == "observation":
            return
        if self._call is not None and "call" not in event:
            event = {**event, "call": self._call}  # the attempt this event belongs to
        self.recorder.event("skill", simulation_time=self.backend.time, event=event)
        try:
            self.trace.on_event(event, self.view_side)
        except Exception as error:  # noqa: BLE001 -- a trace keyframe never fails the skill it records
            self.recorder.event("trace_error", simulation_time=self.backend.time, error=f"{type(error).__name__}: {error}")
        if self.ui is not None:
            self.ui.skill_event({**event, "simulation_time": self.backend.time})
        if self.monitor is not None and event["type"] == "skill_stage":
            self.monitor.set_phase(Phase("tool_call", event.get("skill"), event.get("stage")))

    def capture(self):
        """Hand the monitor a head (and, with monitor.wrist, the active wrist) image when one is due."""
        now = self.backend.time
        if self.monitor is None or not self.monitor.capture_due(now) or not hasattr(self.backend, "ego_image"):
            return
        self.monitor.capture(self.backend.ego_image(), now)
        if self.settings.monitor.wrist:
            self.monitor.capture(self.backend.wrist_view(self.view_side).rgb, now, f"wrist-{self.view_side}")

    def heartbeat(self):
        """Step valid physics while idle/thinking; invalid generations wait for reset."""
        if self.requires_reset or self.frozen():
            time.sleep(.02)
            return
        start = time.monotonic()
        try:
            self.backend.base_velocity(self.backend.epoch, 0.0, 0.0, 0.0)
            if self.status == "Thinking":  # step without publishing frames
                self.backend.advance(self.backend.epoch, 0.02)
                self.observed.update_scene()
                if self.monitor is not None:
                    self.capture()
            else:
                self.observed.advance(self.backend.epoch, 0.02)
        except PhysicsInvalid as error:
            self.physics_failed(error)
            raise
        time.sleep(max(0, 0.02 - (time.monotonic() - start)))

    def prompt(self, task, frame):
        """The turn's prompt. Its opening is the same every turn of a task so the model's prompt cache can serve it
        (instructions, the built-in skills, the coder, scene memory, task, plan instructions); what changes follows
        (skills the coder wrote, hands, coder calls left, recent steps, observation)."""
        settings = vars(self.settings)

        def line(skill):
            return f"{prompt_line(skill.PROMPT, settings)} Refusals: {', '.join(dict.fromkeys(skill.FAILURES))}."
        skills = "\n".join(line(skill) for skill in self.coder.registry.values())
        if self.coder_on:
            skills += (f"\n{CALL}: skill, goal, tried, success, constraints. A coding assistant changes the skill and "
                       f"runs it on the robot (see above); the calls left this task are under Status below.")
        if self.auto:
            tries, rescues = self.settings.coder.tries, self.settings.coder.rescues
            skills += (f" In this session a failed skill call goes to the coding assistant automatically, at most "
                       f"{rescues} time{'s' if rescues != 1 else ''} while that skill keeps failing on a target; its "
                       f"result follows the failure in the recent steps. Once a skill has failed on a target you may "
                       f"call it there at most {tries} more times until a call of it succeeds; use the results to "
                       "decide whether another try can help, else choose something else or done.")
        coded = "\n".join(f"{name} (version {self.coder.skills[name]['version']}, written by the coding assistant "
                          f"this task; it replaces any built-in skill of that name): " + line(self.coder.module(name))
                          for name in self.coder.names())
        left = self.settings.coder.calls - self.coder.calls
        hands = "; ".join(hand_status(side, self.context.carried.get(side)) for side in ("left", "right"))
        recent = self.history[-HISTORY_STEPS:]
        steps = "\n".join(json.dumps(rounded(history_line(entry, full=index >= len(recent) - DETAILED_STEPS)))
                          for index, entry in enumerate(recent))
        return "\n\n".join(part for part in (
            harness.agent_instructions(settings),
            f"Skills:\n{skills}\nAny skill may also refuse with: {', '.join(self.refusals())}.",
            "Initial-scene memory:\n" + annotations(self.semantic_context),
            "Current task:\n" + task,
            harness.plan_instructions(settings) if self.planner is not None else "",
            "Skills the coding assistant wrote this task:\n" + coded if coded else "",
            f"Status: hands now: {hands}." + (f" {CALL}: {left} call{'s' if left != 1 else ''} left this task."
                                              if self.coder_on else ""),
            "Recent steps, oldest first:\n" + (steps or "none"),
            "Current observation:\n" + json.dumps(model_observation(frame))) if part)

    def frozen(self):
        """Whether the operator froze the simulation and the agent (the console's Pause; never while stopping)."""
        paused = getattr(self.controls, "paused", None)
        return bool(paused is not None and paused.is_set() and not self.controls.cancel.is_set())

    def operator_pending(self):
        """Whether the operator paused, or gave a note the agent has not read yet (a done waits for either)."""
        has_notes = getattr(self.controls, "has_notes", None)
        return self.frozen() or bool(has_notes is not None and has_notes())

    def commanded(self):
        """Whether the operator gave a command while the world is paused (the session runs it as the plan's next goal:
        hold_for_operator)."""
        has_notes = getattr(self.controls, "has_notes", None)
        return bool(self.frozen() and has_notes is not None and has_notes())

    def hold_for_operator(self):
        """While the operator's pause lasts the robot waits here and the monitor goes on watching. A command given
        meanwhile becomes the plan's next goal and runs; when that goal closes the robot waits here again
        (after_operator_goal). "continue" (or Resume) goes on with the rest of the plan."""
        while self.frozen():
            if self.controls.cancel.is_set():
                raise Cancelled("Task stopped by operator while paused")
            if self.operator_commands():
                return
            self.update("Paused", self.hold_text or "Paused by the operator. Give a command, or continue.")
            self.paused_tick()
        self.hold_text = ""

    def operator_commands(self):
        """The notes given during the pause: "continue" resumes the plan; any other text becomes the plan's next goal
        (without a plan, the agent's next prompt carries it). Either ends the pause; returns whether one came."""
        notes = self.controls.take_notes() if self.commanded() else []
        if not notes:
            return False
        lines = []
        for note in notes:
            if CONTINUE.fullmatch(note.strip().lower()):
                self.recorder.event("operator", type="continue", text=note, simulation_time=self.backend.time)
                continue
            goal_id = self.planner.insert_operator(note) if self.planner is not None else None
            self.recorder.event("operator", type="command", text=note, goal_id=goal_id, simulation_time=self.backend.time)
            if self.monitor is not None:
                self.monitor.mark("command", note, self.backend.time)
            if goal_id is None:
                lines.append(note)
        if lines:
            self.operator_note = "Operator (the person supervising you; follow it): " + " ".join(lines)
        self.hold_text = ""
        self.controls.paused.clear()
        return True

    def after_operator_goal(self):
        """When the operator's command (an operator goal) closes and no other is waiting, the robot pauses and waits
        for the next command or continue."""
        closed = self.planner.take_operator_closed() if self.planner is not None else None
        if closed is None or self.planner.operator_pending() or not hasattr(self.controls, "pause"):
            return
        status = self.planner.plan.status.get(closed)
        self.hold_text = (f"Your command ({closed}) {'is done' if status == 'done' else 'failed'}. Waiting: give "
                          "another command, or continue with the plan.")
        self.recorder.event("operator", type="hold", goal_id=closed, status=status, simulation_time=self.backend.time)
        self.controls.pause()

    def interruption(self):
        """(code, reason) when the operator or the monitor stops the running skill, else None. The skill's context
        asks until it gets one, then stops asking: that moment the executor took the request (an intervention
        event, "accepted"); how the skill then ended follows in `intervention_ended`."""
        stop = getattr(self.controls, "skill_stop", None)
        taken = None
        if stop is not None and stop.is_set():
            taken = "operator", OPERATOR_STOPPED, "The operator stopped the skill"
        elif self.commanded():  # a command while paused: the skill ends so the command can run as the next goal
            taken = "operator", OPERATOR_STOPPED, "The operator gave a command while paused"
        elif self.monitor is not None and (reason := self.monitor.stop_requested()) is not None:
            taken = "monitor", MONITOR_STOPPED, f"The monitor stopped the skill: {reason}"
        if taken is None:
            return None
        self._stop_taken = taken[0]
        self.skill_event({"type": "intervention", "stage": "accepted", "initiator": taken[0], "action": "stop_skill",
                          "skill": self._running_skill, "reason": taken[2],
                          "note": "The skill executor took the request at its next cancellation check"})
        return taken[1], taken[2]

    def intervention_ended(self, skill, code):
        """How a skill that took a stop request ended: the executor's confirmation, or its failure to stop."""
        initiator, self._stop_taken = self._stop_taken, None
        if initiator is None:
            return
        expected = OPERATOR_STOPPED if initiator == "operator" else MONITOR_STOPPED
        stopped = code == expected
        self.skill_event({"type": "intervention", "stage": "confirmed" if stopped else "failed", "initiator": initiator,
                          "action": "stop_skill", "skill": skill, "code": code,
                          "note": f"The skill ended with {code}" + ("" if stopped else f", not {expected}")})

    def refusals(self):
        return (*COMMON_FAILURES, MONITOR_STOPPED, OPERATOR_STOPPED) if self.monitor is not None else COMMON_FAILURES

    def operator_input(self):
        """The operator's notes since the last turn, as the prompt's operator line; a dismissal clears the alert and
        the plan's mark for it."""
        controls, lines = self.controls, []
        if getattr(controls, "dismissed", None) is not None and controls.dismissed.is_set():
            controls.dismissed.clear()
            goal = self.monitor.dismiss() if self.monitor is not None else None
            if goal is not None and self.planner is not None and self.planner.plan is not None:
                self.planner.plan.flags.pop(goal, None)
            self.recorder.event("operator", type="dismiss", goal_id=goal, simulation_time=self.backend.time)
        for note in controls.take_notes() if hasattr(controls, "take_notes") else ():
            lines.append(note)
            self.recorder.event("operator", type="note", text=note, simulation_time=self.backend.time)
            if self.monitor is not None:
                self.monitor.mark("note", note, self.backend.time)
        return ("Operator (the person supervising you; follow it): " + " ".join(lines)) if lines else ""

    def skill_lines(self):
        settings = vars(self.settings)
        return tuple(prompt_line(skill.PROMPT, settings) for skill in self.coder.registry.values())

    def watch(self, task, images):
        """Apply the monitor's verdicts since the last turn and tell it the goal now; the prompt's monitor line."""
        if self.monitor is None:
            return ""
        verdicts = self.monitor.take_verdicts()
        note = monitored.apply(verdicts, self.planner)
        if self.planner is not None:
            self.planner.phase()  # the active goal after any reopen
        human = next((v for v in verdicts if v.label == "human_needed"), None)
        if human is not None and self.ui is not None:
            self.await_operator(human.text())
            self.monitor.operator_resumed()
        request = self.goal_request(task, images)
        self.monitor.goal(request, request.hands, plan=monitored.plan_snapshot(self.planner))
        self.monitor.set_phase(Phase("planning"))
        return note

    def hands(self) -> str:
        return "; ".join(hand_status(side, self.context.carried.get(side)) for side in ("left", "right"))

    def goal_request(self, task, images, skill=None, arguments=None):
        """What the monitor reads about the goal now: at a turn (IMAGES: its head and top-down views, with the
        monitor's recent findings) or as SKILL starts with ARGUMENTS."""
        monitor = self.monitor
        return monitored.request(self.planner, task, self.skill_lines(), self.hands(),
                                 (images[0], images[-1]) if images else (),
                                 stages=monitor.known_stages() if monitor is not None else {},
                                 findings=monitor.recent_findings() if monitor is not None and skill is None else (),
                                 skill=skill, arguments=arguments, carried=set(self.context.carried))

    def record_turn(self, step, task, images):
        """The turn as a replay and an offline monitor read it: the plan, the hands and the goal request."""
        request = self.goal_request(task, images)
        self.recorder.event("turn", step=step, simulation_time=self.backend.time,
                            plan=monitored.plan_snapshot(self.planner), hands=request.hands,
                            carried=sorted(self.context.carried),
                            request=request_json(request, self.recorder.path))

    def await_operator(self, text):
        """The monitor saw something only a person can fix: hold until the operator presses Resume (or Stop)."""
        self.update("Paused", f"The monitor asks for a person: {text}. Press Resume to continue or Stop.")
        self.recorder.event("monitor", type="operator_pause", simulation_time=self.backend.time, reason=text)
        tick = self.paused_tick if self.settings.agent.pause_while_thinking else self.heartbeat
        self.controls.resumed.clear()
        while not self.controls.resumed.is_set():
            if self.controls.cancel.is_set():
                raise Cancelled("Task stopped by operator during a monitor pause")
            tick()
        self.recorder.event("monitor", type="operator_resume", simulation_time=self.backend.time)

    def paused_tick(self):
        """Wait one heartbeat period without advancing the world (agent.pause_while_thinking)."""
        time.sleep(.02)

    def decide(self, prompt, images, names, attempt):
        """One decision. A busy service is asked again under attempt-retryN. With agent.pause_while_thinking the
        world holds still throughout, retries included, and the held wall time is recorded."""
        agent = self.settings.agent
        tick = self.paused_tick if agent.pause_while_thinking else self.heartbeat
        started, sim = time.monotonic(), self.backend.time
        try:
            for retry in range(agent.busy_retries + 1):
                try:
                    return self.provider.generate(prompt, images=images, names=names,
                                                  attempt=attempt if not retry else attempt.with_name(f"{attempt.name}-retry{retry}"),
                                                  tick=tick, cancelled=self.controls.cancel.is_set)
                except ProviderBusy as busy:
                    if retry == agent.busy_retries:
                        raise
                    self.update("Thinking", f"{busy} Asking again in {agent.busy_wait * (retry + 1):g} s.")
                    until = time.monotonic() + agent.busy_wait * (retry + 1)
                    while time.monotonic() < until and not self.controls.cancel.is_set():
                        tick()
                    if self.controls.cancel.is_set():
                        raise Cancelled("Operator cancelled while waiting to ask again")
            raise AssertionError("unreachable")
        finally:
            if agent.pause_while_thinking:
                held = time.monotonic() - started
                self.paused_seconds += held
                self.recorder.event("physics_paused", reason="agent_decision", decision=attempt.name,
                                    wall_seconds=held, simulation_time=self.backend.time,
                                    simulation_advanced=self.backend.time - sim)

    def run_task(self, task: str) -> dict:
        paused = getattr(self.controls, "paused", None)
        if paused is not None:  # a pause belongs to the task it held (e.g. a console that went away): not this one
            paused.clear()
        task_dir = self.recorder.start_task(task, self.backend.epoch, skills=list(self.skill_lines()))
        self.skill_calls, self._call = 0, None  # skill calls of this task, numbered as the monitor numbers them
        started_wall, started_sim = time.monotonic(), self.backend.time
        outcome = {"status": "error", "steps": 0, "model_claim": None}
        self.paused_seconds = 0.0
        self.history.append({"operator_task": task})
        self.coder.reset(task_dir)  # coded skills live for one task
        self.recovery.reset()
        self.planner = PlanRunner(self.settings.plan.replans) if self.settings.plan.enabled else None
        self.trace.reset(task_dir if self.settings.recording.keyframes else None)
        self.monitor_note = ""
        if self.monitor is not None:
            self.monitor.start_task(task, task_dir)
        try:
            if self.requires_reset:
                raise PhysicsInvalid(self._physics_fault[1])
            self.backend.set_evaluation_sink(self.recorder.physics)
            for step in range(self.settings.agent.max_steps):
                names = tuple(dict.fromkeys((*Prompts(self.coder.registry), *self.coder.names(),
                                             *((CALL,) if self.coder_on else ()), "done")))
                # the response schema lists every decision of the task, the same each turn (a stable prompt the
                # model's cache can serve); the plan phase's choices are checked here instead
                offered = names + (PLAN_DECISIONS if self.planner is not None else ())
                if self.controls.cancel.is_set():
                    raise Cancelled("Task stopped by operator")
                frame = self.observed.snapshot()
                images = self.recorder.agent_images(frame)
                # the monitor's verdicts may reopen a goal, so they go on the plan before this turn's choices
                self.monitor_note = self.watch(task, images) or self.monitor_note
                self.hold_for_operator()
                self.operator_note = self.operator_input() or self.operator_note
                if self.planner is not None:
                    names = self.planner.names(names)
                self.record_turn(step, task, images)
                prompt = self.prompt(task, frame)
                if self.planner is not None and (note := self.planner.note()):
                    prompt += "\n\n" + note
                if self.monitor_note:
                    prompt += "\n\n" + self.monitor_note
                if self.operator_note:
                    prompt += "\n\n" + self.operator_note
                self.update("Thinking")
                if self.monitor is not None:
                    self.monitor.decision(self.backend.time)
                decision = self.decide(prompt, images, offered, task_dir / "decisions" / f"step-{step:03d}")
                if self.controls.cancel.is_set() or frame.epoch != self.backend.epoch:
                    raise Cancelled("Decision invalidated before execution")
                if decision.skill not in offered:
                    raise ProviderError(f"Unsupported skill: {decision.skill}")
                if decision.skill not in names:  # a decision of another plan phase: refused, the agent chooses again
                    result = {"skill": decision.skill, "code": "INVALID_ARGUMENT",
                              "message": f"{decision.skill} is not available now; choose one of: {', '.join(names)}"}
                    outcome["steps"] = step + 1
                    self.history.append({"base_pose": frame.base_pose.tolist(), "decision": asdict(decision),
                                         "epoch": frame.epoch, "frame_sequence": frame.sequence, "result": result})
                    self.recorder.event("result", step=step, simulation_time=self.backend.time, result=result)
                    continue
                self.recorder.event("decision", step=step, epoch=frame.epoch, sequence=frame.sequence,
                                    simulation_time=self.backend.time, decision=decision)
                outcome["steps"] = step + 1
                self.update("Executing", decision.text)
                if self.ui is not None:
                    self.ui.select(decision, frame)
                if decision.skill == "done":
                    self.settle()  # the robot stands still before the final sample; the operator may pause meanwhile
                if decision.skill == "done" and self.operator_pending():
                    # the operator paused or spoke while the agent decided or the robot settled: the task stays open
                    result = {"skill": "done", "code": "OPERATOR_PENDING",
                              "message": "The operator paused or gave a command before the task closed; read it first"}
                    self.history.append({"base_pose": frame.base_pose.tolist(), "decision": asdict(decision),
                                         "epoch": frame.epoch, "frame_sequence": frame.sequence, "result": result})
                    self.recorder.event("result", step=step, simulation_time=self.backend.time, result=result)
                    continue
                if decision.skill == "done":
                    summary = decision.arguments.get("summary", decision.text)
                    if not isinstance(summary, str):
                        raise ProviderError("done.summary must be text")
                    outcome.update(status="model_done", model_claim=summary)
                    if self.monitor is not None and (late := self.monitor.take_verdicts()):
                        outcome["monitor_unapplied"] = [asdict(verdict) for verdict in late]
                    break
                if self.planner is not None and decision.skill in PLAN_DECISIONS:
                    result = self.planner.handle(decision, self.context.carried)
                    self.after_operator_goal()
                    self.history.append({"base_pose": frame.base_pose.tolist(), "decision": asdict(decision),
                                         "epoch": frame.epoch, "frame_sequence": frame.sequence, "result": result})
                    self.recorder.event("result", step=step, simulation_time=self.backend.time, result=result)
                    continue
                if decision.arguments.get("hand") in ("left", "right"):
                    self.view_side = decision.arguments["hand"]
                self.monitor_note = self.operator_note = ""  # the agent has read them
                result = self.act(decision, frame, step)
                if result["code"] == "PHYSICS_INVALID":
                    raise PhysicsInvalid(result["message"])
                if self.auto and self.recovery.wants_coder(stuck_key(decision.skill, decision.arguments),
                                                                          result):
                    result = self.rescue(task, decision, result, step)
                    if result["code"] == "PHYSICS_INVALID":
                        raise PhysicsInvalid(result["message"])
                if self.planner is not None:
                    self.planner.observe(result, self.context.carried)
                    self.after_operator_goal()
            else:
                outcome["status"] = "step_limit"
        except PhysicsInvalid as exc:
            self.physics_failed(exc)
            outcome.update(status="physics_invalid", reason=str(exc), requires_reset=True)
        except Cancelled as exc:
            outcome.update(status="cancelled", reason=str(exc))
        except ProviderError as exc:
            outcome.update(status="provider_error", reason=str(exc))
        except KeyboardInterrupt:
            outcome.update(status="interrupted", reason="KeyboardInterrupt")
            raise
        except Exception as exc:
            outcome.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc())
            raise
        finally:
            cleanup_errors = []
            if self.monitor is not None:
                try:
                    outcome["monitor"] = self.monitor.end_task()
                except Exception as exc:  # noqa: BLE001 -- Cleanup must retain terminal evidence.
                    cleanup_errors.append(f"monitor: {type(exc).__name__}: {exc}")
            try:
                self._halt(outcome["status"])
            except Exception as exc:  # noqa: BLE001 -- Cleanup must retain terminal evidence.
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            try:
                self.recorder.physics(self.backend.evaluation_state())
            except Exception as exc:  # noqa: BLE001 -- Preserve failure if final sensors are lost.
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            self.backend.set_evaluation_sink(None)
            if cleanup_errors:
                outcome.update(status="error", status_before_cleanup=outcome["status"],
                               cleanup_error="; ".join(cleanup_errors))
            outcome.update(wall_seconds=time.monotonic() - started_wall,
                           simulation_seconds=self.backend.time - started_sim,
                           pause_while_thinking=self.settings.agent.pause_while_thinking,
                           physics_paused_seconds=self.paused_seconds)
            self.recorder.end_task(outcome)
            if not self.requires_reset:
                self.update(outcome["status"], outcome.get("reason") or outcome.get("model_claim") or "")
        return outcome

    def act(self, decision, frame, step, rescued=None):
        """Run DECISION (a skill, a coded skill or a coder call) from FRAME; record it and return its result. RESCUED
        is the stuck point an automatic coder call is for."""
        carried_before = {side: dict(carry) for side, carry in self.context.carried.items()}
        key = stuck_key(decision.skill, decision.arguments)
        refused = self.recovery.admit(key) if self.auto and decision.skill != CALL else None
        if refused is not None:
            result = refused
        elif decision.skill == CALL:
            result = self.coder.call(decision.arguments, self)
            if self.auto and result["code"] not in ("INVALID_ARGUMENT", "CODER_LIMIT", "CODER_NOT_YET"):
                self.recovery.coder_done(decision.arguments.get("skill"), result["code"], rescued)
        else:
            self.monitor_skill(decision)
            if getattr(self.controls, "skill_stop", None) is not None:
                self.controls.skill_stop.clear()  # a Stop skill pressed between skills is not for this one
            code, self._running_skill, self._stop_taken = None, decision.skill, None
            try:
                if decision.skill in self.coder.skills:
                    result = self.runner.run(decision.skill, decision.arguments, frame,
                                             skill=self.coder.module(decision.skill)).as_dict()
                else:
                    result = self.runner.run(decision.skill, decision.arguments, frame).as_dict()
                code = result["code"]
            finally:
                self.intervention_ended(decision.skill, code or "ERROR")
                self._running_skill, self._call = None, None
                if code == OPERATOR_STOPPED:
                    self.controls.skill_stop.clear()
                    self.recorder.event("operator", type="stop_skill", skill=decision.skill,
                                        simulation_time=self.backend.time)
                    if self.monitor is not None:
                        self.monitor.mark("stop_skill", f"stopped {decision.skill}", self.backend.time)
                if self.monitor is not None:
                    self.monitor.skill_finished(code, self.backend.time)
        if decision.skill != CALL and refused is None:
            self.recovery.settle(key, result["code"])
        entry = {"base_pose": frame.base_pose.tolist(), "decision": asdict(decision), "epoch": frame.epoch,
                 "frame_sequence": frame.sequence, "result": prompted(result)}
        if decision.skill != CALL and refused is None and result["code"] != "PHYSICS_INVALID":  # for the coder
            entry["what_changed"] = harness.change_report(frame, self.observed.snapshot(), carried_before,
                                                          self.context.carried)
        self.history.append(entry)
        self.recorder.event("result", step=step, simulation_time=self.backend.time, result=result)
        return result

    def monitor_skill(self, decision):
        """Tell the monitor a skill starts in the current goal. The task's skill calls are numbered here as the
        monitor numbers them (Monitor.skill_started, MonitorLink.skill_started), so its readings (Phase.call) and the
        executor's events (their "call") name the same attempt."""
        self.skill_calls += 1
        self._call = self.skill_calls
        task = next((entry["operator_task"] for entry in reversed(self.history) if "operator_task" in entry), "")
        request = self.goal_request(task, (), skill=decision.skill, arguments=decision.arguments)
        self.recorder.event("skill_request", simulation_time=self.backend.time,
                            request=request_json(request, self.recorder.path))
        if self.monitor is not None:
            self.monitor.skill_started(request, self.backend.time)

    def rescue(self, task, decision, failed, step):
        """Auto mode: hand DECISION's FAILED result to the coder as a call the agent did not have to make."""
        if self.coder.calls >= self.settings.coder.calls:
            return failed
        frame = self.observed.snapshot()
        auto = type(decision)(text=f"Automatic coder call after {decision.skill} failed with {failed['code']}.",
                              skill=CALL, arguments=request(task, decision, failed))
        self.recorder.event("decision", step=step, epoch=frame.epoch, sequence=frame.sequence,
                            simulation_time=self.backend.time, decision=auto, automatic=True)
        self.update("Executing", auto.text)
        return self.act(auto, frame, step, rescued=stuck_key(decision.skill, decision.arguments))

    def settle(self):
        """Stand still for agent.done_settle_seconds after the agent's done, before its final physics sample."""
        epoch, tick = self.backend.epoch, self.settings.motion.tick
        self.backend.base_velocity(epoch, 0.0, 0.0, 0.0)
        for _ in range(math.ceil(self.settings.agent.done_settle_seconds / tick)):
            self.observed.advance(epoch, tick)

    def _halt(self, status):
        """Stop the robot after a task; a finished task keeps its arm targets."""
        if self.requires_reset:
            return
        if status == "model_done":
            self.backend.base_velocity(self.backend.epoch, 0.0, 0.0, 0.0)  # settled before the task closed (settle)
        else:
            self.backend.stop(self.backend.epoch)

    def run(self):
        """The idle/task loop. UI threads never step or reset the backend."""
        self.observed.snapshot()
        while not self.controls.closing.is_set():
            command = self.controls.next()
            if command is None:
                try:
                    self.heartbeat()
                except PhysicsInvalid:
                    pass
                continue
            if command.name == "task":
                self.run_task(command.text)
            elif command.name == "stop":
                if not self.requires_reset:
                    self.backend.stop(self.backend.epoch)
                    self.update("Stopped", "Enter another task when ready.")
                self.controls.cancel.clear()
            elif command.name == "reset":
                self.backend.set_evaluation_sink(None)
                if self.ui is not None:
                    self.ui.clear_view()
                try:
                    self.backend.reset()
                    self.observed.snapshot()
                except PhysicsInvalid as error:
                    self.observed.update_scene()
                    self.physics_failed(error)
                    continue
                self._physics_fault = None
                self.context.carried.clear()
                self.history.clear()
                self.coder.reset()
                self.controls.cancel.clear()
                self.recorder.event("scene_reset", epoch=self.backend.epoch)
                self.update("Ready", "Scene reset. Previous plans were invalidated.")
