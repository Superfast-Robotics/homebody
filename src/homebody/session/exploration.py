"""The exploration session: the explorer's task, its model decisions and its live skills.

`ExplorationSession` owns one explorer's provider, live skill context, runner, prepared action and task
recording; the explorer (the runtime) owns the live map, the sensors and what the console shows. `run` is
the task loop. The exploration policy builds each decision's evidence and asks this session to call the
provider and run skills; it never does either itself.
"""
import math
import threading
import time
import traceback
from pathlib import Path

from homebody.primitives.navigation.errors import Abort, ActionRefused, DecisionInterrupted, Stopped
from homebody.real2sim.exploration import vlm_pick
from homebody.real2sim.exploration.decision_types import (
    ActionRequest,
    DecisionContext,
    PrepareAction,
    Proposal,
    Recovery,
)
from homebody.real2sim.exploration.freshness import CAMERA_MAP_MAX_AGE_S
from homebody.real2sim.exploration.livemap import FOOTPRINT_M
from homebody.real2sim.exploration.planner import FRONTIER_PROGRESS_M2, VIEW_LEG_MIN_M, Waypoint
from homebody.real2sim.exploration.policy import ExplorationPolicy
from homebody.real2sim.exploration.settings import current, to_dict
from homebody.real2sim.exploration.skills import REGISTRY
from homebody.real2sim.exploration.vlm_pick import DecisionSession
from homebody.skills.contract import Runner
from homebody.vlm.catalog import model_label

from .live_context import LiveContext, run_live_result
from .live_recorder import LiveRecorder

ROOT = Path(__file__).resolve().parents[3]
MIN_MAP_M2 = 1.0  # known floor before the first leg: the DONE rule is meaningless on no map


class ExplorationSession(DecisionSession):
    """One explorer's task: its decisions, its live skills and its recording."""

    def __init__(self, runtime):
        super().__init__()
        self.exploration, self.settings = runtime, current()
        self.policy = ExplorationPolicy(runtime)
        self.context = LiveContext(runtime, emit=self.live_skill_event)
        self.runner = Runner(self.context, REGISTRY)
        self.thread = self.pending_action = self.recording_error = None
        configuration = to_dict(self.settings) | {"task": runtime._current_brief(),
                                                 "budget_s": runtime.budget_s}
        self.recorder = LiveRecorder(runtime.out_dir / "session", root=ROOT, settings=configuration)
        if isinstance(runtime.pick_hook, vlm_pick.Picker):
            runtime.pick_hook.session = self

    def resolve_context(self, policy):
        """Answer POLICY's requests: one frozen decision, then prepared or executed actions. An
        interrupted decision discards the action it prepared."""
        runtime = self.exploration
        answer, queried = None, False
        try:
            while True:
                try:
                    request = policy.send(answer)
                except StopIteration as finished:
                    if getattr(runtime.source, "observe_only", False):
                        self.cancel_prepared("Observation-only preview")
                    return finished.value
                if isinstance(request, DecisionContext):
                    if queried:
                        raise RuntimeError("A decision context requested a second provider call")
                    queried = True
                    self.cancel_prepared("Replaced by a new decision")
                    if self.policy.offered_actions(self) != tuple(tool.name for tool in request.tools):
                        raise RuntimeError("Offered actions changed after freezing the decision context")
                    answer = runtime.pick_hook(list(request.images), request.prompt, request.on_partial,
                                               request.number, tools=request.tools, **request.limits)
                elif isinstance(request, PrepareAction):
                    self.cancel_prepared("Replaced by another prepared action")
                    answer = self.runner.prepare(request.name, request.arguments, self.context.frame())
                    if self.context.failure is not None:
                        raise self.context.failure
                    if not answer.finished:
                        self.pending_action = answer
                elif isinstance(request, ActionRequest):
                    answer = run_live_result(runtime, request.name, request.arguments)
                    runtime._record_skill(answer)
                else:
                    raise TypeError(f"Unsupported decision request: {type(request).__name__}")
        except BaseException:
            self.cancel_prepared("Decision interrupted before execution")
            raise

    def resolve_proposal(self, proposals):
        """Drive the policy's proposal continuation: each Proposal is one decision, each Recovery a
        local recovery; an interrupted decision is thrown back into the continuation."""
        runtime = self.exploration
        answer = error = None
        while True:
            try:
                request = proposals.throw(error) if error is not None else proposals.send(answer)
            except StopIteration as finished:
                return finished.value
            error = None
            try:
                if isinstance(request, Proposal):
                    options = {"rejection": request.rejection} if request.rejection else {}
                    if request.continuation:
                        options["continuation"] = request.continuation
                    answer = runtime._vlm_leg(request.x, request.y, request.yaw, **options)
                elif isinstance(request, Recovery):
                    answer = runtime._recover_proposal(request.record)
                else:
                    raise TypeError(f"Unsupported policy request: {type(request).__name__}")
            except DecisionInterrupted as interrupted:
                error = interrupted

    def cancel_prepared(self, message):
        """Discard the prepared action, if any, without executing it."""
        attempt, self.pending_action = self.pending_action, None
        if attempt is not None:
            self.runner.cancel(attempt, message)

    def execute_prepared(self, waypoint, *, look, mps):
        """Execute the action prepared for WAYPOINT with LOOK and MPS; None when nothing was prepared
        for it, after discarding a preparation for another waypoint."""
        attempt = self.pending_action
        if attempt is None:
            return None
        if getattr(attempt.plan, "waypoint", None) is not waypoint:
            self.cancel_prepared("Selected waypoint replaced before execution")
            return None
        self.pending_action = None
        attempt.plan.motion = {"look": look, "mps": mps}
        return self.runner.execute(attempt)

    def live_skill_event(self, event):
        """Record one skill event; a failed write stops the exploration."""
        event = dict(event)
        if "plan" in event:
            event["plan"] = getattr(event["plan"], "summary", event["plan"])
        try:
            self.recorder.event("skill", event=event)
        except (OSError, ValueError, TypeError) as error:
            self._recording_failed(error)
            raise

    def _recording_failed(self, error):
        self.recording_error = f"{type(error).__name__}: {error}"
        self.exploration.stop("Session recording failed")
        self.exploration.message = ("Recording failed; the files already written are kept: "
                                    f"{self.recording_error}")

    def start_exploration(self):
        """Start the task on its own thread unless it is running."""
        runtime = self.exploration
        if self.thread is not None and self.thread.is_alive():
            return
        runtime._invalidate_question("new attempt")
        runtime.source.stop_flag.clear()
        runtime.source.pause_flag.clear()
        runtime._resume_event.clear()
        runtime._decision_generation += 1
        self.thread = runtime.thread = threading.Thread(target=self.run_exploration, daemon=True)
        self.thread.start()

    def run_exploration(self, *, safe=True):
        """Record one task around `run` (`run_safe` when SAFE); without a recording nothing runs."""
        runtime = self.exploration
        try:
            self.recorder.start_task(runtime._current_brief(), self.context.epoch)
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            self._recording_failed(error)
            runtime.phase, runtime.stage = "error", "recording failed"
            runtime.stop_reason = runtime.message
            return
        try:
            (run_safe if safe else run)(runtime, self)
        finally:
            try:
                self.cancel_prepared("Exploration ended before execution")
                self.recorder.end_task({"status": runtime.phase, "reason": runtime.stop_reason,
                                        "steps": len(runtime.vlm_picks),
                                        "active_seconds": runtime.active_elapsed(),
                                        "paused_seconds": runtime.paused_elapsed(),
                                        "capture_acceptance": "owned_by_capture_finalization"})
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                self._recording_failed(error)


def run_safe(runtime, session) -> None:
    """`run`, ending the task on a stop, an abort or an unexpected error instead of the thread."""
    try:
        run(runtime, session)
    except Stopped as stopped:
        why = runtime.stop_why if runtime.source.stop_flag.is_set() else str(stopped)
        held = False
        if ("budget" in str(stopped) and not runtime.source.stop_flag.is_set()
                and not getattr(runtime.source, "observe_only", False)):
            try:
                runtime.source.hold_position()
                held = True
            except Stopped as hold_failure:
                why = f"{why}; {hold_failure}"
        runtime.phase, runtime.stop_reason = "stopped", f"stopped: {why}"
        runtime.stage = "held" if held else "stopped"
    except Abort as why:
        runtime.phase, runtime.stop_reason = "aborted", f"abort: {why}"
    except Exception as exc:  # noqa: BLE001 - the task boundary shows an unexpected error on the console
        traceback.print_exc()
        runtime.source.stop_flag.set()
        runtime.phase, runtime.stop_reason = "error", f"error: {type(exc).__name__}: {exc}"
    finally:
        runtime._invalidate_question("attempt ended")
        runtime.ended_at, runtime.ended_wall_s = runtime.source.now(), time.time()
        runtime.plan, runtime.next_wp, runtime.credit = [], None, False
        runtime.message = runtime.stop_reason
        runtime._save()


def run(runtime, session) -> None:
    """The exploration task: a survey from the start, then one leg per decision until it ends."""
    runtime.ended_at = runtime.ended_wall_s = None
    runtime._paused_wall_s = runtime._paused_source_s = 0.0
    runtime._pause_started_wall = runtime._pause_started_source = None
    runtime.t0 = runtime.source.now()
    runtime.t0_wall = time.time()
    runtime.stage = "starting"
    runtime.stops, runtime.trail, runtime.stop_reason = [], [], ""
    runtime.edge_tries = {}
    runtime.refusals = []  # every refused move, with the driver's reason
    runtime.plan_s = []  # seconds the robot stood while each leg was planned
    runtime.routes = []
    runtime.vlm_picks = []
    runtime.previous_action_result, runtime.failed_goals = None, []
    runtime.last_skill_result = None
    runtime._invalidate_question("new attempt")
    runtime.operator_questions = []
    runtime._gallery_not_before = 0.0
    runtime.pending_face = None
    runtime.fails = 0
    if getattr(runtime.source, "observe_only", False):
        runtime.credit = False
        runtime.phase = "preview"
        while not runtime.source.stop_flag.is_set():
            runtime.check()
            map_age = runtime.map_age_s()
            if (map_age is None or not 0 <= map_age <= CAMERA_MAP_MAX_AGE_S
                    or runtime.source.odom_age_s() > 0.5):
                runtime.stage = "waiting for fresh data"
                runtime.message = "Preview: waiting for fresh LiDAR and position data."
            elif runtime.pick_hook is not None:
                x, y, yaw = runtime.source.pose()
                runtime.enclosure = runtime.planner.enclosure()
                leg = runtime._model_leg(x, y, yaw)
                if runtime.stop_reason:
                    return
                if leg is not None and leg[0] == "ask_operator":
                    runtime._ask_operator(runtime.vlm_picks[-1])
                    continue
                if leg is not None and leg[0] == "update_region":
                    runtime._save()
                    continue
                runtime.message = f"Preview only, the robot does not move. The model proposed: {leg}"
            else:
                runtime.message = "Preview only: recording the live map with the model turned off."
            runtime.source.stop_flag.wait(10.0)
        raise Stopped("the preview was stopped")
    if hasattr(runtime.source, "arm"):
        runtime.phase = "arming"
        runtime.source.arm(runtime, lambda msg: setattr(runtime, "message", msg))
    runtime._sync_task_brief()
    runtime.credit = True
    if session.settings.exploration.start_spin:
        runtime.phase, runtime.message = "start spin", "Looking around from the start point."
        runtime._spin()
    runtime.phase = "explore"
    while not runtime.stop_reason:
        if runtime.active_elapsed() >= runtime.budget_s:
            runtime.stop_reason = "incomplete: the exploration time budget ran out"
            break
        x, y, yaw = runtime.source.pose()
        with runtime.lock:  # an empty map is enclosed with nothing to walk: never done on it
            known_m2 = float(runtime.live.known_free().sum()) * 0.05 * 0.05
        if known_m2 < MIN_MAP_M2:
            runtime.message = f"Waiting for the first scans ({known_m2:.1f} m² of floor mapped so far)."
            runtime.check()
            time.sleep(0.2)
            continue
        t_plan = time.time()  # the robot stands while a leg is planned
        with runtime.lock:
            runtime.enclosure = runtime.planner.enclosure()
            enclosed = runtime.enclosure["enclosed"]
            runtime.mapped = runtime.planner.mapped_fraction() if enclosed else 0.0
            if runtime.pick_hook is not None:  # a model's run ends only on its own completion
                runtime.walked = runtime.planner.walked_fraction(
                    runtime.visited, runtime.planner.seen_network()
                )
            else:
                # done when the base has walked most of the skeleton, not when the lidar saw it
                runtime.walked = runtime.planner.walked_fraction(runtime.visited) if enclosed else 0.0
                if enclosed and runtime.walked >= session.settings.exploration.done_fraction:
                    runtime.stop_reason = runtime._geometric_stop(
                        f"walked {100 * runtime.walked:.0f}% of the path network and mapped "
                        f"{100 * runtime.mapped:.0f}% of the area inside the limit"
                    )
                    break
        leg = None
        if runtime.pick_hook is not None:  # the model's time is in vlm_picks[n].seconds, not plan_s
            leg = runtime._model_leg(x, y, yaw)
            if runtime.stop_reason:
                break
            if leg is not None and leg[0] == "ask_operator":
                runtime._ask_operator(runtime.vlm_picks[-1])
                continue
            if leg is not None and leg[0] == "update_region":
                runtime._save()
                continue
            t_plan = time.time()
        if leg is None:  # no model, or no open frontier left for a click to snap to
            leg = runtime._classical_leg(x, y)
        runtime.plan_s.append(round(time.time() - t_plan, 2))
        if leg is None:
            if runtime.enclosure["enclosed"]:
                looked = runtime.enclosure.get("looked_unresolved_m", 0.0)
                runtime.stop_reason = runtime._geometric_stop(
                    "the area is enclosed and every branch of the path network was walked"
                    + (
                        f" ({looked:.1f} m of edge, looked at twice, is still unknown:"
                        " perhaps a low platform or glass)"
                        if looked
                        else ""
                    )
                )
            else:
                runtime.stop_reason = (
                    f"stopped: {runtime.enclosure['open_m']:.1f} m of unexplored edge remains, "
                    "but no route reaches it (each edge was tried twice)"
                )
            break
        try:
            if leg[0] == "look":
                wp = leg[1]
                runtime.plan, runtime.next_wp = [], wp
                runtime.stage, runtime.message = "looking", runtime.vlm_picks[-1]["why"]
                reached = runtime._face(wp.headings[0], recover_blocked=True)
                if reached:
                    runtime._record_action("look", wp.xy, "arrived")
                    runtime.stop_here("look")
            elif leg[0] in ("frontier", "vlm"):
                kind, wp, centre = leg
                if kind == "frontier":
                    runtime.message = (
                        f"Closing the map: {runtime.enclosure['open_m']:.1f} m of unexplored "
                        "edge remains; walking to the nearest."
                    )
                else:
                    pick = runtime.vlm_picks[-1]
                    runtime.message = (
                        f"Decision {pick['n']}: {model_label(session.settings.agent)} chose "
                        f"({wp.xy[0]:.2f}, {wp.xy[1]:.2f}) m. "
                        f"{pick['fallback'] or pick['why']}"
                    )
                x0, y0 = x, y
                face = wp.headings[0] if wp.headings else None
                with runtime.lock:
                    before = 0.0 if centre is None else runtime.planner.known_near(centre)
                gx, gy = wp.xy
                # a frontier view spot this near is looked at, not crawled to; the model's
                # click is walked whenever it is a walk at all (else it re-clicks the same spot)
                near_m = VIEW_LEG_MIN_M if kind == "frontier" else FOOTPRINT_M
                if math.hypot(gx - x0, gy - y0) < near_m and face is not None:
                    reached = runtime._face(face)  # already here: turn to the edge, no crawl
                else:  # the normal drive: turn, walk the route, arrive facing the open edge
                    reached = runtime._walk_to(
                        wp, look=None if face is None else (lambda gx, gy, h=face: h)
                    )
                if reached:
                    runtime._record_action(kind, wp.xy, "arrived")
                    runtime._arrival_face()
                    runtime.stop_here(kind)
                if centre is not None:  # a leg aimed at an edge counts as a try on it
                    key = (round(centre[0]), round(centre[1]))  # one try-count per metre cell
                    with runtime.lock:
                        # a look succeeds when it revealed floor near the edge, even if a new
                        # edge stands farther out
                        gained = runtime.planner.known_near(centre) - before
                        still = runtime.planner.edge_still_there(centre)
                        if reached and (gained >= FRONTIER_PROGRESS_M2 or not still):
                            runtime.edge_tries.pop(key, None)
                        else:
                            runtime.edge_tries[key] = runtime.edge_tries.get(key, 0) + 1
                            if runtime.edge_tries[key] >= 2:
                                runtime.planner.blacklist.append(centre)
            else:
                route, _, why = leg[1]
                runtime.message = f"The area is enclosed; walking the path network ({why})."
                reached = runtime._walk_route(route, Waypoint("voronoi", route[-1], through=True))
                key = runtime.planner.last_leg.get("key")
                if not reached and key is not None:
                    runtime.planner.edge_tries[key] = runtime.planner.edge_tries.get(key, 0) + 1
                if reached:
                    runtime._record_action(leg[0], route[-1], "arrived")
        except ActionRefused as why:
            runtime.check()
            reached = False
            runtime.refusals.append({"t": round(runtime.elapsed(), 1), "why": str(why),
                                  "code": why.code, "arrived": False})
            runtime.message = f"That action was refused: {why}. Asking for another decision."
            runtime._record_action(leg[0], getattr(leg[1], "xy", None), "refused", why)
            runtime._learn_from_ring()
            runtime.plan, runtime.next_wp = [], None
        except Stopped as why:
            facing_result = (runtime.previous_action_result or {}).get("facing_result", {})
            if facing_result.get("outcome") != "stopped":
                runtime._record_action(leg[0], getattr(leg[1], "xy", None), "stopped", why)
            raise
        runtime.fails = 0 if reached else runtime.fails + 1
        runtime.stage = "arrived" if reached else "refused"
        if runtime.vlm_picks and runtime.vlm_picks[-1].get("outcome") is None:
            runtime.vlm_picks[-1]["outcome"] = runtime.stage
        runtime._save()  # after every leg: a killed console still leaves run.json and the map
        # consecutive refusals do not end the run; the next leg is planned
        with runtime.lock:
            runtime.coverage = runtime.planner.coverage()
            runtime.enclosure = runtime.planner.enclosure()

    if runtime.stage == "held":
        return
    runtime.source.hold_position()
    runtime.phase = "done" if runtime.stop_reason.startswith("done:") else "stopped"
    runtime.stage = "held"
    with runtime.lock:
        runtime.coverage = runtime.planner.coverage()
