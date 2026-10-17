"""Explore, record evidence and execute checked model/planner actions.

The model infers a task polygon, and completion needs readable scoped view evidence; motion
authority, geometry, freshness and session limits are checked for every action.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from homebody.helpers.floor import TOO_CLOSE_M, reachable_network
from homebody.helpers.raster import px_to_grid, world_to_px
from homebody.primitives.navigation.contracts import jsonable
from homebody.primitives.navigation.errors import (
    Abort,
    ActionRefused,
    DecisionInterrupted,
    Stopped,
)
from homebody.session.exploration import ExplorationSession
from homebody.session.live_context import run_live, run_live_result
from homebody.session.operator_questions import ask_question

from . import vlm_pick
from .freshness import CAMERA_MAP_MAX_AGE_S, DECISION_FRESH_STABLE_S
from .livemap import FOOTPRINT_M, Boundary, LiveMap, default_boundary
from .planner import (
    RADIUS_M,
    Planner,
    Waypoint,
    padded_spot,
    plan_padded,
    route_refusal,
)
from .policy import decision, proposals
from .settings import current
from .skills import ACTIONS
from .visual_history import (
    MAX_ASSOCIATION_AGE_S,
    VisualHistory,
    completion_refusal,
    select_context_frame,
    select_transit_frames,
)

VISIT_RADIUS_M = 0.6  # a skeleton cell this close to the walked path counts as walked
WALK_MPS = 0.30  # walking speed of every leg
BREACH_ABORT_M = 0.50  # this far outside the boundary aborts exploration
CREDIT_EVERY_S = 0.3
LEG_MIN_S, LEG_SLACK = 60.0, 3.0  # a leg's deadline: 3x its walking time, at least a minute
REPLAY_SCANS = 400


def recovery_eligible(owner):
    """Whether a blocked start may try the checked local departure recovery."""
    return bool(owner.live.scope_ready
                and callable(getattr(owner.source, "recover_departure", None))
                and not getattr(owner.source, "observe_only", False)
                and owner._proposal_budget_available())


class MotionEvents:
    """Show controller events on the explorer's status without granting it motion access."""
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, kind, payload):
        r = self.runtime
        if kind == "status":
            r.stage, r.message = payload["stage"], payload["message"]
        elif kind == "drive.phase" and payload.get("phase") in {"SETTLE", "VERIFY_ARRIVAL"}:
            r.stage, r.message = "verifying arrival", "held; checking settled arrival"
        elif kind == "drive.route" and payload.get("phase") == "execute":
            self._route(payload.get("route_world_xy", []))
            r.radius = payload.get("radius_m", getattr(r, "radius", None))
        elif kind == "drive.route_rejoin":
            if payload["phase"] in ("start", "complete"):
                self._route(payload["route_world_xy"])
            if payload["phase"] == "start":
                r.stage, r.message = "repairing", "checked rejoin to the planned route"
            elif payload["phase"] == "complete":
                r.stage, r.message = "driving", "planned route rejoined"
        elif kind in ("base.localization_wait", "drive.localization_degraded"):
            r.stage, r.message = "waiting for fresh data", payload.get("why", "localization unavailable")
        elif kind in ("base.localization_recovered", "drive.localization_recovered"):
            r.stage, r.message = "repairing", "fresh localization; revalidating the route"
        elif kind == "drive.route_revalidated":
            if payload.get("reason") == "tracking reference repaired":
                self._route(payload["route_world_xy"])
                r.stage, r.message = "driving", "tracking reference repaired"
            else:
                r.stage, r.message = "driving", "localization restored and route revalidated"

    def _route(self, points):
        self.runtime.plan = [tuple(point) for point in points]
        self.runtime.routes.append(points)


class Explorer:
    def __init__(
        self, source, boundary: Boundary | None, out_dir: Path, budget_s: float | None = None
    ):
        self.source = source
        source.attach(MotionEvents(self), self._wait_while_paused)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.visual_history = VisualHistory(self.out_dir / "views")
        settings = current().exploration
        self.budget_s = float(settings.budget_s if budget_s is None else budget_s)
        self.ask_operator_enabled = settings.ask_operator
        self.decision_recovery_wait_s = settings.decision_recovery_wait_s
        self.brief: str | None = None
        self.session = None
        self.lock = threading.RLock()
        self._map_save_lock = threading.Lock()
        self._pause_request_lock = threading.Lock()
        self._resume_event = threading.Event()
        self._decision_generation = 0
        self._pause_started_wall = self._pause_started_source = None
        self._paused_wall_s = self._paused_source_s = 0.0
        self._pause_restore = None
        self.pending_question = None
        self.operator_questions = []
        self.replay: deque = deque(maxlen=REPLAY_SCANS)  # the newest scans, replayed into a new map
        self.map_t: float | None = None
        self.stop_why = "stop pressed"
        self.task_region_generation = 0
        self.set_boundary(boundary)  # also the map, visited cells and pending_face
        self.phase = "idle"
        self.message = ("Press Start exploration. The model works out the task region "
                        "from the first look around and your brief.")
        self.stops: list[dict] = []
        self.trail: list[tuple[float, float, float]] = []
        self.plan: list[tuple[float, float]] = []
        self.next_wp: Waypoint | None = None
        self.radius = RADIUS_M
        self.stop_reason = ""
        self.t0: float | None = None
        self.t0_wall: float | None = None  # wall clock at exploration start, for run.json
        self.ended_at: float | None = None
        self.ended_wall_s: float | None = None
        self.credit = False
        self._last_credit = -1e9
        self.coverage = {">=1": 0.0, ">=2": 0.0, ">=3": 0.0, "patches": 0}
        self.enclosure = {
            "enclosed": False,
            "open_m": 0.0,
            "looked_unresolved_m": 0.0,
            "unknown_m2": 0.0,
        }
        self.thread: threading.Thread | None = None
        # the model picks every leg when this is set (vlm_pick.Picker)
        self.pick_hook = None
        self.vlm_picks: list[dict] = []
        self.previous_action_result: dict | None = None
        # a model-called skill's result newer than previous_action_result, or None
        self.last_skill_result: dict | None = None
        self._gallery_not_before = 0.0
        self._capture_sequence = 0
        self.failed_goals: list[dict] = []
        self.walked = self.mapped = 0.0
        self.stage = "idle"  # asking the model / planning / driving / arrived, for the console
        self.vlm_view: dict = {
            "version": 0,
            "n": 0,
            "image": None,
            "ego": None,
            "head": f"{vlm_pick.model_label_text()} · waiting for the first decision",
            "words": "",
            "since": None,
        }

    # -- map ----------------------------------------------------------------------------------
    def set_boundary(self, zone: Boundary | None) -> None:
        """Select manual permission or automatic scope; boundary edits reset the idle map."""
        with self.lock:
            self.task_region_generation += 1
            self._task_brief = None
            self.zone = zone
            self.boundary = zone if zone is not None else default_boundary()
            self.live = LiveMap(self.boundary)
            self.live.task_region_required = zone is None
            self.task_region_evidence = None
            self._gallery_sample = None
            self._decision_gallery_key = None
            self._recent_route_evidence = {}
            self.pending_face = None
            self.visited = np.zeros(self.live.floor.shape, bool)
            self.planner = Planner(self.live)
            for pts, origin, t in self.replay:
                self.live.add_scan(pts, origin, t)
        (self.out_dir / "boundary.json").write_text(
            json.dumps(
                {
                    "start_frame_points_m": self.boundary.points,
                    "zone": None if zone is None else zone.points,
                    "ownership": "automatic" if zone is None else "manual",
                }
            )
        )

    def reset_frame(self) -> None:
        """Clear the replayed scans and rebuild the map in the new start frame."""
        self._invalidate_question("start frame reset")
        with self.lock:
            self.replay.clear()
            self.map_t = None
            self.set_boundary(self.zone)
            self.visual_history.reset("exploration start frame reset", frame_metadata=
                                          getattr(self.source, "start_frame_metadata", None))

    def map_age_s(self) -> float | None:
        """Age of the newest map evidence in seconds, or None."""
        return None if self.map_t is None else time.time() - self.map_t

    def ingest(self) -> None:
        scans = self.source.scans()
        if not scans:
            return
        x, y, yaw = self.source.pose()
        with self.lock:
            for pts, origin, t in scans:
                self.live.add_scan(pts, origin, t)
                step = max(1, len(pts) // 1500)
                self.replay.append((pts[::step], origin, t))
            self.live.sweep(x, y, FOOTPRINT_M, current().lidar.self_clear_m)
            self.map_t = max(t for _, _, t in scans)
            if not getattr(self.source, "observe_only", False):
                self._visit(x, y)
            if self.credit and self.source.now() - self._last_credit >= CREDIT_EVERY_S:
                self.planner.see((x, y), [yaw], credit=True)
                self._last_credit = self.source.now()
        refresh_geometry = getattr(getattr(self.source, "backend", None),
                                   "refresh_execution_geometry", None)
        if refresh_geometry is not None:
            refresh_geometry()
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.03:
            self.trail.append((x, y, self.source.now()))

    def _visit(self, x: float, y: float) -> None:
        if self.visited.shape != self.live.floor.shape:
            self.visited = np.zeros(self.live.floor.shape, bool)
        u, v = self.live.uv(x, y)
        r = round(VISIT_RADIUS_M / 0.05)
        v0, v1 = max(v - r, 0), min(v + r + 1, self.visited.shape[0])
        u0, u1 = max(u - r, 0), min(u + r + 1, self.visited.shape[1])
        if v0 < v1 and u0 < u1:
            vv, uu = np.ogrid[v0:v1, u0:u1]
            self.visited[v0:v1, u0:u1] |= (vv - v) ** 2 + (uu - u) ** 2 <= r * r

    def elapsed(self) -> float:
        end = self.source.now() if self.ended_at is None else self.ended_at
        return 0.0 if self.t0 is None else end - self.t0

    def paused_elapsed(self) -> float:
        ongoing = (0.0 if self._pause_started_wall is None else
                   time.monotonic() - self._pause_started_wall)
        return self._paused_wall_s + ongoing

    def active_elapsed(self) -> float:
        ongoing = (0.0 if self._pause_started_source is None else
                   self.source.now() - self._pause_started_source)
        return max(0.0, self.elapsed() - self._paused_source_s - ongoing)

    def active_monotonic(self) -> float:
        return time.monotonic() - self.paused_elapsed()

    def request_pause(self) -> bool:
        terminal = {"idle", "done", "stopped", "aborted", "error", "arming"}
        # publish the request without waiting for the map lock
        with self._pause_request_lock:
            if (self.phase in terminal or self.source.stop_flag.is_set()
                    or self.source.pause_flag.is_set()):
                return False
            self._pause_restore = (self.phase, self.stage, self.message)
            self._decision_generation += 1
            self._resume_event.clear()
            self.source.pause_flag.set()
        with self.lock:
            if self.phase in terminal or self.source.stop_flag.is_set():
                return False
            # keep a pause the writer already confirmed
            if self.source.pause_flag.is_set() and self.phase != "paused":
                self.phase, self.stage = "pausing", "pausing"
                self.message = "Pausing… confirming hold"
        return True

    def resume(self) -> bool:
        with self.lock:
            if (self.pending_question is not None or self.phase != "paused"
                    or self.source.stop_flag.is_set()
                    or getattr(self.source, "frame_reset", False)):
                return False
            self._resume_event.set()
            return True

    def _wait_while_paused(self, generation, epoch) -> None:
        """Wait paused until resumed; raise Stopped on a stop, frame change or brief change."""
        live, brief = self.live, self._current_brief()
        self._pause_started_wall, self._pause_started_source = time.monotonic(), self.source.now()
        try:
            with self.lock:
                question = self.pending_question
                self.stage = "waiting for operator" if question is not None else "paused"
                self.message = (question["question"] if question is not None else
                                "Paused and holding position. The sensors and the recording keep running.")
                self.phase = "paused"
            while True:
                if self.source.stop_flag.is_set():
                    raise Stopped(self.stop_why)
                if getattr(self.source, "frame_reset", False) or self.live is not live:
                    raise Stopped("start frame changed while paused")
                if self._current_brief() != brief:
                    raise Stopped("task brief changed while paused; start a new attempt")
                guard = getattr(self.source, "stationary_guard", None)
                if guard is not None:
                    guard(generation)
                motion_epoch = getattr(self.source, "motion_epoch", None)
                if epoch is not None and motion_epoch is not None and motion_epoch() != epoch:
                    raise Stopped("motion frame changed while paused")
                if self._resume_event.is_set():
                    self.source.pause_flag.clear()
                    self.phase, self.stage, self.message = self._pause_restore or (
                        "explore", "resuming", "Resuming with fresh observations")
                    return
                self._resume_event.wait(.1)
        finally:
            self._paused_wall_s += time.monotonic() - self._pause_started_wall
            self._paused_source_s += self.source.now() - self._pause_started_source
            self._pause_started_wall = self._pause_started_source = None

    def _service_pause(self) -> None:
        if not self.source.pause_flag.is_set():
            return
        service = getattr(self.source, "service_pause", None)
        if getattr(self.source, "observe_only", False):
            self._wait_while_paused(None, None)
        elif service is not None:
            service()
        else:
            # an offline source stands still itself before the pause is shown
            self.source.hold_position()
            self._wait_while_paused(None, None)

    def _invalidate_question(self, reason):
        with self.lock:
            question = self.pending_question
            if question is None:
                return
            question.update(status="invalidated", invalidation_reason=reason)
            self.pending_question = None
            self._persist_question(question, "invalidated")

    def _persist_question(self, question, event):
        row = dict(question) | {"event": event, "wall_s": time.time()}
        with (self.out_dir / "operator_questions.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        path = self.out_dir / "vlm" / f"leg_{question['decision']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            saved["operator_question"] = dict(question)
            if question["status"] != "pending":
                saved["outcome"] = question["status"]
            path.write_text(json.dumps(saved, indent=1))

    def submit_operator_answer(self, question_id, text) -> bool:
        with self.lock:
            question = self.pending_question
            if (question is None or question["id"] != question_id
                    or question["status"] != "pending" or self.phase != "paused"
                    or self.source.stop_flag.is_set() or getattr(self.source, "frame_reset", False)
                    or question["brief"] != self._current_brief()
                    or question["generation"] != self.task_region_generation
                    or question["visual_epoch"] != getattr(self.visual_history, "epoch", None)
                    or not isinstance(text, str) or not text.strip()):
                return False
            question.update(status="answered", answer=text.strip(), answered_wall_s=time.time())
            self._persist_question(question, "answered")
            self.pending_question = None
            self._resume_event.set()
            return True

    def _ask_operator(self, rec):
        return run_live(self, "ask_operator", {"record": rec})

    # -- what live skills read (skills.live.LiveWorld) ----------------------------------------
    def pose(self):
        return self.source.pose()

    def base_xy(self):
        if hasattr(self.source, "base_xy"):
            return self.source.base_xy()
        return self.decision_pose()[:2]

    def remaining(self) -> float:
        """Seconds left in the active exploration budget."""
        return self.budget_s-self.active_elapsed()

    def motion_epoch(self):
        """The source's motion epoch; None for a source without one."""
        epoch = getattr(self.source, "motion_epoch", None)
        return None if epoch is None else epoch()

    def stationary_guard(self, *generation):
        """The source's stationary guard: the current generation, or a check that GENERATION is
        still current; None for a source without one."""
        guard = getattr(self.source, "stationary_guard", None)
        return None if guard is None else guard(*generation)

    @property
    def observe_only(self) -> bool:
        return bool(getattr(self.source, "observe_only", False))

    def ask_question(self, rec):
        """Stand still and wait for the answer to REC's question."""
        return ask_question(self, rec)

    def emit(self, kind, **data):
        """Apply a live skill's workflow event to the explorer's state and its decision records."""
        if kind == "status":
            for key in ("stage", "message"):
                if key in data:
                    setattr(self, key, data[key])
        elif kind == "route":
            self.plan, self.next_wp, self.radius = data["points"], data["waypoint"], data["radius"]
            self.routes.append([[round(x, 3), round(y, 3)] for x, y in data["points"]])
        elif kind == "radius":
            self.radius = data["radius"]
        elif kind == "pending_face":
            self.pending_face = data["intention"]
        elif kind == "refusal":
            self.refusals.append({"t": round(self.elapsed(), 1), "why": data["why"]})
        elif kind == "action":
            self._record_action(data["action"], data.get("target"), data["outcome"],
                                data.get("error"))
        elif kind == "gallery_after":
            self._gallery_not_before = data["time"]
        elif kind == "facing":
            result = data["result"]
            rec = self.vlm_picks[-1]
            rec["facing_result"] = result
            if self.previous_action_result is not None:
                self.previous_action_result["facing_result"] = result
            self._patch_decision(rec["n"], facing_result=result)
        elif kind == "recovery_start":
            self.plan, self.next_wp = [], None
            self.stage, self.message = "repairing", "checking another departure from the blocked start"
        elif kind == "recovery_result":
            result, refusal = data["result"], data["refusal"]
            if refusal is not None:
                self._record_action("recovery", None, "refused", refusal)
            else:
                outcome = result.get("outcome") if result else "refused"
                if outcome not in {"route_restored", "departure_restored", "repositioned_goal_unavailable", "refused"}:
                    outcome = "recovered"
                self._record_action("recovery", None, outcome)
                self.previous_action_result["details"] = result
        elif kind == "recovery_record":
            rec = data["record"]
            rec["departure_recovery"] = deepcopy(self.previous_action_result)
            rec["outcome"] = "rejected"
            self._patch_decision(rec["n"], departure_recovery=rec["departure_recovery"],
                                 outcome="rejected")
        elif kind == "completion":
            self.stop_reason = data["reason"]
        elif kind == "help":
            self.stop_reason = data["reason"]
            self.phase, self.stage = "stopped", "held"
        else:
            raise ValueError(f"Unknown exploration skill event: {kind}")

    def _patch_decision(self, number, **values):
        path = self.out_dir / "vlm" / f"leg_{number}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            saved.update(values)
            path.write_text(json.dumps(saved, indent=1))

    def region_snapshot(self):
        """What a task-region admission compares before and after its checks."""
        return {
            "frame": self.live, "manual": self.zone is not None,
            "decision_generation": self._decision_generation,
            "stopped": self.source.stop_flag.is_set(), "context": self.task_scope_context(),
            "brief": self._current_brief(), "task_brief": self._task_brief,
            "generation": self.task_region_generation,
        }

    def publish_region(self, candidate, growth, revision, evidence):
        """Install an admitted task region, or its map growth; returns the accepted record."""
        frame = self.live
        accepted = self.task_scope_context() | {
            "revision": candidate.task_region_revision, "status": "accepted",
            "vertices_start_frame_m": candidate.task_region_points, "evidence": evidence}
        if growth is None:
            frame.publish_task_region(candidate, revision)
        else:
            self.visited = self.planner.publish_growth(growth, self.visited)
            self.boundary = frame.boundary
        self.task_region_evidence = evidence
        self.plan, self.next_wp, self.pending_face = [], None, None
        return accepted

    def persist_region(self, accepted, metrics):
        """Record the accepted region and any growth metrics, then save the map."""
        if metrics is not None:
            with (self.out_dir / "map_growth.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
        with (self.out_dir / "task_regions.jsonl").open("a") as stream:
            stream.write(json.dumps(accepted | {"wall_s": time.time()}) + "\n")
        self.save_map()
        backend = getattr(self.source, "backend", None)
        if backend is not None:
            backend.refresh_execution_geometry()

    def assess_completion(self, assessment, history):
        """Why the scoped visual evidence does not support ASSESSMENT's completion claim, or ''."""
        current, metadata = self.source.context_image()
        odom_age, map_age = self.source.odom_age_s(), self.map_age_s()
        fresh = (history["current_viewpoint"] is not None and current is not None
                 and metadata.get("status") == "fresh" and map_age is not None
                 and 0 <= odom_age <= 0.5 and 0 <= map_age <= CAMERA_MAP_MAX_AGE_S
                 and not getattr(self.source, "frame_reset", False))
        return completion_refusal(
            assessment, history["view_ids"], current_fresh=fresh,
            same_epoch=self.visual_history.epoch == history["epoch"],
            observe_only=getattr(self.source, "observe_only", False),
            coverage=self.visual_history.coverage_assessment,
            readable_ids=self.visual_history.readable_ids())

    # -- exploration --------------------------------------------------------------------------
    def _session(self):
        if self.session is None:
            self.session = ExplorationSession(self)
        return self.session

    def start(self) -> None:
        self._session().start_exploration()

    def stop(self, why: str = "stop pressed") -> None:
        if not self.source.stop_flag.is_set():
            self.stop_why = why
        self.source.stop_flag.set()
        self._invalidate_question(why)
        self._decision_generation += 1
        self._resume_event.set()

    def _run_safe(self) -> None:
        self._session().run_exploration(safe=True)

    def planning_cancelled(self) -> bool:
        """Raise when a locked search must unwind for a stop, frame change or pause."""
        if self.source.stop_flag.is_set():
            raise Stopped(self.stop_why)
        if getattr(self.source, "frame_reset", False):
            raise Abort("SuperOdom frame changed during planning")
        if self.source.pause_flag.is_set():
            raise DecisionInterrupted()
        if self.active_elapsed() >= self.budget_s:
            raise Stopped("incomplete: the exploration time budget ran out while planning")
        return False

    def check(self) -> None:
        if self.source.stop_flag.is_set():
            raise Stopped(self.stop_why)
        if getattr(self.source, "frame_reset", False):
            why = getattr(self.source, "frame_reset_why", "")
            raise Abort(f"SuperOdom restarted: the start frame is gone ({why})")
        self._service_pause()
        x, y, _ = self.source.pose()
        out = self.boundary.distance_outside(x, y)
        if out > BREACH_ABORT_M:
            raise Abort(f"the robot is {out:.2f} m outside the boundary")
        if self.active_elapsed() >= self.budget_s:
            raise Stopped("incomplete: the exploration time budget ran out; holding position")

    def stop_here(self, kind: str) -> None:
        x, y, _ = self.source.pose()
        self.stops.append(
            {
                "n": len(self.stops) + 3,
                "kind": kind,
                "xy": [round(x, 3), round(y, 3)],
                "t": round(self.elapsed(), 1),
            }
        )

    @contextmanager
    def action_capture(self, kind, *, route=None):
        """Retain route evidence on the camera worker during the motion, saved after it."""
        owner, frame = self.visual_history, self.live
        begin = getattr(self.source, "begin_context_retention", None)
        retained = owner is not None and route is not None and begin is not None
        if retained:
            self._capture_sequence += 1
            context = {"kind": kind, "epoch": owner.epoch,
                       "route_id": f"E{owner.epoch:03d}-R{self._capture_sequence:04d}",
                       "started_wall_s": time.time()}
            departure = next((r for r in owner.records
                              if r["id"] == self._decision_gallery_key
                              and r["source_kind"] == "stationary"), None)
            if departure is not None:
                phone = departure["phone"]
                context.update(departure_view_id=departure["id"],
                               departure_camera_frame_identity=[phone.get("sequence"),
                                   phone.get("receiver_decode_time_s")],
                               departure_pose_xyyaw=departure["robot_pose_xyyaw"],
                               departure_pose_sampled_at_s=departure["pose_sampled_at_s"],
                               departure_registered_pose_age_s=departure["registered_pose_age_s"])
            sampler = self._route_evidence_sampler(context, frame, route)
            begin(context, sample_context=sampler)
        try:
            yield
        finally:
            if self.visual_history is not None:
                self._gallery_not_before = time.monotonic() + current().iphone.post_action_buffer_s
            if retained:
                self.source.finish_context_retention(time.time(), tail_s=0.)
                self._save_route_evidence(owner, frame)

    def _route_evidence_sampler(self, context, frame, route):
        """A sampler of the cached pose and progress along ROUTE; it commands no motion."""
        state = {"route": tuple(map(tuple, route)), "revision": 0,
                 "previous": None, "traveled_m": 0.}

        def sample():
            if (getattr(self.source, "frame_reset", False)
                    or self.live is not frame or self.visual_history is None
                    or self.visual_history.epoch != context["epoch"]):
                return {}
            try:
                age_before = self.source.odom_age_s()
                pose = tuple(self.source.pose())
                # the larger of the two ages, so a new registration cannot freshen this pose
                age = max(age_before, self.source.odom_age_s())
            except RuntimeError:
                state["previous"] = None
                return {}
            sampled_at = time.time()
            if (len(pose) != 3 or not np.isfinite([*pose, age]).all()
                    or not 0 <= age <= MAX_ASSOCIATION_AGE_S):
                state["previous"] = None
                return {}
            points = tuple(map(tuple, self.plan)) or state["route"]
            if points != state["route"]:
                state["route"], state["revision"] = points, state["revision"] + 1
            if state["previous"] is not None:
                state["traveled_m"] += math.dist(pose[:2], state["previous"][:2])
            state["previous"] = pose
            progress = getattr(self.source, "context_traveled_m", lambda: None)()
            if progress is not None:
                state["traveled_m"] = progress
            path = np.asarray(points, dtype=float)
            nearest = int(np.argmin(np.linalg.norm(path - pose[:2], axis=1)))
            segment = 0
            if len(path) > 1:
                delta = np.diff(path, axis=0)
                fraction = np.clip(np.sum((pose[:2] - path[:-1]) * delta, axis=1)
                                   / np.maximum(np.sum(delta * delta, axis=1), 1e-12), 0., 1.)
                segment = int(np.argmin(np.linalg.norm(
                    path[:-1] + fraction[:, None] * delta - pose[:2], axis=1)))
            return {"pose": pose, "pose_sampled_at_s": sampled_at,
                    "registered_pose_age_s": age, "traveled_m": state["traveled_m"],
                    "motion_stage": self.stage,
                    "held_since_wall_s": getattr(self.source, "_context_hold_since", None),
                    "route_context": {"route_id": context["route_id"],
                                      "revision": state["revision"], "segment": segment,
                                      "nearest_knot": nearest}}

        return sample

    def _save_route_evidence(self, owner, frame):
        """Save the transit and pre-facing frames retained during the last motion."""
        retention = self.source.retained_context_samples()
        with self.lock:
            if (not retention or getattr(self.source, "frame_reset", False)
                    or self.live is not frame or self.visual_history is not owner
                    or retention.get("context", {}).get("epoch") != owner.epoch):
                return
            snapshot = owner.snapshot()
        latest = {"transit_ids": [], "pre_facing_ids": []}
        selection = {}
        for sample in select_transit_frames(retention, diagnostics=selection):
            key, _ = snapshot.capture_transit(
                sample["rgb"], sample["metadata"], sampled_at=sample["sampled_at"],
                action_interval={k: retention[k] for k in
                                 ("context", "started_wall_s", "ended_wall_s", "tail_s")},
                **{k: sample[k] for k in ("pose", "pose_sampled_at_s",
                   "registered_pose_age_s", "route_context", "traveled_m", "motion_stage")})
            if key is not None:
                latest["transit_ids"].append(key)
        # samples from the current stationary spell, after the post-action buffer
        hold_start = getattr(self.source, "_context_hold_since", None)
        held = [s | {"odom_age_s": s["registered_pose_age_s"]}
                for s in retention.get("samples", []) if s.get("held_since_wall_s") is not None
                and s["held_since_wall_s"] == hold_start
                and s["sampled_at"] - s["held_since_wall_s"] >= current().iphone.post_action_buffer_s]
        if self.pending_face is not None and len(held) >= 2:
            sample, report = select_context_frame(held, retained=True)
            key, _ = snapshot.capture(
                sample["rgb"], sample["metadata"] | {"selection": report},
                sample["pose"], sampled_at=sample["sampled_at"],
                odom_age_s=sample["odom_age_s"], capture_role="pre_facing",
                route_context=sample["route_context"])
            if key is not None:
                latest["pre_facing_ids"].append(key)
        if self.live is frame and self.visual_history is owner:
            self._commit_visual_history(owner, snapshot, frame)
            self._recent_route_evidence = latest
            log = getattr(self.source, "drive_log", None)
            if log is not None:
                log.emit(time.monotonic_ns(), "explore.route_evidence", {
                    "context": retention["context"], "selection": selection,
                    "saved": latest})

    def capture_survey_heading(self):
        """Retain the settled view at this heading, sampled as for a decision."""
        with self.lock:
            owner, frame = self.visual_history, self.live
            snapshot = owner.snapshot() if owner is not None else None
        if owner is None:
            return
        self._gallery_not_before = time.monotonic() + current().iphone.post_action_buffer_s
        self._camera_context()
        gallery = self._gallery_sample
        with self.lock:
            if self.live is not frame:
                raise Abort("map frame changed while sampling survey context")
            if self.visual_history is not owner or owner.epoch != snapshot.epoch:
                raise Stopped("start frame changed while sampling survey context")
        key, _ = snapshot.capture(
            gallery["rgb"], gallery["metadata"], gallery["pose"],
            sampled_at=gallery["sampled_at"], odom_age_s=gallery["odom_age_s"])
        if key is not None:
            self._commit_visual_history(owner, snapshot, frame)

    def require_task_scope(self):
        self._sync_task_brief()
        if getattr(self.live, "task_region_required", False) and not self.live.scope_ready:
            raise ActionRefused("the model must define the task boundary before translation",
                                code="TASK_SCOPE_UNRESOLVED")

    def task_scope_context(self, transform=None, shape=None):
        points = self.zone.points if self.zone is not None else self.live.task_region_points
        result = {"revision": getattr(self.live, "task_region_revision", 0),
                  "generation": self.task_region_generation,
                  "visual_epoch": getattr(self.visual_history, "epoch", None),
                  "brief": self._task_brief,
                  "status": "accepted" if points is not None else "awaiting initial proposal",
                  "frozen": self.zone is not None,
                  "vertices_start_frame_m": points,
                  "evidence": getattr(self, "task_region_evidence", None),
                  "manual_zone_active": self.zone is not None,
                  "ownership": "manual" if self.zone is not None else "automatic"}
        if points is not None and transform is not None:
            pixels = world_to_px(transform, np.asarray(points))
            result["vertices_grid"] = [px_to_grid(*p, shape) for p in pixels]
        return result

    def _current_brief(self):
        return " ".join((self.brief or current().exploration.task).split())

    def _sync_task_brief(self):
        """Apply an edited task brief between actions, keeping the images and maps."""
        brief = self._current_brief()
        if self._task_brief is None:
            self._task_brief = brief
            return
        if self._task_brief == brief:
            return
        self._invalidate_question("task brief changed")
        with self.lock:
            owner, frame = self.visual_history, self.live
            updated = owner.snapshot()
            updated.coverage_assessment = []
            self.live.clear_task_region()
            self._task_brief = brief
            self.task_region_evidence = None
            self.plan, self.next_wp, self.pending_face = [], None, None
            event = self.task_scope_context() | {"reason": "user brief changed"}
        self._commit_visual_history(owner, updated, frame, persist=True)
        with (self.out_dir / "task_regions.jsonl").open("a") as stream:
            stream.write(json.dumps(event | {"wall_s": time.time()}) + "\n")
        self.save_map()
        backend = getattr(self.source, "backend", None)
        if backend is not None:
            backend.refresh_execution_geometry()

    def _apply_task_region(self, update, transform, shape, supplied_ids, frame, context=None):
        if update is None:
            return ""
        result = run_live_result(self, "update_region", {
            "update": update, "transform": transform, "shape": shape,
            "supplied_ids": supplied_ids, "frame": frame, "context": context})
        self._record_skill(result)
        return "" if result.ok else result.message

    def _walk_route(self, route, wp: Waypoint) -> bool:
        session = self._session()
        result = session.runner.run("walk_route", {"route": route, "waypoint": wp, "mps": WALK_MPS},
                                    session.context.frame())
        return session.context.unwrap(result)

    def refresh_planning_radius(self):
        resolve = getattr(self.source, "planning_radius_m", None)
        self.radius = resolve(RADIUS_M) if callable(resolve) else RADIUS_M
        self.planner.radius_m = self.radius
        return {"requested_radius_m": RADIUS_M} if callable(resolve) else {}

    def leg_seconds(self, points) -> float:
        """The deadline for walking POINTS within the remaining budget."""
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points))
        remaining = self.budget_s - self.active_elapsed()
        return max(0., min(remaining, max(LEG_MIN_S, LEG_SLACK * length / WALK_MPS)))

    def _walk_to(self, wp: Waypoint, look=None, mps: float = WALK_MPS) -> bool:
        """Run checked translation through the session's registered skill lifecycle."""
        session = self._session()
        retained = session.execute_prepared(wp, look=look, mps=mps)
        if retained is not None:
            return session.context.unwrap(retained)
        result = session.runner.run(
            "pick_point", {"waypoint": wp, "look": look, "mps": mps}, session.context.frame())
        return session.context.unwrap(result)

    def _learn_from_ring(self) -> None:
        """Mark the points a ring refusal hit as obstacles in the map."""
        pts = getattr(self.source, "ring_hits", None)
        if pts is None or not len(pts):
            return
        self.source.ring_hits = np.zeros((0, 3))
        with self.lock:
            self.live.mark_obstacle(np.asarray(pts, float))
        self.save_map()

    def _face(self, yaw: float, *, recover_blocked: bool = False) -> bool:
        return run_live(self, "look_at", {"yaw": yaw, "recover_blocked": recover_blocked})

    def _arrival_face(self):
        if self.pending_face is not None:
            run_live_result(self, "look_at", {"arrival": True})

    def _spin(self, label: str = "start spin") -> None:
        run_live(self, "survey", {"label": label})

    def _classical_leg(self, x: float, y: float):
        """The planner's own pick: the best frontier while the range is open, else the next
        Voronoi leg."""
        self.require_task_scope()
        self.refresh_planning_radius()
        with self.lock:
            if not self.enclosure["enclosed"]:
                found = self.planner.next_frontier(x, y, self.visited)
                if found is not None:
                    wp, centre, _ = found
                    return ("frontier", wp, centre)
            voronoi = self.planner.next_leg(x, y, self.visited)
            return None if voronoi is None else ("voronoi", voronoi, None)

    def decision_pose(self):
        """Wait stationary for sustained fresh input, without extending a moving action."""
        limit = CAMERA_MAP_MAX_AGE_S
        started = context = stable_since = generation = None
        last_log = -math.inf
        decision_frame = self.live
        previous_stage, previous_message = self.stage, self.message
        log = getattr(self.source, "drive_log", None)
        pause_total = self.paused_elapsed()
        while True:
            try:
                self.check()
                paused = self.paused_elapsed()
                if paused != pause_total:
                    if started is not None:
                        started += paused - pause_total
                    stable_since = None
                    pause_total = paused
                if self.live is not decision_frame:
                    raise Abort("map frame changed during observation recovery")
                if started is not None:
                    if self.t0 is not None and self.active_elapsed() >= self.budget_s:
                        raise Stopped("the exploration time budget ran out while waiting for fresh sensor data")
                    guard = getattr(self.source, "stationary_guard", None)
                    if guard is not None:
                        generation = guard(generation)
            except (Stopped, Abort) as exc:
                if started is not None and log is not None:
                    log.emit(context, "explore.decision_preempted", {"reason": str(exc),
                             "wait_s": time.monotonic() - started})
                raise
            age = self.map_age_s()
            # SimSource poses carry no odometry age
            odom_age = self.source.odom_age_s() if hasattr(self.source, "odom_age_s") else 0.0
            fresh = age is not None and 0 <= age <= limit and 0 <= odom_age <= 0.5
            health_stable = self.source.health_stable_s()
            if fresh and started is None and health_stable >= DECISION_FRESH_STABLE_S:
                return getattr(self.source, "navigation_pose", self.source.pose)()
            now = time.monotonic()
            if started is None:
                started, context = now, time.monotonic_ns()
                wait_s = self.decision_recovery_wait_s
                stable_s = DECISION_FRESH_STABLE_S
                guard = getattr(self.source, "stationary_guard", None)
                if guard is not None:
                    generation = guard()
            stable_since = (now if stable_since is None else stable_since) if fresh else None
            stable_for = (0.0 if stable_since is None else
                          min(now - stable_since, health_stable))
            status = getattr(self.source, "freshness_status", dict)()
            details = {"map_age_s": age if age is not None and math.isfinite(age) else None,
                       "odom_age_s": odom_age if math.isfinite(odom_age) else None,
                       "wait_s": now - started, "limit_s": wait_s,
                       "stable_s": stable_for, "required_stable_s": stable_s,
                       "freshness": status}
            why = (f"the next decision needs fresh LiDAR and position data "
                   f"(map {age} s old, odometry {odom_age} s old)")
            if now - started >= wait_s:
                if log is not None:
                    log.emit(context, "explore.decision_unavailable", details)
                raise Stopped(f"{why}; still not fresh after {wait_s:g} s; holding position")
            if fresh and stable_for >= stable_s:
                if log is not None:
                    log.emit(context, "explore.decision_ready", details)
                self.stage, self.message = previous_stage, previous_message
                return getattr(self.source, "navigation_pose", self.source.pose)()
            if log is not None and now - last_log >= 1.0:
                log.emit(context, "explore.decision_wait", details)
                last_log = now
            reason = status.get("reason", "")
            self.stage = "verifying fresh data" if fresh else "waiting for fresh data"
            self.message = (f"{why}{f'; {reason}' if reason else ''}; "
                            f"waited {now - started:.0f} of {wait_s:g} s; "
                            f"fresh for {stable_for:.1f} of the {stable_s:g} s needed")
            # wait without the map lock
            self.source.stop_flag.wait(min(0.1, wait_s - (now - started)))

    def _model_leg(self, x: float, y: float, yaw: float):
        return self._session().resolve_proposal(proposals(self, x, y, yaw))

    def _proposal_budget_available(self) -> bool:
        return self.active_elapsed() < self.budget_s

    def _recover_proposal(self, rec) -> bool:
        if not recovery_eligible(self):
            return False
        result = run_live_result(self, "departure_recovery", {"record": rec})
        return result.details.get("recovered", False)

    def _proposal_continuation(self, x, y, rejected):
        """Offer one stationary reassessment, with an optional checked walking alternative."""
        if (len(rejected) < 2
                or any(r.get("kind") != "point" or r.get("rejection_code") not in
                       {"goal_blocked", "disconnected", "goal_adjustment", "start_blocked"}
                       for r in rejected)
                or not self._proposal_budget_available()):
            return None
        reassessment = len(rejected) == 2
        actions = [name for name in ACTIONS if name != "pick_point"
                   and (self.zone is None or name != "update_region")
                   and (self.ask_operator_enabled or name != "ask_operator")]
        continuation = {
            "available": True, "query_limit": len(rejected) + 1,
            "reason": "final held reassessment after two rejected walking proposals",
            "available_nonwalking_actions": actions,
            "rejections": [{k: r.get(k) for k in
                            ("n", "xy", "proposed_goal", "rejection_code", "rejection")}
                           for r in rejected],
            "walk_alternative_status": "no useful unvisited route found in bounded probe",
        }
        goals = [r["proposed_goal"] for r in rejected if r.get("proposed_goal") is not None]
        if any(math.dist(a, b) <= current().drive.tolerance_m
               for i, a in enumerate(goals) for b in goals[i+1:]):
            continuation["walk_alternative_status"] = "prior substitute goals are not distinct"
            return continuation if reassessment else None
        excluded = goals + [r["goal"] for r in self.failed_goals]
        with self.lock:
            failure = route_refusal(self.live, (x, y), radius_m=self.planner.radius_m,
                                    cancelled=self.planning_cancelled)
            if failure is not None:
                continuation.update(walk_alternative_status="current departure is not admissible",
                                    start_refusal=failure)
                return continuation if reassessment else None
            network, reach = reachable_network(self.planner, x, y)
            vv, uu = np.nonzero(network & ~self.visited)
            xs, ys = self.live.xy(uu, vv)
            useful = np.hypot(xs - x, ys - y) >= TOO_CLOSE_M
            for gx, gy in excluded:
                useful &= np.hypot(xs - gx, ys - gy) > current().drive.tolerance_m
            candidates = np.flatnonzero(useful)
            for i in np.linspace(0, len(candidates) - 1, min(8, len(candidates)), dtype=int):
                self.planning_cancelled()
                if not self._proposal_budget_available():
                    break
                j = candidates[i]
                goal = padded_spot(self.live, (float(xs[j]), float(ys[j])), reach)
                u, v = self.live.uv(*goal)
                if (not (0 <= u < self.live.W and 0 <= v < self.live.H)
                        or self.visited[v, u] or math.dist((x, y), goal) < TOO_CLOSE_M
                        or any(math.dist(goal, g) <= current().drive.tolerance_m for g in excluded)):
                    continue
                route, radius = plan_padded(self.live, (x, y), goal, self.planner.radius_m,
                                            cancelled=self.planning_cancelled)
                if route is not None:
                    continuation.update(walk_alternative_status="validated alternative available",
                                        reason="a distinct useful walking alternative is validated",
                                        validated_alternative={
                                            "goal_xy_m": list(goal), "frame": "exploration_start",
                                            "radius_m": radius})
                    return continuation
        return continuation if reassessment else None

    def _record_skill(self, result) -> None:
        """Record the result of a model-called skill that does not move the base."""
        self.last_skill_result = {"skill": result.skill, "code": result.code,
                                  "message": result.message}

    def _record_action(self, kind, goal, outcome, failure=None) -> None:
        """Record and save the result of one action."""
        self._gallery_not_before = time.monotonic() + current().iphone.post_action_buffer_s
        result = {"kind": kind, "outcome": outcome, "t": round(self.elapsed(), 3),
                  "goal": None if goal is None else [float(v) for v in goal],
                  "pose_xyyaw": [float(v) for v in self.source.pose()],
                  "code": getattr(failure, "code", None),
                  "why": str(failure) if failure is not None else "action completed",
                  "details": getattr(failure, "details", None) or {}}
        if outcome == "arrived" and kind != "look" and goal is not None:
            drive_result = getattr(self.source, "_last_drive_result", None)
            if drive_result and math.dist(drive_result["goal_xy"], goal) < 1e-6:
                result["details"] = deepcopy(drive_result)
        if kind == "look" and outcome == "arrived":
            turn_result = getattr(self.source, "_last_turn_result", None)
            if turn_result is not None:
                result["details"] = deepcopy(turn_result)
        result = jsonable(result)
        self.previous_action_result, self.last_skill_result = result, None
        if outcome == "arrived" and kind != "look":
            tolerance = current().drive.tolerance_m
            self.failed_goals[:] = [r for r in self.failed_goals if r.get("pose_xyyaw") and
                math.dist(r["pose_xyyaw"][:2], result["pose_xyyaw"][:2]) <= tolerance]
        elif outcome == "refused" and goal is not None and kind != "look":
            self.failed_goals.append({"goal": result["goal"], "why": result["why"],
                                      "code": result["code"],
                                      "pose_xyyaw": result["pose_xyyaw"],
                                      "route_xy": result["details"].get("route_xy")})
        if (kind not in {"home", "recovery"} and self.vlm_picks
                and self.vlm_picks[-1].get("outcome") is None):
            rec = self.vlm_picks[-1]
            rec.update(outcome=outcome, action_result=result)
            path = self.out_dir / "vlm" / f"leg_{rec['n']}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                saved.update(outcome=outcome, action_result=result)
                path.write_text(json.dumps(saved, indent=1))

    def failed_goal_reason(self, goal, route) -> str:
        """Why GOAL along ROUTE repeats a failed destination, or ''."""
        for failed in self.failed_goals:
            if math.dist(goal, failed["goal"]) > current().drive.tolerance_m:
                continue
            previous = failed.get("route_xy")
            if previous and route and vlm_pick.route_changed(previous, route, current().drive.tolerance_m):
                continue
            return (f"previously failed destination within {current().drive.tolerance_m:.2f} m: "
                    f"{failed['why']}; choose another position, a useful look, or request_help")
        return ""

    def _camera_context(self):
        """Sample the phone camera while standing between actions; the selected frame and report."""
        live = self.live
        history, epoch = self.visual_history, self.visual_history.epoch
        guard = getattr(self.source, "stationary_guard", None)
        generation = guard() if guard is not None else None
        buffer_started = time.monotonic()
        previous_stage, previous_message = self.stage, self.message
        while self._gallery_not_before:
            self.check()
            if (self.live is not live or self.visual_history is not history
                    or history.epoch != epoch):
                raise Abort("start frame changed during phone buffer")
            if guard is not None:
                generation = guard(generation)
            remaining = self._gallery_not_before - time.monotonic()
            if remaining <= 0:
                break
            self.stage, self.message = "waiting for camera", (
                f"Holding while the delayed camera video catches up ({max(0., remaining):.1f} s left).")
            self.source.stop_flag.wait(max(0., min(.1, remaining)))
        buffer_wait_s = time.monotonic() - buffer_started
        self.stage, self.message = previous_stage, previous_message
        if self._gallery_not_before:
            self.decision_pose()  # recheck freshness after the buffer wait
            self._gallery_not_before = 0.0
        started, samples = time.monotonic(), []
        refused = (self.previous_action_result or {}).get("outcome") == "refused"
        limit = self.budget_s
        window = max(0., min(1.2, limit - self.active_elapsed()))
        selected = report = None
        for i in range(7):
            if i:
                self.source.stop_flag.wait(max(
                    0., started + min(i * .2, window) - time.monotonic()))
            self.check()
            if (self.live is not live or self.visual_history is not history
                    or history.epoch != epoch):
                raise Abort("start frame changed during phone sampling")
            if guard is not None:
                generation = guard(generation)
            rgb, metadata = self.source.context_image()
            samples.append({"rgb": rgb, "metadata": metadata, "pose": self.source.pose(),
                            "sampled_at": time.time(), "odom_age_s": self.source.odom_age_s()})
            selected = report = None
            elapsed = time.monotonic() - started
            # sample until the window closes
            if elapsed >= window - 1e-6:
                break
        if selected is None:
            selected, report = select_context_frame(samples[-3:])
        self.check()
        if (self.live is not live or self.visual_history is not history
                or history.epoch != epoch):
            raise Abort("start frame changed during phone selection")
        if guard is not None:
            guard(generation)
        self._gallery_sample, gallery_report = select_context_frame(samples, retained=True)
        self._gallery_sample = self._gallery_sample | {
            "metadata": self._gallery_sample["metadata"] | {"selection": gallery_report}}
        report.update(gallery_selection=gallery_report,
                      window_s=time.monotonic() - started, candidate_count=len(samples),
                      sampled_frame_count=len(samples), resampled_after_refusal=refused,
                      post_action_buffer_wait_s=buffer_wait_s)
        return selected, report

    def _observation_readiness(self, context_meta):
        """The current cached input state."""
        source = self.source
        map_age = self.map_age_s()
        odom_age = getattr(source, "odom_age_s", lambda: 0.)()
        scan_t = getattr(source, "ring_t", None)
        scan_age = None if scan_t is None else time.time() - scan_t
        health = getattr(source, "health_stable_s", lambda: math.inf)()
        camera_ready = getattr(source, "context_ready",
                               lambda: context_meta.get("status") == "fresh")()
        check = getattr(getattr(source, "backend", None), "observation_refusal", None)
        refusal = None if check is None else check()
        map_ready = map_age is not None and 0 <= map_age <= CAMERA_MAP_MAX_AGE_S
        inputs_ready = (map_ready and 0 <= odom_age <= .5 and camera_ready
                        and health >= DECISION_FRESH_STABLE_S and refusal is None)
        def finite(value):
            return value if value is not None and math.isfinite(value) else None

        return {"checked_at_s": time.time(), "decision_inputs_ready": inputs_ready,
                "motion_observations_ready": None if check is None else refusal is None,
                "planning_map_ready": map_ready, "rgb_ready": camera_ready,
                "odom_age_s": finite(odom_age), "scan_age_s": finite(scan_age),
                "map_age_s": finite(map_age), "health_stable_s": finite(health),
                "required_stable_s": DECISION_FRESH_STABLE_S,
                "blocking_observation": None if refusal is None else {
                    "reason": str(refusal), "details": getattr(refusal, "details", {})},
                "freshness": getattr(source, "freshness_status", dict)()}

    def _recent_proposal_refusals(self):
        """Recent proposal refusals in this task scope, as context only."""
        current = self.task_scope_context()
        recent = []
        for rec in reversed(self.vlm_picks):
            context = rec.get("proposal_context", {})
            if (not rec.get("rejection") or any(context.get(k) != current.get(k)
                    for k in ("generation", "visual_epoch", "brief", "ownership"))):
                continue
            details = rec.get("rejection_details") or {}
            version = details.get("map_version", context["map_version"])
            recent.append({
                "decision": rec["n"], "action": rec["kind"],
                "requested_xy_m": rec.get("xy"), "code": rec.get("rejection_code"),
                "why": rec["rejection"], "diagnosis": details.get("code"),
                "frame": "exploration_start", "frame_generation": context["generation"],
                "region_revision": context["revision"],
                "map_version": version,
                "permission_changed_since": context["revision"] != current["revision"],
                "map_changed_since": version != self.live.version,
                **{key: details[key] for key in (
                    "radius_m", "outside_map", "scoped_refusal", "scoped_route_available",
                    "unscoped_route_available") if key in details},
            })
            if len(recent) == 4:
                break
        return list(reversed(recent))

    def _vlm_leg(self, x: float, y: float, yaw: float, *, rejection: str = "",
                 continuation=None):
        return self._session().resolve_context(decision(self, x, y, yaw, rejection=rejection,
                                                        continuation=continuation))

    def _geometric_stop(self, reason: str) -> str:
        return f"stopped: {reason}; the camera views were not checked for coverage"

    def _run(self) -> None:
        self._session().run_exploration(safe=False)

    def _skeleton_xy(self) -> list:
        skel = getattr(self.planner, "skeleton", None)
        if skel is None:
            return []
        v, u = np.nonzero(skel)
        xs, ys = self.live.xy(u, v)
        return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(xs[::2], ys[::2])]

    def _commit_visual_history(self, owner, snapshot, live, *, persist=False):
        """Publish SNAPSHOT as the visual history if LIVE is still the map frame."""
        staged = snapshot.stage_index() if persist else None
        published = snapshot.snapshot()
        try:
            with self.lock:
                if self.live is not live:
                    raise Abort("map frame changed while saving visual context; holding position")
                if self.visual_history is not owner or owner.epoch != snapshot.epoch:
                    raise Stopped("start frame changed while saving visual context")
                if staged is not None:
                    staged.replace(snapshot.directory / "index.json")
                self.visual_history = published
                return self.visual_history
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)

    def save_map(self) -> None:
        """Save the map export; one save at a time, without blocking scan ingestion."""
        with self._map_save_lock:
            with self.lock:
                owner = self.live
                frozen = owner.snapshot()
            fd, temporary = tempfile.mkstemp(prefix=".live_map.", suffix=".npz", dir=self.out_dir)
            os.close(fd)
            try:
                frozen.write_artifact(temporary)
                with self.lock:
                    if self.live is owner:
                        os.replace(temporary, self.out_dir / "live_map.npz")
            finally:
                Path(temporary).unlink(missing_ok=True)

    def _save(self) -> None:
        self.save_map()
        cov = self.coverage
        (self.out_dir / "run.json").write_text(
            json.dumps(
                {
                    "phase": self.phase,
                    "stop_reason": self.stop_reason,
                    "elapsed_s": round(self.elapsed(), 1),
                    "active_elapsed_s": round(self.active_elapsed(), 3),
                    "paused_elapsed_s": round(self.paused_elapsed(), 3),
                    "budget_s": self.budget_s,
                    "t0_wall": self.t0_wall,
                    "ended_wall_s": self.ended_wall_s,
                    "enclosure": self.enclosure,
                    "stops": self.stops,
                    "coverage": cov,
                    "boundary": self.boundary.points,
                    "zone": None if self.zone is None else self.zone.points,
                    "task_region": self.task_scope_context(),
                    "operator_questions": self.operator_questions,
                    "radius_m": self.radius,
                    "requested_radius_m": RADIUS_M,
                    "checked_recovery": True,
                    "source": self.source.name,
                    "trail": [list(map(float, p)) for p in self.trail[::5]],
                    "next": None if self.next_wp is None else asdict(self.next_wp),
                    "voronoi_edges": [
                        {
                            "length_m": round(e["length_m"], 2),
                            "unknown": round(e["unknown"], 2),
                            "walked": round(
                                float(self.visited[e["cells"][:, 0], e["cells"][:, 1]].mean()), 2
                            ),
                        }
                        for e in getattr(self.planner, "edges", [])
                    ],
                    "routes": getattr(self, "routes", []),
                    "refusals": getattr(self, "refusals", []),
                    "previous_action_result": self.previous_action_result,
                    "failed_goals": self.failed_goals,
                    "mapped_fraction": getattr(self, "mapped", None),
                    "walked_fraction": getattr(self, "walked", None),
                    "plan_s": getattr(self, "plan_s", []),
                    "vlm": {"picks": self.pick_hook is not None, "legs": self.vlm_picks,
                            "visual_history": self.visual_history.manifest(),
                            "provider": current().agent.provider, "model": vlm_pick.model_name(),
                            "brief": self._current_brief(),
                            "pictures": "vlm/leg_<n>_camera.png, leg_<n>_history_<k>.png, "
                                        "leg_<n>.png and leg_<n>_local.png (as sent)"},
                    "skeleton_xy": self._skeleton_xy(),
                    "saved_at": time.time(),
                },
                indent=1,
            )
        )
