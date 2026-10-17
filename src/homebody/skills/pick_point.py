"""Checked live translation, with route preparation separate from body execution."""
import math
from dataclasses import dataclass

import numpy as np

from homebody.helpers.floor import (
    TOO_CLOSE_M,
    Waypoint,
    click_note,
    goal_adjustment,
    reachable_network,
    snap_to_network,
)
from homebody.helpers.raster import grid_to_px, to_metres, valid_grid
from homebody.primitives.navigation.errors import Abort, ActionRefused, DecisionInterrupted, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions
from homebody.real2sim.exploration import settings

from .contract import Result, SkillFailure

SURFACE = LiveActions
NEEDS = ("walk",)
FAILURES = LIVE_FAILURES
PROMPT = ('pick_point: x and y, a click on the map named by map_view ("overview" or "local", '
          'default overview): walk to that floor, which must be reachable inside the task region by '
          'a corridor wide enough for the planning footprint; face_x and face_y, optional, a point on '
          'the same map to face after arriving. why, target_area and expected_gain say what the view '
          'should add.')
PARAMETERS = {"type": "object", "properties": {
    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
    "y": {"type": "integer", "minimum": 0, "maximum": 1000},
    "face_x": {"type": "integer", "minimum": 0, "maximum": 1000},
    "face_y": {"type": "integer", "minimum": 0, "maximum": 1000},
    "map_view": {"type": "string", "enum": ["overview", "local"]},
    "why": {"type": "string"}, "target_area": {"type": "string"},
    "expected_gain": {"type": "string"}},
    "required": ["x", "y", "why"], "additionalProperties": False}


@dataclass
class Plan:
    epoch: object
    waypoint: object
    points: list | None
    radius: float
    seconds: float
    request: dict
    look: object
    mps: float
    valid: bool = True

    @property
    def summary(self):
        return {"goal_xy": self.waypoint.xy, "route": self.points,
                "radius_m": self.radius, "seconds": self.seconds, "mps": self.mps,
                "has_facing": self.look is not None, "motion_request": self.request}


@dataclass
class AdmittedPoint:
    epoch: object
    waypoint: object
    near: tuple
    decision: int
    motion: dict | None = None
    valid: bool = True

    @property
    def summary(self):
        return {"decision": self.decision, "goal_xy": self.waypoint.xy,
                "coordinate_admission": True}


def prepare(arguments, selected, current, planning):
    if "admission" not in arguments:
        return _prepare_route(arguments, selected, current, planning)
    try:
        return _admit(arguments["admission"], current, planning)
    except (ActionRefused, Stopped, Abort) as error:
        planning.fail(error)


def _admit(payload, current, planning):
    world, routes = planning.world, planning.routes
    answer, rec = payload["answer"], payload["record"]
    decision_live, history = payload["frame"], payload["history"]
    T, rgb = payload["transform"], payload["image"]
    selection, continuation = payload["selection"], payload["continuation"]
    n, near = rec["n"], payload["near"]
    why, leg = "", None
    if (not valid_grid(answer.grid) or
            (answer.face_grid is not None and not valid_grid(answer.face_grid))):
        raise SkillFailure("ACTION_UNAVAILABLE", "invalid map coordinates; use the map's 0..1000 grid",
                           {"near": near})
    x, y, _yaw = world.decision_pose()
    if world.live is not decision_live:
        raise Abort("map frame changed during model decision; holding position")
    px = grid_to_px(answer.grid, rgb.shape)
    gx, gy = to_metres(T, *px)
    rec["px"], rec["xy"] = [round(px[0], 1), round(px[1], 1)], [round(gx, 3), round(gy, 3)]
    near = (gx, gy)
    face_xy = None  # the model's second point: what the robot faces at the goal
    if answer.face_grid is not None:
        fpx = grid_to_px(answer.face_grid, rgb.shape)
        face_xy = to_metres(T, *fpx)
        rec["face_px"] = [round(fpx[0], 1), round(fpx[1], 1)]
        rec["face_xy"] = [round(face_xy[0], 3), round(face_xy[1], 3)]
    world.refresh_planning_radius()
    with world.lock:
        if world.live is not decision_live:
            raise Abort("map frame changed during model admission; holding position")
        rec["click_on"] = click_note(world.live, world.boundary, (gx, gy))
        network, reach = reachable_network(world.planner, x, y)
        shown_planner, shown_network = selection or (world.planner, network)
        if not world.live.task_region_contains(gx, gy):
            why = "requested destination is outside the accepted task boundary"
            rec["rejection_code"] = "task_scope"
            snapped = None
        else:
            route = None
            u, v = world.live.uv(gx, gy)
            requested_checked = (
                world.live._in_grid(u, v) and world.live.known_free()[v, u]
                and math.dist((x, y), (gx, gy)) >= TOO_CLOSE_M)
            if requested_checked:
                route, _ = routes.plan_padded(
                    world.live, (x, y), (gx, gy), world.planner.radius_m,
                    cancelled=world.planning_cancelled)
            if route is not None:
                snapped = ((gx, gy), 0., math.atan2(gy-y, gx-x))
                rec["selection_basis"] = "requested_observed_floor"
            else:
                nearby = routes.nearest_reachable_floor(
                    world.planner, (gx, gy), x, y, reach=reach,
                    cancelled=world.planning_cancelled)
                if nearby is not None:
                    goal, route = nearby
                    snapped = (goal, math.dist(goal, (gx, gy)),
                               math.atan2(goal[1]-y, goal[0]-x))
                    rec["selection_basis"] = "nearby_reachable_floor"
                else:
                    snapped = snap_to_network(
                        shown_planner, (gx, gy), x, y, face_xy, network=shown_network)
                    rec["selection_basis"] = "network_padding"
        if snapped is None and not why:
            failure = routes.route_refusal(
                world.live, (x, y), radius_m=world.planner.radius_m,
                cancelled=world.planning_cancelled)
            if failure is not None:
                why = failure["why"]
                rec.update(rejection_code=failure["code"], rejection_details=failure)
            else:
                why = "no walkable path network to snap to"
                rec["rejection_code"] = "no_network"
        elif snapped is not None:
            spot, snap_m, face = snapped
            goal = spot if route is not None else routes.padded_spot(world.live, spot, reach)
            distance, why = goal_adjustment((gx, gy), goal)
            if why:
                rec["rejection_code"] = "goal_adjustment"
                diagnostic = routes.goal_adjustment_diagnostics(
                    world.live, (x, y), (gx, gy), goal, world.planner.radius_m,
                    cancelled=world.planning_cancelled)
                rec["rejection_details"] = diagnostic
                why += "; " + diagnostic["why"]
            rec["proposed_goal"] = [round(float(v), 3) for v in goal]
            rec["goal_adjustment_m"] = None if distance is None else round(distance, 3)
            rec["snap_xy"] = [round(spot[0], 3), round(spot[1], 3)]
            rec["snap_m"] = round(snap_m, 3)
            if continuation and not why and any(
                    math.dist(goal, r["proposed_goal"]) <= settings.current().drive.tolerance_m
                    for r in continuation["rejections"]
                    if r.get("proposed_goal") is not None):
                why = ("proposal repeats a refused destination; "
                       "choose a distinct viewpoint")
                rec["rejection_code"] = "repeated_proposal"
            if not why:
                if route is None and (not requested_checked or goal != (gx, gy)):
                    route, _ = routes.plan_padded(world.live, (x, y), goal,
                                                  world.planner.radius_m,
                                                  cancelled=world.planning_cancelled)
                if route is None:
                    failure = routes.route_refusal(world.live, (x, y), goal,
                                                   world.planner.radius_m,
                                                   cancelled=world.planning_cancelled)
                    why = ("displayed destination is not currently reachable: "
                           + failure["why"]) if selection else failure["why"]
                    rec.update(rejection_code=failure["code"], rejection_details=failure)
                elif history is not None:
                    why = world.failed_goal_reason(goal, route)
    if not why:
        leg = ("vlm", Waypoint("vlm", goal, [face] if history is None else []), None)
        rec["used"] = True
        rec["snap_xy"] = [round(spot[0], 3), round(spot[1], 3)]
        rec["snap_m"] = round(snap_m, 3)
        rec["face_yaw"] = round(face, 3) if history is None else None
        world.emit("pending_face", intention=(None if history is None or face_xy is None else {
            "target": face_xy, "frame": decision_live,
            "revision": world.live.task_region_revision, "decision": n}))
    if why:
        raise SkillFailure("ACTION_UNAVAILABLE", why, {"near": near})
    return AdmittedPoint(current.epoch, leg[1], near, n)


def _prepare_route(arguments, selected, current, planning):
    world, routes = planning.world, planning.routes
    wp, look, mps = arguments["waypoint"], arguments.get("look"), arguments["mps"]
    try:
        while True:
            try:
                world.require_task_scope()
                request = world.refresh_planning_radius()
                x, y, _ = world.decision_pose()
                with world.lock:
                    if not world.live.task_region_contains(*wp.xy):
                        failure = routes.route_refusal(
                            world.live, (x, y), wp.xy, world.planner.radius_m,
                            cancelled=world.planning_cancelled)
                        raise ActionRefused(
                            "requested destination is outside the accepted task boundary",
                            code="TASK_BOUNDARY", details=failure)
                    if wp.kind == "vlm":
                        goal = wp.xy
                    else:
                        reach = np.isfinite(world.live.geodesic(x, y, world.planner.radius_m))
                        goal = routes.padded_spot(world.live, wp.xy, reach)
                    path, radius = routes.plan_padded(
                        world.live, (x, y), goal, world.planner.radius_m,
                        cancelled=world.planning_cancelled)
                    failure = (routes.route_refusal(
                        world.live, (x, y), goal, world.planner.radius_m,
                        cancelled=world.planning_cancelled) if path is None else None)
            except DecisionInterrupted:
                world.check()
                continue
            break
        if path is None:
            if failure is not None and failure["code"] in {"task_boundary", "task_scope_unresolved"}:
                raise ActionRefused(failure["why"], code="TASK_BOUNDARY", details=failure)
            if wp.kind == "vlm":
                raise ActionRefused("selected model goal is no longer reachable",
                                    code="NO_COLLISION_FREE_PATH")
            return Plan(current.epoch, wp, None, radius, 0., request, look, mps)
        points = routes.route_points(path)
        world.emit("route", points=points, waypoint=wp, radius=radius)
        seconds = world.leg_seconds(points)
        return Plan(current.epoch, wp, points, radius, seconds, request, look, mps)
    except (ActionRefused, Stopped, Abort) as error:
        planning.fail(error)


def execute(plan, ctx):
    world = ctx.world
    if not plan.valid:
        raise SkillFailure("INVALID_ARGUMENT", "prepared route was discarded")
    if isinstance(plan, AdmittedPoint):
        if plan.motion is None:
            raise SkillFailure("INVALID_ARGUMENT", "admitted point has no execution request")
        frame = ctx.frame()
        plan = _prepare_route({"waypoint": plan.waypoint, **plan.motion},
                              frame, frame, ctx.planning())
    if plan.points is None:
        return Result("pick_point", "OK", "No route was selected.", {"reached": False})
    try:
        world.emit("status", stage="driving")
        intention = world.pending_face
        if intention is not None:
            intention["deadline"] = world.active_monotonic() + plan.seconds
            intention["motion_epoch"] = world.motion_epoch()
            intention["authority"] = world.stationary_guard()
        with world.action_capture(plan.waypoint.kind, route=plan.points):
            ctx.actions.walk(plan.points, look=plan.look, mps=plan.mps, radius_m=plan.radius,
                             seconds=plan.seconds, **plan.request)
        world.check()
        return Result("pick_point", "OK", "The walk finished.", {"reached": True})
    except (ActionRefused, Stopped, Abort) as error:
        ctx.fail(error)


def abort(plan, ctx):
    """Discard this route; the synchronous motion primitive has already stopped."""
    if plan is not None:
        plan.valid = False
