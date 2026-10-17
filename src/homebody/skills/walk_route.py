"""Execute a classical route through the same checked live motion primitive."""
from dataclasses import dataclass

from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions

from .contract import Result, SkillFailure

SURFACE = LiveActions
NEEDS = ("walk",)
FAILURES = LIVE_FAILURES
PROMPT = "walk_route: Internal checked classical exploration route."


@dataclass
class Plan:
    points: list | None
    waypoint: object
    radius: float
    request: dict
    mps: float
    valid: bool = True


def prepare(arguments, selected, current, planning):
    world, routes = planning.world, planning.routes
    route, wp, mps = arguments["route"], arguments["waypoint"], arguments["mps"]
    try:
        world.require_task_scope()
        if not route or len(route) < 2:
            return Plan(None, wp, 0., {}, mps)
        points = routes.route_points(route)
        world.emit("route", points=points, waypoint=wp, radius=world.radius)
        request = world.refresh_planning_radius()
        radius = max(world.planner.last_leg.get("radius") or world.radius, world.radius)
        world.emit("radius", radius=radius)
        return Plan(points, wp, radius, request, mps)
    except (ActionRefused, Stopped, Abort) as error:
        planning.fail(error)


def execute(plan, ctx):
    if not plan.valid:
        raise SkillFailure("INVALID_ARGUMENT", "prepared classical route was discarded")
    if plan.points is None:
        return Result("walk_route", "OK", "No route was selected.", {"reached": False})
    world = ctx.world
    try:
        world.emit("status", stage="driving")
        with world.action_capture(plan.waypoint.kind, route=plan.points):
            ctx.actions.walk(plan.points, mps=plan.mps, radius_m=plan.radius,
                             seconds=world.leg_seconds(plan.points), **plan.request)
        world.check()
        return Result("walk_route", "OK", "The route walk finished.", {"reached": True})
    except (ActionRefused, Stopped, Abort) as error:
        ctx.fail(error)


def abort(plan, ctx):
    if plan is not None:
        plan.valid = False
