"""Checked viewing turn through the shared skill lifecycle."""
import math

from homebody.helpers.floor import FACE_MIN_M
from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions

from . import viewing
from .contract import Result
from .live import interruptions

SURFACE = LiveActions
NEEDS = ("turn_hold",)
FAILURES = LIVE_FAILURES
PROMPT = ('look_at: x and y, a click on the map named by map_view: turn in place to face that point, '
          'without walking. why, target_area and expected_gain say what the view should add.')
PARAMETERS = {"type": "object", "properties": {
    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
    "y": {"type": "integer", "minimum": 0, "maximum": 1000},
    "map_view": {"type": "string", "enum": ["overview", "local"]},
    "why": {"type": "string"}, "target_area": {"type": "string"},
    "expected_gain": {"type": "string"}},
    "required": ["x", "y", "why"], "additionalProperties": False}


def prepare(arguments, selected, current, planning):
    if arguments.get("arrival"):
        return {"arrival": True}
    return {"yaw": float(arguments["yaw"]),
            "recover_blocked": bool(arguments.get("recover_blocked", False))}


def execute(plan, ctx):
    with interruptions(ctx.fail):
        if plan.get("arrival"):
            result = arrival_facing(ctx) or {}
            code = result.get("code", "OK") if result.get("outcome") == "refused" else "OK"
            return Result("look_at", code, result.get("why", "The robot faces the requested direction."), result)
        viewing.face(ctx, **plan)
        return Result("look_at", "OK", "The viewing turn finished.", {"reached": True})


def arrival_facing(ctx):
    world = ctx.world
    intention = world.pending_face
    world.emit("pending_face", intention=None)
    if intention is None:
        return
    result = {"target": list(intention["target"]), "outcome": "cancelled"}
    try:
        world.check()
        world.stationary_guard(intention.get("authority"))
        if (world.live is not intention["frame"]
                or world.live.task_region_revision != intention["revision"]
                or world.motion_epoch() != intention.get("motion_epoch")):
            result["why"] = "frame, task boundary or motion authority changed"
            return result
        remaining = min(intention.get("deadline", 0.) - world.active_monotonic(),
                        world.remaining())
        if remaining <= 0:
            result["why"] = "arrival-facing deadline expired"
            return result
        x, y, _ = world.decision_pose()
        if (world.live is not intention["frame"]
                or world.live.task_region_revision != intention["revision"]):
            result["why"] = "frame or task boundary changed while waiting"
            return result
        remaining = min(intention["deadline"] - world.active_monotonic(),
                        world.remaining())
        if remaining <= 0:
            result["why"] = "arrival-facing deadline expired while waiting"
            return result
        world.stationary_guard(intention.get("authority"))
        fx, fy = intention["target"]
        if math.hypot(fx - x, fy - y) < FACE_MIN_M:
            result["why"] = "facing point too close to actual arrival pose"
            return result
        world.emit("status", stage="looking", message="Arrived; turning to the requested view.")
        with world.action_capture("arrival_facing"):
            ctx.actions.turn_hold(math.atan2(fy - y, fx - x), viewing.HOLD_S, seconds=remaining)
        world.check()
        result.update(outcome="arrived", why="The viewing turn finished.")
    except ActionRefused as exc:
        result.update(outcome="refused", why=str(exc), code=exc.code)
    except (Stopped, Abort) as exc:
        result.update(outcome="stopped", why=str(exc))
        raise
    finally:
        world.emit("facing", result=result)
    return result
