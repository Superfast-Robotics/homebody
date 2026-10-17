"""Checked departure repair after a refused proposal; never executes that proposal."""
from dataclasses import dataclass

from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions

from .contract import Result, SkillFailure

SURFACE = LiveActions
NEEDS = ("recover_departure",)
FAILURES = LIVE_FAILURES
PROMPT = "departure_recovery: Internal checked recovery after a rejected walking proposal."


@dataclass
class Plan:
    record: dict
    frame: object
    radius: float
    seconds: float
    request: dict
    valid: bool = True

    @property
    def summary(self):
        return {"decision": self.record["n"], "radius_m": self.radius,
                "seconds": self.seconds,
                "goal_xy": self.record.get("proposed_goal") or self.record.get("xy"),
                **self.request}


def prepare(arguments, selected, current, planning):
    world = planning.world
    try:
        world.check()
        frame = world.live
        world.emit("recovery_start")
        request = world.refresh_planning_radius()
        return Plan(arguments["record"], frame, world.radius,
                    min(30., world.remaining()), request)
    except (ActionRefused, Stopped, Abort) as error:
        planning.fail(error)


def execute(plan, ctx):
    if not plan.valid:
        raise SkillFailure("INVALID_ARGUMENT", "prepared departure recovery was discarded")
    world, rec = ctx.world, plan.record
    refusal = None
    try:
        try:
            result = ctx.actions.recover_departure(
                plan.radius, seconds=plan.seconds,
                goal_xy=rec.get("proposed_goal") or rec.get("xy"), **plan.request)
        except ActionRefused as why:
            result, refusal = None, why
        world.emit("recovery_result", result=result, refusal=refusal, record=rec)
        world.check()
        if world.live is not plan.frame:
            raise Abort("map frame changed during departure recovery; holding position")
        world.emit("recovery_record", record=rec)
        details = {"recovered": bool(result)}
        if refusal is not None:
            return Result("departure_recovery", refusal.code, str(refusal), refusal.details | details)
        return Result("departure_recovery", "OK", "The robot moved back to a safe starting spot.", details)
    except (ActionRefused, Stopped, Abort) as error:
        ctx.fail(error)


def abort(plan, ctx):
    """Discard the prepared request; synchronous recovery owns its stop."""
    if plan is not None:
        plan.valid = False
