"""The all-round survey turn, run by the explorer through the skill lifecycle."""
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions

from . import viewing
from .contract import Result
from .live import interruptions

SURFACE = LiveActions
NEEDS = ("turn_hold",)
OPTIONAL = ("step_to",)
FAILURES = LIVE_FAILURES
PROMPT = "survey: Turn in place through a full circle of settled views; not a model action."


def prepare(arguments, selected, current, planning):
    return {"label": arguments.get("label", "start spin")}


def execute(plan, ctx):
    with interruptions(ctx.fail):
        viewing.survey(ctx, **plan)
        return Result("survey", "OK", "The look-around finished.")
