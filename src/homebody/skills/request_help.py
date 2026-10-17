"""End an incomplete exploration attempt with an explicit request for assistance."""
from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions

from .contract import Result

SURFACE = LiveActions
NEEDS = ()
FAILURES = LIVE_FAILURES
PROMPT = ('request_help: why: hold and report incomplete exploration when no useful admissible '
          'action remains.')
PARAMETERS = {"type": "object", "properties": {"why": {"type": "string", "minLength": 1}},
              "required": ["why"], "additionalProperties": False}


class Plan:
    def __init__(self, why, record):
        self.why, self._record = why, record
        self.summary = {"decision": record["n"], "why": why}


def prepare(arguments, selected, current, planning):
    return Plan(arguments["why"], arguments["record"])


def execute(plan, ctx):
    try:
        ctx.world.decision_pose()
        ctx.world.emit("help", reason=f"incomplete: the model asked the operator for help: {plan.why}")
        plan._record.update(used=True, outcome="held")
        return Result("request_help", "OK", plan.why)
    except (ActionRefused, Stopped, Abort) as error:
        ctx.fail(error)
