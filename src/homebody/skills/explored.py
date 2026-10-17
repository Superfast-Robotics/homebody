"""Accept completion only against the scoped visual-evidence contract."""
from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions
from homebody.real2sim.exploration.settings import current
from homebody.vlm.catalog import model_label

from .contract import Result

SURFACE = LiveActions
NEEDS = ()
FAILURES = LIVE_FAILURES
PROMPT = ('explored: why; coverage_summary; evidence_views, readable view ids; remaining_gaps, '
          'unfinished required coverage, empty to finish; limitations, optional: inaccessible '
          'surfaces, assumptions or exclusions, never an unseen requested area. Ends the exploration.')
PARAMETERS = {"type": "object", "properties": {
    "why": {"type": "string"}, "coverage_summary": {"type": "string"},
    "evidence_views": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    "remaining_gaps": {"type": "array", "items": {"type": "string"}},
    "limitations": {"type": "array", "items": {"type": "string"}}},
    "required": ["why", "coverage_summary", "evidence_views", "remaining_gaps"],
    "additionalProperties": False}


class Plan:
    def __init__(self, inputs):
        self._inputs = inputs
        self.summary = {"decision": inputs["record"]["n"], "walked_legs": inputs["walked_legs"]}


def prepare(arguments, selected, current, planning):
    return Plan(dict(arguments))


def execute(plan, ctx):
    try:
        inputs = plan._inputs
        answer, history, rec = inputs["answer"], inputs["history"], inputs["record"]
        walked_legs = inputs["walked_legs"]
        why = ctx.world.assess_completion(answer.completion, history)
        if not why:
            ctx.world.emit("completion", reason=(
                f"done: {model_label(current().agent)} judged the exploration complete after "
                f"{walked_legs} move{'' if walked_legs == 1 else 's'}. {answer.why}"))
            rec.update(used=True, outcome="accepted")
        else:
            rec["completion_refusal"] = why
            rec["rejection"] = why
        return Result("explored", "INVALID_ARGUMENT" if why else "OK",
                      why or "The exploration is complete.")
    except (ActionRefused, Stopped, Abort) as error:
        ctx.fail(error)
