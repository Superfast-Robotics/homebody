"""Wait for an operator answer without granting scope or restarting the attempt."""

from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions

from .contract import Result

SURFACE = LiveActions
NEEDS = ("hold_position",)
FAILURES = LIVE_FAILURES
PROMPT = ('ask_operator: question and why: hold and ask the operator one question about scope, '
          'direction or priority that the observations cannot settle; the answer informs a fresh '
          'decision.')
PARAMETERS = {"type": "object", "properties": {
    "question": {"type": "string", "minLength": 1}, "why": {"type": "string", "minLength": 1}},
    "required": ["question", "why"], "additionalProperties": False}


class Plan:
    def __init__(self, record):
        self._record = record
        self.summary = {"decision": record["n"], "question": record["question"]}


def prepare(arguments, selected, current, planning):
    return Plan(arguments["record"])


def execute(plan, ctx):
    try:
        ctx.ask(plan._record)
        return Result("ask_operator", "OK", "The operator answered.")
    except (ActionRefused, Stopped, Abort) as error:
        ctx.fail(error)


def action_refusal(answer, enabled=True):
    if not enabled:
        return "ask_operator is disabled; enable it with --enable-ask-operator at launch"
    if answer.task_region_update is not None:
        return "operator question cannot also change the region"
    if not isinstance(answer.question, str) or not answer.question.strip() or not answer.why:
        return "operator question requires nonempty question and why"
    return ""


def admit(answer, record):
    if not isinstance(answer.question, str) or not answer.question.strip() or not answer.why:
        return None, "operator question requires nonempty question and why"
    if answer.task_region_update is not None:
        return None, "operator question cannot also change the region"
    record.update(used=True, outcome="awaiting_answer")
    return ("ask_operator", None, None), ""
