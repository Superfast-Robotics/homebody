"""The dashboard's immutable records and the JSON-safe conversion of session values into them."""
import json
import math
from dataclasses import dataclass, field

TEXT_LIMIT = 2000  # characters of model or result text kept per record
ARGUMENT_LIMIT = 4000  # characters of a decision's JSON arguments kept


@dataclass(frozen=True)
class Interval:
    """A span on one timeline track (goal, skill or thinking); END is None while it runs."""
    id: str
    track: str
    label: str
    start: float
    end: float | None = None
    status: str = "running"
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    id: str
    t: float
    skill: str
    text: str
    arguments: dict
    goal: str | None = None
    tool: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """One decision the executor ran: a skill or a coder call, with its stages and result."""
    id: str
    skill: str
    arguments: dict
    start: float
    goal: str | None = None
    end: float | None = None
    state: str = "running"
    code: str = ""
    message: str = ""
    stages: tuple = ()
    attempt: int = 1  # this skill's call number within its goal
    recovery: bool = False  # run by a recovery branch or the coding assistant
    call: int | None = None  # the task's skill call number the executor and the monitor share (None: not reported)
    number: int = 1  # its goal's attempt number: its goal's calls so far, this one included
    execution: str = ""  # the attempt's identity: "<task id>/call-<call>", or "<task id>/<tool id>" without a call


@dataclass(frozen=True)
class Incident:
    """A recorded event. KIND is a detection (skill, goal, task, monitor) or an action (intervention, recovery),
    with TAG naming the action. ALERT marks the ones that need an operator (an intermediate skill failure is history,
    not an alert). DETECTION, INITIATOR and EXECUTOR say which component found the problem, asked for the recovery
    and ran it, only as the session's events establish them ("" when unknown). CAUSED_BY links an event to the one
    that caused it, only where the runtime makes that link; OUTCOME is what followed, filled in when it happens."""
    id: str
    t: float
    kind: str
    severity: str
    title: str
    reason: str = ""
    source: str = "session"
    node: str | None = None
    tool: str | None = None
    skill: str | None = None
    action: str = ""
    outcome: str = ""
    observation: dict | None = None
    tag: str = ""
    alert: bool = False
    task: str | None = None
    detection: str = ""
    initiator: str = ""
    executor: str = ""
    caused_by: str | None = None
    frame: int | None = None
    state: str = "active"
    acknowledged_at: float | None = None
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    """One monitor reading. An observation judges behaviour (LEVEL expected, unknown, elevated or anomaly; RISK
    and CONFIDENCE in [0, 1] when reported); a heartbeat only shows the monitor is connected. SYNTHETIC marks test
    data, never shown as a real outcome. T is when the monitor made the observation (its own stamp when it gives
    one), RECEIVED when the dashboard got it. ELEVATED_THRESHOLD, when the monitor reports one, separates elevated from
    low risk. DURING_TASK, NODE and SKILL are what ran when it arrived."""
    t: float
    level: str
    source: str
    risk: float | None = None
    threshold: float | None = None
    confidence: float | None = None
    reason: str = ""
    kind: str = "observation"
    received: float | None = None
    elevated_threshold: float | None = None
    synthetic: bool = False
    during_task: bool = False
    node: str | None = None
    skill: str | None = None
    signal: str = ""  # what RISK measures, as the monitor names it (e.g. "stall", "unsafe", "failure:dropped")
    call: int | None = None  # the skill call of its task it watched (Phase.call), None between skills or unknown
    tool: str | None = None  # the attempt with that call, as the store matched it (None: no match, never guessed)
    frame_quality: str = "unknown"  # the robot camera's frame quality at the observation (ui/dashboard/frames.py)
    id: str = ""  # the observation's identity on this dashboard, set by the store
    sim_time: float | None = None  # simulation time of the newest frame the monitor read (None: not reported)
    lag_s: float | None = None  # how far behind the newest frame the monitor read when it was taken
    scope: str = "unknown"  # what it is about, by the monitor's rules: skill_attempt, goal, task, robot, unknown
    scope_goal: str | None = None  # the goal of a goal-scoped (or attempt-scoped) reading
    semantics: str = ""  # what RISK is, in the monitor's own terms ("" when it did not say)
    confidence_semantics: str = ""  # what CONFIDENCE is
    calibration: str = "unknown"  # whether the scores are calibrated probabilities: calibrated, uncalibrated, unknown
    reliability: str = "unknown"  # an independent check of the assessment: reliable, unreliable, unknown
    input_source: str = ""  # what the monitor read
    phase: str = ""  # what the harness was doing, as the monitor saw it (idle, planning, tool_call)
    quality_source: str = ""  # how FRAME_QUALITY was established (which frame, checked by what)
    quality_relation: str = "unknown"  # the checked frame to the detector: detector_input, synchronized, receipt, unknown
    quality_reason: str = ""
    activity: str = ""  # what the decision model sees the robot doing: the held label (monitor/choice.py)
    activity_p: float | None = None  # that label's probability in this reading
    activity_votes: int = 0  # readings of the window in which it was the most probable
    activity_window: int = 0


@dataclass(frozen=True)
class Intervention:
    """A request that a component stop or change what the robot does, and what became of it. STAGES are the
    events that established each step, oldest first: (stage, t, source, note), with stage one of "detected",
    "decided", "requested", "accepted", "confirmed", "rejected", "failed", "cancelled", then "retry_started" and
    "retry_finished" when the plan's goal tried again; a step no event established is absent, never assumed. INITIATOR asked for it, DECIDED_BY chose to, TARGET was asked, INCIDENT is the detection it acted
    on. PHYSICAL is whether the robot was seen to stop: "unverified" unless a motion or safety signal says so."""
    id: str
    t: float
    action: str
    initiator: str
    decided_by: str = ""
    detection: str = ""
    target: str = "Skill executor (session)"
    tool: str | None = None
    skill: str | None = None
    node: str | None = None
    incident: str | None = None
    task: str | None = None
    reason: str = ""
    stages: tuple = ()
    physical: str = "unverified"
    execution: str = ""  # the attempt's identity (ToolCall.execution); "" when no attempt could be matched
    attempt: int | None = None  # its goal's attempt number


@dataclass(frozen=True)
class Instruction:
    id: str
    t: float
    text: str
    state: str
    reason: str = ""


@dataclass(frozen=True)
class Log:
    id: str
    t: float
    severity: str
    source: str
    message: str
    node: str | None = None


LEVELS = ("expected", "unknown", "elevated", "anomaly")
KINDS = ("observation", "heartbeat")
SCOPES = ("skill_attempt", "goal", "task", "robot", "environment", "unknown")
CALIBRATIONS = ("calibrated", "uncalibrated", "unknown")
RELIABILITIES = ("reliable", "unreliable", "unknown")
RELATIONS = ("detector_input", "synchronized", "receipt", "unknown")


def clip(text, limit=TEXT_LIMIT) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def plain(value):
    """VALUE as JSON-safe data: arrays become lists, non-finite floats None, anything else its text."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def arguments(value) -> dict:
    """A decision's arguments, JSON-safe and bounded; oversized arguments keep only their text."""
    if not isinstance(value, dict):
        return {}
    data = {str(key): plain(item) for key, item in value.items()}
    text = json.dumps(data)
    return data if len(text) <= ARGUMENT_LIMIT else {"truncated": clip(text, ARGUMENT_LIMIT)}


def unit(value: dict, name: str):
    number = value.get(name)
    if number is None:
        return None
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0. <= number <= 1.:
        raise ValueError(f"reading {name} must be a number in [0, 1]")
    return float(number)


def reading(value: dict, now: float, **context) -> Reading:
    """A monitor reading from its adapter's dict, with CONTEXT (during_task, node, skill) from the store; refuses
    unknown levels or kinds and numbers outside [0, 1]."""
    level, kind = value.get("level", "unknown"), value.get("kind", "observation")
    if level not in LEVELS:
        raise ValueError(f"reading level must be one of {', '.join(LEVELS)}")
    if kind not in KINDS:
        raise ValueError(f"reading kind must be one of {', '.join(KINDS)}")
    return Reading(t=float(value.get("t", now)), level=level, source=clip(value.get("source", "monitor"), 80),
                   risk=unit(value, "risk"), threshold=unit(value, "threshold"),
                   confidence=unit(value, "confidence"), reason=clip(value.get("reason", ""), 400), kind=kind,
                   received=now, elevated_threshold=unit(value, "elevated_threshold"),
                   synthetic=bool(value.get("synthetic", False)), signal=clip(value.get("signal", ""), 80),
                   call=call_number(value.get("call")),
                   frame_quality=clip(value.get("frame_quality", "unknown"), 24),
                   sim_time=number(value.get("sim_time")), lag_s=number(value.get("lag_s")),
                   scope=choice(value, "scope", SCOPES), scope_goal=text(value.get("scope_goal"), 80),
                   semantics=clip(value.get("semantics", ""), 300),
                   confidence_semantics=clip(value.get("confidence_semantics", ""), 200),
                   calibration=choice(value, "calibration", CALIBRATIONS),
                   reliability=choice(value, "reliability", RELIABILITIES),
                   input_source=clip(value.get("input_source", ""), 200), phase=clip(value.get("phase", ""), 24),
                   quality_source=clip(value.get("quality_source", ""), 200),
                   quality_reason=clip(value.get("quality_reason", ""), 200),
                   quality_relation=choice(value, "quality_relation", RELATIONS),
                   activity=clip(value.get("activity", ""), 80), activity_p=unit(value, "activity_p"),
                   activity_votes=count(value.get("activity_votes")), activity_window=count(value.get("activity_window")),
                   **context)


def number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def text(value, limit) -> str | None:
    return None if value is None else clip(value, limit)


def choice(value: dict, name: str, allowed: tuple) -> str:
    """VALUE's NAME when it is one of ALLOWED, else "unknown" (an unrecognized claim is not passed on as fact)."""
    found = value.get(name, "unknown")
    return found if found in allowed else "unknown"


def count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 64 else 0


def call_number(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None
