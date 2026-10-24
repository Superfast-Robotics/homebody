"""The live monitor (homebody/monitor) on the dashboard: each new reading of its View becomes one dashboard reading.

The risk is the strongest of the signals the monitor's rules weigh (monitor/decide.signals), named in SIGNAL; the
level follows the verdict the reading got: an abort or a reopened goal is an anomaly (the dashboard raises an alert),
a flag or a risk at the threshold is elevated, a reading whose questions were all refused is unknown. REQUEST says the
verdict asked the skill executor to stop the skill: the monitor does so exactly when an abort or reopen falls on the
skill call it watched (Monitor._decide), so a verdict on a reading taken outside that call requests nothing.

Each reading also says what is known about it: SEMANTICS (what its score is: an uncalibrated answer probability of the
decision model, which the rules average before acting), SCOPE (what it is about, by the monitor's own rules: the skill
attempt it watched during a tool call, a goal while the agent plans or for a standing claim, else unknown), SIM_TIME
(the simulation time of the newest frame it read) and INPUT_SOURCE. The model's probabilities are not calibrated and
nothing checks the assessment independently, so CALIBRATION is "uncalibrated" and RELIABILITY "unknown".
"""
from homebody.monitor.decide import signals

SOURCE = "Live monitor"
ALERTS = ("abort_skill", "reopen_goal")  # the verdicts that stop a skill or reopen a goal
INPUT = "Robot cameras: the newest frames the monitor captured (head first), read by the decision model"
AVERAGED = "uncalibrated; the rules act on its running average, not on one reading"


def semantics(signal: str | None) -> str:
    """What a signal's score is, in the decision model's own terms (monitor/decision.py, monitor/decide.signals)."""
    if not signal:
        return "No score: the model answered none of the questions"
    if signal == "stall":
        return f"Probability the model answered 'no change' to the progress question across the frames ({AVERAGED})"
    if signal in ("unsafe", "human_needed"):
        return f"Probability the model answered yes to '{signal.replace('_', ' ')}' ({AVERAGED})"
    if signal.startswith("failure:"):
        return f"Probability the model chose the failure label '{signal[8:].replace('_', ' ')}' ({AVERAGED})"
    if signal.startswith("undone:"):
        return f"One minus the probability the model said goal {signal[7:]}'s claim still holds ({AVERAGED})"
    return f"Probability the model gave for {signal} ({AVERAGED})"


def scope(reading, signal: str | None) -> tuple[str, str | None]:
    """What the reading is about, by the monitor's rules (monitor/decide.observe): a standing claim is its goal's; during
    a skill call the reading watches that attempt (an abort stops only it); while the agent plans the rules only flag
    the goal; otherwise (idle) nothing the dashboard can name."""
    if signal and signal.startswith("undone:"):
        return "goal", signal[7:]
    if reading.phase.name == "tool_call" and reading.phase.call:
        return "skill_attempt", reading.goal_id
    if reading.phase.name == "planning":
        return "goal", reading.goal_id
    return "unknown", None


def strongest(reading, vocabulary) -> tuple[str | None, float | None]:
    """The reading's strongest signal and its probability (None, None when the model answered nothing)."""
    if vocabulary is not None:
        found = signals(reading, vocabulary)
    else:  # no labels yet: the questions that need none
        found = {name: getattr(reading, name) for name in ("unsafe", "human_needed")
                 if getattr(reading, name) is not None}
    if not found:
        return None, None
    name = max(found, key=lambda key: found[key])
    return name, float(found[name])


def verdict_for(view, reading):
    """The non-continue verdict this reading gave, if any."""
    return next((v for v in reversed(view.verdicts) if v.sim_time == reading.sim_time), None)


def feed(view, tau: float) -> dict | None:
    """VIEW's newest reading as a dashboard reading (ui/dashboard/records.reading), or None without one."""
    reading = view.reading
    if reading is None:
        return None
    this_call = reading.phase.name == "tool_call" and reading.phase.call == view.skill_call
    return value(reading, view.vocabulary, verdict_for(view, reading), tau, this_call=this_call)


def value(reading, vocabulary, verdict, tau: float, *, this_call: bool = False) -> dict:
    """READING (with the VERDICT it gave, if any) as a dashboard reading; THIS_CALL: taken during the skill call
    the monitor watches now, so an abort or reopen on it is a stop request."""
    name, risk = strongest(reading, vocabulary)
    activity = max(reading.activity.values(), default=None) if reading.activity else None
    if verdict is not None and verdict.action in ALERTS:
        level = "anomaly"
    elif verdict is not None or (risk is not None and risk >= tau):
        level = "elevated"
    elif risk is None:
        level = "unknown"
    else:
        level = "expected"
    if verdict is not None:
        reason = f"{verdict.action.replace('_', ' ')}: {verdict.text()}"
    elif name is not None:
        reason = f"{name.split(':', 1)[-1].replace('_', ' ')} p={risk:.2f}"
    else:
        reason = "the model answered none of the questions"
    request = verdict is not None and verdict.action in ALERTS and this_call
    call = reading.phase.call if reading.phase.name == "tool_call" and reading.phase.call else None  # the attempt it watched
    kind, goal = scope(reading, name)
    return {"level": level, "kind": "observation", "source": SOURCE, "risk": risk, "threshold": tau,
            "confidence": activity, "reason": reason, "signal": name or "", "call": call,
            "sim_time": float(reading.sim_time), "lag_s": float(reading.lag_s), "scope": kind, "scope_goal": goal,
            "semantics": semantics(name), "confidence_semantics": "Probability of the top activity label (uncalibrated)"
            if activity is not None else "", "calibration": "uncalibrated", "reliability": "unknown",
            "input_source": INPUT, "phase": reading.phase.name,
            **({"request": {"action": "stop_skill", "reason": verdict.text()}} if request else {})}


class MonitorFeed:
    """The monitor's on_update: hands DASHBOARD each reading once, from the monitor's threads."""

    def __init__(self, dashboard, settings):
        self.dashboard, self.tau = dashboard, settings.tau
        self._seen = None

    def __call__(self, view):
        key = None if view.reading is None else (view.calls, view.reading.goal_id, view.reading.sim_time)
        if key is None or key == self._seen:
            return
        self._seen = key
        value = feed(view, self.tau)
        if value is not None:
            self.dashboard.monitor(value)
