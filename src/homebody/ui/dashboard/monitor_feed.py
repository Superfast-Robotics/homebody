"""The live monitor (homebody/monitor) on the dashboard: each new reading of its View becomes one dashboard reading.

The risk is the strongest of the signals the monitor's rules weigh (monitor/decide.signals); the level follows the
verdict the reading got: an abort or a reopened goal is an anomaly (the dashboard raises an alert), a flag or a risk
at the threshold is elevated, a reading whose questions were all refused is unknown.
"""
from homebody.monitor.decide import signals

SOURCE = "Live monitor"
ALERTS = ("abort_skill", "reopen_goal")  # the verdicts that stop a skill or reopen a goal


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
    name, risk = strongest(reading, view.vocabulary)
    verdict = verdict_for(view, reading)
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
    return {"level": level, "kind": "observation", "source": SOURCE, "risk": risk, "threshold": tau,
            "confidence": activity, "reason": reason}


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
