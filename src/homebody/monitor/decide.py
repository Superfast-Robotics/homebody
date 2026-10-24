"""The monitor's rules: readings in, at most one verdict out. Pure; the monitor thread holds the state.

A signal is one probability a reading carries: a failure label of the goal's vocabulary, unsafe, human_needed,
a standing claim no longer holding (`undone:<goal>`) or no progress across the frames (`stall`). Each is averaged
(weight `ema` on the newest reading) and fires when the average stays at or above its threshold for its number of
readings in a row (`stall` instead for `stall_s` simulation seconds: a skill stands still for seconds while it plans,
verifies or squeezes); firing restarts its count, and a reading without the signal (a refused question) restarts it
too. A probability below its threshold never acts. In order:

1. a done goal's claim no longer holds -> reopen that goal (the running skill stops); once per reopen, until
   the goal is done again;
2. unsafe -> abort the running skill, whatever the grace, cooldown or budget;
3. human_needed -> the same, until the operator resumes (then not again for that goal);
4. a failure label, or no progress while a skill runs -> abort the skill, or only flag the goal during the
   agent's planning, within `grace_s` of the skill's start, within `cooldown_s` of an abort, or once the goal has
   used its `max_aborts_per_goal`.
"""
from dataclasses import dataclass, field, replace

from homebody.monitor.targets import Reading, Verdict, Vocabulary

URGENT = ("unsafe", "human_needed")


@dataclass(frozen=True)
class MonitorState:
    goal_id: str | None = None
    ema: dict = field(default_factory=dict)  # signal -> average probability
    streak: dict = field(default_factory=dict)  # signal -> readings in a row at its threshold
    aborts: dict = field(default_factory=dict)  # goal id -> aborts this task
    skill_since: float | None = None  # simulation time the running skill started
    still_since: float | None = None  # simulation time no progress began, while it lasts
    cooldown_until: float = float("-inf")
    reopened: frozenset = frozenset()  # goals reopened and not yet done again
    silenced: frozenset = frozenset()  # (goal id, signal) pairs not to act on again


def skill_started(state: MonitorState, goal_id: str | None, sim_time: float) -> MonitorState:
    return replace(state, goal_id=goal_id if goal_id is not None else state.goal_id, skill_since=sim_time)


def silenced(state: MonitorState, goal_id: str, signal: str) -> MonitorState:
    """STATE no longer acting on SIGNAL for GOAL_ID."""
    return replace(state, silenced=state.silenced | {(goal_id, signal)},
                   ema={k: v for k, v in state.ema.items() if k != signal},
                   streak={k: v for k, v in state.streak.items() if k != signal})


def signals(reading: Reading, vocabulary: Vocabulary) -> dict:
    """The reading's signal probabilities; a refused question gives none."""
    found = {}
    if "activity" not in reading.refusals and reading.activity:
        for name in vocabulary.failures():
            found[f"failure:{name}"] = reading.activity.get(name, 0.0)
    for name in ("unsafe", "human_needed"):
        if getattr(reading, name) is not None and name not in reading.refusals:
            found[name] = getattr(reading, name)
    for goal_id, holds in reading.still_true.items():
        if holds is not None:
            found[f"undone:{goal_id}"] = 1.0 - holds
    if "progress" not in reading.refusals and reading.progress:
        found["stall"] = reading.progress.get("no_change", 0.0)
    return found


def thresholds(signal: str, settings) -> tuple[float, int]:
    if signal == "unsafe":
        return settings.tau_unsafe, settings.ticks_unsafe
    return settings.tau, settings.ticks


def observe(state: MonitorState, reading: Reading, vocabulary: Vocabulary, settings) -> tuple[MonitorState, Verdict]:
    """STATE after READING (of VOCABULARY's goal) and the verdict it gives."""
    if reading.goal_id != state.goal_id:  # a new goal: its signals start afresh
        state = replace(state, goal_id=reading.goal_id, ema={}, streak={})
    working = reading.phase.name == "tool_call"
    in_grace = state.skill_since is not None and reading.sim_time - state.skill_since < settings.grace_s
    values = {name: p for name, p in signals(reading, vocabulary).items()
              if (reading.goal_id, name) not in state.silenced
              and not (name.startswith("undone:") and name.split(":", 1)[1] in state.reopened)}
    if not working or in_grace:
        values.pop("stall", None)  # standing still is expected outside a skill and while one starts
    ema = {name: settings.ema * p + (1 - settings.ema) * state.ema.get(name, p) for name, p in values.items()}
    streak = {}
    for name, average in ema.items():
        tau, _ = thresholds(name, settings)
        streak[name] = state.streak.get(name, 0) + 1 if average >= tau else 0
    still = ema.get("stall", 0.0) >= settings.tau
    still_since = (state.still_since if state.still_since is not None else reading.sim_time) if still else None
    fired = [name for name in streak if name != "stall" and streak[name] >= thresholds(name, settings)[1]]
    if still_since is not None and reading.sim_time - still_since >= settings.stall_s:
        fired.append("stall")
    state = replace(state, ema=ema, streak=streak, still_since=still_since)
    if not fired:
        return state, Verdict("continue", reading.goal_id, sim_time=reading.sim_time)
    name = choose(fired, ema)
    state = replace(state, streak={**streak, name: 0}, still_since=None if name == "stall" else state.still_since)
    t, goal = reading.sim_time, reading.goal_id
    if name.startswith("undone:"):
        undone = name.split(":", 1)[1]
        claim = next((c.text for c in vocabulary.standing if c.goal_id == undone), "its result")
        state = replace(state, cooldown_until=t + settings.cooldown_s, reopened=state.reopened | {undone})
        return state, Verdict("reopen_goal", undone, "undone", ema[name], f"no longer true: {claim}", t)
    label = name.split(":", 1)[1] if name.startswith("failure:") else name
    described = vocabulary.label(label)
    reason = described.description if described else ""
    if name in URGENT:
        action = "abort_skill" if working else "flag"
    else:
        spent = state.aborts.get(goal, 0) >= settings.max_aborts_per_goal
        action = "abort_skill" if working and not in_grace and t >= state.cooldown_until and not spent else "flag"
    if action == "abort_skill":
        state = replace(state, aborts={**state.aborts, goal: state.aborts.get(goal, 0) + 1},
                        cooldown_until=t + settings.cooldown_s)
    return state, Verdict(action, goal, label, ema[name], reason, t)


def choose(fired: list[str], ema: dict) -> str:
    """The signal to act on: a regression first, then unsafe, human_needed, then the most probable."""
    for prefix in ("undone:", "unsafe", "human_needed"):
        matching = [name for name in fired if name.startswith(prefix)]
        if matching:
            return max(matching, key=lambda name: ema[name])
    return max(fired, key=lambda name: ema[name])
