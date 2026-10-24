"""The monitor's rules: when sustained readings abort a skill, flag a goal or reopen one, and when they do not."""
from dataclasses import replace

from homebody.monitor.decide import MonitorState, observe, skill_started
from homebody.monitor.targets import Claim, Label, Phase, Reading, Vocabulary
from homebody.session.config import Monitor

SETTINGS = Monitor(enabled=True, tau=0.8, tau_unsafe=0.9, ticks=3, ticks_unsafe=2, stall_s=3.0, ema=1.0,
                   grace_s=2.0, cooldown_s=3.0, max_aborts_per_goal=2)
VOCAB = Vocabulary("g2", (Label("lifting", "progress", "raises the held cup"),
                          Label("dropped", "failure", "the cup falls out of the hand")),
                   standing=(Claim("g1", "the cup is in the left hand"),))
WORKING = Phase("tool_call", "pick", "Lifting")


def reading(t, **fields):
    return Reading(goal_id="g2", sim_time=t, phase=fields.pop("phase", WORKING), **fields)


def run(readings, *, state=None, settings=SETTINGS, vocab=VOCAB):
    state = state or skill_started(MonitorState(), "g2", -10.0)
    verdicts = []
    for item in readings:
        state, verdict = observe(state, item, vocab, settings)
        verdicts.append(verdict)
    return state, verdicts


def test_low_confidence_continues():
    _, verdicts = run([reading(t, activity={"dropped": 0.6, "lifting": 0.4}) for t in range(10)])
    assert {v.action for v in verdicts} == {"continue"}


def test_sustained_failure_aborts_after_ticks_readings_once():
    _, verdicts = run([reading(t, activity={"dropped": 0.95}) for t in range(4)])
    assert [v.action for v in verdicts] == ["continue", "continue", "abort_skill", "continue"]
    assert verdicts[2].label == "dropped" and verdicts[2].goal_id == "g2"


def test_one_confident_reading_among_low_ones_does_not_fire():
    probs = [0.95, 0.1, 0.95, 0.1, 0.95]
    _, verdicts = run([reading(t, activity={"dropped": p}) for t, p in enumerate(probs)])
    assert {v.action for v in verdicts} == {"continue"}


def test_none_of_these_never_fires():
    _, verdicts = run([reading(t, activity={"none_of_these": 0.99}) for t in range(6)])
    assert {v.action for v in verdicts} == {"continue"}


def test_progress_labels_never_fire():
    _, verdicts = run([reading(t, activity={"lifting": 0.99}) for t in range(6)])
    assert {v.action for v in verdicts} == {"continue"}


def test_failure_in_grace_period_only_flags():
    state = skill_started(MonitorState(), "g2", 0.0)
    _, verdicts = run([reading(t * 0.5, activity={"dropped": 0.95}) for t in range(3)], state=state)
    assert verdicts[2].action == "flag"


def test_abort_budget_then_flag_only():
    items = [reading(10 + t * 2, activity={"dropped": 0.95}) for t in range(12)]
    _, verdicts = run(items)
    actions = [v.action for v in verdicts if v.action != "continue"]
    assert actions[:2] == ["abort_skill", "abort_skill"]
    assert set(actions[2:]) == {"flag"}


def test_cooldown_after_abort_flags_instead():
    items = [reading(10 + t * 0.5, activity={"dropped": 0.95}) for t in range(6)]
    _, verdicts = run(items)
    actions = [v.action for v in verdicts if v.action != "continue"]
    assert actions == ["abort_skill", "flag"]


def test_planning_phase_flags_failures():
    planning = Phase("planning")
    _, verdicts = run([reading(t, activity={"dropped": 0.95}, phase=planning) for t in range(3)])
    assert verdicts[2].action == "flag"


def test_unsafe_aborts_fast_even_in_grace_and_past_budget():
    state = replace(skill_started(MonitorState(), "g2", 0.0), aborts={"g2": 5})
    _, verdicts = run([reading(t * 0.1, unsafe=0.97) for t in range(2)], state=state)
    assert verdicts[1].action == "abort_skill" and verdicts[1].label == "unsafe"


def test_human_needed_aborts():
    _, verdicts = run([reading(t, human_needed=0.95) for t in range(3)])
    assert verdicts[2].action == "abort_skill" and verdicts[2].label == "human_needed"


def test_undone_claim_reopens_its_goal_before_unsafe():
    _, verdicts = run([reading(t, still_true={"g1": 0.05}, unsafe=0.99) for t in range(3)])
    fired = [v for v in verdicts if v.action != "continue"]
    assert fired[0].action == "abort_skill" and fired[0].label == "unsafe"  # unsafe fires at its 2nd reading
    _, verdicts = run([reading(t, still_true={"g1": 0.05}, unsafe=0.99) for t in range(4)],
                      settings=replace(SETTINGS, ticks_unsafe=3))
    assert verdicts[2].action == "reopen_goal" and verdicts[2].goal_id == "g1"  # same reading: regress wins


def test_stall_needs_tool_call_and_stall_s_of_simulation_time():
    stalled = {"no_change": 0.95, "some": 0.05, "clear": 0.0}
    _, verdicts = run([reading(t * 0.4, progress=stalled) for t in range(8)])  # 2.8 s: a skill planning its grasp
    assert {v.action for v in verdicts} == {"continue"}
    _, verdicts = run([reading(t, progress=stalled) for t in range(4)])  # 3 s still
    assert [v.action for v in verdicts] == ["continue"] * 3 + ["flag"]  # a stall is labelled, never a stop
    assert verdicts[-1].label == "stall"
    state, verdicts = run([reading(t, progress=stalled) for t in range(12)])
    assert "abort_skill" not in {v.action for v in verdicts} and not state.aborts  # monitoring goes on; no abort spent
    moving = {"no_change": 0.1, "some": 0.8, "clear": 0.1}
    _, verdicts = run([reading(t, progress=stalled if t != 2 else moving) for t in range(5)])
    assert {v.action for v in verdicts} == {"continue"}  # progress in between restarts the clock
    _, verdicts = run([reading(t, progress=stalled, phase=Phase("planning")) for t in range(6)])
    assert {v.action for v in verdicts} == {"continue"}


def test_refused_question_resets_its_streak():
    items = [reading(0, activity={"dropped": 0.95}), reading(1, activity={"dropped": 0.95}),
             reading(2, refusals=("activity",)), reading(3, activity={"dropped": 0.95})]
    _, verdicts = run(items)
    assert {v.action for v in verdicts} == {"continue"}


def test_new_goal_resets_signals_but_keeps_abort_counts():
    state, _ = run([reading(t, activity={"dropped": 0.95}) for t in range(2)])
    state = replace(state, aborts={"g2": 1})
    other = Reading(goal_id="g3", sim_time=5, activity={"dropped": 0.95}, phase=WORKING)
    state, verdict = observe(state, other, replace(VOCAB, goal_id="g3"), SETTINGS)
    assert verdict.action == "continue" and state.goal_id == "g3" and state.aborts == {"g2": 1}


def test_ema_smooths_readings():
    smooth = replace(SETTINGS, ema=0.5)
    probs = [0.5, 0.95, 0.95, 0.95, 0.95]  # averages 0.5, 0.725, 0.84, 0.89, 0.92
    _, verdicts = run([reading(t, activity={"dropped": p}) for t, p in enumerate(probs)], settings=smooth)
    assert [v.action for v in verdicts] == ["continue"] * 4 + ["abort_skill"]


def test_a_reopened_goal_is_not_reopened_again_until_done_again():
    items = [reading(t, still_true={"g1": 0.05}) for t in range(9)]
    state, verdicts = run(items)
    assert [v.action for v in verdicts].count("reopen_goal") == 1 and "g1" in state.reopened
    state = replace(state, reopened=frozenset())  # g1 done again (Monitor.goal clears it)
    _, verdicts = run([reading(20 + t, still_true={"g1": 0.05}) for t in range(3)], state=state)
    assert verdicts[2].action == "reopen_goal"


def test_after_an_operator_resume_human_needed_is_not_acted_on_again_for_the_goal():
    from homebody.monitor.decide import silenced
    state = silenced(skill_started(MonitorState(), "g2", -10.0), "g2", "human_needed")
    _, verdicts = run([reading(t, human_needed=0.99) for t in range(5)], state=state)
    assert {v.action for v in verdicts} == {"continue"}
