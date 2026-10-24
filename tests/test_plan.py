"""Plan mode: the agent's behavior-tree plan, its contract, the walk through it, and its runner."""
import pytest

from homebody.session.planned import PlanRunner
from homebody.tree.plan import Plan, PlanError, check_holds, validate
from homebody.vlm.provider import Decision


def goal(goal_id, check=None, budget=None):
    node = {"type": "goal", "id": goal_id, "goal": f"reach {goal_id}", "check": check}
    return node if budget is None else {**node, "budget": budget}


TREE = {"type": "sequence", "children": [
    {"type": "fallback", "children": [goal("right", {"name": "holding", "hand": "right"}),
                                      goal("left", {"name": "holding", "hand": "left"})]},
    goal("carry", {"name": "arrived"}), goal("put", {"name": "placed"})]}


@pytest.mark.parametrize("tree, problem", [
    ({"type": "loop"}, "type must be"), ({"type": "sequence", "children": []}, "non-empty"),
    ({"type": "goal", "id": "a", "goal": " ", "check": None}, "goal text"),
    ({"type": "goal", "id": "a", "goal": "x", "check": {"name": "teleport"}}, "check must be"),
    ({"type": "goal", "id": "a", "goal": "x", "check": {"name": "holding"}}, "needs hand"),
    ({"type": "goal", "id": "a", "goal": "x", "check": None, "budget": 40}, "budget"),
    ({"type": "sequence", "children": [goal("a"), goal("a")]}, "unique id"),
])
def test_the_contract_refuses_a_bad_tree_with_its_reason(tree, problem):
    with pytest.raises(PlanError, match=problem):
        validate(tree)


def test_the_walk_follows_sequence_and_fallback():
    plan = Plan(TREE)
    assert plan.next() == ("goal", TREE["children"][0]["children"][0])
    plan.close("right", False, "no grasp")
    assert plan.next()[1]["id"] == "left"
    plan.close("left", True)
    assert plan.next()[1]["id"] == "carry"
    plan.close("carry", True)
    plan.close("put", True)
    assert plan.next()[0] == "done"
    assert "[failed] right: reach right | check: name=holding hand=right (no grasp)" in plan.render()


def test_checks_read_robot_state_and_this_goals_results():
    assert check_holds({"name": "holding", "hand": "right"}, {"right": {"state": "verified"}}, [])
    assert not check_holds({"name": "holding", "hand": "right"}, {"right": {"state": "pending"}}, [])
    assert check_holds({"name": "hand_free", "hand": "left"}, {"right": {}}, [])
    assert check_holds({"name": "placed"}, {}, [{"skill": "place", "code": "OK"}])
    assert not check_holds({"name": "arrived"}, {}, [{"skill": "navigate", "code": "STALL"}])
    assert not check_holds(None, {}, [{"skill": "pick", "code": "OK"}])


def plan_decision(tree):
    return Decision("My plan.", "plan", {"tree": tree})


def test_the_runner_plans_then_closes_goals_by_their_checks():
    runner = PlanRunner(replans=2)
    assert runner.names(("pick", "done")) == ("plan",)
    assert runner.handle(plan_decision(TREE), {})["code"] == "OK"
    assert runner.names(("pick", "navigate", "done")) == ("pick", "navigate", "goal_done", "goal_failed", "done")
    assert "Active goal right" in runner.note()
    runner.observe({"skill": "pick", "code": "OK"}, {"right": {"state": "verified"}})
    assert runner.plan.status["right"] == "done" and runner.goal is None
    runner.phase()
    assert runner.goal["id"] == "carry"


def test_a_spent_budget_fails_the_goal_and_a_failed_tree_is_rewritten_keeping_done_goals():
    runner = PlanRunner(replans=1)
    tree = {"type": "sequence", "children": [goal("a", None, budget=1), goal("b", None, budget=1)]}
    runner.handle(plan_decision(tree), {})
    runner.phase()
    runner.handle(Decision("Reached.", "goal_done", {"reason": "seen it"}), {})
    runner.phase()
    runner.observe({"skill": "pick", "code": "NO_GRASP", "message": "none"}, {})
    assert runner.plan.status["b"] == "failed" and runner.names(("pick", "done")) == ("plan",)
    assert "goal b failed: budget of 1 skill calls spent" in runner.note()
    runner.handle(plan_decision({"type": "sequence", "children": [goal("a"), goal("c")]}), {})
    assert runner.plan.status == {"a": "done", "c": "pending"} and runner.replans == 1
    runner.phase()
    runner.handle(Decision("No.", "goal_failed", {"reason": "blocked"}), {})
    assert runner.names(("pick", "done")) == ("done",)  # no rewrites left


def test_goal_done_needs_its_check_and_bad_plans_end_plan_mode():
    runner = PlanRunner(replans=2)
    runner.handle(plan_decision(TREE), {})
    runner.phase()
    assert runner.handle(Decision("Got it.", "goal_done", {}), {})["code"] == "CHECK_FAILED"
    other = PlanRunner(replans=2)
    assert other.handle(plan_decision({"type": "loop"}), {})["code"] == "PLAN_INVALID"
    assert other.names(("pick", "done")) == ("plan",)
    other.handle(plan_decision({"type": "loop"}), {})
    assert other.abandoned and other.names(("pick", "done")) == ("pick", "done") and other.note() == ""


def test_a_placed_or_arrived_goal_closes_when_a_coder_call_did_the_work():
    by_coder = {"skill": "call_coder", "code": "OK",
                "details": {"target": "place", "executes": [{"version": 2, "code": "OK"}]}}
    assert check_holds({"name": "placed"}, {}, [by_coder])
    assert not check_holds({"name": "arrived"}, {}, [by_coder])  # it changed place, not navigate
    no_run = {"skill": "call_coder", "code": "OK", "details": {"target": "place", "executes": []}}
    failed = {"skill": "call_coder", "code": "CODER_FAILED",
              "details": {"target": "place", "executes": [{"version": 2, "code": "OK"}]}}
    assert not check_holds({"name": "placed"}, {}, [no_run]) and not check_holds({"name": "placed"}, {}, [failed])


def test_the_monitor_flags_a_goal_and_reopens_a_done_one():
    runner = PlanRunner(2)
    runner.handle(Decision("p", "plan", {"tree": TREE}), {})
    runner.phase()
    runner.observe({"skill": "pick", "code": "OK"}, {"right": {"state": "verified"}})  # right closes by its check
    assert runner.phase() == "goal" and runner.goal["id"] == "carry"
    runner.flag("carry", "dropped p=0.93: the cup falls")
    assert "[ACTIVE ⚠ monitor: dropped p=0.93: the cup falls] carry" in runner.note()
    assert runner.reopen("right", "no longer true: the cup is in the right hand")
    assert runner.phase() == "goal" and runner.goal["id"] == "right"
    assert "monitor reopened" in runner.note()
    assert not runner.reopen("put", "not done yet")  # only a done goal reopens
    runner.observe({"skill": "pick", "code": "OK"}, {"right": {"state": "verified"}})
    assert runner.phase() == "goal" and runner.goal["id"] == "carry"
    assert "⚠" not in next(line for line in runner.plan.render().splitlines() if "right:" in line)


def test_flag_and_reopen_without_a_plan_do_nothing():
    runner = PlanRunner(2)
    runner.flag("g1", "x")
    assert not runner.reopen("g1", "x")
