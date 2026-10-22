import json

from evolve import plan_bench

TREE = {"type": "sequence", "children": [
    {"type": "goal", "id": "g1", "goal": "Pick the can.", "check": {"name": "holding", "hand": "right"}, "budget": 4},
    {"type": "goal", "id": "g2", "goal": "Place it.", "check": {"name": "placed"}, "budget": 3}]}


def test_a_tree_reads_as_indented_goals_with_checks_and_budgets():
    text = plan_bench.tree_text(TREE)
    assert text.splitlines()[0] == "SEQUENCE"
    assert "  - goal g1: Pick the can. [check: holding hand=right; budget 4]" in text


def test_a_preference_counts_only_when_it_holds_in_both_orders(tmp_path, monkeypatch):
    case = tmp_path / "cases" / "k1"
    case.mkdir(parents=True)
    (case / "case.json").write_text(json.dumps({"task": "t", "split": "val", "sol": {"tree": TREE}}))
    out = tmp_path / "out"
    (out / "k1").mkdir(parents=True)
    (out / "k1" / "plan.json").write_text(json.dumps({"tree": TREE, "accepted": True}))
    scores = dict.fromkeys(plan_bench.CRITERIA, 5)
    answers = iter([{"A": scores, "B": scores, "preferred": "A", "reason": "r"},
                    {"A": scores, "B": scores, "preferred": "A", "reason": "r"}])
    monkeypatch.setattr(plan_bench, "judge_once", lambda *a: next(answers))
    result = plan_bench.judge_case(case, out)
    assert result["preferred"] == "tie"  # A both times means different plans: position bias, not a preference
    assert {v["preferred"] for v in result["verdicts"]} == {"open", "sol"}
