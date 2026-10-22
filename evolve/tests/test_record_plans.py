import json

from evolve import record_plans


def session(root, split, scenario, task, code="OK"):
    folder = root / split / scenario / "r0" / "session" / "s1"
    (folder / "tasks" / "task-000" / "decisions" / "step-000").mkdir(parents=True)
    (folder / "tasks" / "task-000" / "decisions" / "step-000" / "prompt.txt").write_text(
        f"You control a robot.\n\nCurrent task:\n{task}\n\nThis turn: write the plan (skill plan).")
    events = [{"kind": "decision", "step": 0, "decision": {"skill": "plan", "text": "t", "arguments": {"tree": {"a": 1}}}},
              {"kind": "result", "step": 0, "result": {"skill": "plan", "code": code}}]
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_a_composites_plan_prompt_is_found_in_its_sub_attempt(tmp_path):
    session(tmp_path, "test", "cafe", "Clear the cafe table.")
    folder = next(tmp_path.glob("test/cafe/r0/session/s1"))
    step = folder / "tasks" / "task-000" / "decisions" / "step-000"
    (step / "01-first_planner").mkdir()
    (step / "prompt.txt").rename(step / "01-first_planner" / "prompt.txt")
    assert record_plans.first_plan(folder)["task"] == "Clear the cafe table."


def test_accepted_first_plans_are_recorded_once_per_scenario(tmp_path, capsys):
    results = tmp_path / "b000"
    session(results, "train", "mv_a", "Move the bottle.")
    session(results, "test", "cafe", "Clear the cafe table.")
    session(results, "val", "mv_b", "Move the cup.", code="INVALID_ARGUMENT")
    session(results, "train", "wh_x", "Already recorded.")
    (tmp_path / "known.json").write_text(json.dumps({"plans": [{"task": "Already recorded."}]}))
    record_plans.main([str(results), str(tmp_path / "out.json"), "--source", "Sol", "--skip", str(tmp_path / "known.json")])
    plans = json.loads((tmp_path / "out.json").read_text())["plans"]
    assert [(p["scenario"], p["task"], p["tree"]) for p in plans] == [
        ("cafe", "Clear the cafe table.", {"a": 1}), ("mv_a", "Move the bottle.", {"a": 1})]
    assert "mv_b" in capsys.readouterr().out
