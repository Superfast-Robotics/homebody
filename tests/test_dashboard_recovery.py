"""Recoveries, replans, retries and monitor readings as the dashboard records them, and grounded place labels."""
import json
from types import SimpleNamespace

import pytest

from homebody.ui.dashboard.places import Places
from homebody.ui.dashboard.records import reading
from homebody.ui.dashboard.store import Store
from homebody.ui.dashboard.tree import plan_nodes, recovery_of


class Clock:
    def __init__(self):
        self.t = 100.

    def __call__(self):
        return self.t


TREE = {"type": "sequence", "children": [
    {"type": "fallback", "children": [
        {"type": "goal", "id": "grasp", "goal": "grasp the jar"},
        {"type": "goal", "id": "reposition", "goal": "reposition and grasp"}]},
    {"type": "goal", "id": "place", "goal": "place the jar"}]}
GOALS = {"grasp": "grasp the jar", "reposition": "reposition and grasp", "place": "place the jar"}


def plan(active, tree=TREE, **status):
    return {"tree": tree, "goals": GOALS, "status": {"grasp": "pending", "reposition": "pending", "place": "pending",
                                                    **status}, "reasons": {"grasp": "no grip"}, "active": active,
            "replans": 0, "feedback": ""}


def call(store, skill, code, goal):
    store.decision(SimpleNamespace(skill=skill, text=f"{skill} now", arguments={}), goal=goal)
    store.update("Executing")
    store.skill({"type": "skill_started", "skill": skill})
    store.skill({"type": "skill_finished", "result": SimpleNamespace(skill=skill, code=code, message=code)})


def kinds(store):
    return [(i["kind"], i["title"]) for i in store.snapshot({})["incidents"]]


def test_recovery_of_finds_the_failed_sibling_before_a_fallback_child():
    assert recovery_of(TREE, {"grasp": "failed"}, "reposition") == "grasp"
    assert recovery_of(TREE, {"grasp": "pending"}, "reposition") is None
    assert recovery_of(TREE, {"grasp": "failed"}, "place") is None


def test_a_fallback_recovery_is_an_action_and_fills_the_failures_outcome():
    clock = Clock()
    store = Store(clock=clock)
    store.update("Thinking", task="Pick the jar", plan=plan("grasp"))
    call(store, "pick", "TARGET_LOST", "grasp")
    store.update("Thinking", plan=plan("reposition", grasp="failed"))
    started = [i for i in store.snapshot({})["incidents"] if i["kind"] == "recovery"]
    assert [i["title"] for i in started] == ["Recovery started: reposition and grasp"]
    assert "no grip" in started[0]["reason"] and started[0]["node"] == "reposition"
    goal_failure, = [i for i in store.snapshot({})["incidents"] if i["kind"] == "goal"]
    assert goal_failure["outcome"].startswith("Recovery started")
    call(store, "pick", "OK", "reposition")
    tool = store.snapshot({})["tools"][-1]
    assert tool["recovery"] and tool["attempt"] == 1
    skill = [i for i in store.snapshot({})["intervals"] if i["track"] == "skill"][-1]
    assert skill["detail"]["recovery"]
    store.update("Thinking", plan=plan("place", grasp="failed", reposition="done"))
    state = store.snapshot({})
    assert ("recovery", "Recovery succeeded: goal reposition (reposition and grasp)") in kinds(store)
    started = next(i for i in state["incidents"] if i["title"].startswith("Recovery started"))
    assert started["outcome"] == "Succeeded"
    assert next(i for i in state["incidents"] if i["kind"] == "goal")["outcome"].startswith("Recovery succeeded")
    node = state["tree"]["nodes"]["goal:reposition"]
    assert node["detail"]["recovers"] == "grasp" and node["detail"]["recovers_label"] == "grasp the jar"
    assert state["tree"]["nodes"]["p.0"]["detail"]["path"] == [
        {"label": "grasp the jar", "status": "failure"}, {"label": "reposition and grasp", "status": "success"}]


def test_a_retry_numbers_its_attempt_and_answers_the_last_failure():
    store = Store(clock=Clock())
    store.update("Thinking", task="Pick", plan=plan("grasp"))
    call(store, "pick", "TARGET_LOST", "grasp")
    call(store, "pick", "OK", "grasp")
    tools = store.snapshot({})["tools"]
    assert [t["attempt"] for t in tools] == [1, 2]
    failure, = [i for i in store.snapshot({})["incidents"] if i["kind"] == "skill"]
    assert failure["skill"] == "pick" and failure["outcome"] == "Retried (attempt 2): OK"


def test_a_replan_keeps_the_old_tree_as_superseded_and_marks_the_rewrite():
    store = Store(clock=Clock())
    store.update("Thinking", task="Pick", plan=plan("grasp"))
    store.update("Thinking", plan=plan(None, grasp="failed", reposition="failed"))
    assert store.snapshot({})["tree"]["mode"] == "replanning"
    rewritten = json.loads(json.dumps(TREE))
    rewritten["children"][1]["goal"] = "set the jar down"
    store.update("Thinking", plan={**plan("place", tree=rewritten), "replans": 1, "feedback": "goal grasp failed"})
    state = store.snapshot({})
    old, = state["superseded"]
    assert old["replan"] == 1 and old["tree"]["nodes"]["goal:grasp"]["status"] == "failure"
    assert ("recovery", "Replanned (rewrite 1)") in kinds(store)
    assert state["tree"]["mode"] == "plan" and state["tree"]["nodes"]["goal:place"]["status"] == "running"


def test_readings_record_what_ran_and_only_observed_anomalies_are_incidents():
    store = Store(clock=Clock())
    store.monitor({"level": "expected", "kind": "heartbeat", "source": "monitor"})
    store.update("Thinking", task="Pick", plan=plan("grasp"))
    store.monitor({"level": "anomaly", "kind": "heartbeat", "source": "monitor"})
    store.monitor({"level": "anomaly", "risk": .9, "threshold": .7, "confidence": .8, "source": "force",
                   "reason": "Force limit", "synthetic": True})
    idle, beat, observed = store.snapshot({})["readings"]
    assert not idle["during_task"] and beat["during_task"] and observed["node"] == "grasp"
    incident, = [i for i in store.snapshot({})["incidents"] if i["kind"] == "monitor"]
    assert incident["title"] == "Synthetic: Force limit" and incident["evidence"]["synthetic"] is True
    assert incident["observation"]["confidence"] == .8 and incident["evidence"]["threshold"] == .7
    with pytest.raises(ValueError, match="kind"):
        reading({"kind": "tick"}, 0.)
    with pytest.raises(ValueError, match="confidence"):
        reading({"confidence": True}, 0.)


def test_task_spans_are_kept_for_the_monitoring_tracks():
    clock = Clock()
    store = Store(clock=clock)
    store.update("Thinking", task="One")
    clock.t += 5
    store.update("model_done", "Done")
    store.update("Thinking", task="Two")
    spans = store.snapshot({})["tasks"]
    assert [(s["text"], s["end"]) for s in spans] == [("One", 105.), ("Two", None)]


def test_planning_mode_before_any_tree_and_after_a_task_without_one():
    task = {"text": "Pick", "start": 0., "end": None, "outcome": None}
    assert plan_nodes({"tree": None}, [], task)["mode"] == "planning"
    ended = {**task, "end": 1., "outcome": "cancelled"}
    tree = plan_nodes({"tree": None}, [], ended)
    assert tree["mode"] == "free" and "ended before" in tree["nodes"]["root"]["detail"]["note"]


def test_places_name_only_fixed_fixtures_near_the_robot(tmp_path):
    (tmp_path / "semantics.json").write_text(json.dumps({"entities": [
        {"label": "Island", "category": "island", "bounds_ws_map_m": [[0, 0, 0], [2, 1, 1]]},
        {"label": "Jar", "category": "jar", "portable_candidate": True, "bounds_ws_map_m": [[3, 3, 0], [3.1, 3.1, 1]]},
        {"label": "Room floor", "category": "floor", "bounds_ws_map_m": [[-9, -9, 0], [9, 9, 0]]}]}))
    places = Places.from_scene(tmp_path, "Kitchen")
    assert places.near(2.5, 0.5) == "Kitchen · Near Island"
    assert places.near(3.05, 3.05) is None  # only the portable jar and the floor are near
    assert Places.from_scene(tmp_path / "missing") is None


def test_consecutive_anomalies_from_one_source_are_one_incident_with_their_peak():
    clock = Clock()
    store = Store(clock=clock)
    store.update("Thinking", task="Pick")
    for risk in (.82, .95, .9):
        store.monitor({"level": "anomaly", "risk": risk, "threshold": .8, "source": "force", "reason": "Force"})
        clock.t += .25
    clock.t += 10
    store.monitor({"level": "anomaly", "risk": .85, "threshold": .8, "source": "force", "reason": "Force"})
    store.monitor({"level": "anomaly", "risk": .85, "threshold": .8, "source": "pose", "reason": "Pose"})
    first, second, other = [i for i in store.snapshot({})["incidents"] if i["kind"] == "monitor"]
    assert first["evidence"]["risk"] == .95 and first["evidence"]["readings"] == 3
    assert second["evidence"]["risk"] == .85 and other["source"] == "pose"
