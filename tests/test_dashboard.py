"""The dashboard's record of a session, its behavior tree view and its operator commands, over a recording hub."""
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import fake_dashboard

from homebody.ui.controls import Controls, capabilities
from homebody.ui.dashboard.records import arguments, reading
from homebody.ui.dashboard.store import Store
from homebody.ui.dashboard.tree import composite_status, plan_nodes


class Clock:
    def __init__(self, t=1000.):
        self.t = t

    def __call__(self):
        return self.t


def decision(skill, text="", **args):
    return SimpleNamespace(skill=skill, text=text or f"{skill} now", arguments=args)


def result(skill, code="OK", message="fine"):
    return SimpleNamespace(skill=skill, code=code, message=message)


TREE = {"type": "sequence", "children": [
    {"type": "goal", "id": "g1", "goal": "pick the jar", "check": {"name": "holding", "hand": "right"}},
    {"type": "fallback", "children": [
        {"type": "goal", "id": "g2", "goal": "place in the cart"},
        {"type": "goal", "id": "g3", "goal": "set it down", "budget": 3}]}]}


def plan(active, **status):
    return {"tree": TREE, "goals": {"g1": "pick the jar", "g2": "place in the cart", "g3": "set it down"},
            "status": {"g1": "pending", "g2": "pending", "g3": "pending", **status}, "reasons": {}, "active": active}


def run_task(store, clock):
    store.update("Ready")
    store.update("Thinking", task="Pick the jar", plan=plan("g1"))
    clock.t += 2
    store.decision(decision("pick", point_normalized_1000=[500, 400]), goal="g1")
    store.update("Executing", "Grasping the jar")
    store.skill({"type": "skill_started", "skill": "pick"})
    clock.t += 1
    store.skill({"type": "skill_stage", "skill": "pick", "stage": "Approaching"})
    clock.t += 3
    store.skill({"type": "skill_finished", "result": result("pick", "TARGET_LOST", "No view of the jar")})


def test_a_failed_skill_reaches_its_tool_call_track_alert_and_log():
    clock = Clock()
    store = Store(clock=clock)
    run_task(store, clock)
    state = store.snapshot(capabilities(Controls()))
    tool, = state["tools"]
    assert (tool["skill"], tool["state"], tool["code"], tool["goal"]) == ("pick", "failed", "TARGET_LOST", "g1")
    assert [stage for _, stage in tool["stages"]] == ["Approaching"]
    skill, = [i for i in state["intervals"] if i["track"] == "skill"]
    assert skill["status"] == "failure" and skill["end"] - skill["start"] == 4.
    alert, = [i for i in state["incidents"] if i["kind"] == "skill"]
    assert alert["severity"] == "warning" and alert["state"] == "active" and alert["node"] == "g1"
    assert alert["tool"] == tool["id"] and "No view" in alert["reason"]
    assert any("TARGET_LOST" in entry["message"] and entry["severity"] == "warning" for entry in state["logs"])
    assert state["decisions"][0]["tool"] == tool["id"]


def test_goal_intervals_follow_the_planner_and_a_failed_goal_raises_an_incident():
    clock = Clock()
    store = Store(clock=clock)
    run_task(store, clock)
    clock.t += 1
    store.update("Thinking", plan={**plan("g2", g1="failed"), "reasons": {"g1": "budget spent"}})
    state = store.snapshot({})
    goals = [i for i in state["intervals"] if i["track"] == "goal"]
    assert [(g["detail"]["goal"], g["status"]) for g in goals] == [("g1", "failure"), ("g2", "running")]
    failed, = [i for i in state["incidents"] if i["kind"] == "goal"]
    assert failed["node"] == "g1" and failed["reason"] == "budget spent"


def test_a_new_task_keeps_old_alerts_and_accepts_the_pending_instruction():
    clock = Clock()
    store = Store(clock=clock)
    run_task(store, clock)
    store.update("model_done", "Done")
    store.instruction("Pick the basil", "pending")
    store.update("Thinking", task="Pick the basil")
    state = store.snapshot({})
    assert {i["state"] for i in state["incidents"]} == {"active"}  # only an operator resolves an alert
    assert state["instructions"][-1]["state"] == "running"
    assert state["task"]["text"] == "Pick the basil" and state["task"]["end"] is None


def test_a_cancelled_task_is_not_an_alert_and_closes_running_work():
    clock = Clock()
    store = Store(clock=clock)
    store.update("Thinking", task="Pick")
    store.decision(decision("navigate"))
    store.update("Executing")
    store.skill({"type": "skill_started", "skill": "navigate"})
    store.intervention("stop", "Operator stopped the task")
    assert store.snapshot({})["controls"]["stop_pending"]
    store.update("cancelled", "Task stopped by operator")
    store.update("Stopped", "Enter another task when ready.")
    state = store.snapshot({})
    assert not state["controls"]["stop_pending"] and state["controls"]["accepting"]
    assert state["task"]["outcome"] == "cancelled"
    assert [i["kind"] for i in state["incidents"]] == ["intervention"]
    assert all(t["state"] != "running" for t in state["tools"])
    assert all(i["end"] is not None for i in state["intervals"])


def test_critical_endings_and_acknowledgement():
    store = Store(clock=Clock())
    store.update("Thinking", task="Pick")
    store.update("provider_error", "Claude Code exited 1")
    incident, outcome = store.snapshot({})["incidents"]
    assert outcome["title"] == "Task outcome: Failed" and outcome["alert"]
    assert incident["severity"] == "critical" and incident["title"] == "Provider error"
    assert store.acknowledge(incident["id"]) and not store.acknowledge(incident["id"])
    assert store.snapshot({})["incidents"][0]["state"] == "acknowledged"


def test_old_timeline_records_are_dropped_but_running_ones_stay():
    clock = Clock()
    store = Store(clock=clock)
    store.update("Thinking", task="Long task")
    store.decision(decision("navigate"))
    clock.t += 10_000
    state = store.snapshot({})
    assert not state["decisions"]
    assert [i["track"] for i in state["intervals"]] == ["thinking"]


def test_monitor_readings_are_validated_and_an_anomaly_is_an_incident():
    store = Store(clock=Clock())
    store.monitor({"level": "elevated", "risk": .6, "threshold": .8, "source": "force"})
    store.monitor({"level": "anomaly", "risk": .9, "threshold": .8, "source": "force", "reason": "Force limit"})
    state = store.snapshot({})
    assert [r["level"] for r in state["readings"]] == ["elevated", "anomaly"]
    assert state["incidents"][0]["title"] == "Force limit" and state["incidents"][0]["evidence"]["risk"] == .9
    with pytest.raises(ValueError, match="level"):
        reading({"level": "fine"}, 0.)
    with pytest.raises(ValueError, match="risk"):
        reading({"level": "expected", "risk": 1.5}, 0.)


def test_arguments_become_bounded_json():
    assert arguments({"point": np.array([1, 2]), "bad": float("nan"), "obj": object}) == {
        "point": [1, 2], "bad": None, "obj": str(object)}
    assert set(arguments({"text": "x" * 10_000})) == {"truncated"}
    assert arguments("not a dict") == {}


@pytest.mark.parametrize("kind, statuses, expected", [
    ("sequence", ["success", "success"], "success"),
    ("sequence", ["success", "failure", "pending"], "failure"),
    ("sequence", ["success", "pending"], "pending"),
    ("fallback", ["failure", "success"], "success"),
    ("fallback", ["failure", "failure"], "failure"),
    ("fallback", ["failure", "running"], "running"),
])
def test_composite_status_follows_the_plan_semantics(kind, statuses, expected):
    assert composite_status(kind, statuses) == expected


def test_the_tree_keeps_the_planner_hierarchy_and_attaches_tool_calls_to_goals():
    clock = Clock()
    store = Store(clock=clock)
    run_task(store, clock)
    store.update("Thinking", plan=plan("g2", g1="failed"))
    tree = store.snapshot({})["tree"]
    nodes = tree["nodes"]
    assert tree["mode"] == "plan" and nodes["root"]["children"] == ["p"]
    assert nodes["p"]["children"] == ["goal:g1", "p.1"] and nodes["p.1"]["type"] == "fallback"
    assert nodes["goal:g1"]["status"] == "failure" and nodes["goal:g2"]["status"] == "running"
    assert nodes["p.1"]["status"] == "running" and nodes["p.1"]["detail"]["selected"] == 0
    action, = nodes["goal:g1"]["children"]
    assert nodes[action]["type"] == "action" and nodes[action]["status"] == "failure"
    assert nodes["goal:g1"]["detail"]["used"] == 1 and nodes["goal:g3"]["detail"]["budget"] == 3


def test_without_a_plan_the_tree_lists_the_tasks_actions_and_says_so():
    clock = Clock()
    store = Store(clock=clock)
    assert store.snapshot({})["tree"] is None
    store.update("Thinking", task="Explore")
    store.decision(decision("navigate"))
    tree = store.snapshot({})["tree"]
    assert tree["mode"] == "free" and "plan mode off" in tree["nodes"]["root"]["detail"]["note"]
    assert len(tree["nodes"]["root"]["children"]) == 1
    assert plan_nodes(None, [], None) is None


def test_controls_without_pause_or_stop_skill_are_reported_unsupported():
    assert capabilities(Controls()) == {"pause": False, "stop_skill": False, "stop_task": True, "reset": True}

    class Pausable(Controls):
        def pause(self): ...
        def resume(self): ...
        def stop_skill(self): ...
    assert capabilities(Pausable()) == {"pause": True, "stop_skill": True, "stop_task": True, "reset": True}


def test_instructions_are_queued_only_when_the_agent_can_take_them():
    dashboard = fake_dashboard()
    assert dashboard.command({"type": "submit", "text": "Pick the jar"})["ok"] is False  # still loading
    dashboard.update("Ready")
    assert not dashboard.command({"type": "submit", "text": "   "})["ok"]
    assert not dashboard.command({"type": "submit", "text": "x" * 5000})["ok"]
    reply = dashboard.command({"type": "submit", "text": " Pick the jar "})
    assert reply["ok"] and dashboard.controls.next().text == "Pick the jar"
    dashboard.update("Thinking")
    reply = dashboard.command({"type": "submit", "text": "Another"})
    assert not reply["ok"] and "working on a task" in reply["message"]
    assert dashboard.controls.next() is None
    states = [i["state"] for i in dashboard._snapshot()["instructions"]]
    assert states == ["rejected", "running", "rejected"]


def test_stop_reset_and_unsupported_controls():
    dashboard = fake_dashboard()
    dashboard.update("Ready")
    assert not dashboard.command({"type": "stop_task"})["ok"]
    dashboard.update("Thinking")
    assert dashboard.command({"type": "stop_task"})["ok"] and dashboard.controls.cancel.is_set()
    assert dashboard.controls.next().name == "stop"
    assert not dashboard.command({"type": "pause"})["ok"]
    assert not dashboard.command({"type": "stop_skill"})["ok"]
    assert not dashboard.command({"type": "launch"})["ok"]
    assert dashboard.command({"type": "reset"})["ok"] and dashboard.controls.next().name == "reset"
    assert not dashboard.command({"type": "acknowledge", "incident": "nope"})["ok"]


def test_a_supported_pause_reaches_the_controller():
    calls = []

    class Pausable(Controls):
        def pause(self):
            calls.append("pause")

        def resume(self):
            calls.append("resume")
    dashboard = fake_dashboard(controls=Pausable())
    dashboard.update("Thinking")
    assert dashboard.command({"type": "pause"})["ok"] and calls == ["pause"]
    assert dashboard._snapshot()["controls"]["pause"]


def test_session_state_is_read_for_the_plan_task_and_hands():
    dashboard = fake_dashboard()
    planner = SimpleNamespace(plan=SimpleNamespace(tree=TREE, status={"g1": "pending", "g2": "pending", "g3": "pending"},
                                                   reasons={}), goal={"id": "g1"}, replans=0, replans_allowed=2,
                              feedback="")
    dashboard.session = SimpleNamespace(history=[{"operator_task": "Pick the jar"}], planner=planner,
                                        context=SimpleNamespace(carried={"right": {"state": "verified"}}))
    dashboard.update("Thinking")
    state = dashboard._snapshot()
    assert state["task"]["text"] == "Pick the jar"
    assert state["tree"]["nodes"]["goal:g1"]["status"] == "running"
    assert state["robot"]["hands"] == {"left": "free", "right": "verified"}
    assert state["intervals"][-1]["track"] in ("goal", "thinking")


def test_frames_stream_watched_views_and_store_history():
    clock = Clock()
    dashboard = fake_dashboard(clock=clock)
    frame = SimpleNamespace(epoch=1, sequence=3, time=2.5, rgb=np.zeros((8, 12, 3), np.uint8),
                            depth=np.ones((8, 12)), labels=np.zeros((8, 12), int), base_pose=np.array([1., 2., .5]))
    dashboard.overlays.ego_rgb = lambda f: f.rgb
    dashboard.show(frame)
    view, meta, data = dashboard.hub.frames[0]
    assert view == "robot" and meta["seq"] == 3 and data[:2] == b"\xff\xd8"
    assert dashboard.hub.telemetries[0]["pose"] == [1., 2., .5]
    assert dashboard._snapshot()["history"][0][1] == clock.t
    dashboard.show(frame)  # within the frame period: not sent again
    assert len(dashboard.hub.frames) == 1


def test_once_the_task_ends_no_goal_runs_and_the_root_says_it_ended_early():
    clock = Clock()
    store = Store(clock=clock)
    run_task(store, clock)
    store.update("model_done", "Done", plan=plan("g2", g1="done"))
    nodes = store.snapshot({})["tree"]["nodes"]
    assert nodes["goal:g2"]["status"] == "pending" and nodes["root"]["status"] == "skipped"
    assert "model_done" in nodes["root"]["detail"]["note"]
    assert all(i["end"] is not None for i in store.snapshot({})["intervals"] if i["track"] == "goal")


def test_a_session_answered_decision_is_not_a_target():
    dashboard = fake_dashboard()
    frame = SimpleNamespace(epoch=1, sequence=1, time=0., rgb=np.zeros((4, 4, 3), np.uint8), base_pose=np.zeros(3))
    dashboard.overlays.ego_rgb = lambda f: f.rgb
    dashboard.select(decision("done", summary="ok"), frame)
    assert "target" not in dashboard._snapshot()["robot"]


def test_plan_mode_before_a_plan_says_it_waits_for_one():
    store = Store(clock=Clock())
    store.update("Thinking", task="Pick", plan={"tree": None, "abandoned": False})
    tree = store.snapshot({})["tree"]
    assert tree["mode"] == "planning" and "Generating the execution plan" in tree["nodes"]["root"]["detail"]["note"]
    store.update("Thinking", plan={"tree": None, "abandoned": True})
    assert "plans were invalid" in store.snapshot({})["tree"]["nodes"]["root"]["detail"]["note"]


def test_a_new_task_clears_the_previous_agent_text_and_target():
    store = Store(clock=Clock())
    store.update("Thinking", task="First")
    store.update("Executing", "Grasping")
    store.note_robot(target={"skill": "pick"}, hands={"left": "free"})
    store.update("model_done", "Done")
    store.update("Thinking", task="Second")
    state = store.snapshot({})
    assert state["agent_text"] == "" and "target" not in state["robot"] and state["robot"]["hands"]
