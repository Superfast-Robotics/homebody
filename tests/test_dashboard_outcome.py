"""Task outcome apart from the executor's lifecycle, alerts apart from both, and who detected and recovered what."""
from types import SimpleNamespace

from homebody.ui.dashboard.outcome import evaluate, lifecycle, objectives
from homebody.ui.dashboard.store import Store


class Clock:
    def __init__(self):
        self.t = 100.

    def __call__(self):
        return self.t


HOLD = {"name": "holding", "hand": "right"}
TREE = {"type": "sequence", "children": [
    {"type": "fallback", "children": [
        {"type": "goal", "id": "pick", "goal": "pick the bag", "check": HOLD},
        {"type": "goal", "id": "retry", "goal": "reposition and pick", "check": HOLD}]},
    {"type": "goal", "id": "place", "goal": "place the bag", "check": {"name": "placed"}}]}
GOALS = {"pick": "pick the bag", "retry": "reposition and pick", "place": "place the bag"}
NO_RECOVERY = {"started": 0, "succeeded": 0, "failed": 0, "replans": 0}


def plan(active=None, tree=TREE, **status):
    return {"tree": tree, "goals": GOALS, "reasons": {"pick": "budget of 2 skill calls spent"}, "active": active,
            "status": {"pick": "pending", "retry": "pending", "place": "pending", **status}}


def ended(outcome="model_done"):
    return {"id": "k1", "text": "Tidy", "start": 0., "end": 10., "outcome": outcome}


def call(store, skill, code, goal):
    store.decision(SimpleNamespace(skill=skill, text=skill, arguments={}), goal=goal)
    store.update("Executing")
    store.skill({"type": "skill_started", "skill": skill})
    store.skill({"type": "skill_finished", "result": SimpleNamespace(skill=skill, code=code, message=code)})


def test_objectives_follow_the_success_path_through_a_fallback():
    recovered = plan(pick="failed", retry="done", place="done")["status"]
    assert [g["id"] for g in objectives(TREE, recovered)] == ["retry", "place"]
    assert [g["id"] for g in objectives(TREE, plan()["status"])] == ["pick", "place"]


def test_outcomes_need_verified_objectives_not_the_agents_word():
    verified = plan(pick="done", place="done")
    assert evaluate(ended(), verified, NO_RECOVERY)["state"] == "success"
    recovered = plan(pick="failed", retry="done", place="done")
    result = evaluate(ended(), recovered, {**NO_RECOVERY, "started": 1, "succeeded": 1})
    assert result["state"] == "success_with_recoveries" and result["label"] == "Success with recovery"
    partial = evaluate(ended(), plan(pick="done", place="failed"), NO_RECOVERY)
    assert partial["state"] == "partial" and partial["failed_objectives"] == ["place the bag"]
    assert partial["objectives"] == {"total": 2, "verified": 1, "agent_closed": 0, "failed": 1, "pending": 0}
    assert evaluate(ended(), None, NO_RECOVERY)["state"] == "unverified"  # no plan: "done" is only the agent's word
    unchecked = {**TREE, "children": [{"type": "goal", "id": "place", "goal": "place the bag"}]}
    assert evaluate(ended(), plan(tree=unchecked, place="done"), NO_RECOVERY)["state"] == "unverified"
    assert evaluate(ended("cancelled"), verified, NO_RECOVERY)["state"] == "cancelled"
    assert evaluate(ended("step_limit"), None, NO_RECOVERY)["state"] == "failed"
    assert evaluate({**ended(), "end": None}, verified, NO_RECOVERY)["state"] == "not_evaluated"


def test_lifecycle_is_the_executor_not_the_outcome():
    controls = {"paused": False, "stop_pending": False}
    assert lifecycle("model_done", controls) == "Finished"
    assert lifecycle("step_limit", controls) == "Finished"
    assert lifecycle("Thinking", controls, "planning") == "Planning"
    assert lifecycle("Executing", {**controls, "stop_pending": True}) == "Stopping"
    assert lifecycle("Executing", {**controls, "paused": True}) == "Paused"
    assert lifecycle("cancelled", controls) == "Interrupted"
    assert lifecycle("provider_error", controls) == "Error"
    assert lifecycle("Ready", controls) == "Ready"


def run_recovered_task(store):
    store.update("Thinking", task="Tidy", plan=plan("pick"))
    call(store, "pick", "NO_GRASP", "pick")
    call(store, "pick", "NO_GRASP", "pick")
    store.update("Thinking", plan=plan("retry", pick="failed"))
    call(store, "pick", "OK", "retry")
    store.update("Thinking", plan=plan("place", pick="failed", retry="done"))
    call(store, "place", "OK", "place")
    store.update("Thinking", plan=plan(None, pick="failed", retry="done", place="done"))
    store.update("model_done", "Done")


def test_a_recovered_task_reports_its_outcome_apart_from_alerts():
    store = Store(clock=Clock())
    store.monitor({"level": "anomaly", "source": "force", "reason": "Grip force high"})  # before the task: no context
    run_recovered_task(store)
    state = store.snapshot({"paused": False})
    assert state["lifecycle"] == "Finished"
    outcome = state["outcome"]
    assert outcome["state"] == "success_with_recoveries" and outcome["recoveries"]["succeeded"] == 1
    assert outcome["unresolved_alerts"] == 1  # the anomaly stays an alert: a recovery never resolves one
    alerts = [i for i in state["incidents"] if i["alert"]]
    assert [i["kind"] for i in alerts] == ["monitor"]
    failures = [i for i in state["incidents"] if i["kind"] == "skill"]
    assert len(failures) == 2 and not any(i["alert"] for i in failures)  # history, not alerts


def test_a_fallback_recovery_is_attributed_to_the_tree_and_linked_to_its_cause():
    store = Store(clock=Clock())
    run_recovered_task(store)
    incidents = {i["id"]: i for i in store.snapshot({})["incidents"]}
    goal = next(i for i in incidents.values() if i["kind"] == "goal")
    assert goal["detection"] == "Plan runner (the goal's check or skill budget)"
    started = next(i for i in incidents.values() if i["tag"] == "recovery_start")
    assert started["initiator"] == "Behavior Tree fallback" and started["caused_by"] == goal["id"]
    assert "skill executor" in started["executor"]
    done = next(i for i in incidents.values() if i["tag"] == "recovery_ok")
    assert done["caused_by"] == started["id"]
    assert all(i["task"] == goal["task"] for i in (started, done))


def test_an_anomaly_names_its_detector_and_no_recovery_initiator():
    store = Store(clock=Clock())
    store.update("Thinking", task="Tidy")
    store.monitor({"level": "anomaly", "source": "force", "reason": "Grip force high"})
    anomaly, = store.snapshot({})["incidents"]
    assert anomaly["detection"] == "Monitoring model (force)" and anomaly["initiator"] == ""


def test_only_alerts_are_acknowledged_or_resolved_and_only_by_the_operator():
    store = Store(clock=Clock())
    store.update("Thinking", task="Tidy", plan=plan("pick"))
    call(store, "pick", "NO_GRASP", "pick")
    store.monitor({"level": "anomaly", "source": "force", "reason": "Grip force high"})
    failure, anomaly = store.snapshot({})["incidents"]
    assert not store.acknowledge(failure["id"]) and not store.resolve(failure["id"])
    assert store.acknowledge(anomaly["id"]) and store.resolve(anomaly["id"]) and not store.resolve(anomaly["id"])
    store.update("model_done", "Done")
    store.update("Thinking", task="Next")
    assert store.snapshot({})["incidents"][1]["state"] == "resolved"


def test_the_task_end_frame_is_kept_as_outcome_evidence():
    store = Store(clock=Clock())
    store.update("Thinking", task="Tidy")
    store.frame_id = 7
    store.update("model_done", "Done")
    assert store.snapshot({})["outcome"]["frame"] == 7
