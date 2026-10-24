"""Interventions on the dashboard: a request to stop the running skill, from the live monitor or the operator, through
the stages the components that own them report, and how every panel's data shows them."""
from fakes import fake_dashboard
from test_dashboard import Clock, decision, plan, result

from homebody.ui.dashboard.interventions import ACK_TIMEOUT_S

ABORT = {"level": "anomaly", "kind": "observation", "source": "Live monitor", "risk": 1.0, "threshold": 0.8,
         "reason": "abort skill: stall p=1.00", "signal": "stall",
         "request": {"action": "stop_skill", "reason": "stall p=1.00"}}


def running(clock=None):
    """A dashboard with a pick running under goal g1."""
    dashboard = fake_dashboard(clock=clock or Clock())
    dashboard.update("Ready")
    dashboard.update("Thinking", state={"task": "pick the jar", "plan": plan("g1"), "robot": {}})
    dashboard.select(decision("pick"), None)
    dashboard.update("Executing", "picking", state={"task": "pick the jar", "plan": plan("g1"), "robot": {}})
    dashboard.skill_event({"type": "skill_started", "skill": "pick"})
    return dashboard


def only(dashboard):
    (row,) = dashboard._snapshot()["interventions"]
    return row


def stages(row):
    return [stage for stage, *_ in row["stages"]]


def accepted(initiator="monitor"):
    return {"type": "intervention", "stage": "accepted", "initiator": initiator, "action": "stop_skill",
            "skill": "pick", "note": "taken at the next check"}


def test_a_monitor_stop_goes_requested_accepted_confirmed_and_the_node_stays_running_until_the_skill_ends():
    dashboard = running()
    dashboard.monitor(ABORT)
    row = only(dashboard)
    assert row["state"] == "requested" and stages(row) == ["detected", "requested"]
    assert row["initiator"] == "Live monitor" and row["target"] == "Skill executor (session)"
    assert row["detection"].startswith("Monitoring model") and row["physical"] == "unverified"
    incident = next(i for i in dashboard._snapshot()["incidents"] if i["kind"] == "monitor")
    assert row["incident"] == incident["id"]
    dashboard.monitor(ABORT)  # the next reading repeats the request: the same request
    assert only(dashboard)["id"] == row["id"]
    dashboard.skill_event(accepted())
    state = dashboard._snapshot()
    assert only(dashboard)["state"] == "executing"
    tool = next(n for n in state["tree"]["nodes"].values() if n["type"] == "action")
    assert tool["status"] == "running" and tool["detail"]["interventions"][0]["state"] == "executing"
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", "MONITOR_STOPPED", "stopped")})
    dashboard.skill_event({"type": "intervention", "stage": "confirmed", "initiator": "monitor", "skill": "pick",
                           "code": "MONITOR_STOPPED"})  # the executor saying so again: no second record
    row = only(dashboard)
    assert row["state"] == "confirmed" and stages(row) == ["detected", "requested", "accepted", "confirmed"]
    assert "MONITOR_STOPPED" in row["stages"][-1][2]
    incident = next(i for i in dashboard._snapshot()["incidents"] if i["id"] == row["incident"])
    assert "stopped the skill" in incident["outcome"]


def test_an_unanswered_request_times_out_and_a_late_confirmation_still_shows():
    clock = Clock()
    dashboard = running(clock)
    dashboard.monitor(ABORT)
    clock.t += ACK_TIMEOUT_S + 1
    assert only(dashboard)["state"] == "timed_out"
    dashboard.skill_event(accepted())
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", "MONITOR_STOPPED")})
    assert only(dashboard)["state"] == "confirmed"


def test_a_skill_that_ends_first_cancels_the_request_and_one_that_took_it_but_did_not_stop_fails_it():
    dashboard = running()
    dashboard.monitor(ABORT)
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", "NO_GRASP", "no grasp")})
    row = only(dashboard)
    assert row["state"] == "cancelled" and "before taking the request" in row["stages"][-1][3]

    dashboard = running()
    dashboard.monitor(ABORT)
    dashboard.skill_event(accepted())
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", "PHYSICS_INVALID", "exploded")})
    assert only(dashboard)["state"] == "failed"


def test_a_stale_remote_request_is_rejected_by_the_harness():
    dashboard = running()
    dashboard.monitor(ABORT)
    dashboard.skill_event({"type": "intervention", "stage": "rejected", "initiator": "monitor",
                           "note": "The request named skill call 2; call 3 runs now"})
    row = only(dashboard)
    assert row["state"] == "rejected" and row["stages"][-1][2] == "Harness (monitor link)"


def test_an_acceptance_that_arrives_before_the_request_is_filled_in_when_the_request_comes():
    dashboard = running()
    dashboard.skill_event(accepted())
    row = only(dashboard)
    assert row["stages"][0][1] is None and "did not reach" in row["stages"][0][3]
    dashboard.monitor(ABORT)
    row = only(dashboard)
    assert stages(row) == ["detected", "requested", "accepted"] and row["stages"][1][1] is not None
    assert row["incident"] is not None


def test_the_operators_stop_skill_is_a_request_the_executor_confirms():
    dashboard = running()
    assert dashboard.command({"type": "stop_skill"})["ok"] and dashboard.controls.skill_stop.is_set()
    row = only(dashboard)
    assert row["initiator"] == "Human operator" and row["decided_by"] == "Human operator" and row["incident"] is None
    dashboard.skill_event(accepted("operator"))
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", "OPERATOR_STOPPED")})
    assert only(dashboard)["state"] == "confirmed"
    assert not [i for i in dashboard._snapshot()["incidents"] if i["tag"] == "stop_skill"]  # one record, not two


def test_a_task_that_ends_withdraws_its_open_requests():
    dashboard = running()
    dashboard.monitor(ABORT)
    dashboard.update("cancelled", "stopped")
    assert only(dashboard)["state"] == "cancelled"


def test_a_goal_with_several_attempts_groups_the_earlier_ones_and_keeps_the_newest_under_the_goal():
    dashboard = running()
    for code in ("MONITOR_STOPPED", "NO_GRASP"):
        dashboard.skill_event({"type": "skill_finished", "result": result("pick", code)})
        dashboard.select(decision("pick"), None)
        dashboard.update("Executing", "again", state={"task": "pick the jar", "plan": plan("g1"), "robot": {}})
        dashboard.skill_event({"type": "skill_started", "skill": "pick"})
    nodes = dashboard._snapshot()["tree"]["nodes"]
    goal = nodes["goal:g1"]
    group, newest = goal["children"]
    assert group == "attempts:g1" and nodes[group]["detail"] == {"visual": True, "count": 2, "failed": 2,
                                                                  "codes": ["MONITOR_STOPPED", "NO_GRASP"]}
    assert nodes[newest]["status"] == "running" and nodes[newest]["detail"]["number"] == 3


def test_anomalies_while_the_monitors_stop_is_open_stay_one_incident():
    clock = Clock()
    dashboard = running(clock)
    dashboard.monitor(ABORT)
    for _ in range(4):
        clock.t += 5  # further apart than one episode's readings, but the stop the first asked for is still open
        dashboard.monitor(ABORT)
    incidents = [i for i in dashboard._snapshot()["incidents"] if i["kind"] == "monitor"]
    assert len(incidents) == 1 and incidents[0]["evidence"]["readings"] == 5
