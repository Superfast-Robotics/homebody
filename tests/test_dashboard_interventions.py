"""Interventions on the dashboard: a request to stop the running skill, from the live monitor or the operator, through
the stages the components that own them report, and how every panel's data shows them."""
from fakes import fake_dashboard
from test_dashboard import Clock, decision, plan, result

from homebody.session.config import Dashboard

ACK_TIMEOUT_S = Dashboard().ack_timeout_s

ABORT = {"level": "anomaly", "kind": "observation", "source": "Live monitor", "risk": 1.0, "threshold": 0.8,
         "reason": "abort skill: stall p=1.00", "signal": "stall", "call": 1,
         "request": {"action": "stop_skill", "reason": "stall p=1.00"}}


def running(clock=None):
    """A dashboard with a pick running under goal g1."""
    dashboard = fake_dashboard(clock=clock or Clock())
    dashboard.update("Ready")
    dashboard.update("Thinking", state={"task": "pick the jar", "plan": plan("g1"), "robot": {}})
    dashboard.select(decision("pick"), None)
    dashboard.update("Executing", "picking", state={"task": "pick the jar", "plan": plan("g1"), "robot": {}})
    dashboard.skill_event({"type": "skill_started", "skill": "pick", "call": 1})
    return dashboard


def only(dashboard):
    (row,) = dashboard._snapshot()["interventions"]
    return row


def stages(row):
    return [stage for stage, *_ in row["stages"]]


def accepted(initiator="monitor", call=1):
    return {"type": "intervention", "stage": "accepted", "initiator": initiator, "action": "stop_skill",
            "skill": "pick", "note": "taken at the next check", "call": call}


def test_a_monitor_stop_goes_requested_accepted_confirmed_and_the_node_stays_running_until_the_skill_ends():
    dashboard = running()
    dashboard.monitor(ABORT)
    row = only(dashboard)
    assert row["state"] == "requested" and stages(row) == ["detected", "decided", "requested"]
    assert row["execution"].endswith("/call-1") and row["attempt"] == 1
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
    for call in (1, None):  # the executor saying so again (an older harness names no call): no second record
        dashboard.skill_event({"type": "intervention", "stage": "confirmed", "initiator": "monitor", "skill": "pick",
                               "code": "MONITOR_STOPPED", "call": call})
    row = only(dashboard)
    assert row["state"] == "confirmed" and stages(row) == ["detected", "decided", "requested", "accepted", "confirmed"]
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
    dashboard.skill_event({"type": "intervention", "stage": "rejected", "initiator": "monitor", "call": 1,
                           "note": "The request named skill call 1; call 2 runs now"})
    row = only(dashboard)
    assert row["state"] == "rejected" and row["stages"][-1][2] == "Harness (monitor link)"


def test_an_acceptance_that_arrives_before_the_request_is_filled_in_when_the_request_comes():
    dashboard = running()
    dashboard.skill_event(accepted())
    row = only(dashboard)
    assert row["stages"][0][1] is None and "did not reach" in row["stages"][0][3]
    dashboard.monitor(ABORT)
    row = only(dashboard)
    assert stages(row) == ["detected", "decided", "requested", "accepted"] and row["stages"][2][1] is not None
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
    for call, code in ((2, "MONITOR_STOPPED"), (3, "NO_GRASP")):
        dashboard.skill_event({"type": "skill_finished", "result": result("pick", code)})
        dashboard.select(decision("pick"), None)
        dashboard.update("Executing", "again", state={"task": "pick the jar", "plan": plan("g1"), "robot": {}})
        dashboard.skill_event({"type": "skill_started", "skill": "pick", "call": call})
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


def next_attempt(dashboard, call, code="MONITOR_STOPPED"):
    """End the running attempt with CODE and start the goal's next one as skill call CALL."""
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", code)})
    dashboard.select(decision("pick"), None)
    dashboard.update("Executing", "again", state={"task": "pick the jar", "plan": plan("g1"), "robot": {}})
    dashboard.skill_event({"type": "skill_started", "skill": "pick", "call": call})


def test_an_abort_on_attempt_1_stays_with_attempt_1_after_attempt_2_starts():
    dashboard = running()
    dashboard.monitor(ABORT)
    dashboard.skill_event(accepted())
    next_attempt(dashboard, 2)
    dashboard.monitor(ABORT)  # a late reading the monitor took during call 1, arriving now: still attempt 1's
    state = dashboard._snapshot()
    first, second = [t for t in state["tools"]]
    (row,) = state["interventions"]
    assert row["tool"] == first["id"] and row["state"] == "confirmed" and row["attempt"] == 1
    nodes = state["tree"]["nodes"]
    assert nodes[f"tool:{second['id']}"]["detail"]["interventions"] == []  # the running attempt inherits nothing
    assert all(r["tool"] == first["id"] for r in state["readings"])
    assert [i["tool"] for i in state["incidents"] if i["kind"] == "monitor"] == [first["id"]]


def test_a_stall_on_attempt_2_is_its_own_incident_and_request():
    dashboard = running()
    dashboard.monitor(ABORT)
    next_attempt(dashboard, 2)
    dashboard.monitor({**ABORT, "call": 2})
    state = dashboard._snapshot()
    tools = [t["id"] for t in state["tools"]]
    assert [i["tool"] for i in state["incidents"] if i["kind"] == "monitor"] == tools
    assert [(r["tool"], r["attempt"]) for r in state["interventions"]] == [(tools[0], 1), (tools[1], 2)]


def test_the_goals_next_attempt_after_a_confirmed_stop_is_its_retry_with_its_outcome():
    dashboard = running()
    dashboard.monitor(ABORT)
    dashboard.skill_event(accepted())
    next_attempt(dashboard, 2)
    assert stages(only(dashboard))[-1] == "retry_started"
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", "OK")})
    row = only(dashboard)
    assert row["state"] == "confirmed" and stages(row)[-2:] == ["retry_started", "retry_finished"]
    assert "Attempt 2 ended with OK" in row["stages"][-1][3]


def test_a_reading_that_names_no_attempt_is_bound_to_none():
    dashboard = running()
    dashboard.monitor({**ABORT, "call": None})
    state = dashboard._snapshot()
    assert state["readings"][0]["tool"] is None
    (row,) = state["interventions"]
    assert row["tool"] is None and row["execution"] == "" and "unknown" in row["stages"][-1][3]


def test_an_expired_request_stays_expired_and_a_late_acknowledgement_joins_it():
    clock = Clock()
    dashboard = running(clock)
    dashboard.monitor(ABORT)
    clock.t += ACK_TIMEOUT_S + 5
    dashboard.skill_event({"type": "intervention", "stage": "expired", "initiator": "monitor", "call": 1,
                           "note": "no acknowledgement within the controller's own deadline"})
    row = only(dashboard)
    assert row["state"] == "expired" and stages(row)[-1] == "expired"
    clock.t += 3
    dashboard.skill_event(accepted())  # the acknowledgement arrives after the request ended
    row = only(dashboard)
    assert row["state"] == "expired" and stages(row)[-1] == "accepted"  # no second record; the end stays the end
    requested = next(t for stage, t, *_ in row["stages"] if stage == "requested")
    ack = next(t for stage, t, *_ in row["stages"] if stage == "accepted")
    assert ack - requested > ACK_TIMEOUT_S  # the deadline was exceeded, which stays derivable


def test_recorded_frames_keep_their_simulation_time_and_readings_say_their_quality_is_the_dashboards():
    import numpy as np
    dashboard = running()
    dashboard.remote_frame("robot", np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8), {"sim": 4.5})
    assert dashboard._snapshot()["history"][-1][2] == 4.5
    dashboard.monitor({**ABORT, "sim_time": 4.5})
    assert dashboard.store.readings[-1].quality_relation == "synchronized"
    dashboard.monitor({**ABORT, "request": None})
    assert dashboard.store.readings[-1].quality_relation == "receipt"


def test_a_late_acknowledgement_never_reopens_a_confirmed_stop_or_one_on_another_attempt():
    dashboard = running()
    dashboard.monitor(ABORT)
    dashboard.skill_event(accepted())
    dashboard.skill_event({"type": "skill_finished", "result": result("pick", "MONITOR_STOPPED", "stopped")})
    dashboard.skill_event(accepted())  # the executor repeating itself after the stop was confirmed
    (row,) = dashboard._snapshot()["interventions"]
    assert row["state"] == "confirmed" and stages(row).count("accepted") == 1
    dashboard.skill_event(accepted(call=None))  # no attempt named and none running: a record of its own, not the old one
    rows = dashboard._snapshot()["interventions"]
    assert len(rows) == 2 and rows[0]["id"] == row["id"] and stages(rows[0]) == stages(row)
