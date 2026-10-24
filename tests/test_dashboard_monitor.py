"""The live monitor on the dashboard: its readings as dashboard readings, an abort as an alert the operator resolves
(the monitor's dismiss), notes while the agent works, a pause that keeps the task, and the observer camera."""
import numpy as np
from fakes import fake_dashboard

from homebody.monitor.monitor import View
from homebody.monitor.targets import Label, Phase, Reading, Verdict, Vocabulary
from homebody.session.config import Monitor
from homebody.ui.dashboard.monitor_feed import MonitorFeed, feed

VOCAB = Vocabulary("g1", (Label("reaching", "progress", "moves the hand to the jar"),
                          Label("dropped", "failure", "the jar falls")))


IDLE = Phase()


def view(t, *, dropped=0.1, unsafe=0.05, verdict=None, calls=1, vocabulary=VOCAB, phase=IDLE, still_true=None):
    reading = Reading("g1", t, activity={"reaching": 0.8, "dropped": dropped}, unsafe=unsafe, phase=phase,
                      still_true=still_true or {})
    return View(status="watching", goal_id="g1", vocabulary=vocabulary, reading=reading, calls=calls,
                verdicts=(verdict,) if verdict else ())


def test_a_reading_takes_its_level_from_its_verdict_and_its_risk_from_the_rules_signals():
    calm = feed(view(1.0), tau=0.8)
    assert calm["level"] == "expected" and calm["risk"] == 0.1 and calm["threshold"] == 0.8
    assert calm["confidence"] == 0.8 and "dropped" in calm["reason"]
    assert feed(view(1.0, dropped=0.85), tau=0.8)["level"] == "elevated"
    flagged = feed(view(2.0, dropped=0.9, verdict=Verdict("flag", "g1", "dropped", 0.9, "the jar falls", 2.0)), 0.8)
    assert flagged["level"] == "elevated" and flagged["reason"].startswith("flag: dropped p=0.90")
    aborted = feed(view(3.0, dropped=0.95, verdict=Verdict("abort_skill", "g1", "dropped", 0.95, "", 3.0)), 0.8)
    assert aborted["level"] == "anomaly"
    assert feed(view(4.0, vocabulary=None), 0.8)["risk"] == 0.05  # before the labels: unsafe and human_needed only
    assert feed(View(), 0.8) is None


def test_a_reading_says_what_its_score_means_where_it_applies_and_what_is_not_known():
    working = feed(view(5.0, dropped=0.9, phase=Phase("tool_call", "pick", call=3)), 0.8)
    assert working["sim_time"] == 5.0 and working["call"] == 3
    assert working["scope"] == "skill_attempt" and working["scope_goal"] == "g1"
    assert "dropped" in working["semantics"] and "uncalibrated" in working["semantics"]
    assert working["calibration"] == "uncalibrated" and working["reliability"] == "unknown"
    assert "camera" in working["input_source"]
    planning = feed(view(6.0, phase=Phase("planning")), 0.8)
    assert planning["scope"] == "goal" and planning["call"] is None  # while the agent plans the rules only flag the goal
    assert feed(view(7.0), 0.8)["scope"] == "unknown"  # idle: no skill and no goal the reading is about
    undone = feed(view(8.0, dropped=0.0, unsafe=0.0, still_true={"g0": 0.1}, phase=Phase("tool_call", "pick", call=4)), 0.8)
    assert undone["signal"] == "undone:g0" and undone["scope"] == "goal" and undone["scope_goal"] == "g0"


def test_an_abort_raises_an_alert_and_resolving_it_dismisses_the_monitors_alert():
    dashboard = fake_dashboard()
    on_update = MonitorFeed(dashboard, Monitor())
    dashboard.update("Executing", "picking")
    on_update(view(1.0))
    on_update(view(1.0))  # the same reading again (a status change) is not a new reading
    abort = view(2.0, dropped=0.95, calls=2, verdict=Verdict("abort_skill", "g1", "dropped", 0.95, "", 2.0))
    on_update(abort)
    assert [r.level for r in dashboard.store.readings] == ["expected", "anomaly"]
    (incident,) = [i for i in dashboard.store.incidents.values() if i.kind == "monitor"]
    assert incident.alert and incident.state == "active"
    assert dashboard.command({"type": "resolve", "incident": incident.id})["ok"]
    assert dashboard.controls.dismissed.is_set()


def test_an_instruction_while_the_agent_works_reaches_it_as_a_note():
    dashboard = fake_dashboard()
    dashboard.update("Executing", "picking")
    reply = dashboard.command({"type": "submit", "text": "Use the left hand"})
    assert reply["ok"] and "next decision" in reply["message"]
    assert dashboard.controls.take_notes() == ["Use the left hand"] and dashboard.controls.next() is None


def test_a_command_while_paused_mid_skill_is_the_plans_next_goal():
    dashboard = fake_dashboard()
    dashboard.update("Executing", "picking")  # the pause holds the skill; the status stays Executing until it ends
    dashboard.controls.pause()
    reply = dashboard.command({"type": "submit", "text": "Turn to the cart"})
    assert reply["ok"] and "next goal" in reply["message"]
    assert dashboard.controls.take_notes() == ["Turn to the cart"]


def test_a_pause_keeps_the_task_running_and_resume_ends_a_monitor_pause():
    dashboard = fake_dashboard()
    dashboard.update("Ready")
    dashboard.update("Executing", "picking")
    dashboard.update("Paused", "The monitor asks for a person: unsafe. Press Resume to continue or Stop.")
    assert dashboard.store.busy and dashboard.store.task["end"] is None
    assert dashboard._snapshot()["controls"]["paused"]  # the page offers Resume
    (pause,) = [i for i in dashboard.store.incidents.values() if i.tag == "pause"]
    assert pause.initiator == "Live monitor" and "unsafe" in pause.reason
    assert dashboard.command({"type": "stop_skill"})["ok"]
    assert dashboard.command({"type": "resume"})["ok"] and dashboard.controls.resumed.is_set()


def test_the_observer_view_is_rendered_only_while_watched():
    dashboard = fake_dashboard()
    assert dashboard.camera == "robot"
    dashboard.hub.views = {"observer"}
    assert dashboard.camera == "observer"
    dashboard.show_camera(np.zeros((4, 4, 3), np.uint8), "Observer view")
    view_name, meta, data = dashboard.hub.frames[-1]
    assert view_name == "observer" and meta["label"] == "Observer view" and data[:2] == b"\xff\xd8"
    assert dashboard._snapshot()["views"]["observer"]["available"]
