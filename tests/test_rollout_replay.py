"""A recorded rollout replayed: the reader rebuilds the monitor's view at a time, a rollout replays on the
dashboard, and the dashboard's Rollouts export, import, evaluate and open rollouts."""
import time

from fakes import fake_dashboard
from test_monitor_offline import SETTINGS, Decisions, Vocabularies, rollout

from homebody.monitor.offline import evaluate
from homebody.rollout.app import RolloutsApp
from homebody.rollout.reader import Rollout
from homebody.rollout.replay import replay


def test_the_reader_takes_the_chosen_monitors_readings_and_verdicts(tmp_path):
    root = rollout(tmp_path)
    evaluate(root, SETTINGS, Vocabularies(), Decisions(fail_after=2.0))
    read = Rollout(root, source="offline")
    assert read.source == "offline" and read.has_live and read.has_offline and read.task_name == "task-000"
    assert (read.t0, read.t1) == (0.0, 3.5) and read.unreadable == 0 and read.instruction == "pick the cup"
    assert [r.sim_time for r, _ in read.readings][:2] == [0.5, 1.0] and read.verdicts[0].action == "abort_skill"
    assert read.by_goal and read.records[0]["kind"] == "frame"
    live = Rollout(root)  # the fixture's live readings are not readings: counted, not trusted
    assert live.source == "live" and live.readings == [] and live.unreadable == 1


def settled(app, timeout=10.0):
    """Wait for APP's background job; its message."""
    until = time.monotonic() + timeout
    while app.busy.locked() and time.monotonic() < until:
        time.sleep(0.02)
    return app.message


def test_a_rollout_replays_on_the_dashboard_as_it_ran(tmp_path):
    root = rollout(tmp_path)
    evaluate(root, SETTINGS, Vocabularies(), Decisions(fail_after=2.0))
    dashboard = fake_dashboard()
    replay(dashboard, Rollout(root, source="offline"), SETTINGS.monitor.tau, now=1000.0)
    store = dashboard.store
    assert store.task["text"] == "pick the cup" and store.task["end"] is not None and store.status == "model_done"
    skills = [i for i in store.intervals.values() if i.track == "skill"]
    assert [(i.label, i.status) for i in skills] == [("pick", "failure")]
    levels = [reading.level for reading in store.readings]
    assert "anomaly" in levels and levels[0] == "expected"
    assert any(i.kind == "monitor" and i.alert for i in store.incidents.values())
    assert store.task["start"] == 1000.0 - 3.5 and max(r.t for r in store.readings) <= 1000.0  # it ends "now"
    assert dashboard.history.index() and dashboard._snapshot()["tree"] is not None
    assert dashboard.clock is time.time  # what the operator does next happens now


def test_the_rollouts_export_import_evaluate_and_open(tmp_path):
    source = rollout(tmp_path / "runs")
    decisions = Decisions(fail_after=2.0)
    dashboard = fake_dashboard()
    app = RolloutsApp(dashboard, runs=tmp_path / "runs", imported=tmp_path / "imported", settings=SETTINGS,
                      models=lambda: (Vocabularies(), decisions), replays=True)
    (row,) = dashboard._snapshot()["rollouts"]["list"]
    assert row == {"name": source.name, "tasks": ["task-000"]}
    pick = {"name": source.name, "task": "task-000"}
    assert "sim rollout" in dashboard.command({"type": "rollout_describe", **pick})["message"]
    assert dashboard.command({"type": "rollout_export", **pick})["ok"]
    assert "Exported" in settled(app)
    (archive,) = (tmp_path / "runs/exports").glob("*.tar.gz")
    assert dashboard.command({"type": "rollout_import", "path": str(archive)})["ok"]
    assert "Imported" in settled(app)
    imported = {"name": f"imported/{source.name}", "task": "task-000"}
    app._listed_at = float("-inf")
    assert imported["name"] in [r["name"] for r in dashboard._snapshot()["rollouts"]["list"]]
    estimate = dashboard.command({"type": "rollout_evaluate", **imported})
    assert "decision calls" in estimate["message"] and estimate["calls"] > 0 and not decisions.requests
    assert dashboard.command({"type": "rollout_evaluate", "confirm": True, **imported})["ok"]
    assert "would have made" in settled(app) and decisions.requests
    assert (tmp_path / "imported" / source.name / "tasks/task-000/monitor-offline/report.json").is_file()
    assert dashboard.command({"type": "rollout_open", "source": "offline", **imported})["ok"]
    assert "Showing" in settled(app) and dashboard.store.task["text"] == "pick the cup"
    assert not dashboard.command({"type": "rollout_open", "name": "nonsense"})["ok"]
    assert not dashboard.command({"type": "rollout_import", "path": str(tmp_path / "missing.tar.gz")})["ok"]


def test_a_live_dashboard_does_not_open_replays_over_its_session(tmp_path):
    source = rollout(tmp_path / "runs")
    dashboard = fake_dashboard()
    RolloutsApp(dashboard, runs=tmp_path / "runs", imported=tmp_path / "imported", settings=SETTINGS)
    reply = dashboard.command({"type": "rollout_open", "name": source.name})
    assert not reply["ok"] and "homebody rollout view" in reply["message"]
    assert not dashboard.command({"type": "rollout_evaluate", "name": source.name})["ok"]  # no monitor models


def test_a_rollout_without_turns_replays_its_decisions_without_a_plan(tmp_path):
    import json
    root = rollout(tmp_path, turns=False)
    with (root / "events.jsonl").open("a") as events:
        events.write(json.dumps({"kind": "decision", "simulation_time": 1.0,
                                 "decision": {"text": "Pick it.", "skill": "pick"}}) + "\n")
    dashboard = fake_dashboard()
    replay(dashboard, Rollout(root), SETTINGS.monitor.tau, now=50.0)
    (decision,) = dashboard.store.decisions.values()
    assert decision.skill == "pick" and decision.t == 50.0 - 3.5 + 1.0 and dashboard.store.plan is None


def test_a_replay_skips_frames_and_events_an_imported_rollout_points_elsewhere(tmp_path):
    import json
    root = rollout(tmp_path)
    secret = tmp_path / "secret"
    secret.mkdir()
    with (root / "events.jsonl").open("a") as events:
        events.write(json.dumps({"kind": "frame", "simulation_time": 9.0, "path": str(secret)}) + "\n")
        events.write(json.dumps({"kind": "frame", "simulation_time": 9.5, "path": "../secret"}) + "\n")
        events.write(json.dumps({"kind": "frame", "simulation_time": 9.7}) + "\n")
        events.write(json.dumps({"kind": "monitor", "type": "fired", "verdict": {"nonsense": 1}}) + "\n")
        events.write(json.dumps({"kind": "monitor", "type": "vocabulary", "labels": [{"bad": 1}]}) + "\n")
    replay = Rollout(root)
    assert all(secret not in path.parents and path != secret for _, path in replay.frames)
    assert replay.t1 == 3.5 and replay.verdicts == []


def test_a_replay_dashboard_runs_no_robot_and_shows_the_last_recorded_frame(tmp_path):
    root = rollout(tmp_path)
    dashboard = fake_dashboard()
    dashboard.read_only = True
    replay(dashboard, Rollout(root), SETTINGS.monitor.tau)
    reply = dashboard.command({"type": "submit", "text": "pick it again"})
    assert not reply["ok"] and "recorded rollout" in reply["message"] and dashboard.controls.next() is None
    controls = dashboard._snapshot()["controls"]
    assert not controls["accepting"] and not controls["pause"] and not controls["stop_task"]
    view, meta, data = dashboard.hub.frames[-1]
    assert view == "robot" and meta["label"] == "Last recorded frame" and data[:2] == b"\xff\xd8"
