"""A recorded rollout replayed: the reader rebuilds what the console showed at a time, and the console's Rollouts
folder opens, exports, imports and evaluates rollouts."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from fakes import mocked_console
from test_monitor_offline import SETTINGS, Decisions, Vocabularies, rollout

from homebody.monitor.offline import evaluate
from homebody.rollout.bundle import export
from homebody.rollout.reader import Rollout
from homebody.ui.rollouts import RolloutsPanel


def test_the_reader_rebuilds_the_plan_skill_and_monitor_at_a_time(tmp_path):
    root = rollout(tmp_path)
    evaluate(root, SETTINGS, Vocabularies(), Decisions(fail_after=2.0))
    replay = Rollout(root, source="offline")
    assert replay.source == "offline" and replay.has_live and replay.has_offline
    assert (replay.t0, replay.t1) == (0.0, 3.5) and replay.unreadable == 0
    early, late = replay.view(1.0), replay.view(3.5)
    assert early.plan["active"] == "g1" and early.goal_id == "g1" and early.phase.stage == "Approaching"
    assert early.spans == ((0.0, None, "pick", "g1", None),) and not early.verdicts
    assert [entry[0] for entry in early.history] == [0.5, 1.0]
    assert late.verdicts and late.verdicts[0].action == "abort_skill" and late.history[-1][4] == "failure"
    assert replay.frame_at(1.2).name == "rgb.png" and replay.frame_at(-1) is None
    assert replay.doing_at(1.0) == "pick: Approaching"
    live = Rollout(root)  # the fixture's live readings are not readings: counted, not trusted
    assert live.source == "live" and live.readings == [] and live.unreadable == 1


def panel(monkeypatch, tmp_path, models=None):
    from homebody.session.config import Monitor
    console = mocked_console(monkeypatch, monitor=Monitor(enabled=True))
    return console, RolloutsPanel(console, runs=tmp_path / "runs", imported=tmp_path / "imported",
                                  settings=SETTINGS, models=models)


def test_the_rollouts_folder_opens_a_replay_and_returns_to_live(monkeypatch, tmp_path):
    root = rollout(tmp_path / "runs")
    console, rollouts = panel(monkeypatch, tmp_path)
    assert rollouts.choice.value == root.name and "sim rollout" in rollouts.about.content
    rollouts.open()
    assert console.replaying and "Replay" in console.strip.content and "g1" in console.plan_view.content
    console.update("Thinking")  # the live run goes on underneath without drawing over the replay
    assert "Replay" in console.strip.content and console._state == "Thinking"
    rollouts.draw(2.0)
    assert console.ego.image.shape == (8, 8, 3) and "2.0 s" in console.ego.label
    rollouts.close()
    assert not console.replaying and "Replay" not in console.strip.content


def test_the_rollouts_folder_exports_imports_and_evaluates(monkeypatch, tmp_path):
    source = rollout(tmp_path / "runs")
    decisions = Decisions(fail_after=2.0)
    _, rollouts = panel(monkeypatch, tmp_path, models=lambda: (Vocabularies(), decisions))
    client = MagicMock()
    rollouts.export(client)
    name, content = client.send_file_download.call_args.args
    assert name == f"{source.name}.tar.gz" and content[:2] == b"\x1f\x8b"
    archive = export(source, tmp_path / "out")
    rollouts.upload.value = SimpleNamespace(name=archive.name, content=archive.read_bytes())
    rollouts.import_()
    assert rollouts.choice.value == f"imported/{source.name}" and "Imported" in rollouts.message.content
    rollouts.plan_evaluation()
    assert "decision calls" in rollouts.message.content and rollouts.confirm_button.visible
    rollouts.evaluate()
    assert "interventions it would have made" in rollouts.message.content and decisions.requests
    assert "evaluated offline" in rollouts.about.content
    assert (tmp_path / "imported" / source.name / "tasks/task-000/monitor-offline/report.json").is_file()


def test_a_rollout_without_turns_takes_its_goals_from_the_readings_and_its_decisions_from_the_agent(tmp_path):
    import json
    root = rollout(tmp_path, turns=False)
    with (root / "events.jsonl").open("a") as events:
        events.write(json.dumps({"kind": "decision", "simulation_time": 1.0,
                                 "decision": {"text": "Pick it.", "skill": "pick"}}) + "\n")
    evaluate(root, SETTINGS, Vocabularies(), Decisions(fail_after=99))
    view = Rollout(root, source="offline").view(2.0)
    assert view.plan is None and view.goals == ((0.0, "task"),) and view.decisions == (1.0,)
    assert view.spans[0][3] == "task" and view.skill_call == view.reading.phase.call


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
