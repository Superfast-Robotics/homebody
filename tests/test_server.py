"""The harness server and its client over a real socket: who may connect and command, epochs, the HTTP endpoints,
a pause when the commander leaves, and the client's monitor stopping a skill on the server."""
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_monitor_cadence import VOCAB, Decisions, Vocabularies
from test_rollout_bundle import session as recorded_session
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from homebody.monitor.monitor import Monitor
from homebody.monitor.targets import NodeInfo, Phase, VocabularyRequest
from homebody.server import protocol
from homebody.server.client import HarnessClient, RemoteControls
from homebody.server.links import UiLink
from homebody.server.monitor_link import MonitorLink
from homebody.server.serve import HarnessServer, Hub, load_token
from homebody.session.config import Monitor as MonitorSettings
from homebody.session.config import Server
from homebody.session.settings import Settings
from homebody.ui.controls import Controls

TOKEN = "t" * 32
MONITOR = MonitorSettings(enabled=True, min_period_s=0.01, p95_margin=1.0, window_frames=2, ticks=2, ema=1.0,
                          grace_s=0.0, cooldown_s=0.0, refresh_on_skill=False, capture_hz=10.0)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def harness(tmp_path):
    settings = Settings(monitor=MONITOR, server=Server(port=free_port(), view_hz=1000.0))
    hub, controls = Hub(), Controls()
    ui = UiLink(hub, quality=80, view_hz=1000.0)
    session_dir = recorded_session(tmp_path)
    monitor = MonitorLink(hub, MONITOR, quality=80, root=session_dir)
    server = HarnessServer(settings, controls, ui, monitor, session_dir=session_dir, token=TOKEN, hub=hub)
    server.start()
    yield server
    server.stop()


def url(server, path=""):
    return f"ws://127.0.0.1:{server.port}{path}"


def hello(server, role="commander", token=TOKEN):
    connection = connect(url(server, "/stream"))
    connection.send(protocol.text("hello", token=token, role=role))
    return connection, json.loads(connection.recv(timeout=5))


def wait_for(condition, timeout=5.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until and not condition():
        time.sleep(0.01)
    return condition()


def test_a_bad_token_is_refused_and_a_second_commander_only_watches(harness):
    connection = connect(url(harness, "/stream"))
    connection.send(protocol.text("hello", token="wrong" * 8, role="commander"))
    with pytest.raises(ConnectionClosed):
        connection.recv(timeout=5)
    first, state = hello(harness)
    assert state["role"] == "commander" and state["epoch"] == 1 and state["session"] == harness.session_dir.name
    second, state = hello(harness)
    assert state["role"] == "viewer" and "another client" in state["notice"]
    second.send(protocol.text("submit", text="pick", epoch=1))
    assert json.loads(second.recv(timeout=5))["reason"] == "viewers cannot command"
    first.close()
    second.close()


def test_commands_carry_the_epoch_and_a_stop_ends_it(harness):
    commander, _ = hello(harness)
    commander.send(protocol.text("submit", text="pick the cup", epoch=1))
    assert wait_for(lambda: not harness.controls.queue.empty())
    assert harness.controls.next().text == "pick the cup"
    commander.send(protocol.text("stop", epoch=1))
    assert json.loads(commander.recv(timeout=5))["epoch"] == 2 and harness.controls.cancel.is_set()
    commander.send(protocol.text("pause", epoch=1))
    refused = json.loads(commander.recv(timeout=5))
    assert refused["type"] == "refused" and "epoch 1 is over" in refused["reason"]
    assert not harness.controls.paused.is_set()
    commander.send(protocol.text("camera", camera="observer", epoch=2))
    assert wait_for(lambda: harness.ui.camera == "observer")
    commander.close()


def test_http_answers_health_and_needs_the_token_for_state_and_bundles(harness):
    base = f"http://127.0.0.1:{harness.port}"
    assert urllib.request.urlopen(f"{base}/health", timeout=5).read() == b"ok\n"
    with pytest.raises(urllib.error.HTTPError) as unauthorized:
        urllib.request.urlopen(f"{base}/state", timeout=5)
    assert unauthorized.value.code == 401

    def get(path):
        return urllib.request.urlopen(urllib.request.Request(f"{base}{path}",
                                                             headers={"Authorization": f"Bearer {TOKEN}"}), timeout=10)
    assert json.loads(get("/state").read())["epoch"] == 1
    name = harness.session_dir.name
    assert get(f"/rollouts/{name}.tar.gz").read()[:2] == b"\x1f\x8b"
    assert get(f"/rollouts/{name}-task-000.tar.gz").read()[:2] == b"\x1f\x8b"
    for path in (f"/rollouts/{name}-task-9.tar.gz", "/rollouts/other.tar.gz", "/rollouts/..%2F.tar.gz"):
        with pytest.raises(urllib.error.HTTPError) as missing:
            get(path)
        assert missing.value.code == 404


def test_the_run_pauses_when_its_commander_goes_away(harness):
    commander, _ = hello(harness)
    harness.ui.update("Executing", "picking")
    commander.close()
    assert wait_for(harness.controls.paused.is_set)


def test_the_token_file_is_made_owner_only_and_reused(tmp_path):
    path = tmp_path / "secret/token"
    token = load_token(str(path))
    assert len(token) >= 32 and (path.stat().st_mode & 0o777) == 0o600 and load_token(str(path)) == token


def test_the_clients_monitor_stops_the_running_skill_on_the_server(harness, tmp_path):
    client = HarnessClient(url(harness), TOKEN, folder=tmp_path / "client")
    decisions = Decisions(activity={"missed": 0.95})
    client.monitor = Monitor(MONITOR, Vocabularies(), decisions, sink=client.monitor_event,
                             on_update=client.monitor_updated)
    client.connect()
    link = harness.monitor
    assert wait_for(lambda: link.connected)
    image = tmp_path / "head.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image)
    request = VocabularyRequest("pick the cup", "", NodeInfo("g1", "pick the cup"), (), ("pick",), "free", (image,))
    link.start_task("pick the cup", harness.session_dir / "tasks/task-000")
    link.goal(request, "free")
    link.skill_started(replace(request, skill="pick", images=()), 0.0)
    link.set_phase(Phase("tool_call", "pick", "Approaching"))
    t = 0.0
    while link.stop_requested() is None and t < 20:
        if link.capture_due(t):
            link.capture(np.zeros((8, 8, 3), np.uint8), t)
        t += 0.05
        time.sleep(0.01)
    assert link.stop_requested() and "missed" in link.stop_requested()
    assert wait_for(lambda: bool(link.verdicts))
    verdicts = link.take_verdicts()
    assert verdicts[0].action == "abort_skill" and verdicts[0].goal_id == "g1"
    assert client.monitor.vocabularies and decisions.requests and VOCAB.labels
    assert (tmp_path / "client" / harness.session_dir.name / "task-000/monitor/readings.jsonl").is_file()
    link.skill_finished("MONITOR_STOPPED", t)
    assert link.stop_requested() is None
    link.client_stop("late stop of an earlier call", 0)
    assert link.stop_requested() is None
    client.close()
    assert wait_for(lambda: not link.connected)



def test_malformed_and_binary_client_messages_are_refused_without_dropping_the_commander(harness):
    commander, _ = hello(harness)
    commander.send(b"\x00\x00\x00\x02{}garbage")
    assert "text messages only" in json.loads(commander.recv(timeout=5))["reason"]
    commander.send(protocol.text("monitor_stop", reason="x", call="not a number"))
    commander.send(protocol.text("monitor_verdicts", verdicts=[{"action": "abort_skill"}]))
    replies = [json.loads(commander.recv(timeout=5)) for _ in range(2)]
    # the unnamed stop is refused by the harness (an intervention event); the malformed verdicts are refused outright
    assert sorted(r["type"] for r in replies) == ["refused", "skill"]
    assert next(r for r in replies if r["type"] == "skill")["event"]["stage"] == "rejected"
    commander.send(protocol.text("pause", epoch=1))
    assert wait_for(harness.controls.paused.is_set) and harness.commander is not None
    commander.close()


def test_a_client_monitor_event_is_recorded_under_its_own_kind(tmp_path):
    recorded = []
    link = MonitorLink(Hub(), MONITOR, quality=80, root=tmp_path,
                       sink=lambda kind, **fields: recorded.append((kind, fields)))
    link.client_event({"kind": "task_finished", "status": "model_done", "remote": 1, "type": "fired"})
    link.client_event({"kind": "monitor", "type": "vocabulary", "labels": ["x" * 100_000]})
    assert recorded[0][0] == "remote_task_finished" and "remote" not in recorded[0][1]
    assert recorded[1] == ("remote_monitor", {"type": "vocabulary", "truncated": True})


def test_a_token_file_others_can_read_is_refused(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 40)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="readable by others"):
        load_token(str(path))


def test_the_remote_console_gets_the_skill_stages_but_not_events_that_draw_on_harness_frames(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    console = MagicMock()
    client = HarnessClient("ws://unused", TOKEN, console=console, folder=tmp_path)
    for event in ({"type": "skill_started", "skill": "pick"}, {"type": "skill_prepared", "plan": {"side": "right"}},
                  {"type": "skill_stage", "skill": "pick", "stage": "Approaching"},
                  {"type": "skill_finished", "result": {"skill": "pick", "code": "OK", "message": "held"}}):
        client.handle({"type": "skill", "event": event}, None)
    kinds = [call.args[0]["type"] for call in console.skill_event.call_args_list]
    assert kinds == ["skill_started", "skill_stage", "skill_finished"]
    assert console.skill_event.call_args_list[-1].args[0]["result"].code == "OK"


def test_a_dashboard_elsewhere_shows_the_sessions_state_frames_and_decisions_and_picks_the_camera(tmp_path):
    from fakes import fake_dashboard

    from homebody.server.client import RemoteControls
    recorded = []
    settings = Settings(monitor=MONITOR, server=Server(port=free_port()))
    hub, controls = Hub(), Controls()
    ui = UiLink(hub, quality=80, view_hz=1000.0)
    ui.session = SimpleNamespace(history=[{"operator_task": "pick the jar"}], planner=None,
                                 context=SimpleNamespace(carried={"right": {"state": "held"}}))
    server = HarnessServer(settings, controls, ui, MonitorLink(hub, MONITOR, quality=80, root=tmp_path),
                           session_dir=recorded_session(tmp_path), token=TOKEN, hub=hub,
                           sink=lambda kind, **fields: recorded.append((kind, fields)))
    server.start()
    try:
        client = HarnessClient(url(server), TOKEN, folder=tmp_path / "client")
        remote = RemoteControls(client)
        dashboard = fake_dashboard(controls=remote)
        client.console = dashboard
        client.connect()
        ui.update("Executing", "picking")
        assert wait_for(lambda: dashboard.store.status == "Executing")
        assert dashboard.remote_state["task"] == "pick the jar"
        assert dashboard.store.robot["hands"] == {"left": "free", "right": "held"}
        frame = SimpleNamespace(rgb=np.zeros((8, 8, 3), np.uint8), time=1.5, sequence=3, epoch=1,
                                base_pose=np.array([1.0, 2.0, 0.5]))
        ui.show(frame)
        assert wait_for(lambda: any(view == "robot" for view, _, _ in dashboard.hub.frames))
        assert wait_for(lambda: dashboard.hub.telemetries) and dashboard.hub.telemetries[-1]["pose"] == [1.0, 2.0, 0.5]
        ui.select(SimpleNamespace(text="Pick it", skill="pick", arguments={"point_normalized_1000": [500, 500]}), frame)
        assert wait_for(lambda: any(d.skill == "pick" for d in dashboard.store.decisions.values()))
        dashboard.hub.views = {"observer"}  # the page watches the observer: the harness renders that camera
        assert wait_for(lambda: ui.camera == "observer")
        remote.stop()
        assert wait_for(lambda: any(kind == "operator_command" and f["command"] == "stop" for kind, f in recorded))
        assert recorded[0][0] == "client_connected" and recorded[0][1]["role"] == "commander"
        client.close()
    finally:
        server.stop()


def test_presets_are_read_checked_and_offered_without_host_paths(tmp_path):
    from homebody.server.presets import load
    scene = tmp_path / "warehouse_cart"
    scene.mkdir()
    (scene / "scene.xml").write_text("<mujoco/>")
    (scene / "task_scene.json").write_text(json.dumps({"name": "Warehouse", "default_task": "pick a jar"}))
    assert [p.task for p in load(None, scene)] == ["pick a jar"]
    file = tmp_path / "presets.toml"
    file.write_text('[[preset]]\nname = "One jar"\nscene = "warehouse_cart"\ntask = "pick one jar"\n')
    (preset,) = load(file, scene)
    assert preset.scene == scene.resolve() and preset.public() == {"name": "One jar", "scene": "warehouse_cart",
                                                                    "task": "pick one jar"}
    file.write_text('[[preset]]\nname = "Bad"\nscene = "missing"\ntask = "x"\n')
    with pytest.raises(ValueError, match="not a scene package"):
        load(file, scene)


def test_a_preset_runs_here_or_asks_for_its_scene_and_never_mid_task(harness, tmp_path):
    from homebody.server.presets import Preset
    here, there = harness.session_dir, tmp_path / "kitchen"
    harness.scene = here
    harness.presets = {"Here": Preset("Here", here, "pick here"), "There": Preset("There", there, "pick there")}
    harness.ui.update("Ready")
    assert harness.run_preset({"name": "Here", "text": "pick it, edited"}) is None
    assert harness.controls.next().text == "pick it, edited"
    assert harness.run_preset({"name": "Nope"}) == "no such preset on this harness"
    harness.ui.update("Executing", "picking")
    assert "stop the running task" in harness.run_preset({"name": "There"})
    harness.ui.update("Ready")
    assert harness.run_preset({"name": "There"}) is None and harness.controls.closing.is_set()
    assert harness.take_switch() == (there, "pick there") and harness.take_switch() == (None, None)


def test_the_dashboard_offers_the_harness_presets_and_runs_the_chosen_one(tmp_path):
    from fakes import fake_dashboard

    from homebody.server.client import RemoteControls
    from homebody.server.commands import run_preset
    sent = []
    client = HarnessClient("ws://unused", TOKEN, folder=tmp_path)
    client.send = lambda kind, **fields: sent.append((kind, fields))
    controls = RemoteControls(client)
    dashboard = fake_dashboard(controls=controls)
    dashboard.extras["scenarios"] = lambda: client.presets
    dashboard.handlers["run_scenario"] = lambda request: run_preset(controls, request)
    client.handle({"type": "state", "presets": [{"name": "Apple", "scene": "kitchen_baskets", "task": "put the apple away"},
                                                 "not a preset"]}, None)
    assert [p["name"] for p in dashboard._snapshot()["scenarios"]] == ["Apple"]
    assert dashboard.command({"type": "run_scenario", "name": "Apple", "text": "put it away"})["ok"]
    assert sent == [("preset", {"name": "Apple", "text": "put it away"})]


def test_a_mark_reaches_the_clients_monitor_with_its_kind(tmp_path):
    """An operator note during a task is marked on the remote monitor; its kind once collided with the message's own
    type and ended the task (TypeError, 2026-10-24)."""
    import json
    from types import SimpleNamespace

    from homebody.server.client import HarnessClient
    sent = []
    link = MonitorLink(SimpleNamespace(publish=lambda message, **kw: sent.append(message)), MONITOR, quality=80,
                       root=tmp_path)
    link.client_connected(True)
    link.mark("note", "keep the other products in place", 3.5)
    fields = json.loads(sent[-1])
    assert fields["type"] == "monitor" and fields["call"] == "mark"
    marks = []
    client = HarnessClient.__new__(HarnessClient)
    client.monitor = SimpleNamespace(mark=lambda kind, text, t: marks.append((kind, text, t)))
    client.monitor_call(fields["call"], fields)
    assert marks == [("note", "keep the other products in place", 3.5)]


def test_a_pause_mid_skill_reaches_the_remote_console_at_once(harness, tmp_path):
    client = HarnessClient(url(harness), TOKEN, folder=tmp_path / "client")
    controls = RemoteControls(client)
    client.connect()
    harness.ui.update("Executing", "picking")  # a skill runs: the session reports no Paused status until it ends
    controls.pause()
    assert wait_for(lambda: harness.controls.paused.is_set() and controls.paused.is_set())
    controls.unpause()
    assert wait_for(lambda: not controls.paused.is_set())
    client.close()


def test_a_pause_the_session_lifts_itself_reaches_the_remote_console_with_the_next_status(harness, tmp_path):
    client = HarnessClient(url(harness), TOKEN, folder=tmp_path / "client")
    controls = RemoteControls(client)
    client.connect()
    controls.pause()
    assert wait_for(controls.paused.is_set)
    harness.controls.unpause()  # "continue" while paused: the session lifts the pause, no console command
    harness.ui.session = type("Session", (), {"controls": harness.controls})()
    harness.ui.update("Thinking")
    assert wait_for(lambda: not controls.paused.is_set())
    client.close()
