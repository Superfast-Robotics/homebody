"""The themed console server and the exploration console's ordered shutdown."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from homebody.ui import theme


def test_open_server_themes_docks_and_titles_one_server(monkeypatch):
    import viser

    server = MagicMock()
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(viser, "ViserServer", factory)
    assert theme.open_server("127.0.0.1", 8099, "G1 · iPhone", "explore", style="workflow css") is server
    factory.assert_called_once_with(host="127.0.0.1", port=8099)
    server.gui.configure_theme.assert_called_once()
    server.gui.main_panel.dock_right.assert_called_once_with()
    assert "workflow css" in server.gui.add_html.call_args_list[0].args[0]
    assert "G1 · iPhone" in server.gui.add_html.call_args_list[1].args[0]


@pytest.mark.parametrize("worker_alive", [False, True])
def test_live_session_recorder_closes_after_source_only_once_worker_has_exited(worker_alive):
    import threading

    from homebody.real2sim.exploration.app import close_console

    calls = []
    explorer = SimpleNamespace(
        stop=lambda _: calls.append("stop"),
        thread=SimpleNamespace(join=lambda **_: calls.append("join"), is_alive=lambda: worker_alive),
        session=SimpleNamespace(recorder=SimpleNamespace(close=lambda: calls.append("recorder"))))
    source = SimpleNamespace(close=lambda: calls.append("source"),
                             drive_log=SimpleNamespace(flush=lambda: calls.append("motion log")))
    server = SimpleNamespace(stop=lambda: calls.append("server"))
    assert close_console(threading.Event(), explorer, source, server) == int(worker_alive)
    assert calls == ["stop", "join", "source", "motion log", *(
        [] if worker_alive else ["recorder"]), "server"]


def test_live_recording_failure_is_reported_without_replacing_task_outcome(capsys):
    import threading

    from homebody.real2sim.exploration.app import close_console

    calls = []
    outcome = {"status": "done", "reason": "Evidence accepted"}
    recorder = SimpleNamespace(close=lambda: calls.append("recorder"), outcome=outcome)
    explorer = SimpleNamespace(
        stop=lambda _: None, thread=None,
        session=SimpleNamespace(recorder=recorder, recording_error="disk full"))
    source = SimpleNamespace(close=lambda: calls.append("source"))
    server = SimpleNamespace(stop=lambda: calls.append("server"))
    assert close_console(threading.Event(), explorer, source, server) == 1
    assert calls == ["source", "recorder", "server"]
    assert recorder.outcome == {"status": "done", "reason": "Evidence accepted"}
    assert "session recording failed: disk full" in capsys.readouterr().out
