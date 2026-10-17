"""Interleave actual console/startup methods without workers, services or robot calls."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from homebody.real2sim.exploration import app
from homebody.real2sim.exploration.explorer import Explorer


@pytest.fixture
def console(monkeypatch, tmp_path):
    quit_now = threading.Event()
    source = SimpleNamespace(
        stop_flag=threading.Event(), zero=Mock(), hold_watchdog=Mock(return_value="fake hold"),
        release_watchdog=Mock(), drive_log=SimpleNamespace(flush=Mock()), feeds=Mock(),
        listen_body=Mock(side_effect=AssertionError("body contact")))
    explorer = SimpleNamespace(
        source=source, phase="restarting SLAM", message="", thread=None, stop_why="",
        _decision_generation=0, _resume_event=threading.Event(), _invalidate_question=Mock(),
        save_map=Mock(), reset_frame=Mock())
    explorer.stop = Mock(side_effect=lambda why: Explorer.stop(explorer, why))
    explorer.start = Mock(side_effect=lambda: setattr(explorer, "phase", "exploring"))
    view = app.ExplorationView(explorer, tmp_path, quit_now, bag_dir=str(tmp_path / "bag"))
    view.restart_slam = SimpleNamespace(value=True)
    view.map_display = SimpleNamespace(render=Mock())
    view.rec = SimpleNamespace(content="")
    view.robot = SimpleNamespace(visible=True)
    view.bag_lines = Mock(return_value="synthetic recording state")
    view.streams_line = Mock(return_value=("", ""))
    recorder = {"running": False}
    view.recorder_running = Mock(side_effect=lambda **_kwargs: recorder["running"])
    tracebacks = Mock()
    monkeypatch.setattr(app, "traceback", SimpleNamespace(print_exc=tracebacks))
    monkeypatch.setattr(app, "current", lambda: SimpleNamespace(
        exploration=SimpleNamespace(observe_only=True), body=SimpleNamespace(address="unused")))
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=Mock(return_value=Mock())))
    # One deterministic status iteration is interleaved inside the fake restart/start call.
    monkeypatch.setattr(app, "time", SimpleNamespace(
        time=lambda: 10., monotonic=lambda: 10., thread_time=lambda: 0.,
        sleep=lambda _seconds: quit_now.set()))
    monkeypatch.setattr(app, "EnvironmentView", Mock(return_value=Mock()))

    def start_recorder(*_args, **_kwargs):
        recorder["running"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    process = Mock(side_effect=start_recorder)
    monkeypatch.setattr(app.subprocess, "run", process)

    def tick():
        view.run()
        assert view.status_log.n == 0
        quit_now.clear()

    source.restart_slam = Mock(return_value=iter(("SLAM ready",)))
    return SimpleNamespace(view=view, source=source, explorer=explorer, recorder=recorder,
                           process=process, tracebacks=tracebacks, tick=tick, quit_now=quit_now)


def test_status_during_slam_restart_does_not_stop_before_recording(console):
    def restart():
        yield "restarting"
        console.tick()
        yield "SLAM ready"

    console.source.restart_slam.side_effect = restart
    console.view.restart_then_explore()

    console.explorer.stop.assert_not_called()
    assert not console.source.stop_flag.is_set()
    console.source.zero.assert_called_once_with()
    console.process.assert_called_once()
    console.explorer.reset_frame.assert_called_once_with()
    console.explorer.start.assert_called_once_with()
    assert console.explorer.phase == "exploring"
    console.tracebacks.assert_not_called()


def test_status_during_recorder_start_waits_for_success(console):
    def start_recorder(*_args, **_kwargs):
        console.tick()
        console.recorder["running"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    console.process.side_effect = start_recorder
    console.view.restart_then_explore()

    console.explorer.stop.assert_not_called()
    assert not console.source.stop_flag.is_set()
    console.explorer.start.assert_called_once_with()
    console.tracebacks.assert_not_called()


@pytest.mark.parametrize("already_running", [False, True])
def test_recording_success_arms_detection_of_later_recorder_loss(console, already_running):
    console.recorder["running"] = already_running
    assert console.view.ensure_recording() == ""
    assert console.process.call_count == int(not already_running)
    console.recorder["running"] = False
    console.explorer.phase = "exploring"

    console.tick()

    console.explorer.stop.assert_called_once_with("the recorder exited")
    assert console.source.stop_flag.is_set()


def test_failed_recording_start_does_not_arm_recorder_loss_detection(console):
    console.process.side_effect = None
    console.process.return_value = SimpleNamespace(returncode=1, stdout="", stderr="fake failure")
    assert "fake failure" in console.view.ensure_recording()
    console.view.recorder_running.reset_mock()

    console.tick()

    console.explorer.stop.assert_not_called()
    console.view.recorder_running.assert_not_called()
    console.explorer.start.assert_not_called()


def test_no_recording_mode_can_restart_and_explore(console):
    console.view.bag_dir = ""
    console.view.restart_then_explore()
    console.tick()

    console.explorer.stop.assert_not_called()
    console.explorer.start.assert_called_once_with()
    console.view.recorder_running.assert_not_called()
    console.process.assert_not_called()
    console.tracebacks.assert_not_called()


def test_user_stop_during_restart_exits_without_failure_or_launch(console):
    def restart():
        yield "restarting"
        app.request_shutdown(console.quit_now, console.explorer)
        yield "restart interrupted"

    console.source.restart_slam.side_effect = restart
    console.view.restart_then_explore()

    assert console.quit_now.is_set() and console.source.stop_flag.is_set()
    assert console.explorer.stop_why == "console termination requested"
    console.source.zero.assert_not_called()
    console.process.assert_not_called()
    console.explorer.reset_frame.assert_not_called()
    console.explorer.start.assert_not_called()
    console.tracebacks.assert_not_called()
    assert "failed" not in console.explorer.message.lower()


def test_genuine_restart_failure_stays_visible_and_never_launches(console):
    console.source.restart_slam.side_effect = RuntimeError("synthetic restart failure")

    console.view.restart_then_explore()

    assert "synthetic restart failure" in console.explorer.message
    console.tracebacks.assert_called_once_with()
    console.process.assert_not_called()
    console.explorer.start.assert_not_called()


def test_internal_stop_during_restart_preserves_its_reason(console):
    def restart():
        yield "restarting"
        console.explorer.stop("synthetic localization fault")
        yield "restart interrupted"

    console.source.restart_slam.side_effect = restart
    console.view.restart_then_explore()

    assert not console.quit_now.is_set()
    assert "synthetic localization fault" in console.explorer.message
    assert "STOP pressed" not in console.explorer.message
    console.tracebacks.assert_called_once_with()
    console.source.zero.assert_not_called()
    console.process.assert_not_called()
    console.explorer.start.assert_not_called()
