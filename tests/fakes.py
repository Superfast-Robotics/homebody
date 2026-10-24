"""Shared test fakes: a planning arm with no collision geometry, and a dashboard over a recording
server hub. Used by tests of the skill framework, the dashboard and the session."""

from unittest.mock import MagicMock

import numpy as np
from scipy.spatial.transform import Rotation

from homebody.helpers.geometry import transform
from homebody.session.settings import Topdown
from homebody.skills.contract import SkillFailure
from homebody.ui.console import Console
from homebody.ui.controls import Controls
from homebody.ui.dashboard import Dashboard


class EmptyCollisionGeometry:
    def boxes(self, arm, joints):
        return []

    def hand_boxes(self, arm, joints, fingers):
        return []

    def torso(self):
        return np.array([20, 20, 20]), np.eye(3), np.ones(3)


class PlanningArm:
    lower = np.full(7, -10.)
    upper = np.full(7, 10.)
    shoulder = np.array([0., 0., 1.])
    collision = EmptyCollisionGeometry()
    collision.side = "right"

    def __init__(self):
        self.failure = None
        self.failure_raised = False
        self.solves = []

    def forward(self, joints):
        if self.failure:
            self.failure_raised = True
            raise SkillFailure(self.failure, "Injected approach observation failure")
        return (transform(np.array([0., 0., 1.]) + joints[:3],
                          Rotation.from_rotvec(np.array(joints[3:6])).as_matrix()),
                np.c_[np.eye(6), np.zeros(6)])

    def solve(self, target, seed, **kwargs):
        self.solves.append((target.copy(), seed.copy()))
        return np.r_[target[:3, 3] - [0, 0, 1], Rotation.from_matrix(target[:3, :3]).as_rotvec(), 0.]


class SceneRecorder:
    """Stands in for the 3D view and records what the console asked it to draw."""
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))

    def last(self, name):
        return next(args for called, args, _ in reversed(self.calls) if called == name)


def mocked_console(monkeypatch, settings=None, monitor=None, controls=None):
    """A real Console over a mocked web server, with a recording 3D view."""
    import socket

    import viser
    server = MagicMock()
    for widget in ("add_image", "add_html", "add_markdown", "add_button", "add_text", "add_dropdown", "add_slider",
                   "add_upload_button"):
        getattr(server.gui, widget).side_effect = lambda *args, kind=widget, **kwargs: MagicMock(kind=kind)
    monkeypatch.setattr(socket, "socket", MagicMock())
    monkeypatch.setattr(viser, "ViserServer", MagicMock(return_value=server))
    console = Console(controls or Controls(), topdown_settings=settings or Topdown(pixels_per_metre=100.), monitor=monitor)
    console.scene_view = SceneRecorder()
    return console


class FakeHub:
    """The dashboard's server hub without sockets: records what the dashboard sends."""
    url = "http://127.0.0.1:0"

    def __init__(self, host, port, *, snapshot, on_command, history=None):
        self.snapshot, self.on_command, self.history = snapshot, on_command, history
        self.views, self.frames, self.forgotten, self.telemetries = {"robot"}, [], [], []
        self.changes, self.closed = 0, False

    def start(self):
        return self

    def changed(self):
        self.changes += 1

    def frame(self, view, meta, data):
        self.frames.append((view, meta, data))

    def forget(self, view):
        self.forgotten.append(view)

    def telemetry(self, values):
        self.telemetries.append(values)

    def wanted(self):
        return set(self.views)

    def close(self):
        self.closed = True


def fake_dashboard(settings=None, controls=None, clock=None):
    """A real Dashboard over a FakeHub, drawing in the calling thread, with a recording 3D view."""
    extra = {} if clock is None else {"clock": clock}
    dashboard = Dashboard(controls or Controls(), topdown_settings=settings or Topdown(pixels_per_metre=100.),
                          render_in_background=False, hub=FakeHub, **extra)
    dashboard.scene_view = SceneRecorder()
    return dashboard
