"""The cuRobo client helpers."""
import numpy as np

from homebody.helpers import curobo


def test_thinned_keeps_endpoints_and_limit():
    rows = np.arange(100)[:, None] * np.ones((1, 7))
    kept = curobo.thinned(rows, 10)
    assert len(kept) == 10 and kept[0, 0] == 0 and kept[-1, 0] == 99
    assert len(curobo.thinned(rows[:5], 10)) == 5


class FakeClient:
    """A planning server that records its calls: unconfigured until it is configured."""
    calls = []

    def __init__(self, address, timeout):
        self.address, self.configured = address, False

    def call(self, method, payload=None):
        FakeClient.calls.append((self.address, method))
        if method == "ping":
            return {"configured": self.configured}
        if method == "configure":
            self.configured = True
            return {}
        return {"success": True, "index": 0, "approach": [], "descent": [], "status": "ok"}

    def close(self):
        pass


def test_plan_configures_the_admitted_server_then_plans(monkeypatch, tmp_path):
    from homebody.session.settings import Settings

    monkeypatch.setattr(curobo, "PolicyClient", FakeClient)
    monkeypatch.setattr(curobo.tempfile, "gettempdir", lambda: str(tmp_path))
    FakeClient.calls = []
    planner = curobo.CuroboPlanner(Settings())
    monkeypatch.setattr(planner, "locks", lambda frame, side: {})
    monkeypatch.setattr(planner, "world_pelvis", lambda frame: np.eye(4))
    frame = type("Frame", (), {"depth": np.ones((4, 4)), "labels": np.zeros((4, 4), int), "intrinsics": np.eye(3),
                               "world_camera": np.eye(4), "arm_joints": lambda self, side: np.zeros(7)})()
    reply = planner.plan(frame, "right", [np.eye(4)], [np.eye(4)])
    assert reply["success"]
    assert [method for _, method in FakeClient.calls] == ["ping", "configure", "plan_grasp"]
    assert len({address for address, _ in FakeClient.calls}) == 1 and planner.client is None
