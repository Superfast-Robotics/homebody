"""The warehouse-sim backend's own logic: targets, a lost link as a stopped task, the observer view, and the harness
moving to another simulation server only when idle. (Attaching to a real server is the live check.)"""
import types

import numpy as np
import pytest

from homebody.backends.warehouse import SimLinkLost, SimTarget, WarehouseRobot


def test_a_target_needs_an_endpoint_a_profile_and_a_scene():
    target = SimTarget.from_json({"endpoint": "tcp://127.0.0.1:5720", "profile": "/x/profile.toml",
                                  "scene": "/scenes/warehouse_cart", "roots": {"HOMEBODY_EFFECTIVE_ROOT": "/x"}})
    assert target.scene.name == "warehouse_cart" and target.roots == {"HOMEBODY_EFFECTIVE_ROOT": "/x"}
    with pytest.raises(ValueError, match="profile"):
        SimTarget.from_json({"endpoint": "tcp://127.0.0.1:5720", "scene": "/s"})


def robot(adapter, cameras=("ego", "observer"), snapshot=None):
    client = types.SimpleNamespace(
        snapshot=snapshot or (lambda episode, images, max_age_s: {"cameras": {"observer": {"rgb": np.ones((4, 4, 3))}}}),
        describe=lambda: {"profile": {"id": "homebody-warehouse_cart"}, "controller": {"running": True}})
    target = SimTarget("tcp://127.0.0.1:5720", "/x/profile.toml", "/scenes/warehouse_cart")
    return WarehouseRobot(target, adapter, client, {"cameras": list(cameras), "camera_stream": "tcp://127.0.0.1:5721"})


def test_a_lost_link_stops_the_task_like_failed_physics_and_other_errors_pass_through():
    from homebody.primitives.actions import ActionRejected, PhysicsInvalid

    def timeout(*args):
        raise TimeoutError("advance timed out; query command status before retrying a command")

    def rejected(*args):
        raise ActionRejected("expired Homebody epoch")
    adapter = types.SimpleNamespace(advance=timeout, grip=rejected, clock=types.SimpleNamespace(episode_id="e"))
    backend = robot(adapter)
    with pytest.raises(SimLinkLost, match="5720") as lost:
        backend.advance(1, 0.04)
    assert isinstance(lost.value, PhysicsInvalid)
    with pytest.raises(ActionRejected):
        backend.grip(1, "left", 0.5)


def test_the_observer_view_comes_from_the_server_when_its_profile_has_one():
    adapter = types.SimpleNamespace(clock=types.SimpleNamespace(episode_id="e"), time=3.5)
    assert robot(adapter).overview_frame().shape == (4, 4, 3)
    assert robot(adapter, cameras=("ego",)).overview_frame() is None
    status = robot(adapter).status()
    assert status == {"endpoint": "tcp://127.0.0.1:5720", "profile": "homebody-warehouse_cart",
                      "controller": {"running": True}, "sim_time": 3.5}


def test_the_harness_moves_to_another_simulation_server_only_when_idle(tmp_path):
    from homebody.server.links import UiLink
    from homebody.server.monitor_link import MonitorLink
    from homebody.server.serve import HarnessServer, Hub
    from homebody.session.config import Monitor
    from homebody.session.settings import Settings
    from homebody.ui.console import Controls
    hub = Hub()
    ui = UiLink(hub, quality=80, view_hz=1000.0)
    server = HarnessServer(Settings(), Controls(), ui, MonitorLink(hub, Monitor(), quality=80, root=tmp_path),
                           session_dir=None, token="t" * 32, hub=hub)
    target = {"endpoint": "tcp://127.0.0.1:5730", "profile": "/x/profile.toml", "scene": "/scenes/kitchen_baskets"}
    assert "bad simulation target" in server.attach_sim({"endpoint": "tcp://127.0.0.1:5730"})
    ui.update("Executing", "picking")
    assert "stop the running task" in server.attach_sim(target)
    ui.update("Ready")
    assert server.attach_sim(target) is None and server.controls.closing.is_set()
    next_target, task = server.take_switch()
    assert next_target.endpoint == "tcp://127.0.0.1:5730" and task is None
    assert server.state()["sim"] is None  # this session ran in-process

    class Lost:
        target = SimTarget("tcp://127.0.0.1:5720", "/p", "/s")

        def status(self):
            raise SimLinkLost("simulation server tcp://127.0.0.1:5720 does not answer")
    server.attach(tmp_path, tmp_path, lambda *a, **k: None, sim=Lost())
    assert server.state()["sim"]["connected"] is False and "does not answer" in server.state()["sim"]["error"]


def test_with_a_simulation_server_the_harness_sends_no_images_of_its_own(tmp_path):
    from homebody.server.links import UiLink
    from homebody.server.monitor_link import MonitorLink
    from homebody.session.config import Monitor
    sent = []
    hub = types.SimpleNamespace(publish=lambda message, **kw: sent.append(message))
    ui = UiLink(hub, quality=80, view_hz=1000.0)
    link = MonitorLink(hub, Monitor(), quality=80, root=tmp_path)
    link.client_connected(True)
    ui.send_views = link.send_frames = False
    ui.show_camera(np.zeros((4, 4, 3), np.uint8), "Observer view")
    ui.camera = "observer"
    ui.show_camera(np.zeros((4, 4, 3), np.uint8), "Observer view")
    assert not link.capture_due(1.0) and sent == []


def test_closing_hands_the_server_back_even_when_the_adapter_fails_to_close():
    calls = []

    def broken():
        calls.append("close")
        raise TimeoutError("lost")
    adapter = types.SimpleNamespace(close=broken)
    backend = robot(adapter)
    backend.client.release = lambda: calls.append("release") or {"released": True}
    backend.close()
    assert calls == ["close", "release"]
