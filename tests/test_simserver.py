"""The HomeBody sim server and its client backend over real sockets, with a stand-in for MuJoCo: physics runs at real
time on its own, advance waits for simulated time, commands keep their epoch checks, a physics fault stops the task
until a reset, the cameras stream in warehouse-sim's format, and a harness refuses a server serving another scene."""
import json
import socket
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zmq

from homebody.backends.warehouse import SimTarget
from homebody.primitives.actions import ActionLimits, ActionRejected, PhysicsInvalid
from homebody.primitives.observations import Frame, HandState, Measurement, NavigationMap, WristView
from homebody.session.settings import Settings
from homebody.simserver import client as sim_client
from homebody.simserver.render import InlineRenderer
from homebody.simserver.server import SimServer
from homebody.simserver.stream import decode

ROOT = Path(__file__).resolve().parents[1]


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def hand():
    return HandState(np.zeros(2), 0.0, np.zeros(3), np.zeros(3))


class Backend:
    """MujocoBackend's surface the server uses: time moves one controller period per advance."""

    def __init__(self, settings):
        self.settings, self.epoch, self.time, self.sequence = settings, 1, 0.0, 0
        self.limits = ActionLimits(settings, {"arm_limits": {"left": [[-1] * 7, [1] * 7], "right": [[-1] * 7, [1] * 7]}})
        self.data = type("Data", (), {"qpos": np.zeros(3), "mocap_pos": np.zeros((0, 3)),
                                      "mocap_quat": np.zeros((0, 4)), "time": 0.0})()
        self.commands, self.fail_at, self.sink = [], None, None
        self._intrinsics = np.eye(3)
        self.navigation = NavigationMap(np.zeros((8, 8), bool), np.array([[.1, 0, 0], [0, -.1, 0], [0, 0, 1]]), .1)

    def advance(self, epoch, seconds):
        self.limits.epoch(epoch, self.epoch)
        if self.fail_at is not None and self.time >= self.fail_at:
            raise PhysicsInvalid("Robot fell or the physics solver reported invalid state")
        self.time += seconds
        self.data.time = self.time
        if self.sink is not None:
            self.sink({"time": self.time})

    def measure(self):
        self.sequence += 1
        return Measurement(self.epoch, self.sequence, self.time, np.eye(4), np.eye(4), np.zeros(3), np.zeros(29),
                           {"left": hand(), "right": hand()})

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.limits.epoch(epoch, self.epoch)
        self.commands.append(("base", forward))

    def arm_target(self, epoch, side, joints):
        self.limits.epoch(epoch, self.epoch)
        self.commands.append(("arm", side, list(joints)))

    def grip(self, epoch, side, closure):
        self.limits.epoch(epoch, self.epoch)
        self.commands.append(("grip", side, closure))

    def stop(self, epoch):
        self.commands.append(("stop",))

    def reset(self):
        self.epoch, self.fail_at = self.epoch + 1, None

    def evaluation_state(self):
        return {"time": self.time}

    def set_evaluation_sink(self, callback):
        self.sink = callback
        if callback is not None:
            callback({"time": self.time})


class Mirror:
    def render(self, kind, state, side=None):
        image = np.full((4, 6, 3), 7, np.uint8)
        if kind == "frame":
            return image, np.ones((4, 6), np.float32), np.zeros((4, 6), np.int32)
        if kind == "wrist_view":
            return {"rgb": image, "depth": np.ones((4, 6)), "labels": np.zeros((4, 6), np.int32),
                    "intrinsics": np.eye(3), "world_camera": np.eye(4)}
        if kind == "wrist_views":
            return {"left": image, "right": image}
        if kind == "stream":
            return {"ego": image, "left_wrist": image, "right_wrist": image, "observer": image}
        return image


@pytest.fixture
def served(tmp_path):
    scene = tmp_path / "kitchen_baskets"
    scene.mkdir()
    (scene / "task_scene.json").write_text(json.dumps({"default_task": "pick the apple"}))
    base = Settings.load(ROOT / "configs/simulation.toml")
    settings = replace(base, sim_server=replace(base.sim_server, endpoint=f"tcp://127.0.0.1:{port()}",
                                                stream=f"tcp://127.0.0.1:{port()}", stream_hz=20.0))
    backend = Backend(settings)
    server = SimServer(backend, InlineRenderer(Mirror()), settings, scene).start()
    target = SimTarget(settings.sim_server.endpoint, None, scene, engine="homebody")
    yield server, backend, sim_client.connect(target, ROOT, settings), scene
    server.close()


def test_physics_runs_at_real_time_and_advance_waits_for_simulated_time(served):
    server, backend, robot, _ = served
    time.sleep(0.2)
    assert backend.time > 0.1  # stepping without a client's advance
    start, wall = robot.time, time.monotonic()
    robot.advance(robot.epoch, 0.3)
    assert robot.time - start >= 0.3 - 1e-6
    assert time.monotonic() - wall == pytest.approx(0.3, abs=0.15)
    assert 0.8 < server.real_time_factor() < 1.2


def test_the_robot_calls_reach_the_backend_with_their_epoch_checks(served):
    _, backend, robot, _ = served
    epoch = robot.epoch
    robot.base_velocity(epoch, 0.2, 0.0, 0.0)
    robot.arm_target(epoch, "left", np.zeros(7))
    robot.grip(epoch, "right", 0.5)
    assert backend.commands[:3] == [("base", 0.2), ("arm", "left", [0.0] * 7), ("grip", "right", 0.5)]
    with pytest.raises(ActionRejected, match="expired"):
        robot.base_velocity(epoch + 5, 0.1, 0.0, 0.0)
    with pytest.raises(ActionRejected, match="whole controller periods"):
        robot.advance(epoch, 0.0011)
    assert isinstance(robot.measure(), Measurement)
    frame = robot.snapshot()
    assert isinstance(frame, Frame) and frame.rgb.shape == (4, 6, 3) and frame.navigation.occupied.shape == (8, 8)
    view = robot.wrist_view("left")
    assert isinstance(view, WristView) and view.side == "left"
    assert set(robot.wrist_views()) == {"left", "right"} and robot.overview_frame().shape == (4, 6, 3)


def test_a_physics_fault_stops_the_task_until_a_reset(served):
    server, backend, robot, _ = served
    backend.fail_at = backend.time
    with pytest.raises(PhysicsInvalid, match="fell"):
        robot.advance(robot.epoch, 0.2)
    assert sim_client.describe(server.config.endpoint)["controller"]["running"] is False
    robot.reset()
    assert sim_client.describe(server.config.endpoint)["controller"]["running"] is True
    robot.advance(robot.epoch, 0.04)


def test_physics_samples_reach_the_session_after_each_advance(served):
    _, _, robot, _ = served
    seen = []
    robot.set_evaluation_sink(seen.append)
    robot.advance(robot.epoch, 0.1)
    assert len(seen) > 10 and seen[-1]["time"] > seen[0]["time"]
    robot.set_evaluation_sink(None)


def test_the_cameras_stream_in_warehouse_sims_format(served):
    server, _, _, _ = served
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 3000)
    sub.connect(server.config.stream)
    seen = set()
    deadline = time.monotonic() + 5
    while len(seen) < 4 and time.monotonic() < deadline:
        _, payload = sub.recv_multipart()
        frame = decode(payload)
        seen.add(frame["camera"])
        assert frame["rgb"].shape == (4, 6, 3) and "sim_time_s" in frame["stamp"]
    sub.close(0)
    assert seen == {"ego", "left_wrist", "right_wrist", "observer"}


def test_a_harness_refuses_a_server_serving_another_scene(served, tmp_path):
    server, _, _, _ = served
    other = tmp_path / "warehouse_cart"
    other.mkdir()
    (other / "task_scene.json").write_text("{}")
    with pytest.raises(ValueError, match="another scene"):
        sim_client.connect(SimTarget(server.config.endpoint, None, other, engine="homebody"), ROOT, None)
    described = sim_client.describe(server.config.endpoint)
    assert described["engine"] == "homebody" and described["profile"]["id"] == "homebody-kitchen_baskets"
    assert set(described["cameras"]) == {"ego", "left_wrist", "right_wrist", "observer"}


def test_a_server_that_does_not_answer_is_a_lost_link():
    with pytest.raises(sim_client.SimLinkLost):
        sim_client.Link(f"tcp://127.0.0.1:{port()}", timeout_s=0.3).call("describe")


def test_advance_keeps_a_fixed_control_period_however_long_the_client_works_between_steps(served):
    _, _, robot, _ = served
    epoch = robot.epoch
    robot.advance(epoch, 0.04)
    start = robot.time
    for _ in range(10):
        time.sleep(0.02)  # the client's own work inside its 40 ms period
        robot.advance(epoch, 0.04)
    assert robot.time - start == pytest.approx(0.40, abs=0.03)  # ten periods, not ten periods plus the work
    time.sleep(0.3)  # far behind: the timeline starts again from now rather than catching up
    before = robot.time
    robot.advance(epoch, 0.04)
    assert robot.time - before == pytest.approx(0.04, abs=0.02)
