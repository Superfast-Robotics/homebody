"""The G1's LeRobot dicts and the surface that reads a LeRobot-style robot as HomeBody's Robot: lossless round
trips, the action rules, and identical commands and records through the surface."""
from dataclasses import replace

import numpy as np
import pytest

from homebody.primitives.observations import HandState, WristView
from homebody.robots import g1
from homebody.robots.surface import LeRobotSurface
from homebody.skills.contract import Context, Runner
from tests.test_skills import FakeBackend, frame


def varied_frame():
    rng = np.random.default_rng(3)
    hands = {side: HandState(rng.random(7), 0.4 + i / 10, rng.random(3), rng.random(3))
             for i, side in enumerate(("left", "right"))}
    return replace(frame(epoch=2, sequence=7, time=1.25), rgb=rng.integers(0, 255, (6, 8, 3), dtype=np.uint8),
                   depth=rng.random((6, 8)), labels=rng.integers(-1, 4, (6, 8)), joints=rng.random(29),
                   world_camera=np.diag([1.0, -1.0, -1.0, 1.0]), world_torso=np.eye(4) * 2,
                   base_pose=np.array([0.5, -0.25, 1.0]), hands=hands)


def same_state(a, b):
    assert (a.epoch, a.sequence, a.time) == (b.epoch, b.sequence, b.time)
    for name in ("world_camera", "world_torso", "base_pose", "joints"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for side in ("left", "right"):
        for name in ("joints", "finger_effort_fraction", "finger_closure_gap"):
            np.testing.assert_array_equal(getattr(a.hands[side], name), getattr(b.hands[side], name))
        assert a.hands[side].closure == b.hands[side].closure


def test_a_frame_survives_the_round_trip_through_its_dict():
    original = varied_frame()
    observation = g1.encode_frame(original)
    assert set(observation) == set(g1.observation_features(6, 8))
    decoded = g1.decode_frame(observation, original.navigation)
    same_state(original, decoded)
    for name in ("rgb", "depth", "labels", "intrinsics"):
        np.testing.assert_array_equal(getattr(original, name), getattr(decoded, name))
    assert decoded.navigation is original.navigation


def test_a_measurement_and_a_wrist_view_survive_their_round_trips():
    original = varied_frame()
    state = g1.encode_state(original)
    assert set(state) == set(g1.state_features())
    same_state(original, g1.decode_measurement(state))
    view = WristView(2, 1.25, "left", original.rgb, original.depth, original.labels, original.intrinsics, np.eye(4))
    observation = g1.encode_wrist(view)
    assert set(observation) == set(g1.wrist_features("left", 6, 8))
    back = g1.decode_wrist(observation, "left")
    assert (back.epoch, back.time, back.side) == (2, 1.25, "left")
    np.testing.assert_array_equal(back.depth, view.depth)


def test_the_joint_keys_follow_lerobots_g1_order_with_the_arms_at_15():
    assert len(g1.JOINTS) == 29 and len(set(g1.JOINTS)) == 29
    assert g1.ARM_JOINTS["left"][0] == "kLeftShoulderPitch" == g1.JOINTS[15]
    assert g1.ARM_JOINTS["right"][-1] == "kRightWristYaw" == g1.JOINTS[28]


class Recorder:
    def __init__(self):
        self.epoch, self.calls = 4, []

    def base_velocity(self, *args):
        self.calls.append(("base", *args))

    def arm_target(self, epoch, side, joints):
        self.calls.append(("arm", epoch, side, tuple(joints)))

    def grip(self, *args):
        self.calls.append(("grip", *args))


def test_an_action_commands_what_it_names_at_its_epoch():
    robot = Recorder()
    arm = {f"{joint}.q": float(i) for i, joint in enumerate(g1.ARM_JOINTS["right"])}
    g1.apply_action(robot, {**arm, "left_hand.closure": 0.5, "base.vx": 0.1, "base.vy": 0.0, "base.yaw_rate": 0.2})
    assert ("base", 4, 0.1, 0.0, 0.2) in robot.calls
    assert ("arm", 4, "right", tuple(range(7))) in robot.calls
    assert ("grip", 4, "left", 0.5) in robot.calls
    g1.apply_action(robot, {"epoch": 9, "right_hand.closure": 1.0})
    assert robot.calls[-1] == ("grip", 9, "right", 1.0)


@pytest.mark.parametrize("action", [{"base.vx": 0.1}, {"kLeftElbow.q": 0.3}, {"left_hand.width": 1.0}])
def test_a_partial_base_or_arm_or_an_unknown_key_is_refused(action):
    with pytest.raises(ValueError):
        g1.apply_action(Recorder(), action)


class DictRobot:
    """A LeRobot-style simulated robot over FakeBackend: everything crosses as g1 dicts."""

    def __init__(self, backend):
        self.backend, self.disconnected = backend, False

    epoch = property(lambda self: self.backend.epoch)
    time = property(lambda self: self.backend.time)
    navigation = property(lambda self: self.backend.frame.navigation)
    simulator = property(lambda self: self.backend)

    def get_observation(self):
        return g1.encode_frame(self.backend.snapshot())

    def get_state(self):
        return g1.encode_state(self.backend.measure())

    def send_action(self, action):
        return g1.apply_action(self.backend, action)

    def advance(self, epoch, seconds):
        self.backend.advance(epoch, seconds)

    def stop(self, epoch):
        self.backend.stop(epoch)

    def disconnect(self):
        self.disconnected = True


def test_the_surface_reads_and_commands_the_robot_as_the_backend_would():
    backend = FakeBackend(varied_frame())
    surface = LeRobotSurface(DictRobot(backend))
    same_state(backend.snapshot(), surface.snapshot())
    same_state(backend.measure(), surface.measure())
    surface.base_velocity(surface.epoch, 0.2, 0.0, 0.1)
    surface.advance(surface.epoch, 0.5)
    assert backend.commands == [(0.2, 0.0, 0.1)] and surface.time == pytest.approx(1.75)
    surface.stop(surface.epoch)
    assert backend.stopped == 1
    surface.close()
    assert surface.robot.disconnected


def test_a_skill_runs_the_same_through_the_surface():
    def walk(robot):
        events = []
        start = robot.snapshot()
        result = Runner(Context(robot, robot, {}, emit=events.append)).run(
            "navigate", {"goal_xy_m": [1.4, 1.0], "facing_xy_m": [2.0, 1.0]}, start)
        return result, [event["type"] for event in events]

    direct, through = FakeBackend(), FakeBackend()
    expected = walk(direct)
    assert walk(LeRobotSurface(DictRobot(through))) == expected
    assert direct.commands and through.commands == direct.commands
