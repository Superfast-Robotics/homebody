"""homebody_g1 as LeRobot sees it: registered under its type, LeRobot-shaped features, and the same commands
and records as the simulator it drives. Skipped where LeRobot is not installed (the Python 3.11 seed env)."""
import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.robots.config import RobotConfig
from lerobot_robot_homebody_sim import HomebodyG1, HomebodyG1Config

from homebody.robots import g1
from homebody.robots.surface import LeRobotSurface
from homebody.session.settings import Settings
from tests.test_robot_surface import same_state, varied_frame
from tests.test_skills import FakeBackend


def robot(tmp_path, backend=None):
    return HomebodyG1(HomebodyG1Config(calibration_dir=tmp_path), Settings(), backend=backend)


def test_the_type_is_registered_and_validates_its_options():
    assert RobotConfig.get_choice_class("homebody_g1") is HomebodyG1Config
    assert HomebodyG1Config(root=None).type == "homebody_g1"
    with pytest.raises(ValueError):
        HomebodyG1Config(hands="three_finger")
    with pytest.raises(ValueError):
        HomebodyG1Config(backend="isaac")


def test_its_features_are_lerobot_shaped(tmp_path):
    g1_robot = robot(tmp_path)
    camera = Settings().camera
    assert g1_robot.observation_features["head"] == (camera.height, camera.width, 3)
    assert g1_robot.observation_features["kLeftShoulderPitch.q"] is float
    assert set(g1_robot.action_features) == set(g1.ACTION_FEATURES)
    assert not g1_robot.is_connected and g1_robot.is_calibrated


def test_it_observes_and_commands_its_simulator(tmp_path):
    backend = FakeBackend(varied_frame())
    g1_robot = robot(tmp_path, backend)
    assert g1_robot.is_connected
    same_state(backend.snapshot(), g1.decode_frame(g1_robot.get_observation(), backend.frame.navigation))
    sent = g1_robot.send_action({"base.vx": 0.3, "base.vy": 0.0, "base.yaw_rate": 0.0})
    assert sent["base.vx"] == 0.3 and backend.commands == [(0.3, 0.0, 0.0)]
    g1_robot.advance(g1_robot.epoch, 0.5)
    np.testing.assert_allclose(backend.frame.base_pose[:2], varied_frame().base_pose[:2] +
                               0.15 * np.array([np.cos(1.0), np.sin(1.0)]))


def test_the_harness_surface_drives_it_like_the_backend(tmp_path):
    backend = FakeBackend(varied_frame())
    surface = LeRobotSurface(robot(tmp_path, backend))
    same_state(backend.measure(), surface.measure())
    surface.stop(surface.epoch)
    assert backend.stopped == 1
