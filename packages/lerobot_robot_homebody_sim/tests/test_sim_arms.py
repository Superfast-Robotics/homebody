"""The simulated tier A arms as LeRobot robots: their real followers' keys, units that round-trip, and a scripted
pick that lifts and holds the cube. The models are fetched on first use; without the network these skip."""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("lerobot")

from lerobot.robots.config import RobotConfig
from lerobot_robot_homebody_sim.arms import ARMS
from lerobot_robot_homebody_sim.sim_arm import SimArm, SimArmConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from arm_check import LIFTED, pick

FOLLOWER = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}


@pytest.fixture(params=sorted(ARMS))
def arm(request):
    try:
        ARMS[request.param].xml_path()
    except (subprocess.CalledProcessError, OSError) as error:
        pytest.skip(f"no {request.param} model: {error}")
    return request.param


def test_each_arm_is_a_registered_type_with_its_followers_keys(arm, tmp_path):
    config = RobotConfig.get_choice_class(f"{arm}_follower_sim")(calibration_dir=tmp_path)
    robot = SimArm(config)
    assert {key.removesuffix(".pos") for key in robot.action_features} == FOLLOWER
    assert robot.observation_features["front"] == (config.height, config.width, 3)


def test_a_commanded_pose_reads_back_in_the_followers_units(arm, tmp_path):
    robot = SimArm(SimArmConfig(arm=arm, calibration_dir=tmp_path))
    robot.connect()
    target = dict(robot.get_state())
    target["shoulder_pan.pos"] += 10.0
    target["gripper.pos"] = 60.0
    robot.send_action(target)
    robot.advance(robot.epoch, 1.5)
    state = robot.get_state()
    assert state["shoulder_pan.pos"] == pytest.approx(target["shoulder_pan.pos"], abs=1.5)
    assert state["gripper.pos"] == pytest.approx(60.0, abs=5.0)
    with pytest.raises(RuntimeError):
        robot.send_action({"epoch": robot.epoch - 1, **target})
    robot.disconnect()


def test_a_scripted_pick_lifts_and_holds_the_cube(arm, tmp_path):
    status, rise = pick(arm, ARMS[arm], tmp_path)
    assert status.code == "OK" and rise >= LIFTED
