"""The simulated tier A arms as LeRobot robots: their real followers' keys, units that round-trip, and a scripted
pick that lifts and holds the cube. The models are fetched on first use; without the network these skip."""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("lerobot")

from lerobot.robots.config import RobotConfig
from lerobot_robot_homebody_sim.arms import ARMS
from lerobot_robot_homebody_sim.rigs import RIGS
from lerobot_robot_homebody_sim.sim_arm import SimArm, SimArmConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from arm_check import LIFTED, pick

@pytest.fixture(params=sorted(RIGS))
def rig(request):
    try:
        for mount in RIGS[request.param].mounts:
            ARMS[mount.arm].xml_path()
    except (subprocess.CalledProcessError, OSError) as error:
        pytest.skip(f"no {request.param} model: {error}")
    return request.param


def followers_keys(rig):
    return {f"{m.prefix}{motor}.pos" for m in RIGS[rig].mounts for motor in (*ARMS[m.arm].motors, "gripper")}


def test_each_rig_is_a_registered_type_with_its_followers_keys(rig, tmp_path):
    config = RobotConfig.get_choice_class(f"{rig}_follower_sim")(calibration_dir=tmp_path)
    robot = SimArm(config)
    assert set(robot.action_features) == followers_keys(rig)
    assert robot.observation_features["front"] == (config.height, config.width, 3)


def test_a_commanded_pose_reads_back_in_the_followers_units(rig, tmp_path):
    robot = SimArm(SimArmConfig(rig=rig, calibration_dir=tmp_path))
    robot.connect()
    mount = RIGS[rig].mounts[-1]
    first, (closed, opened) = mount.prefix + ARMS[mount.arm].motors[0], ARMS[mount.arm].gripper_units
    target = dict(robot.get_state())
    target[f"{first}.pos"] += 10.0
    target[f"{mount.prefix}gripper.pos"] = closed + 0.6 * (opened - closed)
    robot.send_action(target)
    robot.advance(robot.epoch, 1.5)
    state = robot.get_state()
    assert state[f"{first}.pos"] == pytest.approx(target[f"{first}.pos"], abs=1.5)
    assert state[f"{mount.prefix}gripper.pos"] == pytest.approx(target[f"{mount.prefix}gripper.pos"],
                                                                abs=0.05 * abs(opened - closed))
    with pytest.raises(RuntimeError):
        robot.send_action({"epoch": robot.epoch - 1, **target})
    robot.disconnect()


def test_each_arm_picks_and_holds_its_cube(rig, tmp_path):
    for mount in RIGS[rig].mounts:
        status, rise = pick(rig, mount.prefix, tmp_path)
        assert status.code == "OK" and rise >= LIFTED, (mount.prefix, status, rise)
