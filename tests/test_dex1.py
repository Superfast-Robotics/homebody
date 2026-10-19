"""The Dex1-1 gripper profile is a separate robot: its own model, calibration and pick and place,
chosen by settings.robot.hands, while the default (Dex3) robot and skills stay as they were."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from homebody.backends.mujoco import hand_state
from homebody.helpers.kinematics import Arm
from homebody.session.config import Robot
from homebody.session.settings import Settings
from homebody.skills import registry as registry_module
from homebody.skills.registry import REGISTRY, registry

ROOT = Path(__file__).resolve().parents[1]
DEX1 = ROOT / "assets/robot/dex1"
needs_profile = pytest.mark.skipif(not (DEX1 / "calibration.json").is_file(),
                                   reason="assets/robot/dex1 not built (tools/build_dex1_hand.py)")


def test_the_default_robot_is_dex3_in_the_original_folder():
    assert Settings().robot.hands == "dex3"
    assert Robot().directory(ROOT) == ROOT / "assets/robot"
    assert Robot("dex1").directory(ROOT) == ROOT / "assets/robot/dex1"
    with pytest.raises(ValueError):
        Robot("dex5")


def test_the_dex1_settings_file_selects_the_gripper():
    settings = Settings.load(ROOT / "configs/simulation.toml", override=ROOT / "configs/dex1.toml")
    assert settings.robot.hands == "dex1"
    assert Settings.load(ROOT / "configs/simulation.toml").robot.hands == "dex3"


def test_dex1_swaps_only_pick_and_place_and_dex3_keeps_the_built_in_modules():
    dex1 = registry("dex1")
    assert registry("dex3") is REGISTRY and tuple(dex1) == tuple(REGISTRY)
    for name in REGISTRY:
        swapped = name in registry_module.HAND_SKILLS["dex1"]
        assert (dex1[name].__name__ == f"homebody.skills.dex1.{name}") == swapped
        assert REGISTRY[name].__name__ == f"homebody.skills.{name}"
    with pytest.raises(ValueError):
        registry("dex5")


def test_a_two_finger_hand_state_reads_closure_and_per_jaw_evidence():
    ray = np.array([-0.04, -0.04])
    state = hand_state([-0.02, -0.01], [-0.04, -0.04], ray, [5.0, 0.0], [16.0, 16.0],
                       ["left_hand_finger_1_joint", "left_hand_finger_2_joint"],
                       {"thumb": [0], "index": [1], "middle": [1]})
    assert state.joints.shape == (2,)
    assert state.closure == pytest.approx(0.375)
    assert state.finger_effort_fraction == pytest.approx([5 / 16, 0, 0])
    assert state.finger_closure_gap == pytest.approx([0.5, 0.75, 0.75])


@needs_profile
def test_the_dex1_model_compiles_with_one_servo_per_calibrated_finger():
    calibration = json.loads((DEX1 / "calibration.json").read_text())
    robot = DEX1 / "g1_grasp"
    xml = ET.parse(robot / "g1_closd_grasp_deploy.xml").getroot()
    xml.find("compiler").set("meshdir", str(robot / "meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    for side in ("left", "right"):
        assert not any("_hand_thumb_" in model.joint(i).name and side in model.joint(i).name
                       for i in range(model.njnt))
        for name, opened in zip(calibration["hand_joints"][side], calibration["hand_open"][side]):
            joint = model.joint(name).id
            assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
            assert model.jnt_range[joint, 0] <= opened <= model.jnt_range[joint, 1]
            assert np.count_nonzero(model.actuator_trnid[:, 0] == joint) == 1
        squeezed = np.array(calibration["hand_open"][side]) + 1.12 * calibration["hand_close_fraction"] * np.array(
            calibration["hand_closed"][side])
        assert np.all(squeezed >= model.jnt_range[[model.joint(n).id for n in calibration["hand_joints"][side]], 0])


@needs_profile
def test_dex1_finger_boxes_slide_together_as_the_jaws_close():
    arm = Arm(DEX1 / "g1_29dof_with_hand.urdf", "right")
    joints = np.zeros(len(arm.names))

    def gap(closed):
        boxes = arm.collision.hand_boxes(arm, joints, [closed, closed])
        return np.linalg.norm(boxes[0][0] - boxes[1][0])

    assert gap(-0.03) == pytest.approx(gap(0.0) - 0.06, abs=1e-3)
