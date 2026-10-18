"""Arm kinematics and the arm pick / place phase machines, on a small synthetic arm (no model download)."""
import numpy as np
import pytest

pytest.importorskip("draccus")
mujoco = pytest.importorskip("mujoco")

from lerobot_skills import ADVANCE, FAILED, SUCCEEDED, run_skill
from lerobot_skills.arm import ArmPick, ArmPickConfig, ArmPlace, ArmPlaceConfig
from lerobot_skills.kinematics import ArmKinematics, ArmSpec

# A yaw base, two pitch links and a wrist pitch: its tool site points along its local +x, like the SO-101's.
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="base"><joint name="pan" axis="0 0 1" range="-3 3"/><geom type="cylinder" size="0.03 0.02"/>
  <body pos="0 0 0.1"><joint name="lift" axis="0 1 0" range="-2 2"/><geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.01"/>
  <body pos="0.15 0 0"><joint name="elbow" axis="0 1 0" range="-2.5 2.5"/><geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.01"/>
  <body pos="0.15 0 0"><joint name="wrist" axis="0 1 0" range="-2.5 2.5"/><geom type="box" size="0.02 0.01 0.01"/>
  <site name="tool" pos="0.05 0 0"/></body></body></body></body></worldbody></mujoco>"""


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text(ARM)
    return ArmSpec(("pan", "lift", "elbow", "wrist"), "grip", str(path), "tool", home=(0.0, -30.0, 60.0, 60.0))


DOWN = np.array([0.3, 0.3, 0.6, np.pi / 2 - 0.9])  # the pitches sum to 90 degrees: the tool points straight down


def test_ik_puts_the_tool_at_a_point_pointing_down(spec):
    kinematics = ArmKinematics(spec)
    target, rotation = kinematics.forward(DOWN)
    assert rotation[:, 0] == pytest.approx([0, 0, -1], abs=1e-6)
    q = kinematics.solve(target, spec.to_radians(spec.home))
    assert q is not None
    position_error, axis_error = kinematics.error(q, target, np.array([0, 0, -1.0]))
    assert position_error < 0.005 and axis_error < np.radians(8)
    assert kinematics.solve([0.9, 0.0, 0.0], spec.to_radians(spec.home)) is None  # beyond the links' length


class Arm:
    """Joints that reach their targets at once; a gripper that stops at `blocked` when something is in the jaws."""

    def __init__(self, spec, blocked=None):
        self.spec, self.blocked, self.epoch = spec, blocked, 1
        self.state = {**{f"{j}.pos": float(v) for j, v in zip(spec.joints, spec.home)}, "grip.pos": 100.0}
        self.advanced = 0.0

    def get_observation(self):
        return dict(self.state)

    def send_action(self, action):
        for key, value in action.items():
            if key == "grip.pos" and self.blocked is not None:
                value = max(value, self.blocked)
            if key in self.state:
                self.state[key] = value
        return action

    def advance(self, epoch, seconds):
        self.advanced += seconds


def test_a_pick_moves_in_phases_and_reports_the_hold(spec):
    for blocked, expected in ((20.0, (SUCCEEDED, "OK")), (None, (FAILED, "NO_GRASP"))):
        arm, skill = Arm(spec, blocked), ArmPick(ArmPickConfig(), spec)
        skill.reset(arm.get_observation(), {"target": ArmKinematics(spec).forward(DOWN)[0].tolist()})
        stages = []
        while not skill.done:
            action = skill.select_action(arm.get_observation())
            stages.append(skill.stage)
            if action:
                assert set(action) == {*(f"{j}.pos" for j in spec.joints), "grip.pos", ADVANCE}
                seconds = action.pop(ADVANCE)
                arm.send_action(action)
                arm.advance(1, seconds)
        assert (skill.status.state, skill.status.code) == expected
        assert [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]][:4] == ["approach", "open", "descend",
                                                                                        "close"]
        assert arm.advanced == pytest.approx(1.5 + 0.8 + 1.0 + 0.8 + 1.5 + 0.8, abs=0.1)


def test_an_unreachable_target_fails_before_moving(spec):
    arm, skill = Arm(spec), ArmPick(ArmPickConfig(), spec)
    skill.reset(arm.get_observation(), {"target": [0.9, 0.0, 0.0]})
    assert skill.status.code == "UNREACHABLE"
    assert run_skill(skill, arm).code == "UNREACHABLE" and arm.advanced == 0


def test_a_place_opens_over_its_target(spec):
    arm, skill = Arm(spec, blocked=20.0), ArmPlace(ArmPlaceConfig(), spec)
    arm.state["grip.pos"] = 20.0
    skill.reset(arm.get_observation(), {"target": ArmKinematics(spec).forward(DOWN)[0].tolist()})
    assert run_skill(skill, arm).state == SUCCEEDED and arm.state["grip.pos"] == 100.0
