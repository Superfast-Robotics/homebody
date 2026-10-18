"""arm_check.py [ARM ...]: phase 3's check of each simulated arm (lerobot_robot_homebody_sim.arms.ARMS).

1. Reach: over a grid on the table (base frame), where the arm's IK can put the jaws pointing down, at the
   table (z = half a cube) and a hand's height above it.
2. Pick: the cube at the arm's pick spot, picked with lerobot_skills' ArmPick from its true position (no vision
   yet), then held still for two seconds. Passes when the skill reports holding and the cube is still lifted
   (its centre at least 4 cm above where it started) at the end.
Prints one line per arm and exits 1 if any arm fails. Needs no camera (the skill reads only joint positions)."""
import sys
from pathlib import Path

import numpy as np
from lerobot_robot_homebody_sim.arms import ARMS
from lerobot_robot_homebody_sim.sim_arm import CUBE, SimArm, SimArmConfig
from lerobot_skills import run_skill
from lerobot_skills.arm import ArmPick, ArmPickConfig
from lerobot_skills.kinematics import ArmKinematics

LIFTED = 0.04


class Joints:
    """The robot with observations of its motors only (no rendering)."""

    def __init__(self, robot):
        self.robot = robot

    epoch = property(lambda self: self.robot.epoch)

    def get_observation(self):
        return self.robot.get_state()

    def send_action(self, action):
        return self.robot.send_action(action)

    def advance(self, epoch, seconds):
        self.robot.advance(epoch, seconds)


def reach(embodiment):
    spec = embodiment.spec()
    kinematics, seed = ArmKinematics(spec), spec.to_radians(spec.home)
    grid = [(x, y) for x in np.arange(0.08, 0.42, 0.02) for y in np.arange(-0.24, 0.25, 0.02)]
    table = [p for p in grid if kinematics.solve([*p, CUBE / 2], seed) is not None]
    raised = [p for p in grid if kinematics.solve([*p, 0.10], seed) is not None]
    radii = [np.hypot(*p) for p in table]
    return len(table), len(raised), len(grid), (min(radii), max(radii)) if radii else (0, 0)


def pick(name, embodiment, calibration_dir):
    robot = SimArm(SimArmConfig(arm=name, calibration_dir=Path(calibration_dir)))
    robot.connect()
    start = robot.object_position("cube")
    skill = ArmPick(ArmPickConfig(), embodiment.spec())
    port = Joints(robot)
    skill.reset(port.get_observation(), {"target": [*start[:2], CUBE / 2]})
    status = run_skill(skill, port) if not skill.done else skill.status
    robot.advance(robot.epoch, 2.0)
    rise = float(robot.object_position("cube")[2] - start[2])
    robot.disconnect()
    return status, rise


def main(names):
    import tempfile
    failed = False
    with tempfile.TemporaryDirectory() as calibration:
        for name in names or ARMS:
            embodiment = ARMS[name]
            table, raised, total, (near, far) = reach(embodiment)
            status, rise = pick(name, embodiment, calibration)
            ok = status.code == "OK" and rise >= LIFTED
            failed |= not ok
            print(f"{name}: reach {table}/{total} table points ({near:.2f}-{far:.2f} m), {raised}/{total} at 10 cm; "
                  f"pick {status.code} ({status.message}), cube rose {rise * 100:.1f} cm -> {'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
