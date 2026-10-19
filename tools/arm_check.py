"""arm_check.py [RIG ...]: the check of each simulated rig (lerobot_robot_homebody_sim.rigs.RIGS), arm by arm.

1. Reach: over a grid on the table (the arm's base frame), where the arm's IK can put the jaws pointing down, at
   the table (z = half a cube) and 10 cm above it.
2. Pick: the cube at the arm's pick spot, picked with lerobot_skills' ArmPick from its true position (no vision
   yet), then held still for two seconds. Passes when the skill reports holding and the cube is still lifted
   (its centre at least 4 cm above where it started) at the end.
Prints one line per arm and exits 1 if any fails. Needs no camera (the skill reads only joint positions)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
from lerobot_robot_homebody_sim.arms import ARMS
from lerobot_robot_homebody_sim.rigs import RIGS
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


def reach(arm):
    spec = ARMS[arm].spec()
    kinematics, seed = ArmKinematics(spec), spec.to_radians(spec.home)
    span = max(0.42, 1.4 * np.hypot(*ARMS[arm].pick_xy))
    grid = [(x, y) for x in np.arange(-span, span + 1e-9, 0.02) for y in np.arange(-span, span + 1e-9, 0.02)
            if 0.06 < np.hypot(x, y) <= span]
    table = [p for p in grid if kinematics.solve([*p, CUBE / 2], seed) is not None]
    raised = [p for p in grid if kinematics.solve([*p, 0.10], seed) is not None]
    radii = [np.hypot(*p) for p in table]
    return len(table), len(raised), len(grid), (min(radii), max(radii)) if radii else (0, 0)


def pick(rig, prefix, calibration_dir):
    """Pick the cube in front of the arm under PREFIX; the skill's status and how far the cube rose."""
    robot = SimArm(SimArmConfig(rig=rig, calibration_dir=Path(calibration_dir)))
    robot.connect()
    cube = "cube" if len(RIGS[rig].mounts) == 1 else f"{prefix}cube"
    start = robot.object_position(cube)
    base = np.linalg.inv(robot.arm_frame(prefix)) @ [*start, 1.0]  # the cube's centre, on a table or the floor
    skill = ArmPick(ArmPickConfig(), robot.arm_spec(prefix))
    port = Joints(robot)
    skill.reset(port.get_observation(), {"target": base[:3].tolist()})
    status = run_skill(skill, port) if not skill.done else skill.status
    robot.advance(robot.epoch, 2.0)
    rise = float(robot.object_position(cube)[2] - start[2])
    robot.disconnect()
    return status, rise


def main(names):
    failed = False
    with tempfile.TemporaryDirectory() as calibration:
        for rig in names or RIGS:
            for mount in RIGS[rig].mounts:
                table, raised, total, (near, far) = reach(mount.arm)
                status, rise = pick(rig, mount.prefix, calibration)
                ok = status.code == "OK" and rise >= LIFTED
                failed |= not ok
                print(f"{rig}{' ' + mount.prefix.rstrip('_') if mount.prefix else ''}: reach {table}/{total} table points "
                      f"({near:.2f}-{far:.2f} m), {raised}/{total} at 10 cm; pick {status.code} ({status.message}), "
                      f"cube rose {rise * 100:.1f} cm -> {'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
