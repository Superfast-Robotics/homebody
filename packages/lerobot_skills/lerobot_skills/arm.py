"""Pick and place for any arm with an ArmSpec, as phase machines in the policy form. Targets are positions in the
arm's base frame; every step is a joint-position action (the robot's units) run for `step_seconds`.

pick: move above the target, open the gripper there, descend, close, lift, then check the hold: the gripper stopped
short of closing (something between the jaws). place: move above the target, descend, open, rise."""
from dataclasses import dataclass

import numpy as np

from .kinematics import ArmKinematics, ArmSpec
from .skill import ADVANCE, FAILED, SUCCEEDED, Skill, SkillConfig, Status


@dataclass
class ArmMotionConfig(SkillConfig):
    approach_height: float = 0.06  # m above the target the move starts its descent from
    step_seconds: float = 1 / 30
    move_seconds: float = 1.5      # each transfer between waypoints
    descend_seconds: float = 1.0
    gripper_seconds: float = 0.8   # opening or closing, held still


@SkillConfig.register_subclass("arm_pick")
@dataclass
class ArmPickConfig(ArmMotionConfig):
    grasp_height: float = 0.0      # the jaws' point relative to the target when they close
    lift_height: float = 0.08
    hold_margin: float = 3.0       # gripper units above closed that mean the jaws are blocked


@SkillConfig.register_subclass("arm_place")
@dataclass
class ArmPlaceConfig(ArmMotionConfig):
    release_height: float = 0.03   # the jaws' point above the target when they open


class _Phases(Skill):
    """Plays a list of phases: (name, arm joint targets in radians or None to hold, gripper value, seconds)."""

    def __init__(self, config, spec: ArmSpec, kinematics: ArmKinematics | None = None):
        super().__init__(config)
        self.spec, self.kinematics = spec, kinematics or ArmKinematics(spec)
        self.phases, self.phase, self.tick, self.start, self.gripper = [], 0, 0, None, None

    def _joints(self, observation):
        return self.spec.to_radians([observation[f"{joint}.pos"] for joint in self.spec.joints])

    def _waypoints(self, observation, positions):
        """Joint angles reaching each of POSITIONS in turn, each solved from the last; None when one is out of
        reach."""
        q, solved = self._joints(observation), []
        for position in positions:
            q = self.kinematics.solve(position, q)
            if q is None:
                return None
            solved.append(q)
        return solved

    def select_action(self, observation):
        if self.done:
            return {}
        if self.phase >= len(self.phases):
            self._finish(observation)
            return {}
        _, target, gripper, seconds = self.phases[self.phase]
        if self.tick == 0:
            self.start = self._joints(observation)
            self.gripper = observation[f"{self.spec.gripper}.pos"] if gripper is None else gripper
        steps = max(1, round(seconds / self.config.step_seconds))
        self.tick += 1
        q = self.start if target is None else self.start + (target - self.start) * min(1.0, self.tick / steps)
        if self.tick >= steps:
            self.phase, self.tick = self.phase + 1, 0
        action = {f"{joint}.pos": float(value) for joint, value in zip(self.spec.joints, self.spec.from_radians(q))}
        return {**action, f"{self.spec.gripper}.pos": float(self.gripper), ADVANCE: self.config.step_seconds}

    @property
    def stage(self):
        return self.phases[self.phase][0] if self.phase < len(self.phases) else "done"

    def _finish(self, observation):
        self.status = Status(SUCCEEDED, "OK")


class ArmPick(_Phases):
    config_class, name = ArmPickConfig, "arm_pick"

    def reset(self, observation=None, arguments=None):
        """ARGUMENTS: {"target": [x, y, z]}, the object's point to close the jaws on, in the base frame."""
        self.status, self.phase, self.tick = Status(), 0, 0
        c, target = self.config, np.asarray(arguments["target"], dtype=float)
        up = np.array([0.0, 0.0, 1.0])
        waypoints = self._waypoints(observation, [target + up * c.approach_height, target + up * c.grasp_height,
                                                  target + up * c.lift_height])
        if waypoints is None:
            self.status = Status(FAILED, "UNREACHABLE", f"no arm pose puts the jaws at {target.round(3).tolist()}")
            return
        above, grasp, lift = waypoints
        s = self.spec
        held = observation[f"{s.gripper}.pos"]  # an open jaw can hang low enough to sweep the table on the way
        self.phases = [("approach", above, held, c.move_seconds), ("open", None, s.gripper_open, c.gripper_seconds),
                       ("descend", grasp, s.gripper_open, c.descend_seconds),
                       ("close", None, s.gripper_closed, c.gripper_seconds), ("lift", lift, s.gripper_closed, c.move_seconds),
                       ("settle", None, s.gripper_closed, c.gripper_seconds)]

    def _finish(self, observation):
        measured = observation[f"{self.spec.gripper}.pos"]
        blocked = abs(measured - self.spec.gripper_closed) > self.config.hold_margin
        self.status = (Status(SUCCEEDED, "OK", f"holding: the gripper stopped at {measured:.1f}") if blocked else
                       Status(FAILED, "NO_GRASP", f"the gripper closed fully ({measured:.1f}): nothing between the jaws"))


class ArmPlace(_Phases):
    config_class, name = ArmPlaceConfig, "arm_place"

    def reset(self, observation=None, arguments=None):
        """ARGUMENTS: {"target": [x, y, z]}, the point to release the held object over, in the base frame."""
        self.status, self.phase, self.tick = Status(), 0, 0
        c, target = self.config, np.asarray(arguments["target"], dtype=float)
        up = np.array([0.0, 0.0, 1.0])
        waypoints = self._waypoints(observation, [target + up * c.approach_height, target + up * c.release_height,
                                                  target + up * c.approach_height])
        if waypoints is None:
            self.status = Status(FAILED, "UNREACHABLE", f"no arm pose puts the jaws over {target.round(3).tolist()}")
            return
        above, release, rise = waypoints
        s, held = self.spec, observation[f"{self.spec.gripper}.pos"]
        self.phases = [("approach", above, held, c.move_seconds), ("descend", release, held, c.descend_seconds),
                       ("open", None, s.gripper_open, c.gripper_seconds), ("rise", rise, s.gripper_open, c.move_seconds)]
