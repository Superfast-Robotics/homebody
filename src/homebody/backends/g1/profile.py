"""The G1 description: the AMO body profile,
joint slices, and the arm limits, gains and Dex3 closure ray from `hand_math`.

ArmGate is the arm's five ordered clamps, without the DDS wire.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from holosoma_inference.g1_control.config import amo_profile

from homebody.backends.g1.hand_math import (
    ARM_EFFORT,
    ARM_LIMITS,
    GAINS,
    HAND_CLOSED,
    HAND_JOINTS,
    HAND_MAX_RATE,
    HAND_OPEN,
    LIMIT_MARGIN,
    grip_from_hand_q,
    hand_q_from_grip,
)
from homebody.real2sim.exploration.settings import current

__all__ = [
    "ARM_EFFORT",
    "ARM_LIMITS",
    "ARM_SLICE",
    "BODY_JOINTS",
    "GAINS",
    "HAND_CLOSED",
    "HAND_JOINTS",
    "HAND_MAX_RATE",
    "HAND_OPEN",
    "LEFT_ARM",
    "LIMIT_MARGIN",
    "PROFILE",
    "RIGHT_ARM",
    "WATCHDOG_S",
    "ArmGate",
    "grip_from_hand_q",
    "hand_q_from_grip",
]


def _hands_down_idle() -> tuple[float, ...]:
    """Both arms' idle pose, shoulders rolled out by `roll`, hands at the sides."""
    roll = 0.20
    left = [0.2, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0]
    right = left.copy()
    left[1] += roll
    right[1] -= roll
    return tuple(float(v) for v in left + right)


# --- the 29-dof body order the AMO kernel speaks
_base = amo_profile()
# Lowstate stop after 25 repeated policy ticks; torso 0.58-0.75 m.
PROFILE = replace(
    _base,
    robot=replace(_base.robot, idle_arm_positions=_hands_down_idle()),
    loop=replace(_base.loop, repeated_state_stop_ticks=25, torso_height_range_m=(0.58, 0.75)),
)
BODY_JOINTS: tuple[str, ...] = tuple(PROFILE.robot.joint_names)
assert len(BODY_JOINTS) == 29, len(BODY_JOINTS)
LEFT_ARM = slice(15, 22)
RIGHT_ARM = slice(22, 29)
ARM_SLICE = {"left": LEFT_ARM, "right": RIGHT_ARM}

WATCHDOG_S = 0.25  # s; a longer gap between steps repeats the last command
ARM_JOINT_RATE_RAD_S = round(1.5 * 0.35, 6)  # the arm gate's joint-rate limit
IDLE_SETTLE_RAD_S = 0.2  # rate a stopped ramp eases to the idle pose


def arm_effort() -> np.ndarray:
    """The per-joint effort cap: the table's, the wrists from the hands settings."""
    out = np.asarray(ARM_EFFORT, dtype=float).copy()
    out[5:7] = current().hands.wrist_effort_nm
    return out


class ArmGate:
    """The arm's five ordered clamps; the caller writes the clamped command to the actuators."""

    def __init__(
        self,
        side: str = "right",
        gains: str = "track",
        max_joint_vel: float | None = None,
        watchdog_s: float = WATCHDOG_S,
    ):
        self.side = side
        self.kp, self.kd = GAINS[gains]
        self.arm_lower, self.arm_upper = ARM_LIMITS[side]
        self.max_joint_vel = ARM_JOINT_RATE_RAD_S if max_joint_vel is None else max_joint_vel
        self.watchdog_s = watchdog_s
        self._q_cmd = np.zeros(7)
        self._weight = 0.0
        self._weight_target = 0.0  # disengaged until arm()
        self._ramp_rate = 0.5  # weight per second
        self._cmd_rate = 0.0
        self._meas_rate = 0.0
        self._gap_rate = 0.0  # how fast |q_cmd - q_meas| grows
        self._q_meas_prev: np.ndarray | None = None
        self._gap_prev: float | None = None

    def prime(self, q_arm) -> None:
        self._q_cmd = np.asarray(q_arm, dtype=float).copy()

    def arm(self, weight_target: float = 1.0, ramp_s: float = 2.0) -> None:
        self._weight_target = float(weight_target)
        self._ramp_rate = 1.0 / max(ramp_s, 1e-3)  # 0 to 1 in ramp_s seconds

    def __call__(self, q_arm, q_meas, dt: float, *, force: bool = False) -> dict:
        stale = dt > self.watchdog_s

        # 1. authority + stiffness ramp (one value drives both)
        step_w = self._ramp_rate * max(dt, 0.0)
        if self._weight < self._weight_target:
            self._weight = min(self._weight_target, self._weight + step_w)
        elif self._weight > self._weight_target:
            self._weight = max(self._weight_target, self._weight - step_w)

        # 2. speed clamp
        want = self._q_cmd.copy() if (stale and not force) else np.asarray(q_arm, dtype=float)
        max_delta = self.max_joint_vel * min(max(dt, 0.0), self.watchdog_s)
        q_prev = self._q_cmd.copy()
        self._q_cmd = np.clip(want, self._q_cmd - max_delta, self._q_cmd + max_delta)

        # 3. joint-limit clamp — a separate table, independent of the IK
        self._q_cmd = np.clip(
            self._q_cmd, self.arm_lower + LIMIT_MARGIN, self.arm_upper - LIMIT_MARGIN
        )
        at_limit = (self._q_cmd <= self.arm_lower + LIMIT_MARGIN + 1e-9) | (
            self._q_cmd >= self.arm_upper - LIMIT_MARGIN - 1e-9
        )

        q_meas = None if q_meas is None else np.asarray(q_meas, dtype=float)
        q_out = self._q_cmd.copy()
        if q_meas is not None:
            # 4. effort clamp, against the effective kp
            kp_eff = self.kp * max(self._weight, 1e-3)
            lim = arm_effort() / kp_eff
            q_out = np.clip(q_out, q_meas - lim, q_meas + lim)
            # 5. not-following: command, measured and gap rates
            gap = float(np.max(np.abs(self._q_cmd - q_meas)))
            if self._q_meas_prev is not None and self._gap_prev is not None and dt > 1e-6:
                a = min(1.0, dt / 0.5)
                self._cmd_rate += a * (
                    float(np.max(np.abs(self._q_cmd - q_prev))) / dt - self._cmd_rate
                )
                self._meas_rate += a * (
                    float(np.max(np.abs(q_meas - self._q_meas_prev))) / dt - self._meas_rate
                )
                self._gap_rate += a * ((gap - self._gap_prev) / dt - self._gap_rate)
            self._q_meas_prev = q_meas.copy()
            self._gap_prev = gap

        return {
            "q_out": q_out,
            "authority": self._weight,
            "stale": stale,
            "kp": self.kp * self._weight,
            "kd": self.kd,
            "at_limit": at_limit,
            "cmd_rate": self._cmd_rate,
            "meas_rate": self._meas_rate,
            "not_following": bool(
                self._cmd_rate > 0.05
                and self._meas_rate < 0.1 * self._cmd_rate
                and self._gap_rate > 0.5 * self._cmd_rate
            ),
        }
