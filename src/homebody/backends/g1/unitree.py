"""Deferred FAR Unitree SDK adapter for the package-owned G1 kernel.

The external ``far-unitree-sdk==0.1.5`` binding is imported only when real hardware is
explicitly constructed; callers retain sole-writer responsibility.
"""
from __future__ import annotations

import math

from holosoma_inference.g1_control.config import G1ControllerProfile
from holosoma_inference.g1_control.contracts import G1State, LowCommand


class UnitreeG1Port:
    """RobotPort over the FAR SDK's low-state and low-command transport."""

    def __init__(self, profile: G1ControllerProfile, interface_name: str) -> None:
        if not interface_name or interface_name == "auto":
            raise ValueError("Unitree G1 port requires one resolved network interface")
        try:
            import unitree_interface
        except ImportError as exc:
            raise ImportError("Unitree G1 hardware requires far-unitree-sdk==0.1.5") from exc
        self._profile = profile
        self._stiffness_level = 1.0
        self._damping_level = 1.0
        self._robot = unitree_interface.create_robot(
            interface_name, unitree_interface.RobotType.G1, unitree_interface.MessageType.HG)
        self._robot.set_control_mode(unitree_interface.ControlMode.PR)

    @property
    def stiffness_level(self) -> float:
        return self._stiffness_level

    @stiffness_level.setter
    def stiffness_level(self, value: float) -> None:
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("stiffness level must be finite")
        self._stiffness_level = converted

    @property
    def damping_level(self) -> float:
        return self._damping_level

    @damping_level.setter
    def damping_level(self, value: float) -> None:
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("damping level must be finite")
        self._damping_level = converted

    def read_state(self) -> G1State:
        sample = self._robot.read_low_state()
        q = tuple(float(value) for value in sample.motor.q)
        dq = tuple(float(value) for value in sample.motor.dq)
        mapping = self._profile.robot.joint_to_motor
        return G1State(
            base_position=(0.0, 0.0, 0.0),
            quaternion_wxyz=tuple(float(value) for value in sample.imu.quat),
            joint_positions=tuple(q[index] for index in mapping),
            base_linear_velocity=(0.0, 0.0, 0.0),
            base_angular_velocity=tuple(float(value) for value in sample.imu.omega),
            joint_velocities=tuple(dq[index] for index in mapping),
        )

    def write_command(self, command: LowCommand) -> None:
        mapping = self._profile.robot.joint_to_motor
        fields = [[0.0] * 29 for _ in range(5)]
        for joint_index, motor_index in enumerate(mapping):
            fields[0][motor_index] = command.positions[joint_index]
            fields[1][motor_index] = command.velocities[joint_index]
            fields[2][motor_index] = command.feedforward_torques[joint_index]
            fields[3][motor_index] = command.stiffness[joint_index] * self._stiffness_level
            fields[4][motor_index] = command.damping[joint_index] * self._damping_level
        output = self._robot.create_zero_command()
        output.q_target, output.dq_target, output.tau_ff, output.kp, output.kd = fields
        self._robot.write_low_command(output)

    def wireless_message(self):
        return self._robot.read_wireless_controller()

    def close(self) -> None:
        for name in ("close", "shutdown", "stop"):
            function = getattr(self._robot, name, None)
            if callable(function):
                function()
                return

    @staticmethod
    def wireless_key(message) -> str | None:
        if message is None:
            return None
        mapping = {
            1: "R1", 2: "L1", 3: "L1+R1", 4: "start", 8: "select",
            10: "L1+select", 16: "R2", 32: "L2", 64: "F1", 128: "F2",
            256: "A", 264: "select+A", 512: "B", 520: "select+B", 768: "A+B",
            1024: "X", 1032: "select+X", 1280: "A+X", 1536: "B+X",
            2048: "Y", 2056: "select+Y", 2304: "A+Y", 2560: "B+Y",
            3072: "X+Y", 4096: "up", 4097: "R1+up", 4104: "select+up",
            4352: "A+up", 4608: "B+up", 5120: "X+up", 6144: "Y+up",
            8192: "right", 8193: "R1+right", 8200: "select+right",
            8448: "A+right", 9216: "X+right", 10240: "Y+right",
            16384: "down", 16385: "R1+down", 16392: "select+down",
            16640: "A+down", 16896: "B+down", 17408: "X+down", 18432: "Y+down",
            32768: "left", 32769: "R1+left", 32776: "select+left",
            33024: "A+left", 33792: "X+left", 34816: "Y+left",
        }
        return mapping.get(int(getattr(message, "keys", 0)))


__all__ = ["UnitreeG1Port"]
