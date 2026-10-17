"""The real G1 controller kernel type and the reason its policy stopped, shared by the driver and
the body host."""

from __future__ import annotations

import functools


def body_stop_reason(status) -> str:
    """Why the kernel's policy is not running, from its status: the link watchdog, an
    inference error, or '' for a plain stop."""
    if getattr(status, "stopped_by_watchdog", False):
        return ("the kernel's link watchdog stopped the policy — the robot state repeated "
                "for its limit of ticks (a stale lowstate link, or a robot held perfectly still)")
    error = getattr(status, "last_error", None)
    if error:
        return f"the policy stopped on an inference error: {error}"
    return ""


@functools.cache
def _real_controller_kernel_type():
    """Build the real-only subclass of the vendored kernel: live arm targets and gains, and a
    measured-pose stop published from the last validated state once the state link fails."""

    from holosoma_inference.g1_control.contracts import VelocityCommand
    from holosoma_inference.g1_control.runtime import G1ControllerKernel, TickResult

    class RealG1ControllerKernel(G1ControllerKernel):
        def _command_for_state(self, state):
            """The vendored command with live arm targets when the policy is inactive and the
            arm gate's gains on the arm joints; the initial-pose ramp is unchanged."""
            command = super()._command_for_state(state)
            if self.ramping or self._arm_targets is None:
                return command
            positions = list(command.positions)
            stiffness, damping = list(command.stiffness), list(command.damping)
            if not self._active:
                offsets = self.profile.robot.joint_offsets_rad
                for i, target in enumerate(self._arm_targets):
                    positions[15 + i] = float(target) + offsets[15 + i]
            gains = getattr(self, "_arm_gains", None)
            if gains is not None:
                for i, (kp, kd) in enumerate(zip(*gains)):
                    stiffness[15 + i], damping[15 + i] = float(kp), float(kd)
            return command.__class__(
                positions=tuple(positions), velocities=command.velocities,
                feedforward_torques=command.feedforward_torques,
                stiffness=tuple(stiffness), damping=tuple(damping))

        def republish_arms(self, command) -> None:
            """Rewrite the last command with this tick's arm targets and gains, so the arms
            publish at 250 Hz between 50 Hz policy ticks."""
            if command is None or self.ramping or self._arm_targets is None:
                return
            positions = list(command.positions)
            stiffness, damping = list(command.stiffness), list(command.damping)
            offsets = self.profile.robot.joint_offsets_rad
            for i, target in enumerate(self._arm_targets):
                positions[15 + i] = float(target) + offsets[15 + i]
            gains = getattr(self, "_arm_gains", None)
            if gains is not None:
                for i, (kp, kd) in enumerate(zip(*gains)):
                    stiffness[15 + i], damping[15 + i] = float(kp), float(kd)
            self.port.write_command(command.__class__(
                positions=tuple(positions), velocities=command.velocities,
                feedforward_torques=command.feedforward_torques,
                stiffness=tuple(stiffness), damping=tuple(damping)))

        @property
        def ramping(self) -> bool:
            """Whether the initial-pose ramp is still travelling."""
            return bool(self._initializing and self._initialization_tick
                        < self.profile.loop.initialization_ticks)

        def set_arm_gains(self, kp14, kd14) -> None:
            """Set the arm joints' gains; None returns them to the profile's."""
            self._arm_gains = None if kp14 is None else (tuple(map(float, kp14)),
                                                          tuple(map(float, kd14)))

        def publish_stopped_hold(self, state):
            """Publish one measured-pose stop without reading the failed state link."""

            self._active = False
            self._initializing = False
            self._walk_enabled = False
            self._velocity = VelocityCommand()
            self._velocity_age_ticks = 0
            self._velocity_stale = True
            callback_error = None
            try:
                self.policy.on_stop()
            except BaseException as exc:  # noqa: BLE001 -- the callback is diagnostic only
                callback_error = f"policy on_stop failed: {type(exc).__name__}: {exc}"
            finally:
                self._active = False
                self._initializing = False
                self._walk_enabled = False
                self._velocity = VelocityCommand()
                self._velocity_age_ticks = 0
                self._velocity_stale = True

            if callback_error is not None:
                self._last_error = callback_error
            command = self._command_for_state(state)
            self.port.write_command(command)
            self._command_ticks += 1
            return TickResult(state=state, command=command, status=self.status)

    return RealG1ControllerKernel
