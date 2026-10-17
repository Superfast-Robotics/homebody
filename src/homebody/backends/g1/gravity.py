"""Dynamics-only gravity compensation for the AMO G1 profile.

RNEA bias torques from the URDF's dynamics model (``pin.buildModelFromUrdf``), reduced to
the 29 profile joints. Pinocchio and NumPy are imported only when this is constructed.
Derived from Psi0 revision ``a32e57a3fabb8590c80677f9cd3d1fc3db60eb06`` (Apache-2.0).
"""

from __future__ import annotations

from pathlib import Path

from holosoma_inference.g1_control.config import RobotProfile


class PinocchioGravityCompensator:
    """Return 29 motor-order RNEA bias torques for a commanded pose."""

    def __init__(self, robot: RobotProfile, urdf_path: str | Path) -> None:
        # Imported here: NumPy, then Torch, then Pinocchio.
        import numpy as np
        import pinocchio as pin

        self._np = np
        self._pin = pin
        full_model = pin.buildModelFromUrdf(str(urdf_path))
        retained = set(robot.joint_names)
        locked = [
            full_model.getJointId(name)
            for name in full_model.names[1:]
            if name not in retained
        ]
        model = pin.buildReducedModel(full_model, locked, np.zeros(full_model.nq))
        if model.nq != 29 or model.nv != 29:
            raise RuntimeError(f"reduced G1 model has nq={model.nq}, nv={model.nv}; expected 29")
        order = np.asarray(
            [model.idx_qs[model.getJointId(name)] for name in robot.joint_names],
            dtype=np.int64,
        )
        if sorted(order.tolist()) != list(range(29)):
            raise RuntimeError("reduced G1 model order is not a 29-joint permutation")
        self._model = model
        self._data = model.createData()
        self._order = order
        self._zeros = np.zeros(model.nv)

    def torques(self, joint_targets: tuple[float, ...]) -> tuple[float, ...]:
        np = self._np
        motor_order = np.asarray(joint_targets, dtype=np.float64)
        if motor_order.shape != (29,):
            raise ValueError("gravity target must contain 29 joint positions")
        model_order = np.zeros(29, dtype=np.float64)
        model_order[self._order] = motor_order
        model_torque = self._pin.rnea(
            self._model,
            self._data,
            model_order,
            self._zeros,
            self._zeros,
        )
        return tuple(np.asarray(model_torque)[self._order].tolist())
