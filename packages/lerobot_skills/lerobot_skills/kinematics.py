"""An arm's kinematics from its MJCF: forward kinematics of its end-effector site and inverse kinematics for a
position plus one approach axis (the jaws pointing a given way; the roll about that axis is free, so a five-joint
arm can reach it too). Damped least squares on MuJoCo's site Jacobians, within the joint ranges."""
from dataclasses import dataclass

import mujoco
import numpy as np

AXIS_TOLERANCE = np.radians(8)


@dataclass(frozen=True)
class ArmSpec:
    """What an arm skill needs to know about an arm: its LeRobot motor names (arm joints in kinematic order, then
    the gripper), their units, its MJCF (the arm alone, its base at the origin: the frame skill targets are in)
    and its end-effector site, whose APPROACH_AXIS (local) points out between the jaws."""
    joints: tuple[str, ...]
    gripper: str
    model_xml: str
    ee_site: str
    approach_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    degrees: bool = True       # joint values in degrees, else radians
    gripper_open: float = 100.0
    gripper_closed: float = 0.0
    home: tuple[float, ...] = ()  # a resting pose of the arm joints, in their units
    model_joints: tuple[str, ...] = ()  # the MJCF's names for JOINTS, when they differ

    def to_radians(self, values):
        values = np.asarray(values, dtype=float)
        return np.deg2rad(values) if self.degrees else values

    def from_radians(self, q):
        q = np.asarray(q, dtype=float)
        return np.rad2deg(q) if self.degrees else q


class ArmKinematics:
    def __init__(self, spec: ArmSpec):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(spec.model_xml)
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(name).id for name in (spec.model_joints or spec.joints)]
        self.qpos = self.model.jnt_qposadr[ids]
        self.dofs = self.model.jnt_dofadr[ids]
        self.low, self.high = self.model.jnt_range[ids, 0], self.model.jnt_range[ids, 1]
        self.site = self.model.site(spec.ee_site).id
        self.axis = np.asarray(spec.approach_axis, dtype=float)

    def forward(self, q):
        """The site's position and rotation at arm joint angles Q (radians)."""
        self.data.qpos[self.qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def error(self, q, position, approach):
        """Position error (m) and approach-axis error (radians) at Q."""
        pos, rotation = self.forward(q)
        axis = rotation @ self.axis
        return float(np.linalg.norm(position - pos)), float(np.arccos(np.clip(axis @ approach, -1, 1)))

    def solve(self, position, q0, approach=(0.0, 0.0, -1.0), tolerance=0.004, axis_tolerance=AXIS_TOLERANCE,
              axis_weight=0.15, damping=0.02, iterations=400):
        """Joint angles (radians) putting the site at POSITION with its approach axis along APPROACH, from seeds
        Q0 then the joint-range midpoint; None when no seed converges within the tolerances."""
        position, approach = np.asarray(position, float), np.asarray(approach, float)
        approach = approach / np.linalg.norm(approach)
        for seed in (np.asarray(q0, float), (self.low + self.high) / 2):
            q = np.clip(seed.copy(), self.low, self.high)
            for _ in range(iterations):
                pos, rotation = self.forward(q)
                axis = rotation @ self.axis
                error_pos, error_axis = position - pos, np.cross(axis, approach)
                if (np.linalg.norm(error_pos) < tolerance and
                        np.arccos(np.clip(axis @ approach, -1, 1)) < axis_tolerance):
                    return q
                jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
                free_roll = np.eye(3) - np.outer(axis, axis)  # turning about the approach axis costs nothing
                jacobian = np.vstack([jacp[:, self.dofs], axis_weight * free_roll @ jacr[:, self.dofs]])
                error = np.concatenate([error_pos, axis_weight * free_roll @ error_axis])
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(6), error)
                norm = np.linalg.norm(step)
                q = np.clip(q + (step * 0.2 / norm if norm > 0.2 else step), self.low, self.high)
        return None
