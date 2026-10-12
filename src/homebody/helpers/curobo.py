"""Client for the cuRobo planning server (tools/curobo_server.py): collision-free goal-set
planning of an arm to one of several pre-poses, then down to the matching pose, against an ESDF
of the current depth image only (no simulator state).

Poses go to the server in the pelvis frame; the robot's own pixels and the EXCLUDE labels (the
target, or the carried object) are removed from the depth so the planner may reach them.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from homebody.remote.g1 import ARM_JOINT_SUFFIXES, G1Kinematics, homebody_hand_names
from homebody.remote.wire import PolicyClient, PolicyWireError

URDF = Path(__file__).resolve().parents[3] / "assets/robot/g1_29dof_with_hand.urdf"
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
TARGET_MASK_GROWTH_PX = 4


class PlannerUnavailable(RuntimeError):
    """The planning server cannot be reached or configured."""


class CuroboPlanner:
    """SETTINGS is the full Settings (its `curobo` group, and the arms' rest pose)."""

    def __init__(self, settings):
        self.all_settings, self.settings = settings, settings.curobo
        self.kinematics = G1Kinematics(URDF)
        self.client = PolicyClient(self.settings.endpoint, self.settings.timeout)

    def locks(self, joints):
        """The posture the non-planning joints are fixed at: the waist where it stands, both arms at
        rest (where an idle arm is kept) and the hands open flat, as pick opens before it moves."""
        locks = {name: float(value) for name, value in zip(self.kinematics.names, joints) if name in WAIST_JOINTS}
        for side in ("left", "right"):
            rest = self.all_settings.placement.rest_arm(side)
            locks.update(zip([f"{side}_{suffix}_joint" for suffix in ARM_JOINT_SUFFIXES], map(float, rest)))
            locks.update(dict.fromkeys(homebody_hand_names(side), 0.0))
        return locks

    def ensure_configured(self, joints):
        """Build the server's per-arm planners once, other joints locked at `locks`."""
        try:
            status = self.client.call("ping")
            if not status.get("configured"):
                self.client.call("configure", {"locks": self.locks(joints)})
        except PolicyWireError as error:
            raise PlannerUnavailable(str(error)) from error

    def world_pelvis(self, frame):
        return frame.world_torso @ np.linalg.inv(self.kinematics.torso_in_pelvis(frame.joints))

    def plan(self, frame, side, pre_world, pose_world, exclude_labels=()):
        """The server's reply for the arm's goal set of (pre, pose) world poses, best first."""
        self.ensure_configured(frame.joints)
        pelvis_world = np.linalg.inv(self.world_pelvis(frame))
        depth = np.array(frame.depth, dtype=np.float32)
        excluded = frame.labels == -1
        for label in exclude_labels:
            # Grow the target's mask: its silhouette pixels read as background and would leave a
            # shell of phantom obstacle exactly where the fingers close.
            excluded |= binary_dilation(frame.labels == label, iterations=TARGET_MASK_GROWTH_PX)
        depth[excluded] = np.nan
        count = self.settings.max_candidates
        request = {"side": side, "start": np.asarray(frame.arm_joints(side), dtype=np.float64),
                   "pre": pelvis_world @ np.asarray(pre_world)[:count],
                   "palm": pelvis_world @ np.asarray(pose_world)[:count],
                   "depth": depth, "intrinsics": np.asarray(frame.intrinsics, dtype=np.float64),
                   "camera": pelvis_world @ frame.world_camera}
        try:
            return self.client.call("plan_grasp", request)
        except PolicyWireError as error:
            raise PlannerUnavailable(str(error)) from error

    def close(self):
        self.client.close()


def thinned(rows, limit):
    """At most LIMIT rows of a joint path, endpoints kept."""
    rows = np.asarray(rows)
    if len(rows) <= limit:
        return rows
    return rows[np.unique(np.linspace(0, len(rows) - 1, limit).round().astype(int))]
