"""G1 conventions an external policy expects: joint and hand order, pelvis-frame wrist poses
from the URDF, and the rot6d rotation encoding."""
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import numpy as np

from homebody.helpers.geometry import axis_rotation, joint_origin, skew

# GR00T REAL_G1 hand vectors list the Dex3 joints alphabetically (from its dataset statistics:
# four finger joints closing over [0, 1.5] rad, then the thumb).
GROOT_HAND_ORDER = ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
WAIST = slice(12, 15)
ARM_JOINT_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
                      "wrist_yaw")
ARM = {"left": slice(15, 22), "right": slice(22, 29)}


@cache
def controller_joint_names() -> tuple[str, ...]:
    """The 29 joint names in the order Measurement.joints reports them."""
    from holosoma_inference.g1_control.config import amo_profile
    return tuple(amo_profile().robot.joint_names)


@cache
def calibration() -> dict:
    """The robot calibration a hardware backend also ships (hand joints and closed pose)."""
    import json
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "assets/robot/calibration.json").read_text())


def homebody_hand_names(side: str) -> tuple[str, ...]:
    """SIDE's seven hand joint names in HandState.joints order (the robot calibration's)."""
    return tuple(calibration()["hand_joints"][side])


def hand_closed(side: str) -> np.ndarray:
    """SIDE's closure-1 hand pose in HandState order, as grip() scales it."""
    values = calibration()
    ray = np.array(values["hand_closed"][side]) * values["hand_close_fraction"]
    ray[:3] *= values["thumb_close_fraction"]
    return ray


def _hand_permutation(side):
    names = homebody_hand_names(side)
    return np.array([names.index(f"{side}_hand_{part}_joint") for part in GROOT_HAND_ORDER])


def hand_to_groot(side: str, values) -> np.ndarray:
    """SIDE's hand joints from HomeBody order to GR00T order."""
    return np.asarray(values, dtype=float)[_hand_permutation(side)]


def hand_from_groot(side: str, values) -> np.ndarray:
    """SIDE's hand joints from GR00T order to HomeBody order."""
    result = np.empty(7)
    result[_hand_permutation(side)] = np.asarray(values, dtype=float)
    return result


def rot6d(rotation) -> np.ndarray:
    """GR00T's rot6d: the first two rows of the rotation matrix, row-major."""
    return np.asarray(rotation, dtype=float)[:2, :].reshape(6)


class Chain:
    """URDF forward kinematics from link ROOT to link TIP."""

    def __init__(self, urdf_root, root: str, tip: str):
        by_child = {j.find("child").get("link"): j for j in urdf_root.findall("joint")}
        chain, link = [], tip
        while link != root:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.joints = list(reversed(chain))
        self.origins = [joint_origin(j) for j in self.joints]
        self.cross = [None if j.get("type") == "fixed" else
                      skew(np.fromstring(j.find("axis").get("xyz"), sep=" ")) for j in self.joints]
        self.names = [j.get("name") for j in self.joints]

    def pose(self, positions: dict) -> np.ndarray:
        """TIP in ROOT given joint POSITIONS by name (missing movable joints read zero)."""
        pose = np.eye(4)
        for name, origin, cross in zip(self.names, self.origins, self.cross):
            pose = pose @ origin
            if cross is not None:
                motion = np.eye(4)
                motion[:3, :3] = axis_rotation(cross, float(positions.get(name, 0.0)))
                pose = pose @ motion
        return pose


class G1Kinematics:
    """Pelvis-frame wrist poses of the G1 from its 29 controller-ordered joints."""

    def __init__(self, urdf: Path):
        root = ET.parse(urdf).getroot()
        self.wrists = {side: Chain(root, "pelvis", f"{side}_wrist_yaw_link") for side in ("left", "right")}
        self.torso = Chain(root, "pelvis", "torso_link")
        self.wrist_palm = {side: Chain(root, f"{side}_wrist_yaw_link", f"{side}_hand_palm_link").pose({})
                           for side in ("left", "right")}
        self.names = controller_joint_names()

    def torso_in_pelvis(self, joints) -> np.ndarray:
        return self.torso.pose(dict(zip(self.names, np.asarray(joints, dtype=float))))

    def palm_target_in_torso(self, joints, side: str, wrist_in_pelvis) -> np.ndarray:
        """The torso-frame palm pose (what Arm.solve takes) that puts SIDE's wrist yaw link at
        WRIST_IN_PELVIS, with the waist at its measured JOINTS."""
        return np.linalg.inv(self.torso_in_pelvis(joints)) @ wrist_in_pelvis @ self.wrist_palm[side]

    def wrist_in_pelvis(self, joints, side: str) -> np.ndarray:
        return self.wrists[side].pose(dict(zip(self.names, np.asarray(joints, dtype=float))))

    def wrist_9d(self, joints, side: str) -> np.ndarray:
        pose = self.wrist_in_pelvis(joints, side)
        return np.r_[pose[:3, 3], rot6d(pose[:3, :3])]
