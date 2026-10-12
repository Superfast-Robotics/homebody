"""Why an arm pose fails: out of reach, wrong orientation, or no IK solution from the seed."""
import math

import numpy as np

from homebody.helpers.geometry import inverse


def palm_in_torso(frame, arm, side):
    """The palm's current pose in the torso frame."""
    return inverse(frame.world_torso) @ frame.palm(arm, side)


def diagnose_ik(arm, pose, seed):
    """Solve torso-frame POSE from SEED and say why it fails. Returns a dict: ok, reason
    ("solved", "beyond_reach", "orientation", "no_solution"), joints (or None),
    shoulder_distance_m, reach_m and, for "orientation", the position-only joints."""
    pose, seed = np.asarray(pose, float), np.asarray(seed, float)
    distance = float(np.linalg.norm(pose[:3, 3] - arm.shoulder))
    report = {"ok": False, "joints": None, "shoulder_distance_m": round(distance, 3), "reach_m": round(float(arm.reach), 3)}
    if distance > arm.reach:
        return {**report, "reason": "beyond_reach"}
    joints = arm.solve(pose, seed)
    if joints is not None:
        return {**report, "ok": True, "reason": "solved", "joints": joints}
    position = arm.solve(pose, seed, position_only=True)
    if position is not None:
        reached = arm.forward(position)[0]
        cos = np.clip((np.trace(reached[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)
        return {**report, "reason": "orientation", "position_joints": position,
                "orientation_error_deg": round(math.degrees(math.acos(cos)), 1)}
    return {**report, "reason": "no_solution"}


def reach_report(arm, poses, seed):
    """Counts of diagnose_ik reasons over POSES, for a one-line summary of why a set fails."""
    counts = {}
    for pose in poses:
        reason = diagnose_ik(arm, pose, seed)["reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return counts
