"""The skill toolkit: IK diagnosis reasons and trace serialisation."""
from pathlib import Path

import numpy as np

from homebody.helpers.kinematics import Arm
from homebody.toolkit import diagnose_ik, reach_report, trace

URDF = Path(__file__).resolve().parents[1] / "assets/robot/g1_29dof_with_hand.urdf"


def arm():
    return Arm(URDF, "right")


def test_diagnose_solved_at_a_reached_pose():
    right = arm()
    seed = np.zeros(7)
    pose = right.forward(np.array([0.2, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]))[0]
    report = diagnose_ik(right, pose, seed)
    assert report["ok"] and report["reason"] == "solved"
    assert np.allclose(right.forward(report["joints"])[0][:3, 3], pose[:3, 3], atol=0.01)


def test_diagnose_beyond_reach():
    right = arm()
    pose = np.eye(4)
    pose[:3, 3] = right.shoulder + np.array([right.reach + 0.2, 0, 0])
    report = diagnose_ik(right, pose, np.zeros(7))
    assert report["reason"] == "beyond_reach" and report["shoulder_distance_m"] > report["reach_m"]


def test_reach_report_counts_reasons():
    right = arm()
    far = np.eye(4)
    far[:3, 3] = right.shoulder + np.array([0, 0, -(right.reach + 0.3)])
    counts = reach_report(right, [far, far], np.zeros(7))
    assert counts == {"beyond_reach": 2}


def test_trace_emits_plain_data():
    events = []

    class Ctx:
        _skill_name = "pick"

        def emit(self, event):
            events.append(event)

    trace(Ctx(), "grasp_search", tried=np.int64(4), pose=np.eye(2), why={"a": np.float32(0.5)})
    assert events == [{"type": "skill_trace", "skill": "pick", "what": "grasp_search",
                       "data": {"tried": 4, "pose": [[1.0, 0.0], [0.0, 1.0]], "why": {"a": 0.5}}}]
