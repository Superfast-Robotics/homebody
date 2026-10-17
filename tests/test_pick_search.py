"""Pick-local collision recovery and nominal servo descent validation."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.session.settings import Settings
from homebody.skills import pick


def search():
    result = pick.GraspSearch.__new__(pick.GraspSearch)
    result.control = Settings()
    result.settings = result.control.grasp
    result.cancelled = lambda: False
    result.blocked = []
    result.blockers = Counter()
    result.torso = np.eye(4)
    return result


def pose(x=0):
    value = np.eye(4)
    value[0, 3] = x
    return value


def test_equivalent_retry_uses_failed_phase_and_pose_tolerances():
    subject = search()
    old = SimpleNamespace(pre=pose(), palm=pose())
    near = SimpleNamespace(pre=pose(.1), palm=pose(.0017))
    assert subject.equivalent(near, old, 'descent')
    assert not subject.equivalent(near, old, 'approach')
    near.palm = pose(.03)
    assert not subject.equivalent(near, old, 'descent')
    near.palm = pose()
    near.palm[:3, :3] = np.diag([-1., -1., 1.])
    assert not subject.equivalent(near, old, 'descent')


def test_servo_checks_each_command_segment_and_rejects_curved_obstruction(monkeypatch):
    subject = search()
    subject.arm = object()
    checked = []
    subject.descent = SimpleNamespace(path_violation=lambda a, b: checked.append((a, b)) or 'finger blocked')
    monkeypatch.setattr(pick, 'servo_step', lambda *args: (np.array([.1, .2]), .1, 0.))
    assert subject.servo_descent(np.zeros(2), pose()) is None
    np.testing.assert_allclose(checked[0][1], [.1, .2])
    assert subject.blockers['finger blocked'] == 1


def test_servo_descent_converges_with_execution_step_model():
    subject = search()

    def forward(joints):
        return pose(joints[0]), np.eye(6)

    subject.arm = SimpleNamespace(forward=forward, lower=np.full(6, -2.), upper=np.full(6, 2.))
    checked = []
    subject.descent = SimpleNamespace(path_violation=lambda a, b: checked.append(b) and None)
    assert subject.servo_descent(np.zeros(6), pose(.06)) == pytest.approx(0.)
    assert len(checked) > 1
    assert abs(checked[-1][0] - .06) < subject.control.motion.palm_tolerance


def test_servo_stall_and_cancel_are_bounded(monkeypatch):
    subject = search()
    subject.arm = object()
    subject.descent = SimpleNamespace(path_violation=lambda *args: None)
    monkeypatch.setattr(pick, 'servo_step', lambda arm, joints, *args: (joints, .1, 0.))
    assert subject.servo_descent(np.zeros(2), pose()) is None
    assert subject.blockers['descent servo did not converge'] == 1
    subject.cancelled = lambda: True
    assert subject.servo_descent(np.zeros(2), pose()) is None
    assert subject.blockers['descent servo did not converge'] == 1


def test_recovery_prefers_clearance_before_insertion_tier(monkeypatch):
    subject = search()
    subject.blocked = [(object(), 'descent')]
    subject.seed = np.zeros(2)
    subject.arm = subject.approach = object()
    subject.counts = {'approach_clear': 0}
    tight = pick.ReachableGrasp(SimpleNamespace(tier=0, width=.02), np.zeros(2), .0)
    roomy = pick.ReachableGrasp(SimpleNamespace(tier=1, width=.03), np.ones(2), .016)
    monkeypatch.setattr(pick, 'approach_path', lambda *args, **kwargs: (np.zeros(2),))
    candidate, _ = subject.best([tight, roomy])
    assert candidate is roomy.candidate


def test_finger_margin_uses_all_fingers_and_existing_hit_threshold():
    subject = search()
    subject.arm = object()
    clearance = SimpleNamespace(
        geometry=SimpleNamespace(hand_boxes=lambda *args: [0, 1]), fingers=np.zeros(2),
        settings=SimpleNamespace(cloud_margin=.008),
        hits=lambda box, margin: np.array([box == 1 and margin > .02]),
        blocked=lambda hits: np.count_nonzero(hits) >= 1)
    assert subject.finger_margin(clearance, np.zeros(2)) == pytest.approx(.008)


def test_equivalent_candidate_is_excluded_before_ik():
    subject = search()
    candidate = SimpleNamespace(pre=pose(), palm=pose())
    subject.blocked = [(candidate, 'descent')]
    subject.counts = dict.fromkeys(subject.GATES, 0)
    subject.arm = SimpleNamespace(solve=lambda *args: pytest.fail('Excluded retry reached IK'))
    assert list(subject.reachable([candidate])) == []
    assert subject.blockers['equivalent blocked grasp'] == 1


def test_recovery_margin_is_worst_along_descent(monkeypatch):
    subject = search()
    subject.blocked = [(object(), 'descent')]
    subject.arm = object()
    subject.descent = SimpleNamespace(path_violation=lambda *args: None)
    margins = iter([.016, 0., .008])
    monkeypatch.setattr(subject, 'finger_margin', lambda *args: next(margins))
    steps = iter([(np.array([.1]), .1, 0.), (np.array([.2]), .05, 0.), (np.array([.2]), 0., 0.)])
    monkeypatch.setattr(pick, 'servo_step', lambda *args: next(steps))
    assert subject.servo_descent(np.zeros(1), pose()) == pytest.approx(0.)
