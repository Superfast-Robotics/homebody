"""Measured-endpoint correction is bounded and never substitutes for pose convergence."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from homebody.motion import arm
from homebody.session.settings import Settings
from homebody.skills import pick
from homebody.skills.contract import SkillFailure


@pytest.mark.parametrize('outcome', ['clear', 'unsolved', 'large', 'blocked'])
def test_correction_uses_current_torso_and_measured_seed(monkeypatch, outcome):
    measured = np.array([.1, -.2])
    solved = measured + (.6 if outcome == 'large' else .2)
    torso = np.eye(4)
    torso[:3, :3] = Rotation.from_euler('z', .2).as_matrix()
    torso[:3, 3] = [.1, -.1, .03]
    frame = SimpleNamespace(world_torso=torso, arm_joints=lambda side: measured, frame_id='new')
    plan = SimpleNamespace(epoch=1, side='right', pre=np.eye(4))
    offset = np.array([.02, .01, 0.])
    solves, checks, moves, events = [], [], [], []

    def solve(goal, seed):
        solves.append((goal, seed))
        return None if outcome == 'unsolved' else solved

    def clearance(*args, **kwargs):
        assert kwargs == {'exempt_start': True}

        def violation(start, end):
            checks.append((start, end))
            return 'finger blocked' if outcome == 'blocked' else None
        return SimpleNamespace(path_violation=violation)

    ctx = SimpleNamespace(observe=lambda epoch: frame, arms={'right': SimpleNamespace(solve=solve)},
                          settings=Settings(), emit=events.append)
    target = SimpleNamespace(update=lambda state: offset)
    monkeypatch.setattr(pick, 'Clearance', clearance)
    monkeypatch.setattr(pick, 'move_path', lambda *args, **kwargs: moves.append((args, kwargs)))
    pick.correct_alignment(plan, ctx, target)
    expected = plan.pre.copy()
    expected[:3, 3] += offset
    assert len(solves) == 1
    np.testing.assert_allclose(solves[0][0], np.linalg.inv(torso) @ expected, atol=1e-12)
    np.testing.assert_allclose(solves[0][1], measured, atol=1e-12)
    np.testing.assert_allclose(plan.pre, np.eye(4), atol=1e-12)
    assert len(checks) == int(outcome in ('clear', 'blocked'))
    assert len(moves) == int(outcome == 'clear')
    assert events[-1]['applied'] == (outcome == 'clear')
    if moves:
        assert moves[0][1] == {'tracked_target': target}  # live checking remains enabled
        np.testing.assert_allclose(moves[0][0][3], [solved], atol=1e-12)


def test_alignment_failure_never_descends_or_retries_correction(monkeypatch):
    calls = []
    plan = SimpleNamespace(epoch=1, side='right', pre=np.eye(4), approach_joints=[])
    ctx = SimpleNamespace(stage=lambda stage: None)
    target = SimpleNamespace(begin_descent=lambda: calls.append('descend'))
    monkeypatch.setattr(pick, 'correct_alignment', lambda *args: calls.append('correction'))

    def servo(*args, **kwargs):
        calls.append('servo')
        raise SkillFailure('ARM_NOT_REACHED', 'full pose not converged')

    monkeypatch.setattr(pick, 'servo', servo)
    with pytest.raises(SkillFailure, match='full pose'):
        pick.approach(plan, ctx, target)
    assert calls == ['correction', 'servo']


def test_timeout_reports_measured_pose_joint_and_limit_residuals(monkeypatch):
    settings = Settings()
    measured = np.array([.1, -.2])
    palm = np.eye(4)
    palm[:3, :3] = Rotation.from_euler('z', .3).as_matrix()
    palm[0, 3] = .04
    state = SimpleNamespace(time=0., world_torso=np.eye(4), arm_joints=lambda side: measured,
                            palm=lambda model, side: palm)
    model = SimpleNamespace(lower=np.array([-1., -1.]), upper=np.array([1., 1.]))
    events = []
    ctx = SimpleNamespace(settings=settings, arms={'right': model}, emit=events.append,
                          check=lambda epoch: None,
                          actions=SimpleNamespace(arm_target=lambda *args: None))
    ctx.advance = lambda *args: setattr(state, 'time', settings.motion.arm_timeout + .01)
    monkeypatch.setattr(arm, 'Reads', lambda *args: SimpleNamespace(next=lambda: state))
    monkeypatch.setattr(arm, 'FreshEvidence', lambda *args: SimpleNamespace(update=lambda *args: None))
    monkeypatch.setattr(arm, 'servo_step', lambda *args: (measured + .05, .04, .3))
    assert arm.track(ctx, 1, 'right', np.eye(4), None, None) is None
    event = events[-1]
    assert event['type'] == 'arm_servo_timeout'
    assert event['position_error_m'] == pytest.approx(.04)
    assert event['rotation_error_rad'] == pytest.approx(.3)
    assert event['joint_tracking_error_rad'] == pytest.approx(.05)
    assert event['joint_limit_margin_rad'] == pytest.approx(.8)
