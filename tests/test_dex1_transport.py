"""Loaded rotation selection, fallback and carry-state contracts."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from homebody.helpers.geometry import points_in, transform
from homebody.motion.carry import CarryState
from homebody.session.settings import Settings
from homebody.skills.contract import SkillFailure
from homebody.skills.dex1 import transport


def test_cost_uses_payload_and_calibrated_pad():
    palm = transform([.3, 0., 1.], np.eye(3))
    record = {'local_points': np.array([[.1, 0., 0.], [.2, 0., 0.]]),
              'palm_to_payload': transform([.15, 0., 0.], np.eye(3))}
    assert transport.pose_cost(record, palm, np.eye(4), np.array([.1, 0., 0.])) == pytest.approx((.5, .05))


@pytest.mark.parametrize('after, expected', [((.38, .1), True), ((.4, .095), False),
                                            ((.41, .05), False), ((.35, .11), False)])
def test_gain_cannot_trade_radius_for_lever(after, expected):
    assert transport.improves((.4, .1), after) is expected


def test_turns_preserve_pad_pivot_and_bound_rotation():
    palm = transform([.1, 0., 1.], np.eye(3))
    pad = np.array([.117, 0., 0.])
    frame = SimpleNamespace(world_torso=np.eye(4))
    options = list(transport.transport_options(frame, palm, None, Settings(), pad))
    assert len(options) == 12
    for target, turn in options:
        np.testing.assert_allclose(points_in(target, pad), points_in(palm, pad), atol=1e-12)
        assert Rotation.from_matrix(target[:3, :3]).magnitude() == pytest.approx(turn)
        assert turn <= transport.MAX_TURN + 1e-12


def setup_context():
    palm = transform([.3, 0., 1.], np.eye(3))
    frame = SimpleNamespace(epoch=1, frame_id=(1, 2), world_torso=np.eye(4),
                            palm=lambda *args: palm, arm_joints=lambda _: np.zeros(2),
                            hands={'right': SimpleNamespace(closure=.5)})
    record = {'state': CarryState.VERIFIED, 'label': 7, 'compact_pending': True,
              'local_points': np.array([[.1, 0., 0.]]),
              'palm_to_payload': transform([.1, 0., 0.], np.eye(3))}
    events = []
    ctx = SimpleNamespace(settings=Settings(), arms={'right': object()}, carried={'right': record},
                          observe=lambda _: frame, check=lambda _: None, stage=lambda _: None,
                          emit=events.append, planning=lambda: SimpleNamespace(jaw_samples={'right': [
                              {'closure': .5, 'palm_to_jaw': transform([.117, 0., 0.], np.eye(3))}]}))
    return ctx, frame, record, events


@pytest.mark.parametrize('baseline_exists', [False, True])
@pytest.mark.parametrize('blocked', [False, True])
def test_search_checks_loaded_path_and_preserves_extraction(monkeypatch, baseline_exists, blocked):
    ctx, frame, record, _ = setup_context()
    target = transform([.2, 0., 1.], np.eye(3))
    baseline = SimpleNamespace(palm=frame.palm(), rows=((.1, .1),)) if baseline_exists else None
    starts, checks = [], []

    def solve(pose, start, restarts):
        starts.append(start)
        assert restarts is False
        return np.array([.2, .2])

    def clearance(*args, **kwargs):
        assert args[5] is record
        return SimpleNamespace(path_violation=lambda start, end: checks.append((start, end)) or
                               ('blocked' if blocked else None))

    ctx.arms['right'] = SimpleNamespace(solve=solve)
    monkeypatch.setattr(transport, 'Clearance', clearance)
    monkeypatch.setattr(transport, 'in_view', lambda *args: True)
    monkeypatch.setattr(transport, 'transport_options', lambda *args: iter([(target, .2)]))
    result = transport.prepare_transport(frame, 'right', ctx, baseline)
    assert checks
    np.testing.assert_allclose(starts[0], [.1, .1] if baseline_exists else [0., 0.], atol=1e-12)
    if blocked:
        assert result is None
    else:
        assert result.palm is target
        np.testing.assert_allclose(result.rows[-1], [.2, .2], atol=1e-12)
        if baseline_exists:
            np.testing.assert_allclose(result.rows[0], baseline.rows[0], atol=1e-12)


@pytest.mark.parametrize('selected, baseline_exists, status', [
    (False, False, 'verified'), (False, True, 'verified'),
    (True, False, 'verified'), (True, True, 'verified'), (True, True, 'uncertain')])
def test_execution_fallback_reporting_and_retention(monkeypatch, selected, baseline_exists, status):
    ctx, frame, record, events = setup_context()
    baseline = SimpleNamespace(palm=frame.palm(), rows=((0., 0.),)) if baseline_exists else None
    alternative = transport.TransportPlan(frame.palm(), ((.1, .1),), (.5, .1), (.4, .08), .2)
    calls = []
    monkeypatch.setattr(transport, 'continuing', lambda *args: True)
    monkeypatch.setattr(transport, 'prepare_compact', lambda *args: baseline)
    monkeypatch.setattr(transport, 'prepare_transport', lambda *args: alternative if selected else None)
    monkeypatch.setattr(transport, 'live_check', lambda *args: 'loaded-check')
    monkeypatch.setattr(transport, 'follow_rows', lambda *args: calls.append(args))
    monkeypatch.setattr(transport, 'servo_best_effort', lambda *args, **kwargs: frame)

    def recover(*args):
        record['state'] = CarryState.VERIFIED if status == 'verified' else CarryState.UNCERTAIN
        return status, frame

    monkeypatch.setattr(transport, 'recover', recover)
    if status != 'verified':
        with pytest.raises(SkillFailure, match='did not retain'):
            transport.compact_carry(ctx, 1, 'right')
    else:
        assert transport.compact_carry(ctx, 1, 'right') is frame
        assert record['state'] == CarryState.VERIFIED
    active = [event for event in events if event['type'] == 'mechanism_active']
    assert bool(active) is selected
    if selected or baseline_exists:
        assert calls[0][3] is (alternative.rows if selected else baseline.rows)
        assert calls[0][4] == 'loaded-check'
    else:
        assert not calls
    if selected:
        assert active[0]['radius_after_m'] == pytest.approx(.4)
        assert active[0]['translation_available'] is baseline_exists


def test_lost_hold_cannot_start_transport(monkeypatch):
    ctx, _, record, _ = setup_context()
    monkeypatch.setattr(transport, 'continuing', lambda *args: False)
    with pytest.raises(SkillFailure, match='currently verified'):
        transport.compact_carry(ctx, 1, 'right')
    assert record['state'] == CarryState.UNCERTAIN
