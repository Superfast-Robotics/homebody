"""Navigation repair uses sensed contact and motion without changing shared recovery."""
from dataclasses import dataclass, fields, replace
from itertools import product

import numpy as np
import pytest

from homebody.motion.carry import CarryState, Recovery, geometry, recover
from homebody.primitives.observations import Frame, HandState
from homebody.skills import navigate
from homebody.skills.contract import Context, SkillFailure
from homebody.toolkit.navigation_carry import recover_navigation
from tests.fakes import PlanningArm
from tests.test_skills import FakeBackend, frame

SURFACE = np.array(list(product((-.02, .02), (-.02, .02), (-.04, .04))))


def hand(thumb=True, gaps=True):
    return HandState(np.zeros(7), 1., np.array([.2 if thumb else 0., .2, .2]),
                     np.full(3, .2) if gaps else np.zeros(3))


@dataclass(frozen=True)
class CarryFrame(Frame):
    shifted: bool = False
    follows: bool = True
    separated: bool = False

    def target_points(self, label):
        assert label == 1  # No alternative identity may be used for recovery.
        surface = np.vstack((SURFACE, SURFACE + [.12, 0., 0.])) if self.shifted else SURFACE
        delta = self.arm_joints('right')[:3] if self.follows else np.zeros(3)
        return surface + delta + [1. if self.separated else 0., 0., 1.]


class CarryBackend(FakeBackend):
    def __init__(self, *, shifted=False, thumb=False):
        source = frame()
        initial = CarryFrame(**{f.name: getattr(source, f.name) for f in fields(Frame)}, shifted=shifted)
        super().__init__(replace(initial, hands={'right': hand(thumb), 'left': hand(False)}))
        self.squeezes, self.moves = [], []
        self.restore = True
        self.stall = False
        self.stale = False

    def grip(self, epoch, side, closure):
        assert epoch == self.epoch
        self.squeezes.append(closure)
        if self.restore:
            self.frame = replace(self.frame, hands={**self.frame.hands, side: hand()})

    def advance(self, epoch, seconds):
        if not self.stale:
            super().advance(epoch, seconds)

    def arm_target(self, epoch, side, joints):
        assert epoch == self.epoch
        self.moves.append(joints)
        if not self.stall:
            measured = self.frame.joints.copy()
            measured[22:29] = joints
            self.frame = replace(self.frame, joints=measured)


def setup(**kwargs):
    backend = CarryBackend(**kwargs)
    arms = {side: PlanningArm() for side in ('left', 'right')}
    ctx = Context(backend, backend, arms)
    record = geometry(backend.frame, arms['right'], 'right', SURFACE + [0., 0., 1.])
    record.update(epoch=1, label=1, state=CarryState.UNCERTAIN, lift_verified=True)
    ctx.carried['right'] = record
    return backend, ctx, record


def test_reclose_restores_fresh_retention_once_without_recapture():
    backend, ctx, record = setup(shifted=True)
    old_points = record['local_points'].copy()
    result = recover_navigation(ctx, 1, 'right')
    assert result.status == Recovery.VERIFIED
    assert result.gate == 'retention_verified'
    assert backend.squeezes == pytest.approx([1. + ctx.settings.grasp.squeeze] * len(backend.squeezes))
    assert backend.squeezes and not backend.moves
    np.testing.assert_allclose(record['local_points'], old_points)
    assert all(np.allclose(v, 0.) for v in backend.commands)
    count = len(backend.squeezes)
    # Another navigation invocation cannot repeat the same repair experiment.
    repeated = recover_navigation(ctx, 1, 'right')
    assert repeated.status == Recovery.UNCERTAIN
    assert repeated.gate == 'probe_required'
    assert repeated.history == {'reclose': 'retention_verified'}
    assert len(backend.squeezes) == count


def test_missing_opposition_after_squeeze_is_not_more_waiting():
    backend, ctx, record = setup()
    backend.restore = False
    result = recover_navigation(ctx, 1, 'right', probe=True)
    assert result.gate == 'opposing_retention'
    assert result.status == Recovery.UNCERTAIN
    assert not backend.moves
    assert record['state'] == CarryState.UNCERTAIN
    count = len(backend.squeezes)
    repeated = recover_navigation(ctx, 1, 'right')
    assert repeated.gate == 'opposing_retention'
    assert len(backend.squeezes) == count


def test_effort_without_blocked_closure_does_not_authorize_repair():
    backend, ctx, _ = setup(shifted=True)
    backend.frame = replace(backend.frame, hands={**backend.frame.hands, 'right': hand(gaps=False)})
    result = recover_navigation(ctx, 1, 'right', probe=True)
    assert result.gate == 'no_blocked_contact'
    assert not backend.squeezes and not backend.moves


@pytest.mark.parametrize('visible', [True, False])
def test_visibility_and_separation_gates(visible):
    backend, ctx, record = setup(shifted=True, thumb=True)
    record['navigation_recovery'] = {'reclose': 'retention_verified'}
    backend.frame = replace(backend.frame, separated=visible,
                            labels=backend.frame.labels if visible else np.zeros_like(backend.frame.labels))
    result = recover_navigation(ctx, 1, 'right', probe=True)
    assert result.gate == ('visible_separation' if visible else 'visibility')
    assert not backend.moves and not backend.squeezes


def test_repeated_sensor_packet_cannot_verify_after_squeeze():
    backend, ctx, record = setup(shifted=True)
    backend.stale = True
    result = recover_navigation(ctx, 1, 'right')
    assert result.status == Recovery.UNCERTAIN and result.gate == 'freshness'
    assert record['state'] == CarryState.UNCERTAIN


def test_probe_can_run_independently_and_refresh_only_after_measured_motion():
    backend, ctx, record = setup(shifted=True, thumb=True)
    old_half = record['payload_half'].copy()
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert result.status == Recovery.VERIFIED and result.gate == 'co_motion_verified'
    assert backend.moves and not backend.squeezes
    assert np.max(record['payload_half']) > np.max(old_half)
    assert record['state'] == CarryState.VERIFIED
    assert result.history == {'probe': 'co_motion_verified'}
    # Shared recovery can verify the refreshed geometry without a changed predicate.
    status, _ = recover(ctx, 1, 'right')
    assert status == Recovery.VERIFIED


@pytest.mark.parametrize('stalled', [True, False])
def test_commanded_lift_or_nonmoving_object_cannot_refresh(stalled):
    backend, ctx, record = setup(shifted=True, thumb=True)
    old_points = record['local_points'].copy()
    backend.stall = stalled
    backend.frame = replace(backend.frame, follows=stalled)
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert result.status == Recovery.UNCERTAIN
    assert result.gate == ('measured_hand_motion' if stalled else 'measured_target_motion')
    np.testing.assert_allclose(record['local_points'], old_points)
    count = len(backend.moves)
    repeated = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert repeated.gate == 'probe_already_attempted'
    assert len(backend.moves) == count


def test_probe_requires_overlap_with_original_identity_capture():
    backend, ctx, record = setup(shifted=True, thumb=True)
    record['local_points'] = record['local_points'] + [0., .2, 0.]
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert result.gate == 'capture_overlap'
    assert not backend.moves


def test_probe_collision_uses_expanded_geometry_and_blocks_motion(monkeypatch):
    from homebody.toolkit import navigation_carry
    backend, ctx, record = setup(shifted=True, thumb=True)
    old_half = record['payload_half'].copy()

    class Blocked:
        def __init__(self, frame, arms, side, settings, excluded, candidate):
            assert excluded == (record['label'],)
            assert np.max(candidate['payload_half']) > np.max(old_half)

        def path_violation(self, start, end):
            return 'blocked'

    monkeypatch.setattr(navigation_carry, 'Clearance', Blocked)
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert result.gate == 'probe_clearance: blocked'
    assert not backend.moves
    np.testing.assert_allclose(record['payload_half'], old_half)


def test_probe_rechecks_collision_during_motion(monkeypatch):
    from homebody.toolkit import navigation_carry
    backend, ctx, record = setup(shifted=True, thumb=True)
    old_points = record['local_points'].copy()

    class ObstacleAppears:
        def __init__(self, *args):
            pass

        def path_violation(self, start, end):
            return 'new obstruction' if backend.moves else None

    monkeypatch.setattr(navigation_carry, 'Clearance', ObstacleAppears)
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert result.status == Recovery.UNCERTAIN
    assert 'new obstruction' in result.gate
    assert result.history == {'probe': 'COLLISION'}
    assert len(backend.moves) == 1 and backend.stopped == 1
    np.testing.assert_allclose(record['local_points'], old_points)


def test_target_rise_without_stable_relative_surface_cannot_refresh(monkeypatch):
    _, ctx, record = setup(shifted=True, thumb=True)
    old_points = record['local_points'].copy()
    target_points = CarryFrame.target_points

    def slipping(view, label):
        points = target_points(view, label)
        if view.arm_joints('right')[2] > ctx.settings.grasp.lift_evidence:
            points += [2 * ctx.settings.grasp.target_surface_tolerance, 0., 0.]
        return points

    monkeypatch.setattr(CarryFrame, 'target_points', slipping)
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert result.status == Recovery.UNCERTAIN
    assert result.gate == 'co_motion_residual'
    np.testing.assert_allclose(record['local_points'], old_points)


def test_epoch_mismatch_refuses_before_actuation():
    backend, ctx, record = setup()
    record['epoch'] = 0
    with pytest.raises(SkillFailure, match='older scene'):
        recover_navigation(ctx, 1, 'right')
    assert not backend.commands and not backend.squeezes


def test_shared_recovery_remains_passive():
    backend, ctx, _ = setup(shifted=True)
    status, _ = recover(ctx, 1, 'right')
    assert status == Recovery.UNCERTAIN
    assert not backend.squeezes and not backend.moves


def test_confirmed_loss_is_cleared_without_active_repair():
    backend, ctx, _ = setup()
    empty = HandState(np.zeros(7), 1., np.zeros(3), np.zeros(3))
    backend.frame = replace(backend.frame, hands={**backend.frame.hands, 'right': empty})
    result = recover_navigation(ctx, 1, 'right', probe=True)
    assert result.status == Recovery.CLEARED
    assert 'right' not in ctx.carried
    assert not backend.squeezes and not backend.moves


def test_probe_needs_new_frames_after_motion(monkeypatch):
    backend, ctx, record = setup(shifted=True, thumb=True)
    old_points = record['local_points'].copy()
    send = backend.arm_target

    def freeze_after_command(epoch, side, joints):
        send(epoch, side, joints)
        backend.stale = True

    monkeypatch.setattr(backend, 'arm_target', freeze_after_command)
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert backend.moves
    assert result.status == Recovery.UNCERTAIN and result.gate == 'freshness'
    np.testing.assert_allclose(record['local_points'], old_points)


def test_probe_stops_on_live_retention_loss_and_remembers_attempt(monkeypatch):
    backend, ctx, record = setup(shifted=True, thumb=True)
    old_points = record['local_points'].copy()
    send = backend.arm_target

    def lose_contact(epoch, side, joints):
        send(epoch, side, joints)
        backend.frame = replace(backend.frame, hands={**backend.frame.hands, side: hand(False)})

    monkeypatch.setattr(backend, 'arm_target', lose_contact)
    result = recover_navigation(ctx, 1, 'right', reclose=False, probe=True)
    assert result.status == Recovery.UNCERTAIN
    assert result.gate.startswith('probe_interrupted:')
    assert result.history == {'probe': 'HOLD_UNCERTAIN'}
    assert backend.stopped == 1
    np.testing.assert_allclose(record['local_points'], old_points)


def test_navigation_raises_with_gate_and_history_even_when_called_by_another_skill():
    backend, ctx, record = setup(shifted=True)
    backend.restore = False
    plan = navigate.Plan(1, np.array([1.1, 1.]), None, [np.array([1.1, 1.])])
    # Pick calls navigate.execute directly, so refusal must still interrupt its caller.
    with pytest.raises(SkillFailure, match='opposing_retention') as failure:
        navigate.execute(plan, ctx)
    assert failure.value.code == 'HOLD_UNCERTAIN'
    assert str(record['navigation_recovery']) in str(failure.value)
    assert 'grip' in navigate.NEEDS
