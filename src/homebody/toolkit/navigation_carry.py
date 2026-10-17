"""Bounded repair of an already lifted carry during navigation only."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from homebody.helpers.collision import Clearance
from homebody.helpers.geometry import inverse, points_in
from homebody.helpers.trajectory import joint_rows
from homebody.motion.arm import close_or_open, follow_rows
from homebody.motion.carry import (
    CarryState,
    Evidence,
    Recovery,
    evidence,
    in_view,
    payload_points,
    recapture,
    recover,
)
from homebody.primitives.observations import Frame, FreshEvidence
from homebody.skills.contract import SkillFailure


@dataclass(frozen=True)
class NavigationRecovery:
    status: Recovery
    frame: Frame
    gate: str
    history: dict


def recover_navigation(ctx, epoch, side, *, reclose=True, probe=False):
    """Stop, observe, optionally reclose once, then optionally probe shifted geometry.

    Actions are remembered on this carry record, including failed/interrupted actions.
    PROBE is opt-in and independent of RECLOSE. Neither flag changes shared recovery.
    """
    record = ctx.carried[side]
    if record['epoch'] != epoch:
        raise SkillFailure('STALE_EPOCH', 'Grasp evidence belongs to an older scene')
    ctx.check(epoch)
    history = record.setdefault('navigation_recovery', {})

    def finish(status, frame, gate):
        return NavigationRecovery(status, frame, gate, dict(history))

    ctx.actions.base_velocity(epoch, 0., 0., 0.)
    ctx.advance(epoch, ctx.settings.motion.settle_seconds)
    status, frame = recover(ctx, epoch, side)
    if status != Recovery.UNCERTAIN:
        return finish(status, frame, 'passive_' + status)
    if not record.get('lift_verified'):
        return finish(status, frame, 'not_lifted')

    settings, arm = ctx.settings.grasp, ctx.arms[side]
    if reclose and 'reclose' not in history:
        gate = plausible(frame, arm, side, record, settings)
        if gate is not None:
            return finish(status, frame, gate)
        history['reclose'] = 'started'
        baseline = frame
        ctx.stage('Restoring carry pressure')
        close_or_open(ctx, epoch, side, 1. + settings.squeeze, settings.close_seconds)
        status, frame, gate = sample_retention(ctx, epoch, side, record, baseline)
        history['reclose'] = gate
        if status == Recovery.VERIFIED:
            # One tactile restoration; keep the capture and all later transport checks.
            record['state'] = CarryState.VERIFIED
        return finish(status, frame, gate)

    if not probe:
        gate = 'reclose_disabled'
        if 'reclose' in history:
            gate = 'probe_required' if frame.hands[side].retained(settings) else 'opposing_retention'
        return finish(Recovery.UNCERTAIN, frame, gate)
    if 'probe' in history:
        return finish(Recovery.UNCERTAIN, frame, 'probe_already_attempted')
    status, frame, gate = sample_retention(ctx, epoch, side, record, frame)
    if status != Recovery.VERIFIED:
        return finish(status, frame, gate)
    if not frame.visible(record['label']):
        return finish(Recovery.UNCERTAIN, frame, 'visibility')
    local = points_in(inverse(frame.palm(arm, side)), frame.target_points(record['label']))
    overlap = np.mean(cKDTree(record['local_points']).query(local)[0] <= settings.recovery_match)
    if overlap < settings.occlusion_visible_match:
        return finish(Recovery.UNCERTAIN, frame, 'capture_overlap')

    # Expand only a temporary collision envelope until motion proves the new surface.
    candidate = dict(record)
    recapture(candidate, frame, arm, side, local)
    palm = frame.palm(arm, side).copy()
    target = palm.copy()
    target[2, 3] += settings.lift_evidence + 2 * settings.target_surface_tolerance
    seed = frame.arm_joints(side)
    joints = arm.solve(inverse(frame.world_torso) @ target, seed)
    if joints is None:
        return finish(Recovery.UNCERTAIN, frame, 'probe_ik')
    if not in_view(frame, payload_points(candidate, target)):
        return finish(Recovery.UNCERTAIN, frame, 'probe_view')
    checker = Clearance(frame, ctx.arms, side, ctx.settings.collision, (record['label'],), candidate)
    reason = checker.path_violation(seed, joints)
    if reason is not None:
        return finish(Recovery.UNCERTAIN, frame, 'probe_clearance: ' + reason)

    def check(state, reference):
        nonlocal checker
        if not state.hands[side].retained(settings):
            raise SkillFailure('HOLD_UNCERTAIN', 'Carry probe lost opposing retention')
        if isinstance(state, Frame):
            if not state.visible(record['label']):
                raise SkillFailure('HOLD_UNCERTAIN', 'Carry probe lost target visibility')
            checker = Clearance(state, ctx.arms, side, ctx.settings.collision, (record['label'],), candidate)
        return checker.path_violation(state.arm_joints(side), reference)

    history['probe'] = 'started'
    ctx.stage('Testing carry co-motion')
    rows = joint_rows(seed, joints, ctx.settings.physics.arm_rate, ctx.settings.servo.tick)
    try:
        follow_rows(ctx, epoch, side, rows, check)
        status, after, gate = sample_retention(ctx, epoch, side, record, frame, motion=(palm, local))
    except SkillFailure as error:
        history['probe'] = error.code
        if error.code in ('HOLD_UNCERTAIN', 'COLLISION'):
            ctx.actions.stop(epoch)
            return finish(Recovery.UNCERTAIN, ctx.observe(epoch), 'probe_interrupted: ' + str(error))
        raise
    history['probe'] = gate
    if status == Recovery.VERIFIED:
        recapture(record, after, arm, side,
                  points_in(inverse(after.palm(arm, side)), after.target_points(record['label'])))
        record['state'] = CarryState.VERIFIED
    return finish(status, after, gate)


def plausible(frame, arm, side, record, settings):
    """A blocked, loaded finger and no visible separation permit a stationary reclose."""
    hand = frame.hands[side]
    contact = ((hand.finger_effort_fraction > settings.hold_effort_fraction) &
               (hand.finger_closure_gap > settings.hold_closure_gap))
    if not np.any(contact):
        return 'no_blocked_contact'
    if frame.visible(record['label']):
        local = points_in(inverse(frame.palm(arm, side)), frame.target_points(record['label']))
        payload = points_in(inverse(record['palm_to_payload']), local)
        outside = np.maximum(np.abs(payload) - record['payload_half'], 0.)
        if np.min(np.linalg.norm(outside, axis=1)) > settings.recovery_separation:
            return 'visible_separation'
    return None


def co_motion_gate(frame, arm, side, record, settings, palm, local):
    """Measured hand travel AND fresh target travel, with stable palm-relative surfaces."""
    if not frame.visible(record['label']):
        return 'visibility'
    current = frame.palm(arm, side)
    if current[2, 3] - palm[2, 3] < settings.lift_evidence:
        return 'measured_hand_motion'
    actual = frame.target_points(record['label'])
    before = points_in(palm, local)
    if np.median(actual[:, 2]) - np.median(before[:, 2]) < settings.lift_evidence:
        return 'measured_target_motion'
    predicted = points_in(current, local)
    residual = max(np.median(cKDTree(predicted).query(actual)[0]),
                   np.median(cKDTree(actual).query(predicted)[0]))
    if residual > settings.target_surface_tolerance:
        return 'co_motion_residual'
    return None


def sample_retention(ctx, epoch, side, record, baseline, motion=None):
    """A bounded fresh window. Loss uses the unchanged shared evidence predicate."""
    settings, arm = ctx.settings.grasp, ctx.arms[side]
    samples = FreshEvidence(epoch, baseline.sequence, baseline.time)
    gate, frame = 'freshness', baseline
    for _ in range(settings.recovery_samples):
        ctx.advance(epoch, settings.recovery_tick)
        frame = ctx.observe(epoch)
        current = evidence(frame, arm, side, record, settings)
        if current == Evidence.LOST:
            value, failed = 'lost', 'lost'
        elif not frame.hands[side].retained(settings):
            value, failed = None, 'opposing_retention'
        else:
            failed = (co_motion_gate(frame, arm, side, record, settings, *motion) if motion is not None
                      else plausible(frame, arm, side, record, settings))
            value = 'held' if failed is None else None
        count = samples.update(frame, value)
        if count is None:
            continue
        gate = failed or 'confirmations'
        if count >= settings.recovery_confirmations:
            if value == 'lost':
                ctx.carried.pop(side)
                return Recovery.CLEARED, frame, 'lost'
            return Recovery.VERIFIED, frame, 'co_motion_verified' if motion is not None else 'retention_verified'
    return Recovery.UNCERTAIN, frame, gate
