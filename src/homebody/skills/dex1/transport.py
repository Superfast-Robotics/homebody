"""Dex1 loaded pose selection after lift verification, before walking."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from homebody.helpers.collision import Clearance
from homebody.helpers.geometry import inverse, points_in, rotation_error
from homebody.helpers.trajectory import joint_rows
from homebody.motion.arm import follow_rows, live_check, servo_best_effort
from homebody.motion.carry import (
    CarryState,
    continuing,
    in_view,
    payload_points,
    prepare_compact,
    recover,
)
from homebody.skills.contract import SkillFailure

MAX_TURN = np.pi / 6
MIN_GAIN_M = 0.01
MAX_CANDIDATES = 12


@dataclass(frozen=True)
class TransportPlan:
    palm: np.ndarray
    rows: tuple
    before: tuple
    after: tuple
    turn: float


def pose_cost(record, palm, world_torso, pad):
    """Observed radial envelope and gravity lever arm about the calibrated pad centre.

    This is a geometric proxy, not a mass/friction estimate. Include the wrist and
    pad in the envelope so rotating a compact payload cannot swing the hand out.
    """
    payload = payload_points(record, palm)
    support = points_in(palm, pad)
    envelope = points_in(inverse(world_torso), np.vstack((payload, support, palm[:3, 3])))
    radius = float(np.max(np.linalg.norm(envelope[:, :2], axis=1)))
    center = (palm @ record['palm_to_payload'])[:3, 3]
    lever = float(np.linalg.norm((center - support)[:2]))
    return radius, lever


def improves(before, after):
    """Require a material gain without trading away either transport objective."""
    return (all(a <= b + 0.002 for a, b in zip(after, before)) and
            sum(before) - sum(after) >= MIN_GAIN_M)


def transport_options(frame, palm, baseline, settings, pad):
    """Bounded world-axis turns about the pad, at radial extraction endpoints.

    Also search when the translation-only compactor found nothing. No extra lift,
    closure or insertion is used; grasp-time geometry and orientation stay intact.
    """
    torso = inverse(frame.world_torso)
    local = torso @ palm
    radius = np.linalg.norm(local[:2, 3])
    positions = [palm.copy() if baseline is None else baseline.palm.copy()]
    if radius > settings.grasp.carry_palm_radius + settings.grasp.carry_compact_min_delta:
        for rise in (0., settings.grasp.lift_height):
            endpoint = local.copy()
            endpoint[:2, 3] *= settings.grasp.carry_palm_radius / radius
            endpoint = frame.world_torso @ endpoint
            endpoint[2, 3] = palm[2, 3] + rise
            positions.append(endpoint)
    for origin in positions:
        pivot = points_in(origin, pad)
        for axis in np.eye(3):
            for angle in (-MAX_TURN, -MAX_TURN / 2, MAX_TURN / 2, MAX_TURN):
                target = origin.copy()
                target[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix() @ palm[:3, :3]
                target[:3, 3] = pivot - target[:3, :3] @ pad
                yield target, abs(angle)


def prepare_transport(frame, side, ctx, baseline):
    """Search only improving poses; bound IK work and check the complete loaded path."""
    arm, record, settings = ctx.arms[side], ctx.carried[side], ctx.settings
    samples = ctx.planning().jaw_samples.get(side)
    if not samples:
        return None
    sample = min(samples, key=lambda row: abs(row['closure'] - frame.hands[side].closure))
    pad = np.asarray(sample['palm_to_jaw'])[:3, 3]
    palm = frame.palm(arm, side)
    reference = palm if baseline is None else baseline.palm
    before = pose_cost(record, reference, frame.world_torso, pad)
    candidates = []
    for target, turn in transport_options(frame, palm, baseline, settings, pad):
        after = pose_cost(record, target, frame.world_torso, pad)
        if improves(before, after) and in_view(frame, payload_points(record, target)):
            candidates.append((sum(after), turn, target, after))
    candidates.sort(key=lambda row: (row[0], row[1]))
    seed = frame.arm_joints(side)
    clearance = Clearance(frame, ctx.arms, side, settings.collision, (record['label'],),
                          record, exempt_start=True)
    for _, turn, target, after in candidates[:MAX_CANDIDATES]:
        ctx.check(frame.epoch)
        # If the original extraction is clear, withdraw along it before turning.
        # Otherwise test a simultaneous loaded withdrawal/turn from measured joints.
        start = seed if baseline is None else np.asarray(baseline.rows[-1])
        solved = arm.solve(inverse(frame.world_torso) @ target, start, restarts=False)
        if solved is None or clearance.path_violation(start, solved) is not None:
            continue
        rows = (() if baseline is None else baseline.rows) + tuple(
            joint_rows(start, solved, settings.physics.arm_rate / 2, settings.servo.tick))
        return TransportPlan(target, rows, before, after, turn)
    return None


def compact_carry(ctx, epoch, side):
    """Choose a transport pose without altering shared carry/place/navigation logic."""
    frame = ctx.observe(epoch)
    record = ctx.carried[side]
    if not continuing(frame, ctx.arms[side], side, record, ctx.settings.grasp):
        record['state'] = CarryState.UNCERTAIN
        raise SkillFailure('HOLD_UNCERTAIN', 'Compact carry requires a currently verified grasp')
    baseline = prepare_compact(frame, side, ctx.arms, record, ctx.settings)
    selected = prepare_transport(frame, side, ctx, baseline)
    ctx.emit({'type': 'dex1_transport_search', 'selected': selected is not None,
              'translation_available': baseline is not None, 'frame_id': frame.frame_id})
    if selected is None and baseline is None:
        return frame
    if selected is not None:
        ctx.emit({'type': 'mechanism_active', 'mechanism': 'loaded_transport_pose', 'side': side,
                  'radius_before_m': selected.before[0], 'radius_after_m': selected.after[0],
                  'lever_before_m': selected.before[1], 'lever_after_m': selected.after[1],
                  'turn_rad': selected.turn, 'translation_available': baseline is not None})
    plan = baseline if selected is None else selected
    record['state'] = CarryState.UNCERTAIN
    ctx.emit({'type': 'carry_prepared', 'phase': 'compact', 'plan': plan})
    ctx.stage('Compacting carry')
    ctx.check(epoch)
    follow_rows(ctx, epoch, side, plan.rows, live_check(ctx, side, (record['label'],)))
    frame = servo_best_effort(ctx, epoch, side, plan.palm, exclude_labels=(record['label'],))
    actual = frame.palm(ctx.arms[side], side)
    ctx.emit({'type': 'dex1_transport_arrival',
              'position_error_m': float(np.linalg.norm(actual[:3, 3] - plan.palm[:3, 3])),
              'rotation_error_rad': float(np.linalg.norm(rotation_error(plan.palm[:3, :3], actual[:3, :3])))})
    ctx.stage('Verifying compact carry')
    status, frame = recover(ctx, epoch, side)
    ctx.emit({'type': 'dex1_transport_retention', 'status': status,
              'orientation_selected': selected is not None})
    if status != 'verified':
        raise SkillFailure('HOLD_UNCERTAIN', 'Compact carry did not retain repeated current hold evidence')
    # Preserve navigation's pending-compaction signal if the measured wrist is still extended.
    local = inverse(frame.world_torso) @ frame.palm(ctx.arms[side], side)
    if np.linalg.norm(local[:2, 3]) <= (ctx.settings.grasp.carry_palm_radius +
                                      ctx.settings.grasp.carry_compact_min_delta):
        record.pop('compact_pending', None)
    return frame
