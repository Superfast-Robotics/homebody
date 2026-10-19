"""Dex1 placement geometry and checks, using calibration and measured state only.

Jaw calibration gives pad frames, not pad mesh extents or friction. Collision
extents come from the robot's calibrated collision interface, never guessed pads.
No contact-location sensor or loaded global planner is exposed by this interface.
"""
import numpy as np

from homebody.helpers.collision import Clearance
from homebody.helpers.geometry import points_in
from homebody.helpers.trajectory import joint_rows
from homebody.motion.arm import follow_rows, settle_joints, track
from homebody.primitives.observations import Frame, FreshEvidence
from homebody.skills.contract import SkillFailure


def jaw_frame(samples, closure):
    if not samples:
        raise SkillFailure('ARM_NOT_REACHED', 'Placement requires calibrated jaw frames; none are available')
    sample = min(samples, key=lambda row: abs(row['closure'] - closure))
    pose = np.asarray(sample['palm_to_jaw'], dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)):
        raise SkillFailure('ARM_NOT_REACHED', 'Placement jaw calibration is invalid')
    return pose


def opposed(hand, settings):
    """There are two independent jaws; middle is an alias of index."""
    return bool(np.all(hand.finger_effort_fraction[:2] > settings.hold_effort_fraction))


def opened(hand, settings):
    """Both prismatic joints must return near zero, the measured open reference."""
    return (hand.closure <= settings.placement.open_closure_limit and
            np.max(np.abs(hand.joints)) <= 0.002 and
            not np.any(hand.finger_effort_fraction[:2] > settings.grasp.hold_effort_fraction))


def withdrawal(palm, jaw, record, margin, hand_points=None):
    """Back the pad plane beyond the payload's near face along the insertion axis."""
    axis = jaw[:3, 2]
    payload = record['palm_to_payload']
    radius = np.abs(axis @ payload[:3, :3]) @ record['payload_half']
    near = axis @ payload[:3, 3] - radius
    tip = axis @ jaw[:3, 3]
    if hand_points is not None:
        tip = max(tip, np.max((hand_points - palm[:3, 3]) @ (palm[:3, :3] @ axis)))
    distance = max(0., tip - near) + margin
    target = palm.copy()
    target[:3, 3] -= (palm[:3, :3] @ axis) * distance
    return target


def opening_clear(frame, planning, side, label):
    """Check the whole jaw-opening sweep against depth and the other arm."""
    hand = frame.hands[side]
    for fraction in np.linspace(0., 1., 9):
        gate = Clearance(frame, planning.arms, side, planning.settings.collision,
                         (label,), fingers=hand.joints * (1. - fraction), exempt_start=True)
        reason = gate.violation(frame.arm_joints(side))
        if reason:
            return reason
    return None


def unloaded_check(ctx, side):
    """Keep the released object's visible depth; do not attach its old box to the wrist."""
    scene = []

    def check(state, reference):
        if isinstance(state, Frame):
            scene[:] = [Clearance(state, ctx.arms, side, ctx.settings.collision, exempt_start=True)]
        return scene[0].path_violation(state.arm_joints(side), reference)
    return check


def extract(ctx, epoch, side, target):
    frame = track(ctx, epoch, side, target, None, unloaded_check(ctx, side))
    if frame is None:
        raise SkillFailure('ARM_NOT_REACHED', 'Open jaws did not finish calibrated axial withdrawal')
    return frame


def return_checked(ctx, epoch, side):
    frame = ctx.observe(epoch)
    rest = ctx.settings.placement.rest_arm(side)
    rows = joint_rows(frame.arm_joints(side), rest, ctx.settings.physics.arm_rate,
                      ctx.settings.servo.tick)
    follow_rows(ctx, epoch, side, rows, unloaded_check(ctx, side))
    joints = settle_joints(ctx, epoch, side, rest)
    if np.max(np.abs(joints - rest)) > ctx.settings.placement.rest_tolerance:
        raise SkillFailure('ARM_NOT_REACHED', 'Measured arm did not return near its rest pose')


def hand_extent(frame, arm, side):
    """World corners of measured jaw collision boxes, for release bookkeeping."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    boxes = arm.collision.hand_boxes(arm, frame.arm_joints(side), frame.hands[side].joints)
    return np.vstack([points_in(frame.world_torso, center + (corners * half) @ rotation.T)
                      for center, rotation, half in boxes])


def require_hold(ctx, epoch, side):
    """Fresh bilateral effort, with the grasp's existing squeeze command unchanged."""
    evidence = FreshEvidence(epoch)
    settings = ctx.settings.grasp
    for index in range(settings.recovery_samples):
        if index:
            ctx.advance(epoch, settings.recovery_tick)
        frame = ctx.observe(epoch)
        held = opposed(frame.hands[side], settings)
        count = evidence.update(frame, held)
        if held and count is not None and count >= settings.recovery_confirmations:
            return frame
    raise SkillFailure('HOLD_UNCERTAIN', 'Both Dex1 jaws did not sustain hold evidence')
