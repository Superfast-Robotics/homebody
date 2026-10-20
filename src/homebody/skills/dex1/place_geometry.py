"""Dex1 placement geometry and checks, using calibration and measured state only.

Jaw calibration gives pad frames, not pad mesh extents or friction. Collision
extents come from the robot's calibrated collision interface, never guessed pads.
No contact-location sensor or loaded global planner is exposed by this interface.
"""
import numpy as np

from homebody.helpers.collision import Clearance
from homebody.helpers.geometry import inverse, points_in
from homebody.helpers.trajectory import retime_path
from homebody.motion.arm import follow_rows, guard, track
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


def return_clearance(arm, world_torso, fingers, joints, released, margin):
    """Signed clearance from the remembered release cylinder, padded for open jaws.

    The stored radius already includes the hand's horizontal reach. Vertical
    clearance uses actual jaw corners, including changes in wrist orientation.
    """
    center, radius, top = released
    palm = world_torso @ arm.forward(joints)[0]
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    boxes = arm.collision.hand_boxes(arm, joints, fingers)
    low = min(np.min(points_in(world_torso, c + (corners * h) @ r.T)[:, 2])
              for c, r, h in boxes)
    return max(float(np.linalg.norm(palm[:2, 3] - center[:2]) - radius - margin),
               float(low - top - margin))


def return_path_check(frame, arm, side, released, settings, start, end):
    """Allow escape from the envelope, never motion back into it.

    A partly cleared start may move out monotonically; once outside, every
    subsequent sample must stay outside. This also covers depth occlusion.
    """
    if released is None:
        return None
    count = max(1, int(np.ceil(np.max(np.abs(end - start)) / settings.joint_step)))
    previous = return_clearance(arm, frame.world_torso, frame.hands[side].joints,
                                start, released, settings.cloud_margin)
    for fraction in np.linspace(0., 1., count + 1)[1:]:
        value = return_clearance(arm, frame.world_torso, frame.hands[side].joints,
                                 start + fraction * (end - start), released, settings.cloud_margin)
        if value < min(previous, 0.) - 1e-6:
            return 'empty hand re-enters released object envelope'
        previous = value
    return None


def return_waypoints(palm, home, hand_points, released, observed, margin):
    """Lift above nearby object/support depth, then retreat toward rest.

    Overhead depth remains an obstacle in Clearance, not a height to climb over.
    No fixed scene coordinates or guessed finger dimensions enter this route.
    """
    center, radius, top = released
    nearby = np.linalg.norm(observed[:, :2] - center[:2], axis=1) <= radius + margin
    nearby &= observed[:, 2] <= max(top, np.max(hand_points[:, 2])) + margin
    if np.any(nearby):
        top = max(top, float(np.max(observed[nearby, 2])))
    lift = palm.copy()
    lift[2, 3] += max(0., top + 2 * margin - np.min(hand_points[:, 2]))
    retreat = lift.copy()
    toward_home = home[:2, 3] - center[:2]
    distance = np.linalg.norm(toward_home)
    retreat[:2, 3] = center[:2] + toward_home * min(1., (radius + 2 * margin) / max(distance, 1e-9))
    return lift, retreat


def return_checked(ctx, epoch, side, released=None):
    frame = ctx.observe(epoch)
    arm, settings = ctx.arms[side], ctx.settings
    rest = settings.placement.rest_arm(side)
    start = frame.arm_joints(side)
    gate = Clearance(frame, ctx.arms, side, settings.collision, exempt_start=True)

    def blocked(a, b):
        return (return_path_check(frame, arm, side, released, settings.collision, a, b) or
                gate.path_violation(a, b))

    path = [start, rest]
    reason = blocked(start, rest)
    if reason and released is not None:
        palm = frame.palm(arm, side)
        home = frame.world_torso @ arm.forward(rest)[0]
        waypoints = return_waypoints(palm, home, hand_extent(frame, arm, side), released,
                                     points_in(frame.world_torso, gate.observed_points),
                                     settings.collision.cloud_margin)
        ctx.emit({'type': 'mechanism_active', 'mechanism': 'clearance_return', 'side': side,
                  'direct_reason': reason, 'released_center': released[0].tolist(),
                  'released_radius': float(released[1]), 'released_top': float(released[2]),
                  'waypoints': [p[:3, 3].tolist() for p in waypoints]})
        path = [start]
        previous = palm
        for waypoint in waypoints:
            count = max(1, int(np.ceil(np.linalg.norm(waypoint[:3, 3] - previous[:3, 3]) / .025)))
            for fraction in np.linspace(0., 1., count + 1)[1:]:
                target = previous.copy()
                target[:3, 3] += fraction * (waypoint[:3, 3] - previous[:3, 3])
                q = arm.solve(inverse(frame.world_torso) @ target, path[-1], restarts=False)
                if q is None:
                    raise SkillFailure('ARM_NOT_REACHED', 'Empty-arm clearance waypoint is unreachable')
                reason = blocked(path[-1], q)
                if reason:
                    raise SkillFailure('COLLISION', f'Empty-arm clearance route: {reason}')
                path.append(q)
            previous = waypoint
        reason = blocked(path[-1], rest)
        path.append(rest)
    if reason:
        raise SkillFailure('COLLISION', f'Empty-arm return: {reason}')

    live = unloaded_check(ctx, side)

    def check(state, reference):
        return (return_path_check(state, arm, side, released, settings.collision,
                                  state.arm_joints(side), reference) or live(state, reference))

    ctx.stage('Returning arm through checked clearance')
    rows = retime_path(np.asarray(path), settings.physics.arm_rate, settings.servo.tick)
    follow_rows(ctx, epoch, side, rows, check)
    # Keep the same live gate while the measured arm catches the final reference.
    started = ctx.observations.time
    while ctx.observations.time - started < settings.motion.arm_timeout:
        state = ctx.observe(epoch)
        if np.max(np.abs(state.arm_joints(side) - rest)) <= settings.placement.rest_tolerance:
            return
        guard(ctx, side, check, state, rest, 'return settling')
        ctx.check(epoch)
        ctx.actions.arm_target(epoch, side, rest)
        ctx.advance(epoch, settings.servo.tick)
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
