"""Dex1 loaded motion: keep robot padding, use a smaller payload depth allowance."""
import numpy as np

from homebody.helpers.collision import Clearance as ArmClearance
from homebody.helpers.convex import point_hits
from homebody.motion.arm import reached_or_stopped, step_to, track
from homebody.primitives.observations import Frame
from homebody.skills.contract import SkillFailure


class Clearance(ArmClearance):
    """Observed payload hulls need not inherit the robot link's full 8 mm padding.

    Keep 4 mm of depth allowance, the full measured hull, robot collisions and
    swept-path checks. Unloaded jaws (including released-object returns) retain
    the original margin.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._payload_start is not None:
            self._payload_start = self.hits(self.payload_box(self.initial_joints),
                                            min(self.settings.cloud_margin, 0.004))

    def payload_hits(self, box):
        margin = min(self.settings.cloud_margin, 0.004)
        hits = self.hits(box, margin)
        if self._payload_start is not None:
            hits &= ~self._payload_start
        hull = self.carried.get('payload_hull')
        if hull is not None and np.any(hits):
            local = (self.points[hits] - box[0]) @ box[1]
            hits[np.flatnonzero(hits)] = point_hits(hull, local, margin)
        return hits


def live_check(ctx, side, exclude_labels=()):
    scene = []

    def check(state, reference):
        if isinstance(state, Frame):
            scene[:] = [Clearance(state, ctx.arms, side, ctx.settings.collision,
                                  exclude_labels, ctx.carried.get(side), exempt_start=True)]
        return scene[0].path_violation(state.arm_joints(side), reference)
    return check


def servo(ctx, epoch, side, target, *, tracked_target=None, exclude_labels=()):
    frame = track(ctx, epoch, side, target, tracked_target, live_check(ctx, side, exclude_labels))
    if frame is None:
        raise SkillFailure('ARM_NOT_REACHED', 'Measured palm did not converge before the servo deadline')
    return frame


def servo_best_effort(ctx, epoch, side, target, *, exclude_labels=(), position_only=False):
    frame = track(ctx, epoch, side, target, None, live_check(ctx, side, exclude_labels),
                  best_effort=True, position_only=position_only)
    return reached_or_stopped(ctx, epoch, frame)


def move_joints(ctx, epoch, side, target, exclude_labels=()):
    step_to(ctx, epoch, side, target, live_check(ctx, side, exclude_labels))
