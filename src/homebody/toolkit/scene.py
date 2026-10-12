"""Collision gates and sensed geometry near a target."""
import numpy as np

from homebody.helpers.collision import Clearance
from homebody.helpers.geometry import inverse, points_in, unproject


def clearance(frame, arms, side, settings, exempt_labels=(), hand="current", carried=None, exempt_start=False):
    """A Clearance gate for SIDE's arm against FRAME's depth (EXEMPT_LABELS removed). HAND is
    "current" (the fingers as measured, so a preshaped hand is checked as it is) or "open".
    Use .violation(joints) and .path_violation(a, b); each returns None or the reason."""
    fingers = np.zeros_like(frame.hands[side].joints) if hand == "open" else None
    return Clearance(frame, arms, side, settings.collision, tuple(exempt_labels), carried=carried,
                     exempt_start=exempt_start, fingers=fingers)


def points_near(frame, center_torso, radius=0.25, exclude_labels=()):
    """Sensed depth points in the torso frame within RADIUS of CENTER_TORSO, labels in
    EXCLUDE_LABELS removed: container walls and neighbours around a target."""
    mask = frame.labels >= 0
    for label in exclude_labels:
        mask &= frame.labels != label
    points = points_in(inverse(frame.world_torso) @ frame.world_camera, unproject(frame.depth, frame.intrinsics, mask))
    return points[np.linalg.norm(points - np.asarray(center_torso, float), axis=1) < radius]
