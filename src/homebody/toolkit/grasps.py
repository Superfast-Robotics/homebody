"""Grasp poses at which a calibrated jaw opening fits the target's visible width."""
from dataclasses import dataclass

import numpy as np

from homebody.helpers.geometry import inverse
from homebody.helpers.grasp import candidates


@dataclass(frozen=True)
class Grasp:
    """A torso-frame palm pose GRASP with PRE above it. Lower TIER inserts deeper (ranks first);
    JAW_IN_PALM is where the fingers meet in the palm frame; WIDTH is the visible width closed on."""
    tier: int
    width: float
    pre: np.ndarray
    grasp: np.ndarray
    jaw_in_palm: np.ndarray


def calibrated_grasps(frame, label, jaw_samples, grasp_settings, limit=48):
    """Top grasps of LABEL's visible points for one hand's JAW_SAMPLES, best first."""
    points = frame.target_points(label)
    apertures = [sample["aperture"] for sample in jaw_samples]
    limits = min(apertures) - grasp_settings.aperture_error, max(apertures) + grasp_settings.aperture_error
    torso = inverse(frame.world_torso)
    found = []
    for sample in jaw_samples:
        palm_to_jaw = np.asarray(sample["palm_to_jaw"])
        for tier, pre, palm, _ in candidates(points, palm_to_jaw, limits, grasp_settings):
            width = float(np.ptp(points @ (palm @ palm_to_jaw)[:3, 0]))
            if abs(sample["aperture"] - width) <= grasp_settings.aperture_error:
                found.append(Grasp(int(tier), width, torso @ pre, torso @ palm, palm_to_jaw[:3, 3].copy()))
    return sorted(found, key=lambda g: g.tier)[:limit]
