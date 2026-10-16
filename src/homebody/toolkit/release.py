"""Bounded payload-centred release alternatives; depth ranks, never certifies, landing."""
import numpy as np
from scipy.spatial.transform import Rotation

from homebody.helpers.collision import under_box


def release_alternatives(palm, record, points, angle, height, margin):
    """Twelve tilted poses at unchanged payload XY, ranked by visible rim intrusion.

    POINTS are world depth excluding the payload. ANGLE and HEIGHT are existing search
    allowances. Unknown depth has neutral rank; every pose still needs IK and collision
    checks. Rotation is about the payload centre, not the wrist.
    """
    attachment = record["palm_to_payload"]
    center = (palm @ attachment)[:3, 3]
    candidates = []
    for dz in (0., height, -height):
        for axis in (np.array([1., 0., 0.]), np.array([0., 1., 0.])):
            for sign in (-1., 1.):
                candidate = palm.copy()
                candidate[:3, :3] = Rotation.from_rotvec(sign * angle * axis).as_matrix() @ palm[:3, :3]
                candidate[:3, 3] = center + [0., 0., dz] - candidate[:3, :3] @ attachment[:3, 3]
                payload = candidate @ attachment
                half = record["payload_half"]
                bottom = payload[2, 3] - np.abs(payload[2, :3]) @ half
                box = (payload[:3, 3], payload[:3, :3], half)
                beneath = points[under_box(points, box, margin)]
                intrusion = max(0., float(np.max(beneath[:, 2]) - bottom)) if len(beneath) else 0.
                candidates.append((intrusion, abs(dz), candidate))
    return [item[2] for item in sorted(candidates, key=lambda item: item[:2])]
