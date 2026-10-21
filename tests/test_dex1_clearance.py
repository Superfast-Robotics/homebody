"""Payload-only padding leaves actual intersections and robot padding protected."""
from types import SimpleNamespace

import numpy as np

from homebody.helpers.collision import Clearance as ArmClearance
from homebody.helpers.convex import convex_shape
from homebody.skills.dex1.clearance import Clearance


def test_payload_margin_keeps_intersections_and_depth_allowance():
    gate = Clearance.__new__(Clearance)
    gate.settings = SimpleNamespace(cloud_margin=.008)
    gate.points = np.array([[.009, 0., 0.], [.013, 0., 0.], [.016, 0., 0.]])
    vertices = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)])
    gate.carried = {'payload_hull': convex_shape(vertices)}
    gate._payload_start = None
    box = np.zeros(3), np.eye(3), np.ones(3) * .01
    assert gate.payload_hits(box).tolist() == [True, True, False]
    assert ArmClearance.payload_hits(gate, box).tolist() == [True, True, True]
    # The inherited jaw/arm query still sees the full original margin.
    assert gate.hits(box, gate.settings.cloud_margin).tolist() == [True, True, True]
