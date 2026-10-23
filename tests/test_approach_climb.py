from types import SimpleNamespace

import numpy as np
import pytest

from homebody.skills.dex1.pick import start_climb


def frame(palm_z, bottom, visible=True):
    palm = np.eye(4)
    palm[2, 3] = palm_z
    arm = SimpleNamespace(forward=lambda joints: (palm, None))
    current = SimpleNamespace(world_torso=np.eye(4), visible=lambda label: visible,
                              target_points=lambda label: np.array([[.5, 0., bottom], [.5, 0., bottom + .03]]))
    return current, arm


def test_a_palm_below_the_support_must_first_rise_over_it():
    current, arm = frame(.65, .74)
    assert start_climb(current, arm, None, 15, .05) == pytest.approx(.14)


def test_a_palm_already_over_the_support_or_an_unseen_target_needs_no_climb():
    assert start_climb(*frame(.85, .74), None, 15, .05) == 0.
    assert start_climb(*frame(.65, .74, visible=False), None, 15, .05) == 0.
