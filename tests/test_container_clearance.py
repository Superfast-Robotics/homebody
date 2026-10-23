from types import SimpleNamespace

import numpy as np
import pytest

from homebody.helpers.collision import in_corridor, swept_rise
from homebody.session.settings import Settings
from homebody.skills.dex1.place import support_release


def basket(floor=.70, rim=.93, near=-.87, far=-1.38):
    """A basket's floor and its near and far walls, as a support's sensed points."""
    xs, ys = np.meshgrid(np.linspace(far, near, 27), np.linspace(-.80, -.46, 18))
    bottom = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, floor)]
    walls = [np.c_[np.full(18, x), np.linspace(-.80, -.46, 18), np.full(18, z)]
             for x in (near, far) for z in np.linspace(floor, rim, 6)]
    return np.vstack([bottom, *walls])


def test_corridor_covers_the_segment_and_its_ends():
    points = np.array([[.5, .05, 0.], [.5, .2, 0.], [-.05, 0., 0.], [1.2, 0., 0.]])
    np.testing.assert_array_equal(in_corridor(points, np.zeros(3), np.array([1., 0.]), .1),
                                  [True, False, True, False])


def test_rise_clears_a_rim_in_the_path_and_ignores_floor_and_overhead():
    rim = np.array([[-.87, -.63, .93], [-.87, -.60, .0], [-.87, -.63, 1.5]])
    start, end = np.array([-.75, -.63]), np.array([-1.0, -.63])
    assert swept_rise(rim, start, end, .84, .96, .04, .04) == pytest.approx(.13)
    assert swept_rise(rim, start, end, .98, 1.1, .04, .04) == 0.
    assert swept_rise(rim, start, start + [.5, 0.], .84, .96, .04, .04) == 0.


def test_basket_release_drops_from_over_the_rim_and_keeps_the_floor_as_landing():
    settings = Settings()
    points = basket()
    current = SimpleNamespace(target_points=lambda label: points, world_torso=np.eye(4))
    current.world_torso[:3, 3] = [-.55, -.63, .75]
    carried = {"palm_to_payload": np.eye(4), "payload_half": np.array([.03, .03, .05])}
    palm, release = support_release(current, 3, np.array([-1.12, -.63, .70]), .015, carried, np.eye(4), settings)
    assert release[2] == pytest.approx(.70)
    assert palm[2, 3] - .05 == pytest.approx(.93 + settings.placement.rim_clearance)


def test_flat_support_keeps_the_requested_height():
    settings = Settings()
    xs, ys = np.meshgrid(np.linspace(-.3, .3, 20), np.linspace(-.3, .3, 20))
    points = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, .75)]
    current = SimpleNamespace(target_points=lambda label: points, world_torso=np.eye(4))
    current.world_torso[:3, 3] = [-.5, 0., .75]
    carried = {"palm_to_payload": np.eye(4), "payload_half": np.array([.03, .03, .05])}
    palm, _ = support_release(current, 3, np.array([0., 0., .75]), .015, carried, np.eye(4), settings)
    assert palm[2, 3] == pytest.approx(.75 + .015 + .05)
