"""Navigation and displays share a pelvis centre without display-side hardware access."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from homebody.backends.g1 import real
from homebody.primitives.navigation.errors import ObservationUnavailable
from homebody.real2sim.exploration import explore_backend, sources
from homebody.real2sim.exploration.livemap import Boundary, LiveMap
from homebody.real2sim.exploration.planner import RADIUS_M


@pytest.fixture
def robot(monkeypatch):
    clock = SimpleNamespace(monotonic=lambda: 100., time=lambda: 1000.)
    monkeypatch.setattr(real, "time", clock)
    monkeypatch.setattr(explore_backend, "time", clock)
    source = sources.RobotSource.__new__(sources.RobotSource)
    source.localization = SimpleNamespace(
        pose=lambda: (1., 2., .4), pose_age_s=lambda: .1,
        zeroed=lambda: True, frame_reset=False)
    source._body = source.frame_epoch = None
    backend = explore_backend.ExploreRealBackend.__new__(explore_backend.ExploreRealBackend)
    source.backend = backend
    source._ctx = Mock(return_value=SimpleNamespace())
    backend._explore_source = source
    backend.explorer = SimpleNamespace(visual_history=SimpleNamespace(epoch=7))
    backend._term_stamp = {}
    backend._imu_yaw = (.2, 100.)
    backend._tilt = (.02, .08)
    backend._q_waist = (.1, .02, -.03)
    backend._yaw_offset = .1
    backend._yaw_offset_at = 100.
    backend._imu_yaw_mirrored = False
    backend._yaw_pairs = []
    backend._yaw_disagree = 0
    backend._twist_cmd = (0., 0., 0.)
    backend._twist_at = 100.
    backend._twist_nonzero_at = 0.
    backend._travel_m = 0.
    backend._turn_rad = 0.
    backend._twist_zero_since = None
    backend._lower_last_command = 100.
    backend._pose_held = None
    backend._pose_reject_since = None
    backend._control_error = None
    backend._ensure_port = Mock(side_effect=AssertionError("must not acquire hardware"))
    return source, backend


def test_navigation_and_display_use_pelvis_while_camera_pose_stays_torso(robot, monkeypatch):
    source, backend = robot
    monkeypatch.setattr(sources.gait, "pose", lambda ctx: backend.base_pose())
    expected = source.navigation_pose()
    assert np.linalg.norm(np.array(expected[:2]) - source.pose()[:2]) > .02
    assert source.footprint_pose() == pytest.approx(expected)
    assert source.base_xy() == pytest.approx(expected[:2])
    assert source.feeds().pose() == pytest.approx(expected)
    assert source.pose() == (1., 2., .4)
    backend._ensure_port.assert_not_called()


def test_display_uses_current_localization_without_refreshing_driver_or_pose_gate(robot):
    source, backend = robot
    backend.base_pose = Mock(side_effect=AssertionError("must not refresh driver pose"))
    backend._execution_base_pose_sample = (7, 0., 0., (100., 100., 0.))
    before = (backend._yaw_offset, backend._yaw_offset_at, backend._pose_held,
              backend._pose_reject_since, backend._term_stamp.copy())
    first = source.feeds().pose()
    source.localization.pose = lambda: (1.2, 2.1, .4)
    second = source.feeds().pose()
    assert np.subtract(second, first) == pytest.approx((.2, .1, 0.))
    assert (backend._yaw_offset, backend._yaw_offset_at, backend._pose_held,
            backend._pose_reject_since, backend._term_stamp) == before
    source._ctx.assert_not_called()
    backend.base_pose.assert_not_called()
    backend._ensure_port.assert_not_called()


def test_near_wall_display_and_driver_agree_on_reachable_floor(robot, monkeypatch):
    source, backend = robot
    monkeypatch.setattr(sources.gait, "pose", lambda ctx: backend.base_pose())
    live = LiveMap(Boundary(((-3., -3.), (3., -3.), (3., 4.), (-3., 4.))), margin_m=0.)
    live.floor[:] = 1
    vv, uu = np.mgrid[:live.H, :live.W]
    xs, _ = live.xy(uu, vv)
    wall = xs >= 1.299
    live.hits[wall] = 3
    live.last_hit[wall] = live.now
    live.version += 1
    torso = source.pose()
    pelvis = source.navigation_pose()
    assert not np.isfinite(live.geodesic(*torso[:2], RADIUS_M)).any()
    driver_reach = np.isfinite(live.geodesic(*pelvis[:2], RADIUS_M))
    display_reach = np.isfinite(live.geodesic(*source.footprint_pose()[:2], RADIUS_M))
    assert driver_reach.any()
    np.testing.assert_array_equal(display_reach, driver_reach)


@pytest.mark.parametrize(("x", "rejected_since", "accepted"), [
    (1.01, 99.9, True),  # A formerly rejected stream returned within the motion allowance.
    (2., 100. - real.POSE_HOLD_S - .01, True),  # A persistent re-registration is now admissible.
    (2., 99.9, False),
])
def test_display_matches_pending_xy_acceptance_without_mutating_gate(
        robot, x, rejected_since, accepted):
    source, backend = robot
    source.localization.pose = lambda: (x, 2., .4)
    backend._pose_held = (np.array([1., 2.]), 99.9)
    backend._pose_reject_since = rejected_since
    before = (backend._pose_held[0].copy(), backend._pose_held[1],
              backend._pose_reject_since, backend._travel_m, backend._turn_rad)
    displayed = source.footprint_pose()
    assert backend._pose_xy_accepted(np.array([x, 2.]), 100.) == accepted
    np.testing.assert_array_equal(backend._pose_held[0], before[0])
    assert (backend._pose_held[1], backend._pose_reject_since,
            backend._travel_m, backend._turn_rad) == before[1:]
    assert displayed == pytest.approx(backend.base_pose())


@pytest.mark.parametrize("unavailable", ["stale_localization", "lost_localization", "frame_reset",
                                         "unzeroed", "stale_body", "missing_tilt", "body_error"])
def test_armed_display_refuses_unavailable_pose_without_torso_fallback(robot, unavailable):
    source, backend = robot
    if unavailable == "stale_localization":
        source.localization.pose_age_s = lambda: real.LOC_ODOM_AGE_MAX_S + .1
    elif unavailable == "lost_localization":
        source.localization.pose_age_s = lambda: float("inf")
    elif unavailable == "frame_reset":
        source.localization.frame_reset = True
    elif unavailable == "unzeroed":
        source.localization.zeroed = lambda: False
    elif unavailable == "stale_body":
        backend._imu_yaw = (.2, 99.4)
    elif unavailable == "missing_tilt":
        backend._tilt = None
    else:
        backend._control_error = RuntimeError("body disconnected")
    with pytest.raises(ObservationUnavailable):
        source.footprint_pose()
    assert source.feeds().pose() is None
    backend._ensure_port.assert_not_called()
    source._ctx.assert_not_called()


def test_display_refuses_frame_change_during_snapshot(robot):
    source, _ = robot
    source.motion_epoch = Mock(side_effect=[7, 8])
    with pytest.raises(ObservationUnavailable, match="frame changed"):
        source.footprint_pose()


def test_unarmed_preview_has_no_body_correction(robot):
    source, _ = robot
    source.backend = None
    assert source.navigation_pose() == source.pose()
    assert source.footprint_pose() == source.pose()
    assert source.feeds().pose() == source.pose()
    source._ctx.assert_not_called()
