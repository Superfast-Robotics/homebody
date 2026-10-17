"""The model and operator see why a known floor has no admitted departure."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from homebody.helpers.floor import departure_status
from homebody.real2sim.exploration import app, vlm_pick
from homebody.real2sim.exploration.livemap import Boundary, LiveMap
from homebody.real2sim.exploration.planner import Planner


def room():
    boundary = Boundary(((-2., -2.), (2., -2.), (2., 2.), (-2., 2.)))
    live = LiveMap(boundary, margin_m=.5)
    live.floor[:] = 1
    vv, uu = np.indices(live.floor.shape)
    x, _ = live.xy(uu, vv)
    wall = (x >= 0.) & (x <= .15)
    live.hits[wall] = 6
    live.first_hit[wall] = 0.
    live.last_hit[wall] = 20.
    live.hit_zmin[wall], live.hit_zmax[wall] = .3, 1.3
    live.maxz[wall] = 1.3
    live.now, live.version = 20., 1
    return live


def test_blocked_start_explains_empty_reach_without_erasing_known_floor():
    live = room()
    known = live.known_free().copy()
    clear = (-.5, 0., 0.)
    blocked = (-.28, 0., 0.)
    assert live.collision_checker(.25).is_safe(*blocked[:2])
    assert not live.collision_checker(.30).is_safe(*blocked[:2])
    assert departure_status(live, clear, .30)["status"] == "available"
    assert departure_status(live, blocked, .30)["status"] == "starting_footprint_blocked"
    assert not np.isfinite(live.geodesic(*blocked[:2], .30)).any()
    np.testing.assert_array_equal(live.known_free(), known)


def test_missing_pose_and_missing_permission_are_not_obstacle_claims():
    live = room()
    assert departure_status(live, None, .30)["status"] == "pose_unavailable"
    live.task_region_required = True
    assert departure_status(live, (-.5, 0., 0.), .30)["status"] == "region_required"
    live.task_region_required = False
    assert departure_status(live, (-1.98, 0., 0.), .30)["status"] == "outside_region"


def test_map_and_3d_overlay_agree_and_visibility_does_not_change_admission():
    live = room()
    explorer = SimpleNamespace(
        lock=threading.RLock(), live=live, planner=Planner(live),
        visited=np.zeros_like(live.floor, bool), trail=[], zone=live.boundary,
        boundary=live.boundary)
    state, scene = {}, Mock()
    for pose, expected in [((-.5, 0., 0.), "available"),
                           ((-.28, 0., 0.), "starting_footprint_blocked")]:
        image, _, _ = vlm_pick.render(explorer, pose)
        tint = np.all(image[:-vlm_pick.LEGEND_PX] == vlm_pick.REACHABLE_FLOOR, axis=2)
        status = app.update_reachable_floor(scene, state, live, pose, .30, map_id=1)
        assert status["status"] == expected
        if expected == "available":
            assert tint.any()
        else:
            assert not tint.any()
            assert not state["handle"].image[..., 3].any()
        hidden = app.update_reachable_floor(
            scene, state, live, pose, .30, map_id=1, visible=False)
        assert hidden == status
        assert not state["handle"].visible
    absent = app.update_reachable_floor(scene, state, live, None, .30, map_id=1)
    assert absent["status"] == "pose_unavailable"
    assert not state["handle"].visible
