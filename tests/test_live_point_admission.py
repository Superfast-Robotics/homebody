"""Held reassessment uses configured goal separation without commanding the body."""
import math
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from homebody.helpers.floor import Waypoint
from homebody.real2sim.exploration import explorer as explorer_module
from homebody.real2sim.exploration import settings
from homebody.real2sim.exploration.livemap import Boundary
from homebody.real2sim.exploration.skills import REGISTRY
from homebody.real2sim.exploration.vlm_pick import Answer
from homebody.session.live_context import LiveContext
from homebody.skills.contract import Runner


@pytest.mark.parametrize(("tolerance", "distance", "admitted"), [
    (.125, .0625, False),
    (.125, .125, False),
    (.125, .1875, True),
    (.5, .375, False),
    (.5, .625, True),
])
def test_continuation_goal_separation_uses_configured_tolerance(
        tmp_path, monkeypatch, tolerance, distance, admitted):
    configured = settings.Settings()
    monkeypatch.setitem(settings._installed, "settings", replace(
        configured, drive=replace(configured.drive, tolerance_m=tolerance)))
    monkeypatch.setattr(explorer_module, "time", SimpleNamespace(
        time=lambda: 10., monotonic=lambda: 10.))
    commands = Mock()
    source = SimpleNamespace(
        stop_flag=threading.Event(), pause_flag=threading.Event(), attach=lambda *_: None,
        pose=lambda: (0., 0., 0.), base_xy=lambda: (0., 0.),
        now=lambda: 10., odom_age_s=lambda: .1, health_stable_s=lambda: math.inf,
        walk=commands.walk, turn_hold=commands.turn_hold, hold_position=commands.hold_position)
    boundary = Boundary(((-2., -2.), (4., -2.), (4., 4.), (-2., 4.)))
    explorer = explorer_module.Explorer(source, boundary, tmp_path)
    # Synthetic free floor isolates admission; the real planner still checks the route.
    explorer.live.floor[:] = 1
    explorer.live.version += 1
    explorer.map_t = 10.
    events = []
    context = LiveContext(explorer, emit=events.append)
    runner = Runner(context, REGISTRY)
    record = {"n": 3}
    payload = {
        "answer": Answer("point", (250, 250), "inspect a distinct reachable area"),
        "record": record, "frame": explorer.live, "history": {"epoch": 0},
        "transform": np.array([[.04, 0., 0.], [0., .04, 0.], [0., 0., 1.]]),
        "image": np.zeros((101, 101, 3), np.uint8), "selection": None, "near": None,
        "continuation": {"rejections": [
            {"proposed_goal": [2., 2.]}, {"proposed_goal": [1. + distance, 1.]}]},
    }

    attempt = runner.prepare("pick_point", {"admission": payload}, context.frame())

    assert record["selection_basis"] == "requested_observed_floor"
    assert record["proposed_goal"] == [1., 1.]
    if admitted:
        assert not attempt.finished
        assert attempt.plan.waypoint.xy == (1., 1.) and record["used"]
        assert "rejection_code" not in record
        assert [event["type"] for event in events] == [
            "skill_started", "skill_stage", "skill_prepared"]
        runner.cancel(attempt, "Preparation test ends without motion")
        assert attempt.result.code == "CANCELLED"
    else:
        assert attempt.finished and attempt.result.code == "ACTION_UNAVAILABLE"
        assert "repeats a refused destination" in attempt.result.message
        assert record["rejection_code"] == "repeated_proposal"
        assert not record.get("used", False)
        assert [event["type"] for event in events] == [
            "skill_started", "skill_stage", "skill_finished"]
    assert len([event for event in events if event["type"] == "skill_finished"]) == 1
    assert commands.mock_calls == []


def test_route_preparation_uses_same_pelvis_as_point_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(explorer_module, "time", SimpleNamespace(
        time=lambda: 10., monotonic=lambda: 10.))
    raw_pose = (1., 2., .4)
    pelvis_pose = (.974, 2.012, .3)
    commands = Mock()
    source = SimpleNamespace(
        stop_flag=threading.Event(), pause_flag=threading.Event(), attach=lambda *_: None,
        pose=lambda: raw_pose, now=lambda: 10.,
        odom_age_s=lambda: .1, health_stable_s=lambda: math.inf,
        walk=commands.walk, turn_hold=commands.turn_hold, hold_position=commands.hold_position)
    owner = explorer_module.Explorer(
        source, Boundary(((-3., -3.), (3., -3.), (3., 4.), (-3., 4.))), tmp_path)
    owner.routes = []
    source.navigation_pose = Mock(side_effect=lambda: (
        pytest.fail("pose acquisition must not hold the ingestion lock")
        if owner.lock._is_owned() else pelvis_pose))
    live = owner.live
    live.floor[:] = 1
    vv, uu = np.mgrid[:live.H, :live.W]
    xs, _ = live.xy(uu, vv)
    wall = xs >= 1.299
    live.hits[wall] = 3
    live.last_hit[wall] = live.now
    live.version += 1
    owner.map_t = 10.
    radius = owner.planner.radius_m
    assert not np.isfinite(live.geodesic(*raw_pose[:2], radius)).any()
    assert np.isfinite(live.geodesic(*pelvis_pose[:2], radius)).any()
    context = LiveContext(owner)
    runner = Runner(context, REGISTRY)
    waypoint = Waypoint("vlm", (-.2, 2.), [])

    attempt = runner.prepare("pick_point", {"waypoint": waypoint, "mps": .3}, context.frame())

    assert not attempt.finished
    source.navigation_pose.assert_called_once_with()
    assert attempt.plan.points[0] == pytest.approx(pelvis_pose[:2])
    assert attempt.plan.points[-1] == pytest.approx(waypoint.xy)
    assert attempt.plan.radius == radius
    assert source.pose() == raw_pose
    runner.cancel(attempt, "Route-preparation test ends without motion")
    assert commands.mock_calls == []
