"""Recovery and turn paths carry the held planning footprint through the observation handoff."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.motion.exploration import gait
from homebody.motion.exploration.localizer import PoseGate
from homebody.primitives.navigation.context import Ctx, Deadline
from homebody.primitives.navigation.contracts import DriveFailure, HardwareUnavailable
from homebody.primitives.navigation.errors import ActionRefused, Stopped
from homebody.real2sim.exploration.explorer import Explorer
from homebody.real2sim.exploration.livemap import Boundary
from homebody.real2sim.exploration.sources import RobotSource
from homebody.skills.navigate_real import navigation


@pytest.fixture
def rig():
    occupied = np.zeros((101, 101), bool)
    occupied[:, 50] = True  # A wall at x=0, in an otherwise open five-metre map.
    art = SimpleNamespace(occupancy=occupied, center_allowed=np.ones_like(occupied),
                          T_map_px=np.array([[.05, 0., -2.5], [0., .05, -2.5], [0., 0., 1.]]))
    state = SimpleNamespace(time=0., pose=(.5, 0., 0.), version=1, commands=[], events=[],
                            on_step=lambda: None, stopped=False, frame=1)

    def guard(epoch, **_):
        if state.stopped or state.frame != epoch:
            raise Stopped("test authority ended")

    def observe(term):
        if term == "base_pose":
            return state.pose
        if term == "map":
            return art
        raise AssertionError(term)

    observe._owns_pose_consistency = True

    def step(channel, command):
        assert channel == "lower_body"
        guard(1)
        state.commands.append(command)
        state.time = round(state.time + .004, 9)
        state.on_step()

    ctx = Ctx(observe, step, lambda: state.time, Deadline(lambda: state.time, 5.),
              lambda _: None, lambda kind, payload: state.events.append((kind, payload)),
              PoseGate())
    port = navigation.NavigationPort(
        navigation.NavigationCapabilities(.25, .30, (.33,), .5, 2.),
        epoch=lambda: 1, guard=guard, fresh=lambda *_args, **_kwargs: guard(1),
        hold=lambda c: gait.hold(c, .2, 0., 0., 0.), stage=lambda *_: None,
        planning_radius=lambda r: r, recovery_map=lambda: (art, occupied.copy(), state.version),
        map_age=lambda: .01, stationary_guard=lambda *_: 1,
        motion_refusal=lambda *_: None, stopped=lambda: state.stopped,
        require_ready=lambda: None, can_refresh_observations=True)
    nav = navigation.Navigator(port)
    source = object.__new__(RobotSource)
    source.backend = SimpleNamespace(explorer=SimpleNamespace(live=SimpleNamespace(scope_ready=True)))
    source.pose_gate, source.turn_drift_m = ctx.gate, .5
    source.motion_epoch, source._motion_guard = port.epoch, guard
    source._fresh_for_motion = port.fresh
    source._motion_stage = lambda *_: None
    return state, art, ctx, nav, source


def continuation(state):
    return [payload for kind, payload in state.events if kind == "recovery.continuation"][-1]


@pytest.fixture
def recorded_explorer(tmp_path, rig):
    state, _, _, _, source = rig
    source.now = lambda: state.time
    source.pose = lambda: state.pose
    owner = Explorer(source, Boundary(((-2., -2.), (2., -2.), (2., 2.), (-2., 2.))), tmp_path)
    source.backend.explorer = owner
    owner.brief = "Explore the synthetic workspace."
    owner.phase, owner.t0, owner.t0_wall = "exploring", 0., 100.
    owner.vlm_picks = [{"n": 3, "reason": "Inspect the remaining space."}]
    leg = tmp_path / "vlm" / "leg_3.json"
    leg.parent.mkdir()
    leg.write_text(json.dumps(owner.vlm_picks[0]))
    return owner, leg


def test_usable_recovery_keeps_original_confirmation_hold(rig):
    state, _, ctx, nav, _ = rig
    result = nav._recovery_departure(ctx, .30, None)
    assert result["outcome"] == "departure_restored"
    assert result["held_start"]["planning_clear"]
    assert result["pose"] == [.5, 0., 0.]
    assert state.time == .2 and len(state.commands) == 50
    assert all((c.vx, c.vy, c.wz) == (0., 0., 0.) for c in state.commands)


@pytest.mark.parametrize("changing", ["pose", "map"])
def test_recovery_rechecks_clearance_changed_during_existing_hold(rig, changing):
    state, art, ctx, nav, _ = rig

    def change():
        if changing == "pose":
            state.pose = (.29, 0., 0.)
        else:
            art.occupancy[:, 55] = True
            state.version = 2

    state.on_step = change
    assert nav._recovery_departure(ctx, .30, None) is None
    assert continuation(state)["departure_verdict"] == "starting_footprint_blocked"
    assert state.time == .2


def test_one_new_clear_sample_is_not_sustained_held_clearance(rig):
    state, _, ctx, nav, _ = rig
    state.pose = (.29, 0., 0.)
    state.on_step = lambda: setattr(state, "pose", (.5, 0., 0.))
    assert nav._recovery_departure(ctx, .30, None) is None
    result = continuation(state)
    assert result["pose"] == [.5, 0., 0.]
    assert result["departure_verdict"] == "planning_clearance_not_retained_while_held"
    assert result["outcome"] == "no_usable_departure"


def test_recovery_rechecks_new_obstacles_after_planning(rig, monkeypatch):
    state, art, ctx, nav, _ = rig

    def prepare(*_, **__):
        art.occupancy[:, 55] = True
        state.version = 2
        raise DriveFailure("NO_COLLISION_FREE_PATH", "new evidence blocks the route")

    monkeypatch.setattr(navigation.drive, "prepare", prepare)
    assert nav._recovery_departure(ctx, .30, (1., 0.)) is None
    result = continuation(state)
    assert result["map_version"] == 2
    assert result["departure_verdict"] == "starting_footprint_blocked"


@pytest.mark.parametrize("interruption", ["stop", "frame", "deadline", "hold_failure"])
def test_confirmation_never_admits_after_interruption(rig, interruption):
    state, _, ctx, nav, _ = rig

    def interrupt():
        if interruption == "stop":
            state.stopped = True
        elif interruption == "frame":
            state.frame = 2
        elif interruption == "hold_failure":
            raise HardwareUnavailable("failed hold")

    state.on_step = interrupt
    if interruption == "deadline":
        ctx.deadline = Deadline(ctx.now, .1)
    with pytest.raises((Stopped, HardwareUnavailable)):
        nav._recovery_departure(ctx, .30, None)
    assert not any(kind == "recovery.continuation" for kind, _ in state.events)


def test_viewing_turn_refuses_execution_clear_but_planning_blocked_endpoint(rig):
    state, art, ctx, _, source = rig
    state.on_step = lambda: setattr(state, "pose", (.29, 0., 0.))
    execution, _ = gait.occupancy_grid(art, .25, exact_footprint=True)
    planning, _ = gait.occupancy_grid(art, .30, exact_footprint=True)
    assert execution.is_safe(.29, 0.) and not planning.is_safe(.29, 0.)
    with pytest.raises(ActionRefused) as caught:
        source._turn_hold(ctx, 0., 1., [None, None])
    assert caught.value.code == "LOOK_BLOCKED"
    assert caught.value.details["planning_radius_m"] == .30
    assert source._last_turn_result is None
    assert not any(kind == "turn.result" for kind, _ in state.events)
    assert all((c.vx, c.vy, c.wz) == (0., 0., 0.) for c in state.commands)


def test_healthy_viewing_turn_keeps_original_hold_and_result(rig):
    state, _, ctx, _, source = rig
    source._turn_hold(ctx, 0., 1., [None, None])
    assert source._last_turn_result["completed"]
    assert state.time == 1. and len(state.commands) == 250


def test_viewing_pose_persists_completed_leg_and_run(rig, recorded_explorer):
    state, _, ctx, _, source = rig
    owner, leg = recorded_explorer
    state.pose = np.array([.5, 0., 0.], dtype=np.float64)
    source._turn_hold(ctx, 0., 1., [None, None])
    assert state.time == 1. and len(state.commands) == 250

    owner._record_action("look", None, "arrived")
    owner._save()
    assert source._last_turn_result["completed"] is True
    assert type(source._last_turn_result["remaining_rad"]) is float
    assert owner.previous_action_result["details"]["completed"] is True
    saved_leg = json.loads(leg.read_text())
    saved_run = json.loads((owner.out_dir / "run.json").read_text())
    assert saved_leg["outcome"] == "arrived"
    assert saved_leg["action_result"]["details"]["completed"] is True
    assert saved_run["previous_action_result"] == saved_leg["action_result"]
    assert saved_run["vlm"]["legs"][0]["action_result"] == saved_leg["action_result"]
    assert (owner.out_dir / "live_map.npz").is_file()


@pytest.mark.parametrize("outcome", ["arrived", "refused"])
def test_action_details_are_normalized_before_state_and_persistence(recorded_explorer, outcome):
    owner, leg = recorded_explorer
    details = {"goal_xy": np.array([1., 0.]), "route_xy": np.array([[.5, 0.], [1., 0.]]),
               "completed": np.bool_(outcome == "arrived"), "attempts": np.int64(2),
               "clearance_m": np.float32(.3), "checks": {"accepted": np.array([True, False])}}
    owner.source.navigation_state._last_drive_result = details
    failure = (ActionRefused("destination blocked", code="GOAL_BLOCKED", details=details)
               if outcome == "refused" else None)
    owner._record_action("walk", (1., 0.), outcome, failure)
    result = owner.previous_action_result
    assert type(result["details"]["completed"]) is bool
    assert type(result["details"]["attempts"]) is int
    assert type(result["details"]["clearance_m"]) is float
    assert result["details"]["checks"] == {"accepted": [True, False]}
    assert result["details"]["route_xy"] == [[.5, 0.], [1., 0.]]
    assert result["details"]["clearance_m"] == pytest.approx(.3)
    assert isinstance(details["completed"], np.bool_)  # Do not mutate the controller result.
    owner._save()
    saved_leg = json.loads(leg.read_text())
    saved_run = json.loads((owner.out_dir / "run.json").read_text())
    assert saved_leg["outcome"] == outcome
    assert saved_leg["action_result"] == saved_run["previous_action_result"] == result
    assert saved_run["vlm"]["legs"][0]["action_result"] == result
    if outcome == "refused":
        assert saved_run["failed_goals"][0]["route_xy"] == [[.5, 0.], [1., 0.]]
        assert saved_leg["action_result"]["code"] == "GOAL_BLOCKED"


def test_settled_pose_waits_for_accepted_observation_instead_of_prediction(rig):
    state, _, ctx, nav, _ = rig
    ctx.observe._owns_pose_consistency = False
    state.pose = np.array([.5, 0., 0.], dtype=np.float64)
    assert nav._settled_recovery_pose(ctx) == (.5, 0., 0.)
    state.pose = np.array([1.5, 0., 0.], dtype=np.float64)
    pose = nav._settled_recovery_pose(ctx)
    assert pose == (1.5, 0., 0.)
    assert all(type(value) is float for value in pose)
    assert state.time == 1. and len(state.commands) == 250
    assert all((c.vx, c.vy, c.wz) == (0., 0., 0.) for c in state.commands)
    assert ctx.gate.waits == 1
    assert [kind for kind, _ in state.events] == ["base.localization_jump", "base.localization_settled"]


def test_settled_pose_still_refuses_expired_verification(rig):
    state, _, ctx, nav, _ = rig
    ctx.observe._owns_pose_consistency = False
    state.pose = np.array([.5, 0., 0.], dtype=np.float64)
    nav._settled_recovery_pose(ctx)
    state.pose = np.array([1.5, 0., 0.], dtype=np.float64)
    ctx.deadline = Deadline(ctx.now, .5)
    with pytest.raises(ActionRefused) as caught:
        nav._settled_recovery_pose(ctx)
    assert caught.value.code == "OBSERVATIONS_UNAVAILABLE"
    assert "deadline" in str(caught.value)
    assert not any(kind == "base.localization_settled" for kind, _ in state.events)


@pytest.mark.parametrize("change,code", [("drift", "TURN_DRIFT"), ("sweep", "LOOK_BLOCKED")])
def test_final_fresh_pose_keeps_original_turn_motion_checks(rig, change, code):
    state, art, ctx, _, source = rig
    if change == "sweep":
        # Both endpoints are clear, but a short reported jump crosses an occupied cell.
        art.occupancy[:, 50] = False
        art.occupancy[:, 60] = True  # Wall x=.5, between the execution-clear endpoints.
        state.pose = (.2, 0., 0.)
        source.turn_drift_m = 1.

    def refresh(*_, **__):
        if state.time >= 1.:
            state.pose = (1.1, 0., 0.) if change == "drift" else (.8, 0., 0.)

    source._fresh_for_motion = refresh
    with pytest.raises(ActionRefused) as caught:
        source._turn_hold(ctx, 0., 1., [None, None])
    assert caught.value.code == code
    assert source._last_turn_result is None


def test_blocked_turn_uses_existing_recovery_without_claiming_finished_view(rig):
    state, _, ctx, _, source = rig
    state.on_step = lambda: setattr(state, "pose", (.29, 0., 0.))
    recoveries = []

    def recover(active, radius, **kwargs):
        recoveries.append((active, radius, active.deadline.remaining(), kwargs))
        return {"outcome": "departure_restored", "pose": [.5, 0., 0.]}

    source.navigation._checked_escape = recover
    with pytest.raises(ActionRefused) as caught:
        source._turn_hold(ctx, 0., 1., [None, None], recover_blocked=True)
    assert caught.value.code == "LOOK_REPOSITIONED"
    assert not caught.value.details["completed"]
    assert len(recoveries) == 1 and recoveries[0][0] is ctx
    assert recoveries[0][1:] == (.30, 3.8, {"requested_radius_m": .30})
    assert source._last_turn_result is None
    assert not any(kind == "turn.result" for kind, _ in state.events)


def test_initial_survey_does_not_require_unselected_task_permission(rig):
    state, _, ctx, _, source = rig
    source.backend.explorer.live.scope_ready = False
    state.pose = (.29, 0., 0.)
    source._turn_hold(ctx, 0., 1., [None, None])
    assert source._last_turn_result["completed"]
    assert state.time == 1.
