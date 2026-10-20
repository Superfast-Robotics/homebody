from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.primitives.observations import HandState
from homebody.session.settings import Settings
from homebody.session.skill_module import Loader
from homebody.skills.contract import SkillFailure
from homebody.skills.dex1 import place
from homebody.skills.dex1.place_geometry import jaw_frame, opened, opposed, withdrawal


def jaw():
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    pose[0, 3] = .12
    return pose


def record():
    payload = np.eye(4)
    payload[:3, 3] = [.12, .01, 0.]
    return {'palm_to_payload': payload, 'payload_half': np.array([.04, .03, .06])}


def test_axial_withdrawal_preserves_rotation_and_clears_near_face():
    palm = np.eye(4)
    palm[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = withdrawal(palm, jaw(), record(), .02)
    np.testing.assert_allclose(result[:3, 3], [0, -.06, 0], atol=1e-12)
    np.testing.assert_allclose(result[:3, :3], palm[:3, :3])
    np.testing.assert_allclose(palm[:3, 3], 0.)


def test_calibration_selects_measured_closure_and_refuses_missing():
    samples = [{'closure': 0., 'palm_to_jaw': np.eye(4)},
               {'closure': .8, 'palm_to_jaw': jaw()}]
    np.testing.assert_allclose(jaw_frame(samples, .7), jaw())
    with pytest.raises(SkillFailure, match='calibrated jaw'):
        jaw_frame([], .5)


def test_two_independent_jaws_and_open_evidence():
    settings = Settings()
    hand = HandState(np.zeros(2), 0., np.zeros(3), np.zeros(3))
    assert opened(hand, settings)
    assert not opened(replace(hand, joints=np.array([0., .01])), settings)
    assert not opposed(replace(hand, finger_effort_fraction=np.array([1., 0., 1.])), settings.grasp)
    assert opposed(replace(hand, finger_effort_fraction=np.array([1., 1., 0.])), settings.grasp)
    assert not opened(replace(hand, finger_effort_fraction=np.ones(3)), settings)


def test_collision_search_continues_to_next_candidate():
    class Arm:
        def solve(self, pose, seed):
            return pose[:3, 3].copy()

    gate = SimpleNamespace(path_violation=lambda a, b: 'blocked' if b[2] > -.05 else None)
    search = place.ReleaseSearch((0.,), (0., .1), (0.,), 0., 0.)
    _, selected, _, joints = place.release_pose(
        Arm(), np.eye(4), np.eye(4), np.zeros(3), np.zeros(3), search, clearance=gate)
    assert joints is not None
    assert selected[2, 3] == pytest.approx(-.1)


def test_insertion_uses_calibrated_forward_axis():
    class Arm:
        def solve(self, pose, seed):
            return pose[:3, 3].copy()

    search = place.ReleaseSearch((0.,), (0.,), (0.,), 0., 0.)
    pre, target, _, _ = place.release_pose(
        Arm(), np.eye(4), np.eye(4), np.zeros(3), np.zeros(3), search,
        jaw=jaw(), carried=record())
    assert pre[0, 3] < target[0, 3]
    assert pre[2, 3] == pytest.approx(target[2, 3])


def test_dex1_place_loads_as_editable_copy(tmp_path):
    path = Path(place.__file__)
    loaded = Loader(tmp_path).load('place', path, 1)
    assert loaded.PROMPT == place.PROMPT
    assert loaded.FAILURES == place.FAILURES


def test_unseen_release_does_not_claim_placement(monkeypatch):
    settings = Settings()
    hand = HandState(np.zeros(2), 0., np.zeros(3), np.zeros(3))
    ctx = SimpleNamespace(settings=settings, carried={'right': record()},
                          observe=lambda epoch: SimpleNamespace(hands={'right': hand}))
    plan = place.Plan(1, 'right', 3, np.zeros(3), None, np.eye(4), np.eye(4))
    monkeypatch.setattr(place, 'complete_return', lambda *args: None)
    with pytest.raises(SkillFailure) as error:
        place.unseen_release(plan, ctx, settings.grasp.recovery_confirmations)
    assert error.value.code == 'PLACEMENT_NOT_VERIFIED'
    assert 'right' not in ctx.carried


@pytest.mark.parametrize('effort, fresh, succeeds', [(1., True, True), (0., True, False), (1., False, False)])
def test_hold_requires_positive_fresh_evidence(effort, fresh, succeeds):
    from homebody.skills.dex1.place_geometry import require_hold

    settings = Settings()
    hand = HandState(np.ones(2) * .01, .5, np.ones(3) * effort, np.ones(3) * .1)
    frame = SimpleNamespace(epoch=1, sequence=0, time=0., frame_id=(1, 0), hands={'right': hand})

    def advance(*args):
        if fresh:
            frame.frame_id = (1, frame.frame_id[1] + 1)
            frame.sequence += 1
            frame.time += .1

    ctx = SimpleNamespace(settings=settings, observe=lambda epoch: frame, advance=advance)
    if succeeds:
        assert require_hold(ctx, 1, 'right') is frame
    else:
        with pytest.raises(SkillFailure, match='sustain hold'):
            require_hold(ctx, 1, 'right')


def test_withdrawal_clears_actual_jaw_tip():
    target = withdrawal(np.eye(4), jaw(), record(), .02, np.array([[.15, 0., 0.]]))
    assert target[0, 3] == pytest.approx(-.09)


def test_opening_checks_intermediate_jaw_positions(monkeypatch):
    from homebody.skills.dex1 import place_geometry

    visited = []

    def gate(*args, fingers, **kwargs):
        visited.append(fingers.copy())
        return SimpleNamespace(violation=lambda q: 'rim' if np.allclose(fingers, .01) else None)

    monkeypatch.setattr(place_geometry, 'Clearance', gate)
    frame = SimpleNamespace(hands={'right': SimpleNamespace(joints=np.ones(2) * .02)},
                            arm_joints=lambda side: np.zeros(7))
    planning = SimpleNamespace(arms=None, settings=Settings())
    assert place_geometry.opening_clear(frame, planning, 'right', 3) == 'rim'
    assert len(visited) > 2


def test_uncertain_abort_stops_without_moving(monkeypatch):
    from homebody.motion.carry import CarryState

    stopped = []
    ctx = SimpleNamespace(observations=SimpleNamespace(epoch=1),
                          carried={'right': {'state': CarryState.UNCERTAIN}},
                          actions=SimpleNamespace(stop=stopped.append))
    plan = place.Plan(1, 'right', 3, np.zeros(3), None, np.eye(4), np.eye(4))
    monkeypatch.setattr(place, 'servo', lambda *args, **kwargs: pytest.fail('uncertain motion'))
    place.abort(plan, ctx)
    assert stopped == [1]


def return_fixture():
    class Arm:
        collision = SimpleNamespace(hand_boxes=lambda arm, q, fingers: [(q, np.eye(3), np.full(3, .02))])

        def forward(self, q):
            pose = np.eye(4)
            pose[:3, 3] = q
            return pose, None

        def solve(self, pose, seed, **kwargs):
            return pose[:3, 3].copy()

    arm = Arm()
    q = np.array([.5, 0., .2])
    frame = SimpleNamespace(world_torso=np.eye(4), hands={'right': SimpleNamespace(joints=np.zeros(2))},
                            arm_joints=lambda side: q.copy(), palm=lambda arm, side: arm.forward(q)[0])
    released = (np.array([.65, 0., .2]), .25, .2)
    return arm, q, frame, released


def test_return_envelope_allows_escape_but_blocks_reentry():
    from homebody.skills.dex1.place_geometry import return_path_check

    arm, q, frame, released = return_fixture()
    settings = Settings().collision
    assert return_path_check(frame, arm, 'right', released, settings, q, q + [0, 0, .1]) is None
    assert return_path_check(frame, arm, 'right', released, settings, q + [0, 0, .1], q)
    assert return_path_check(frame, arm, 'right', released, settings, q, q - [0, 0, .01])


def test_return_waypoint_uses_jaw_bottom_and_local_support():
    from homebody.skills.dex1.place_geometry import return_waypoints

    palm = np.eye(4)
    palm[:3, 3] = [.5, 0., .2]
    hand = np.array([[.5, 0., .15], [.5, 0., .25]])
    released = (np.array([.65, 0., .2]), .25, .2)
    observed = np.array([[.6, 0., .24], [.6, 0., 1.5], [3., 0., 1.]])
    lift, retreat = return_waypoints(palm, np.eye(4), hand, released, observed, .01)
    assert lift[2, 3] == pytest.approx(.31)
    np.testing.assert_allclose(retreat[:3, 3], [.38, 0., .31], atol=1e-12)
    np.testing.assert_allclose(lift[:3, :3], palm[:3, :3])


@pytest.mark.parametrize('mode', ['route', 'clear', 'blocked'])
def test_return_routes_before_motion_and_retains_live_checks(monkeypatch, mode):
    from homebody.skills.dex1 import place_geometry as geometry

    arm, q, frame, released = return_fixture()
    settings = Settings()
    settings = SimpleNamespace(collision=settings.collision, physics=settings.physics, servo=settings.servo,
                               motion=settings.motion, placement=SimpleNamespace(
                                   rest_arm=lambda side: np.zeros(3), rest_tolerance=.03))
    if mode == 'clear':
        released = (np.array([5., 0., .2]), .25, .2)
    events, checked, paths = [], [], []
    ctx = SimpleNamespace(observe=lambda epoch: frame, arms={'right': arm}, settings=settings,
                          emit=events.append, stage=lambda stage: None, observations=SimpleNamespace(time=0.))
    monkeypatch.setattr(geometry, 'Clearance', lambda *args, **kwargs: SimpleNamespace(
        observed_points=np.empty((0, 3)),
        path_violation=lambda a, b: 'visible rim' if mode == 'blocked' else None))
    monkeypatch.setattr(geometry, 'unloaded_check', lambda *args: lambda state, ref: checked.append(ref))

    def follow(ctx, epoch, side, rows, check):
        paths.extend(rows)
        for row in rows:
            assert check(frame, row) is None
            q[:] = row

    monkeypatch.setattr(geometry, 'follow_rows', follow)
    if mode == 'blocked':
        with pytest.raises(SkillFailure, match='visible rim'):
            geometry.return_checked(ctx, 1, 'right', released)
        assert not paths
        assert events[0]['type'] == 'mechanism_active'
        return
    geometry.return_checked(ctx, 1, 'right', released)
    if mode == 'clear':
        assert not events
        np.testing.assert_allclose(q, np.zeros(3), atol=1e-12)
        return
    assert events[0]['type'] == 'mechanism_active'
    assert events[0]['mechanism'] == 'clearance_return'
    assert len(checked) == len(paths)
    assert max(row[2] for row in paths) > .23
    np.testing.assert_allclose(q, np.zeros(3), atol=1e-12)


def test_complete_return_passes_release_envelope(monkeypatch):
    envelope = (np.ones(3), .2, .5)
    plan = place.Plan(1, 'right', 3, np.zeros(3), None, np.eye(4), np.eye(4), released=envelope)
    calls = []
    monkeypatch.setattr(place, 'return_checked', lambda *args: calls.append(args))
    assert place.complete_return(plan, object()) is None
    assert calls[0][-1] is envelope
