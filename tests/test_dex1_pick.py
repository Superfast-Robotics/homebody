"""Dex1 grasp geometry and sensor decisions without a simulator."""
import inspect
from collections import Counter
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.helpers.geometry import transform
from homebody.primitives.observations import HandState
from homebody.session.settings import Settings
from homebody.session.skill_module import Loader
from homebody.skills.contract import SkillFailure
from homebody.skills.dex1 import pick


def calibration():
    jaw = transform([.117, 0., 0.], np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]))
    return [{'closure': float(closure), 'aperture': width, 'palm_to_jaw': jaw}
            for closure, width in zip(np.linspace(1., 0., 18), np.linspace(.006, .095, 18))]


def cloud(extents=(.06, .04, .08)):
    return np.array(list(product(*[np.linspace(-v / 2, v / 2, 7) for v in extents])))


def candidates(points, forward=None, cancelled=lambda: False):
    return list(pick._width_fitted_candidates(points, calibration(), Settings().grasp, cancelled, forward))


def test_forward_pads_and_pregrasp_follow_calibration():
    options = candidates(cloud())
    assert options
    for option in options:
        jaw = option.palm @ calibration()[0]['palm_to_jaw']
        direction = jaw[:3, 2]
        np.testing.assert_allclose(option.palm[:3, 0], direction, atol=1e-10)
        np.testing.assert_allclose(option.pre[:3, 3] - option.palm[:3, 3], -.1 * direction, atol=1e-10)
        assert direction[0] >= .25 - 1e-9
        assert direction[2] >= -np.sqrt(3) / 2 - 1e-9
        assert np.linalg.det(option.palm[:3, :3]) == pytest.approx(1.)
        assert .006 < option.width <= .087 + 1e-9


def test_thin_wide_target_uses_vertical_closure_at_front_edge():
    points = cloud((.2, .2, .02))
    options = candidates(points)
    assert options
    level = [o for o in options if abs(o.palm[2, 0]) < 1e-9]
    assert level
    for option in level:
        jaw = option.palm @ calibration()[0]['palm_to_jaw']
        assert abs(jaw[2, 0]) == pytest.approx(1.)
        depth = points @ jaw[:3, 2]
        near = np.quantile(depth, Settings().grasp.shape_quantiles[0])
        assert near < jaw[:3, 3] @ jaw[:3, 2] <= near + .055 + 1e-9


def test_no_aperture_padding_or_too_thin_grasp_and_cancellation():
    assert not candidates(cloud((.1, .1, .1)))
    assert not candidates(cloud((.001, .001, .001)))
    with pytest.raises(SkillFailure, match='cancelled'):
        candidates(cloud(), cancelled=lambda: True)


def test_world_translation_and_heading_rotate_the_grasp():
    points = cloud()
    turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    offset = np.array([1.2, -.4, .8])
    transformed = candidates(points @ turn.T + offset, [0., 1., 0.])
    for option in candidates(points):
        expected = transform(offset, turn) @ option.palm
        assert any(np.allclose(expected, candidate.palm, atol=1e-9) and
                   candidate.contact_overlap == pytest.approx(option.contact_overlap, abs=1e-12)
                   for candidate in transformed)


def test_overlap_requires_both_patches_and_clips_to_pad_neighbourhood():
    settings = SimpleNamespace(shape_quantiles=(0., 1.), pad_band=.02, pad_band_points=4)
    patch = np.array(list(product([-.04, .04], [-.04, .04])))
    left = np.column_stack((np.full(4, -.03), patch))
    right = np.column_stack((np.full(4, .03), patch))
    assert pick.contact_overlap(np.vstack((left, right)), settings) == pytest.approx(.04 ** 2)
    # A large near-side patch cannot compensate for disjoint opposing depth.
    right[:, 2] += .1
    assert pick.contact_overlap(np.vstack((left, right)), settings) == pytest.approx(0.)
    right[:, 2] -= .1
    assert pick.contact_overlap(np.vstack((left, right[:3])), settings) == pytest.approx(0.)
    assert pick.contact_overlap(np.vstack((left, right)) + [0., .1, 0.], settings) == pytest.approx(0.)


def test_candidates_measure_overlap_at_the_calibrated_pad_origin():
    points = cloud()
    options = candidates(points)
    assert any(option.contact_overlap > 0 for option in options)
    for option in options:
        jaw = option.palm @ calibration()[0]['palm_to_jaw']
        local = (points - jaw[:3, 3]) @ jaw[:3, :3]
        assert option.contact_overlap == pytest.approx(pick.contact_overlap(local, Settings().grasp), abs=1e-12)


@pytest.mark.parametrize('case', ['overlap', 'clearance', 'tie', 'descent_blocked', 'route_blocked'])
def test_overlap_ranking_keeps_clearance_gates_and_original_fallbacks(monkeypatch, case):
    subject = pick.GraspSearch.__new__(pick.GraspSearch)
    subject.settings = Settings().grasp
    subject.seed = np.zeros(2)
    subject.arm = subject.approach = object()
    subject.cancelled = lambda: False
    subject.blocked = []
    subject.blockers = Counter()
    subject.counts = {'approach_clear': 0}
    shape = pick.visible_shape(cloud(), subject.settings.shape_quantiles)
    original = pick.GraspCandidate(0, .02, np.eye(4), np.eye(4), shape, .0001)
    seated = replace(original, tier=1, width=.03, palm=np.eye(4), contact_overlap=.0004)
    if case == 'tie':
        seated = replace(seated, contact_overlap=original.contact_overlap)
    first = pick.ReachableGrasp(original, np.zeros(2), .01 if case == 'clearance' else 0.)
    second = pick.ReachableGrasp(seated, np.ones(2))
    monkeypatch.setattr(subject, 'servo_descent',
                        lambda joints, palm: None if case == 'descent_blocked' and palm is seated.palm else 0.)
    monkeypatch.setattr(pick, 'approach_path',
                        lambda arm, seed, joints, *args, **kwargs:
                        None if case == 'route_blocked' and joints is second.pre_joints else (joints,))
    selected, _ = subject.best([first, second])
    assert selected is (seated if case == 'overlap' else original)


@pytest.mark.parametrize('height', [.046, .059, .061, .12])
@pytest.mark.parametrize('case', ['pitch', 'same_tier', 'clearance', 'descent_blocked', 'route_blocked'])
def test_short_target_prefers_reachable_insertion_before_overlap(monkeypatch, height, case):
    subject = pick.GraspSearch.__new__(pick.GraspSearch)
    subject.settings = Settings().grasp
    subject.seed = np.zeros(2)
    subject.arm = subject.approach = object()
    subject.cancelled = lambda: False
    subject.blocked = []
    subject.blockers = Counter()
    subject.counts = {'approach_clear': 0}
    # Real generated poses: 30-degree full-depth insertion versus the 60-degree
    # tier selected in the jar recordings. Translate to shelf height as in use.
    options = candidates(cloud((.048, .048, height)) + [1., -.2, .76])
    shallow = replace(next(o for o in options if o.tier == 1), contact_overlap=.0001)
    steep = replace(next(o for o in options if o.tier == 2), contact_overlap=.0004)
    if case == 'same_tier':
        steep = replace(steep, tier=shallow.tier)
    first = pick.ReachableGrasp(shallow, np.zeros(2))
    second = pick.ReachableGrasp(steep, np.ones(2), .01 if case == 'clearance' else 0.)
    monkeypatch.setattr(subject, 'servo_descent',
                        lambda joints, palm: None if case == 'descent_blocked' and palm is shallow.palm else 0.)
    monkeypatch.setattr(pick, 'approach_path',
                        lambda arm, seed, joints, *args, **kwargs:
                        None if case == 'route_blocked' and joints is first.pre_joints else (joints,))
    selected, _ = subject.best([second, first])
    assert selected is (shallow if height < .06 and case == 'pitch' else steep)


def hand(efforts):
    return HandState(np.array([-.02, -.02]), .45, np.array(efforts), np.full(3, .5))


@pytest.mark.parametrize('efforts, held', [([.4, .4, .4], True), ([.4, 0, 0], False),
                                          ([0, .4, .4], False), ([.4, 0, .4], False)])
def test_two_physical_jaws_are_required(efforts, held):
    assert pick.opposed_jaws(hand(efforts), Settings().grasp) is held


def test_wide_object_still_gets_full_squeeze(monkeypatch):
    calls = []
    monkeypatch.setattr(pick, 'close_or_open', lambda *args: calls.append(args) or 'frame')
    ctx = SimpleNamespace(settings=Settings(), stage=lambda _: None)
    assert pick.squeeze(SimpleNamespace(epoch=1, side='left'), ctx,
                        SimpleNamespace(hands={'left': hand([.4, .4, .4])})) == 'frame'
    assert calls[0][3] == pytest.approx(1.12)


@pytest.mark.parametrize('fresh, efforts, passes', [(True, [.4, .4, .4], True),
                                                   (False, [.4, .4, .4], False),
                                                   (True, [.4, 0, 0], False)])
def test_contact_confirmation_requires_fresh_bilateral_evidence(fresh, efforts, passes):
    frames = [SimpleNamespace(epoch=1, sequence=i if fresh else 0, time=i * .1 if fresh else 0.,
                              hands={'right': hand(efforts)}) for i in range(5)]
    iterator = iter(frames[1:])
    ctx = SimpleNamespace(settings=Settings(), advance=lambda *args: None, observe=lambda _: next(iterator))
    plan = SimpleNamespace(epoch=1, side='right')
    if passes:
        pick.require_contact(plan, ctx, frames[0])
    else:
        with pytest.raises(SkillFailure) as error:
            pick.require_contact(plan, ctx, frames[0])
        assert error.value.code == 'LIFT_NOT_VERIFIED'


def test_partial_finish_corridor_follows_insertion_not_world_vertical():
    settings = Settings()
    tracker = pick.PickTarget.__new__(pick.PickTarget)
    tracker.ctx = SimpleNamespace(settings=settings)
    tracker.limits = settings.blind_finish
    goal = np.eye(4)
    pre = goal.copy()
    pre[0, 3] = -.1
    tracker.plan = SimpleNamespace(pre=pre, palm=goal)
    tracker._require_corridor(pre, goal)
    off = goal.copy()
    off[2, 3] = .08
    with pytest.raises(SkillFailure):
        tracker._require_corridor(off, goal)
    off = goal.copy()
    off[0, 3] = .08
    with pytest.raises(SkillFailure):
        tracker._require_corridor(off, goal)


def test_lift_uses_collision_checked_payload_preserving_target(monkeypatch):
    target = np.eye(4)
    calls = []
    monkeypatch.setattr(pick, 'servo_best_effort', lambda *args, **kwargs: calls.append((args, kwargs)))
    ctx = SimpleNamespace(carried={'right': {'label': 4, 'lift_target': target}})
    pick.lift_pending(ctx, 1, 'right')
    assert calls[0][0][3] is target
    assert calls[0][1] == {'exclude_labels': (4,)}


def test_dex1_pick_loads_as_editable_copy(tmp_path):
    path = tmp_path / 'pick.py'
    path.write_text(inspect.getsource(pick))
    module = Loader(tmp_path / 'lib').load('pick', path, 1)
    assert module.PROMPT == pick.PROMPT
    assert module.FAILURES == pick.FAILURES
    assert list(module._width_fitted_candidates(cloud(), calibration(), Settings().grasp, lambda: False))


def test_low_target_stance_allows_raised_oblique_wrist():
    settings = Settings()
    shape = pick.Shape(np.array([.4, 0., .60]), np.array([.37, -.02, .56]), np.array([.43, .02, .64]))
    current = SimpleNamespace(world_torso=np.eye(4))
    planning = SimpleNamespace(settings=settings, arms={'right': SimpleNamespace(shoulder=np.array([0., 0., 1.]))},
                               jaw_samples={'right': calibration()})
    distance = pick.grasp_distance('right', shape, current, planning)
    assert .117 < distance < settings.grasp.grasp_range + .117 + 1e-9


@pytest.mark.parametrize('lag, descending, retries', [(.019, True, 1), (.35, True, 0),
                                                     (.019, False, 0), (np.nan, True, 0)])
def test_insertion_retry_is_low_lag_descent_only_and_bounded(monkeypatch, lag, descending, retries):
    tracker = SimpleNamespace(descending=descending)
    monkeypatch.setattr(pick, 'PickTarget', lambda *args: tracker)
    calls = []

    def failed(*args):
        raise SkillFailure('ARM_NOT_REACHED', 'timeout', {'joint_tracking_error_rad': lag})

    monkeypatch.setattr(pick, 'approach', failed)
    monkeypatch.setattr(pick, 'replan', lambda plan, *args: (calls.append(plan) or plan, tracker))
    ctx = SimpleNamespace(settings=Settings(), stage=lambda _: None)
    with pytest.raises(SkillFailure, match='timeout'):
        pick.approach_or_replan(object(), ctx)
    assert len(calls) == retries


def test_insertion_budget_survives_collision_retry_and_checks_enclosure(monkeypatch):
    tracker = SimpleNamespace(descending=True)
    monkeypatch.setattr(pick, 'PickTarget', lambda *args: tracker)
    errors = iter([SkillFailure('COLLISION', 'blocked'),
                   SkillFailure('ARM_NOT_REACHED', 'timeout', {'joint_tracking_error_rad': .02}), None])

    def approach(*args):
        error = next(errors)
        if error:
            raise error
        return 'frame'

    calls = []
    monkeypatch.setattr(pick, 'approach', approach)
    monkeypatch.setattr(pick, 'replan', lambda plan, *args: (plan, tracker))
    monkeypatch.setattr(pick, 'require_jaw_enclosure', lambda *args: calls.append(args))
    ctx = SimpleNamespace(settings=Settings(), stage=lambda _: None)
    assert pick.approach_or_replan(object(), ctx)[2] == 'frame'
    assert len(calls) == 1


@pytest.mark.parametrize('offset, passes', [(0., True), (.107, False)])
def test_recovered_insertion_requires_measured_enclosure(offset, passes):
    samples = calibration()
    palm = np.eye(4)
    palm[0, 3] = offset
    points = cloud() + np.array([.117, 0, 0])
    frame = SimpleNamespace(palm=lambda *args: palm, hands={'right': SimpleNamespace(closure=0.)})
    ctx = SimpleNamespace(settings=Settings(), arms={'right': object()},
                          planning=lambda: SimpleNamespace(jaw_samples={'right': samples}))
    target = SimpleNamespace(update=lambda _: None, track=SimpleNamespace(last_supported_points=points))
    plan = SimpleNamespace(side='right')
    if passes:
        pick.require_jaw_enclosure(plan, ctx, target, frame)
    else:
        with pytest.raises(SkillFailure, match='did not enclose'):
            pick.require_jaw_enclosure(plan, ctx, target, frame)


def test_timeout_diagnostics_are_forwarded_without_changing_context():
    events = []
    ctx = SimpleNamespace(emit=events.append, settings=Settings())
    monitor = pick.InsertionContext(ctx)
    event = {'type': 'arm_servo_timeout', 'joint_tracking_error_rad': .019}
    monitor.emit(event)
    assert monitor.timeout['joint_tracking_error_rad'] == pytest.approx(.019)
    assert events == [event]
    assert monitor.settings is ctx.settings


def test_timeout_replan_excludes_failed_insertion_and_uses_fresh_frame(monkeypatch):
    frames = iter([SimpleNamespace(epoch=1, sequence=i, time=float(i), frame_id=(1, i)) for i in (1, 2)])
    fresh = []
    target = SimpleNamespace(descending=True, blocked=[], supported=lambda _: True, rebased=lambda p: target)
    plan = SimpleNamespace(epoch=1, side='right', label=4)
    ctx = SimpleNamespace(actions=SimpleNamespace(stop=lambda _: None), stage=lambda _: None,
                          observe=lambda _: next(frames), advance=lambda _: None, emit=lambda _: None,
                          planning=lambda: 'planning')

    def plan_target(side, label, frame, planning, blocked):
        fresh.append(frame.sequence)
        assert blocked == [(plan, 'descent')]
        assert planning == 'planning'
        return plan

    monkeypatch.setattr(pick, 'plan_target', plan_target)
    pick.replan(plan, ctx, target, SkillFailure('ARM_NOT_REACHED', 'timeout'), 2)
    assert fresh == [2]


def test_failed_insertion_is_excluded_before_ik():
    from collections import Counter

    search = pick.GraspSearch.__new__(pick.GraspSearch)
    search.control = Settings()
    search.cancelled = lambda: False
    search.counts = dict.fromkeys(search.GATES, 0)
    search.blockers = Counter()
    failed = SimpleNamespace(pre=np.eye(4), palm=np.eye(4))
    search.blocked = [(failed, 'descent')]
    search.arm = SimpleNamespace(solve=lambda *args: pytest.fail('Failed insertion reached IK'))
    assert list(search.reachable([failed])) == []
    assert search.blockers['equivalent blocked grasp'] == 1


def test_lift_baseline_uses_same_quantile_as_recovery():
    # Raw max at 1.2 m used to hide a real 17 cm lift of this 4 cm jar.
    from homebody.motion.carry import evidence

    settings = Settings()
    points = cloud((.04, .04, .04)) + [0., 0., .75]
    points = np.vstack((points, [0., 0., 1.2]))
    palm = np.eye(4)
    hand = HandState(np.ones(2) * -.02, .5, np.ones(3) * .5, np.ones(3) * .5)
    frame = SimpleNamespace(epoch=1, palm=lambda *args: palm, target_points=lambda _: points,
                            visible=lambda _: True, hands={'right': hand})
    ctx = SimpleNamespace(settings=settings, arms={'right': object()}, carried={})
    pick.begin_pending(ctx, frame, 'right', 7, np.eye(4))
    record = ctx.carried['right']
    assert record['baseline_upper'] < .8
    points = points + [0., 0., .17]
    palm[2, 3] = .17
    assert evidence(frame, ctx.arms['right'], 'right', record, settings.grasp) == 'held'
    palm[2, 3] = 0.
    assert evidence(frame, ctx.arms['right'], 'right', record, settings.grasp) != 'held'


@pytest.mark.parametrize('across, passes', [(.045, True), (.1, False)])
def test_enclosure_allows_pad_overlap_but_not_a_missed_target(across, passes):
    points = cloud() + [.117, 0., across]
    frame = SimpleNamespace(palm=lambda *args: np.eye(4), hands={'right': SimpleNamespace(closure=0.)})
    ctx = SimpleNamespace(settings=Settings(), arms={'right': object()},
                          planning=lambda: SimpleNamespace(jaw_samples={'right': calibration()}))
    target = SimpleNamespace(update=lambda _: None, track=SimpleNamespace(last_supported_points=points))
    if passes:
        pick.require_jaw_enclosure(SimpleNamespace(side='right'), ctx, target, frame)
    else:
        with pytest.raises(SkillFailure, match='did not enclose'):
            pick.require_jaw_enclosure(SimpleNamespace(side='right'), ctx, target, frame)


def test_recheck_defers_compaction_and_keeps_verified_ownership(monkeypatch):
    from homebody.skills.dex1.transport import CarryState

    ctx = SimpleNamespace(carried={'right': {'label': 7, 'state': CarryState.VERIFIED}}, stage=lambda _: None)
    monkeypatch.setattr(pick, 'recover', lambda *args: ('verified', None))
    result = pick.recheck(pick.RecoveryPlan(1, 'right', 7), ctx)
    assert result.code == 'OK'
    assert ctx.carried['right']['compact_pending']
    assert ctx.carried['right']['state'] == CarryState.VERIFIED


@pytest.mark.parametrize('status', ['verified', 'uncertain'])
def test_pick_defers_only_after_verified_lift(monkeypatch, status):
    frame = SimpleNamespace(target_points=lambda _: cloud(), palm=lambda *args: np.eye(4))
    plan = SimpleNamespace(epoch=1, side='right', label=7)
    target = SimpleNamespace(track=SimpleNamespace(last_supported_points=cloud()))
    ctx = SimpleNamespace(stage=lambda _: None, settings=Settings(), arms={'right': object()},
                          carried={'right': {'baseline_upper': 0., 'baseline_palm_height': 0.}})
    monkeypatch.setattr(pick, 'close_or_open', lambda *args: frame)
    monkeypatch.setattr(pick, 'approach_or_replan', lambda *args: (plan, target, frame))
    monkeypatch.setattr(pick, 'begin_pending', lambda *args, **kwargs: None)
    monkeypatch.setattr(pick, 'squeeze', lambda *args: frame)
    monkeypatch.setattr(pick, 'require_contact', lambda *args: None)
    monkeypatch.setattr(pick, 'lift_pending', lambda *args: frame)
    monkeypatch.setattr(pick, 'recover', lambda *args: (status, frame))
    if status == 'verified':
        assert pick.grasp(plan, ctx).code == 'OK'
        assert ctx.carried['right']['compact_pending']
    else:
        with pytest.raises(SkillFailure, match='did not verify lift'):
            pick.grasp(plan, ctx)
        assert 'compact_pending' not in ctx.carried['right']
