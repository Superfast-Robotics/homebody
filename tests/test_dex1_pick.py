"""Dex1 grasp geometry and sensor decisions without a simulator."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.helpers.geometry import transform
from homebody.primitives.observations import HandState
from homebody.session.settings import Settings
from homebody.skills.contract import SkillFailure
from homebody.skills.dex1 import pick


def calibration():
    jaw = transform([.117, 0., 0.], np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]))
    return [{'aperture': width, 'palm_to_jaw': jaw} for width in np.linspace(.006, .095, 18)]


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
        assert any(np.allclose(expected, candidate.palm, atol=1e-9) for candidate in transformed)


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





def test_low_target_stance_allows_raised_oblique_wrist():
    settings = Settings()
    shape = pick.Shape(np.array([.4, 0., .60]), np.array([.37, -.02, .56]), np.array([.43, .02, .64]))
    current = SimpleNamespace(world_torso=np.eye(4))
    planning = SimpleNamespace(settings=settings, arms={'right': SimpleNamespace(shoulder=np.array([0., 0., 1.]))},
                               jaw_samples={'right': calibration()})
    distance = pick.grasp_distance('right', shape, current, planning)
    assert .117 < distance < settings.grasp.grasp_range + .117 + 1e-9
