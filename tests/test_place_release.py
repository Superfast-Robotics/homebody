from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.skills import place
from homebody.toolkit.release import release_alternatives


def record():
    attachment = np.eye(4)
    attachment[:3, 3] = [.12, .04, .03]
    return {"palm_to_payload": attachment, "payload_half": np.array([.04, .02, .08]), "label": 7}


def test_alternatives_preserve_payload_xy_and_do_not_mutate():
    palm = np.eye(4)
    carried = record()
    poses = release_alternatives(palm, carried, np.empty((0, 3)), .4, .08, .01)
    assert len(poses) == 12
    for pose in poses:
        np.testing.assert_allclose((pose @ carried['palm_to_payload'])[:2, 3], [.12, .04])
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(palm, np.eye(4))


def test_visible_rim_prefers_height_clearance_but_sparse_depth_is_neutral():
    palm = np.eye(4)
    carried = record()
    points = np.array([[.12, .04, .025]])
    poses = release_alternatives(palm, carried, points, .4, .15, .01)
    assert (poses[0] @ carried['palm_to_payload'])[2, 3] == pytest.approx(.18)
    unknown = release_alternatives(palm, carried, np.empty((0, 3)), .4, .15, .01)
    assert (unknown[0] @ carried['palm_to_payload'])[2, 3] == pytest.approx(.03)


def test_alternative_precedes_pullback_and_rotates_about_payload():
    palm = np.eye(4)
    palm[0, 3] = .5
    alternative = release_alternatives(palm, record(), np.empty((0, 3)), .4, .08, .01)[0]

    class Arm:
        def solve(self, pose, seed):
            # Only a tilted wrist or a pulled-back upright wrist solves.
            return np.zeros(7) if abs(pose[2, 1]) > .1 or pose[0, 3] < .45 else None

    search = place.ReleaseSearch((0., .1), (0.,), (0.,), .02, 0.)
    _, selected, _, joints = place.release_pose(
        Arm(), np.eye(4), palm, record()['palm_to_payload'][:3, 3], np.zeros(7), search, [alternative])
    assert joints is not None
    np.testing.assert_allclose(selected, alternative)


def test_successful_direct_pose_does_not_try_alternatives():
    class Arm:
        def solve(self, pose, seed):
            return np.zeros(7)

    def unexpected():
        raise AssertionError('unnecessary fallback')
        yield

    palm = np.eye(4)
    search = place.ReleaseSearch((0., .1), (0.,), (0.,), .02, 0.)
    _, selected, _, _ = place.release_pose(
        Arm(), np.eye(4), palm, np.zeros(3), np.zeros(7), search, unexpected())
    np.testing.assert_allclose(selected, palm)


def test_retry_retains_world_target_only_for_same_metric_request(monkeypatch):
    carried = record()
    target = np.array([.6, .2, 1.])
    carried['place_target'] = (((.5, .1), 0.), target)
    planning = SimpleNamespace(carried={'right': carried}, arms={'right': None},
                               settings=SimpleNamespace(grasp=None))
    monkeypatch.setattr(place, 'destination', lambda *args: (None, np.array([.5, .1]), 0.))
    monkeypatch.setattr(place, 'continuing', lambda *args: True)
    seen = []

    def plan(*args, target_world=None):
        seen.append(target_world)
        return place.Plan(1, 'right', 7, target, None, np.eye(4), np.eye(4))

    monkeypatch.setattr(place, 'plan_release', plan)
    place.prepare({'hand': 'right'}, None, None, planning)
    np.testing.assert_allclose(seen[-1], target)
    carried['place_target'] = (((.4, .1), 0.), target)
    place.prepare({'hand': 'right'}, None, None, planning)
    assert seen[-1] is None


def test_verification_reference_does_not_follow_selected_pose():
    target = np.array([.6, .2, 1.])
    plan = place.Plan(1, 'right', 7, target, None, np.eye(4), np.eye(4), False)
    plan = replace(plan, short_m=.2, measured=np.array([.4, .2, 1.]))
    ctx = SimpleNamespace(settings=SimpleNamespace(placement=SimpleNamespace(
        xy_tolerance=.1, support_height_tolerance=.05)))
    settling = place.Settling(plan, ctx, np.zeros(3))
    shape = SimpleNamespace(center=np.array([.4, .2, .5]), lower=np.array([.4, .2, .4]))
    assert not settling.near(shape)
    np.testing.assert_allclose(plan.point, target)


def test_plan_records_original_goal_and_search_displacement(monkeypatch):
    from homebody.session.settings import Settings

    carried = record()
    current = SimpleNamespace(epoch=1, world_torso=np.eye(4),
                              palm=lambda *args: np.eye(4), arm_joints=lambda *args: np.zeros(7))
    planning = SimpleNamespace(carried={'right': carried}, arms={'right': None}, settings=Settings())
    monkeypatch.setattr(place, 'Clearance', lambda *args, **kwargs: SimpleNamespace(
        observed_points=np.empty((0, 3)), path_violation=lambda *args: None))

    def shifted(arm, torso, palm, offset, seed, search, alternatives):
        candidate = palm.copy()
        candidate[0, 3] -= .12
        return candidate, candidate, np.zeros(7), np.zeros(7)

    monkeypatch.setattr(place, 'release_pose', shifted)
    plan = place.plan_release('right', None, np.array([.5, .1]), 0., current, current, planning)
    np.testing.assert_allclose(plan.point, [.5, .1, 0.])
    np.testing.assert_allclose(plan.selected_point, [.38, .1, 0.], atol=1e-12)
    assert plan.short_m == pytest.approx(.12)


def test_measured_misalignment_refuses_before_opening_and_saves_target(monkeypatch):
    from homebody.session.settings import Settings
    from homebody.skills.contract import SkillFailure

    carried = record()
    palm = np.eye(4)
    target = np.array([.5, .1, 1.])
    plan = place.Plan(1, 'right', 7, target, np.zeros(7), palm, palm,
                      target_key=((.5, .1), 0.))
    measured_palm = palm.copy()
    measured_palm[0, 3] += 1.
    ctx = SimpleNamespace(carried={'right': carried}, settings=Settings(), arms={'right': None},
                          stage=lambda *args: None, observe=lambda *args: SimpleNamespace(
                              palm=lambda *args: measured_palm))
    monkeypatch.setattr(place, 'move_joints', lambda *args: None)
    monkeypatch.setattr(place, 'servo', lambda *args, **kwargs: None)
    monkeypatch.setattr(place, 'attached_center', lambda *args: carried['palm_to_payload'][:3, 3])
    monkeypatch.setattr(place, 'close_or_open', lambda *args: pytest.fail('opened before alignment check'))
    with pytest.raises(SkillFailure, match='Payload XY'):
        place.execute(plan, ctx)
    np.testing.assert_allclose(carried['place_target'][1], target)
