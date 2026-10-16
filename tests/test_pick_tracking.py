"""Bounded supported-to-occluded tracking and preservation of recovery failures."""
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.session.settings import Settings
from homebody.skills import pick
from homebody.skills.contract import SkillFailure


@pytest.fixture
def tracking(monkeypatch):
    settings = Settings()
    points = np.array([[x, y, z] for x in (-0.02, 0.02)
                       for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
    pose = np.eye(4)
    plan = SimpleNamespace(points=points, center=np.zeros(3), palm=pose,
                           label=1, side="right", epoch=1)
    events = []
    ctx = SimpleNamespace(settings=settings, arms={"right": object()}, emit=events.append)
    target = pick.PickTarget(plan, ctx)
    state = SimpleNamespace(time=1., frame_id="fresh", base_pose=np.zeros(3),
                            world_camera=pose.copy(), visible=lambda label: True,
                            target_points=lambda label: points,
                            palm=lambda arm, side: pose)
    target._see(state, pose)
    assert target.mode == "supported"
    monkeypatch.setattr(target, "_arm_hides", lambda frame, points: True)
    return target, state, pose, events


@pytest.mark.parametrize("mode", ["supported", "partial"])
def test_next_camera_packet_can_be_occluded(tracking, mode):
    target, state, palm, _ = tracking
    target.mode = mode
    target.begin_descent()
    state.time += target.ctx.settings.servo.perception_period
    state.visible = lambda label: False
    target._see(state, palm)
    target.update(state)
    assert target.mode == "occluded"
    assert target.window is not None
    assert target.last_supported_time == pytest.approx(1.)


@pytest.mark.parametrize("failure", ["old", "missing_history", "far", "rays", "corridor"])
def test_occlusion_requires_all_admission_gates(tracking, monkeypatch, failure):
    target, state, palm, events = tracking
    if failure == "old":
        state.time += (target.ctx.settings.servo.perception_period +
                       target.limits.occluded_recency + 0.01)
    elif failure == "missing_history":
        target.last_supported_time = None
    elif failure == "far":
        palm = palm.copy()
        palm[2, 3] = target.limits.occluded_reach + 0.01
    elif failure == "rays":
        monkeypatch.setattr(target, "_arm_hides", lambda frame, points: False)
    else:
        palm = palm.copy()
        palm[:3, :3] = np.diag([-1., -1., 1.])
    with pytest.raises(SkillFailure, match="tracking=") as error:
        target._admit_hidden(state, np.empty((0, 3)), palm, target.plan.palm)
    assert error.value.code == "TARGET_LOST"
    assert target.window is None
    assert events[-1]["type"] == "target_tracking_failure"
    assert events[-1]["preceding_mode"] == "supported"
    assert not events[-1]["target_visible"]


def test_hidden_packets_do_not_renew_blind_budget(tracking):
    target, state, palm, _ = tracking
    state.visible = lambda label: False
    state.time += target.ctx.settings.servo.perception_period
    target._see(state, palm)
    window = target.window
    state.time += target.ctx.settings.motion.arm_timeout + 0.01
    target._see(state, palm)
    assert target.window is window
    with pytest.raises(SkillFailure, match="motion or time bound"):
        target.update(state)


def test_hidden_motion_still_checks_travel_and_rays(tracking, monkeypatch):
    target, state, palm, _ = tracking
    state.visible = lambda label: False
    target._see(state, palm)
    palm[2, 3] += target.window.travel_limit + 0.001
    with pytest.raises(SkillFailure, match="motion or time bound"):
        target.update(state)
    monkeypatch.setattr(target, "_arm_hides", lambda frame, points: False)
    with pytest.raises(SkillFailure, match="measured arm occlusion"):
        target._see(state, palm)


def test_recovery_failure_keeps_original_tracking_reason(tracking, monkeypatch):
    target, _, _, _ = tracking
    ctx = target.ctx
    ctx.stage = lambda name: None
    monkeypatch.setattr(pick, "PickTarget", lambda plan, ctx: target)

    def approach(*args):
        raise SkillFailure("TARGET_LOST", 'tracking={"preceding_mode": "supported"}')

    def replan(*args):
        raise SkillFailure("ARM_NOT_REACHED", "The arm hiding the target did not return")

    monkeypatch.setattr(pick, "approach", approach)
    monkeypatch.setattr(pick, "replan", replan)
    with pytest.raises(SkillFailure) as error:
        pick.approach_or_replan(target.plan, ctx)
    assert error.value.code == "ARM_NOT_REACHED"
    assert "Initial approach failure [TARGET_LOST]" in str(error.value)
    assert '"preceding_mode": "supported"' in str(error.value)
    assert "recovery failure [ARM_NOT_REACHED]" in str(error.value)


@pytest.mark.parametrize("measured,explained,exposed,allowed", [
    (1., 1., 0., True), (0., 1., 0., False),
    (1., 0., 0., False), (1., 1., 1., False),
])
def test_measured_occlusion_gates_and_diagnostics(tracking, monkeypatch,
                                                measured, explained, exposed, allowed):
    target, state, _, _ = tracking
    from homebody.helpers.target import ArmOcclusion

    arm = SimpleNamespace(collision=SimpleNamespace(boxes=lambda *args: [], hand_boxes=lambda *args: []))
    target.ctx.arms["right"] = arm
    state.arm_joints = lambda side: np.zeros(7)
    state.hands = {"right": SimpleNamespace(joints=np.zeros(7))}
    monkeypatch.setattr(pick, "arm_occlusion_evidence",
                        lambda *args: ArmOcclusion(measured, explained, exposed))
    assert bool(pick.PickTarget._arm_hides(target, state, np.empty((0, 3)))) is allowed
    assert target.occlusion["measured_fraction"] == pytest.approx(measured)
    assert target.occlusion["explained_missing_fraction"] == pytest.approx(explained)
    assert target.occlusion["exposed_missing_fraction"] == pytest.approx(exposed)
