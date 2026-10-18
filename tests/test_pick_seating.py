"""Pick seating must add command preload and verify transport, not just effort."""
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.motion.carry import CarryState, Evidence
from homebody.session.settings import Settings
from homebody.skills import pick
from homebody.skills.contract import SkillFailure


class View:
    def __init__(self, shift=0., effort=0.2):
        self.points = np.array([[0., 0., 0.], [0., 0.01, 0.]]) + [shift, 0., 0.]
        self.hands = {'right': SimpleNamespace(closure=0.4,
                                              finger_effort_fraction=np.full(3, effort))}
        self.world_torso = np.eye(4)

    def palm(self, arm, side):
        return np.eye(4)

    def target_points(self, label):
        return self.points

    def arm_joints(self, side):
        return np.zeros(7)


def setup():
    plan = SimpleNamespace(epoch=1, side='right', label=2)
    arm = SimpleNamespace(lower=np.full(7, -1.), upper=np.ones(7),
                          solve=lambda pose, seed: np.ones(7) * 0.1)
    ctx = SimpleNamespace(settings=Settings(), arms={'right': arm},
                          carried={'right': {'state': CarryState.VERIFIED}},
                          stage=lambda text: None, emit=lambda event: None)
    return plan, ctx


def test_squeeze_increments_command_even_when_measured_closure_is_small(monkeypatch):
    plan, ctx = setup()
    frame, commands = View(), []
    monkeypatch.setattr(pick, 'evidence', lambda *args: Evidence.ENCLOSED)
    monkeypatch.setattr(pick, 'close_or_open', lambda ctx, epoch, side, closure, seconds:
                        commands.append(closure) or frame)
    assert pick.squeeze(plan, ctx, frame) is frame
    assert commands == pytest.approx([1.06, 1.12])


@pytest.mark.parametrize('initial', [Evidence.UNKNOWN, Evidence.LOST, Evidence.ENCLOSED])
def test_squeeze_stops_when_contact_evidence_disappears(monkeypatch, initial):
    plan, ctx = setup()
    states, commands = iter([initial, Evidence.LOST]), []
    monkeypatch.setattr(pick, 'evidence', lambda *args: next(states))
    monkeypatch.setattr(pick, 'close_or_open', lambda *args: commands.append(args[3]) or View())
    pick.squeeze(plan, ctx, View())
    assert commands == pytest.approx([1.06] if initial == Evidence.ENCLOSED else [])


@pytest.mark.parametrize('shift,effort,held', [(0.03, 0.2, Evidence.HELD),
                                             (0., 0., Evidence.HELD),
                                             (0., 0.2, Evidence.UNKNOWN),
                                             (0., 0.2, Evidence.LOST)])
def test_transport_rejects_slip_missing_view_or_effort(monkeypatch, shift, effort, held):
    plan, ctx = setup()
    monkeypatch.setattr(pick, 'Frame', View)
    monkeypatch.setattr(pick, 'evidence', lambda *args: held)
    check = pick.TransportCheck(plan, ctx, View())
    with pytest.raises(SkillFailure, match='Carry motion lost'):
        check.update(View(shift, effort))
    assert ctx.carried['right']['state'] == CarryState.UNCERTAIN


@pytest.mark.parametrize('blocked', [False, True])
def test_transport_requires_clear_motion_and_checks_both_legs(monkeypatch, blocked):
    plan, ctx = setup()
    motions, recoveries = [], []
    monkeypatch.setattr(pick, 'Frame', View)
    monkeypatch.setattr(pick, 'evidence', lambda *args: Evidence.HELD)
    monkeypatch.setattr(pick, 'Clearance', lambda *args, **kw: SimpleNamespace(
        path_violation=lambda *args: 'obstacle' if blocked else None))
    monkeypatch.setattr(pick, 'payload_points', lambda *args: np.zeros((2, 3)))
    monkeypatch.setattr(pick, 'in_view', lambda *args: True)

    def servo(ctx, epoch, side, goal, *, tracked_target, exclude_labels):
        assert exclude_labels == (plan.label,)
        tracked_target.update(View())
        motions.append(goal.copy())
        return View()

    def recover(*args):
        recoveries.append(True)
        ctx.carried['right']['state'] = CarryState.VERIFIED
        return 'verified', View()

    monkeypatch.setattr(pick, 'servo', servo)
    monkeypatch.setattr(pick, 'recover', recover)
    if blocked:
        with pytest.raises(SkillFailure, match='No clear'):
            pick.check_transport(plan, ctx, View())
        assert not motions and not recoveries
    else:
        pick.check_transport(plan, ctx, View())
        assert len(motions) == len(recoveries) == 2
        assert motions[0][0, 3] == pytest.approx(-0.04)
        np.testing.assert_allclose(motions[1], np.eye(4), atol=1e-9)
