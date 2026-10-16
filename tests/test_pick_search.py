"""Planning budgets and exact gates must survive current-stance probing."""
from collections import Counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from homebody.skills import pick
from homebody.skills.contract import SkillFailure
from homebody.toolkit.search import SearchBudget


def test_budget_reserves_time_and_bounds_children():
    now = [10.]
    budget = SearchBudget(20., reserve=5., clock=lambda: now[0])
    child = budget.child(2.)
    now[0] += 2.1
    assert child() and not budget()
    assert child.report()['elapsed_seconds'] == pytest.approx(2.1)
    last = budget.child(100.)
    now[0] = 25.
    assert budget() and last()
    assert SearchBudget(1., reserve=2.)()


def test_budget_propagates_cancellation():
    cancelled = [False]
    budget = SearchBudget(100., cancelled=lambda: cancelled[0])
    child = budget.child(10.)
    cancelled[0] = True
    assert child() and budget.report()['cancelled']


def search():
    result = object.__new__(pick.GraspSearch)
    result.seed = np.zeros(1)
    result.torso = np.eye(4)
    result.arm = NS(shoulder=np.zeros(3), reach=1.)
    result.settings = NS(approach_start_lifts_m=())
    result.approach = object()
    result.counts = dict.fromkeys(result.GATES, 0)
    result.blockers = Counter()
    result.endpoint_failures = {}
    result.exhausted = False
    result.routes_tried = result.routes_incomplete = 0
    result.budget = SearchBudget(100.)
    result.cancelled = result.budget
    return result


def test_routes_before_exhausting_candidates(monkeypatch):
    probe = search()
    consumed = []

    def candidates():
        for i in range(100):
            consumed.append(i)
            yield NS(candidate=NS(tier=0, width=.05), pre_joints=np.zeros(1))

    monkeypatch.setattr(pick, 'approach_path', lambda *a, **kw: (np.zeros(1),))
    assert probe.best(candidates()) is not None
    assert len(consumed) == 1
    assert probe.routes_tried == 1
    assert probe.report()['alternatives_unexplored']


def test_route_timeout_leaves_other_candidates_available(monkeypatch):
    probe = search()
    now = [0.]
    probe.budget = SearchBudget(10., clock=lambda: now[0])
    probe.cancelled = probe.budget

    def route(*args, **kwargs):
        if probe.routes_tried == 1:
            now[0] += 3.1
            assert args[5]()
            return None
        return (np.zeros(1),)

    monkeypatch.setattr(pick, 'approach_path', route)
    candidates = [NS(candidate=NS(tier=0, width=.05), pre_joints=np.zeros(1)) for _ in range(2)]
    assert probe.best(candidates) is not None
    assert probe.routes_incomplete == 1
    assert not probe.report()['search_complete']


@pytest.mark.parametrize('failed_gate', ['pre', 'closing', 'descent', None])
def test_exact_endpoint_and_descent_gates(failed_gate):
    probe = search()
    calls = []

    def solve(pose, seed):
        calls.append(pose)
        if failed_gate == ('pre' if len(calls) == 1 else 'closing'):
            return None
        return np.zeros(1)

    probe.arm.solve = solve
    probe.descent = NS(path_violation=lambda *a: 'blocked' if failed_gate == 'descent' else None)
    pose = np.eye(4)
    pose[0, 3] = 1.2
    found = list(probe.reachable([NS(pre=pose, palm=pose)]))
    assert bool(found) == (failed_gate is None)
    if failed_gate in ('pre', 'closing'):
        assert probe.endpoint_failures[failed_gate]['reach_deficit_m'] == pytest.approx(.2)
    if failed_gate == 'descent':
        assert probe.blockers['blocked'] == 1
    assert probe.report()['search_complete']


def test_prepare_prefers_validated_grasp_to_stance(monkeypatch):
    monkeypatch.setattr(pick, 'click_argument', lambda *a: (1, None))
    monkeypatch.setattr(pick, 'plan_target', lambda *a, **kw: 'validated')
    monkeypatch.setattr(pick, 'approach_stance', lambda *a: pytest.fail('unnecessary base adjustment'))
    planning = NS(carried={}, cancelled=lambda: False)
    assert pick.prepare({}, None, None, planning) == 'validated'


def test_heuristic_refusal_keeps_probe_evidence(monkeypatch):
    monkeypatch.setattr(pick, 'click_argument', lambda *a: (1, None))

    def probe(*a, **kw):
        raise SkillFailure('NO_GRASP', 'closing IK failed; alternatives unexplored')

    def stance(*a):
        raise SkillFailure('UNREACHABLE', 'preferred range')

    monkeypatch.setattr(pick, 'plan_target', probe)
    monkeypatch.setattr(pick, 'approach_stance', stance)
    with pytest.raises(SkillFailure, match='closing IK failed; alternatives unexplored'):
        pick.prepare({}, None, None, NS(carried={}, cancelled=lambda: False))


def test_expired_search_does_not_claim_endpoint_impossibility():
    probe = search()
    probe.budget = SearchBudget(0.)
    probe.cancelled = probe.budget
    probe.arm.solve = lambda *a: pytest.fail('IK after planning deadline')
    assert list(probe.reachable([NS(pre=np.eye(4), palm=np.eye(4))])) == []
    report = probe.report()
    assert report['budget_exhausted'] and report['alternatives_unexplored']
    assert not report['search_complete']
    assert report['gates']['width'] == 0


def test_failed_probe_can_still_choose_clear_stance(monkeypatch):
    monkeypatch.setattr(pick, 'click_argument', lambda *a: (1, None))

    def probe(*a, **kw):
        raise SkillFailure('NO_GRASP', 'route blocked')

    monkeypatch.setattr(pick, 'plan_target', probe)
    monkeypatch.setattr(pick, 'approach_stance', lambda *a: 'clear stance')
    assert pick.prepare({}, None, None, NS(carried={}, cancelled=lambda: False)) == 'clear stance'
