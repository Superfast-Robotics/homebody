"""Semantic actions use the same terminal-event lifecycle without introducing motion."""
import json
import threading
from types import SimpleNamespace

import pytest

from homebody.real2sim.exploration.explorer import Explorer
from homebody.real2sim.exploration.skills import REGISTRY
from homebody.session.live_context import LiveContext, run_live_result
from homebody.skills.contract import Runner


@pytest.fixture
def action_owner():
    events = []
    owner = SimpleNamespace(
        live=SimpleNamespace(task_region_revision=3), lock=threading.RLock(),
        source=SimpleNamespace(stop_flag=threading.Event(), hold_position=lambda: None),
        zone=None, _decision_generation=4, check=lambda: None,
        _current_brief=lambda: "Explore the room", _task_brief="Explore the room",
        task_region_generation=7, task_scope_context=dict,
        decision_pose=lambda: (1., 2., .3), stop_reason="", phase="explore", stage="asking",
        previous_action_result={"outcome": "arrived", "goal": [1., 2.]})
    # The explorer's own region snapshot and event effects, over this minimal state.
    owner.region_snapshot = lambda: Explorer.region_snapshot(owner)
    owner.emit = lambda kind, **data: Explorer.emit(owner, kind, **data)
    context = LiveContext(owner)
    context.emit = events.append
    owner.session = SimpleNamespace(context=context, runner=Runner(context, registry=REGISTRY))
    return owner, events


def terminal(events, name):
    assert [event["type"] for event in events] == [
        "skill_started", "skill_stage", "skill_prepared", "skill_finished"]
    assert events[0]["skill"] == name
    assert events[1]["stage"] == "Preparing"
    # The recorder can retain the public plan without traversing live map/state references.
    json.dumps(events[2]["plan"].summary, allow_nan=False)
    assert events[-1]["result"].skill == name
    return events[-1]["result"]


def test_region_refusal_retains_permission_and_emits_one_terminal_result(action_owner):
    owner, events = action_owner
    frame = owner.live
    result = run_live_result(owner, "update_region", {
        "update": {"reason": "missing evidence"}, "transform": object(), "shape": (20, 30),
        "supplied_ids": [], "frame": frame, "context": None})
    assert result is terminal(events, "update_region")
    assert result.code == "INVALID_ARGUMENT" and "supplied evidence IDs" in result.message
    assert owner.live is frame and frame.task_region_revision == 3
    assert owner.stop_reason == "" and owner.previous_action_result["outcome"] == "arrived"


def test_help_records_incomplete_without_replacing_last_motion_result(action_owner):
    owner, events = action_owner
    record = {"n": 2, "used": False}
    previous = dict(owner.previous_action_result)
    result = run_live_result(owner, "request_help", {"why": "Doorway remains blocked", "record": record})
    assert result is terminal(events, "request_help") and result.ok
    assert owner.stop_reason == "incomplete: the model asked the operator for help: Doorway remains blocked"
    assert (owner.phase, owner.stage) == ("stopped", "held")
    assert record == {"n": 2, "used": True, "outcome": "held"}
    assert owner.previous_action_result == previous
