"""What an execute changed, in words, and executes refused on a view or arguments the coder did not check."""
import math
from types import SimpleNamespace

import numpy as np

from homebody.harness.change_report import change_report
from homebody.session.coder import CoderCall


def frame(x=0.0, yaw=0.0, closure=0.0, effort=0.0):
    hand = SimpleNamespace(closure=closure, finger_effort_fraction=np.full(3, effort))
    return SimpleNamespace(base_pose=np.array([x, 0.0, yaw]), hands={"left": hand, "right": hand})


def test_a_still_robot_reports_no_motion_and_unchanged_hands():
    lines = change_report(frame(), frame(), {}, {})
    assert lines[0] == "base: did not move"
    assert "left hand: closure 0.00 -> 0.00, finger effort 0.00 -> 0.00; holds nothing" in lines


def test_motion_closure_effort_and_a_new_hold_are_reported():
    after = frame(x=0.3, yaw=math.radians(30), closure=0.8, effort=0.45)
    lines = change_report(frame(), after, {}, {"right": {"state": "verified"}})
    assert lines[0] == "base: moved 0.30 m, turned 30 deg"
    assert "right hand: closure 0.00 -> 0.80, finger effort 0.00 -> 0.45; holds nothing -> holds its object (grasp verified)" in lines


def test_an_uncertain_grasp_is_named():
    lines = change_report(frame(), frame(), {}, {"left": {"state": "uncertain"}})
    assert any(line.endswith("holds nothing -> grasp uncertain") for line in lines)


def call(view=1, checked=None):
    """A CoderCall with only what execute's preconditions read."""
    c = object.__new__(CoderCall)
    c.settings = SimpleNamespace(executes=2, checks=6)
    c.executes, c.view, c.checked, c.deadline = [], view, checked, math.inf
    c.code_digest = lambda: "d"
    return c


def test_execute_refuses_a_check_made_before_the_robot_moved():
    answer = call(view=2, checked={"digest": "d", "view": 1, "arguments": {"hand": "left"}}).execute({"hand": "left"})
    assert "robot moved" in answer["error"] and "observation/" in answer["error"]


def test_execute_refuses_arguments_other_than_the_checked_ones():
    answer = call(checked={"digest": "d", "view": 1, "arguments": {"hand": "left"}}).execute({"hand": "right"})
    assert "arguments you checked" in answer["error"]
