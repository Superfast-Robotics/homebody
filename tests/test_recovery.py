"""Stuck points in the coder's auto mode: how many coder calls and agent retries a failing skill gets."""
from homebody.session.recovery import Recovery, request
from homebody.vlm.provider import Decision


def test_a_stuck_point_gets_three_coder_calls_and_three_retries():
    recovery = Recovery(3)
    coder_calls = 0
    for call in range(4):
        assert recovery.admit("pick") is None
        recovery.settle("pick", "NO_GRASP")
        if recovery.wants_coder("pick", {"code": "NO_GRASP"}):
            coder_calls += 1
            recovery.coder_done("pick", "CODER_FAILED")
    assert coder_calls == 3
    refused = recovery.admit("pick")
    assert refused["code"] == "RECOVERY_LIMIT" and refused["details"]["failures"] == 4
    assert recovery.admit("navigate") is None  # another skill is not limited


def test_success_ends_the_stuck_point():
    recovery = Recovery(3)
    recovery.settle("pick", "NO_GRASP")
    recovery.coder_done("pick", "OK")
    assert "pick" not in recovery.points
    recovery.settle("pick", "NO_GRASP")
    recovery.admit("pick")
    recovery.settle("pick", "OK")
    assert "pick" not in recovery.points


def test_bad_arguments_are_not_a_stuck_point():
    recovery = Recovery(3)
    recovery.settle("pick", "INVALID_ARGUMENT")
    assert not recovery.wants_coder("pick", {"code": "INVALID_ARGUMENT"}) and "pick" not in recovery.points


def test_the_automatic_request_carries_the_failed_call():
    decision = Decision("The cup is on the shelf.", "pick", {"hand": "right"})
    ask = request("Put the cup away.", decision, {"code": "NO_GRASP", "message": "no grasp passed"})
    assert ask["skill"] == "pick" and "NO_GRASP: no grasp passed" in ask["tried"]
    assert "Put the cup away." in ask["goal"] and ask["success"].startswith("pick returns OK")


def test_a_final_refusal_is_left_to_the_agent():
    recovery = Recovery(3)
    recovery.settle("navigate", "CARRY_CLEARED")
    assert not recovery.wants_coder("navigate", {"code": "CARRY_CLEARED", "details": {"final": True}})
    assert recovery.wants_coder("navigate", {"code": "CARRY_CLEARED", "details": {}})
