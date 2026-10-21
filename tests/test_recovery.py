"""Stuck points in the coder's auto mode: how many coder calls and agent retries a failing skill gets."""
from homebody.session.recovery import Recovery, request, stuck_key
from homebody.vlm.provider import Decision

PICK = stuck_key("pick", {"hand": "right", "point_normalized_1000": [420, 610]})
NAVIGATE = stuck_key("navigate", {"goal_map_1000": [300, 300]})


def test_a_stuck_point_gets_three_coder_calls_and_three_retries():
    recovery = Recovery(3, 3)
    coder_calls = 0
    for call in range(4):
        assert recovery.admit(PICK) is None
        recovery.settle(PICK, "NO_GRASP")
        if recovery.wants_coder(PICK, {"code": "NO_GRASP"}):
            coder_calls += 1
            recovery.coder_done("pick", "CODER_FAILED")
    assert coder_calls == 3
    refused = recovery.admit(PICK)
    assert refused["code"] == "RECOVERY_LIMIT" and refused["details"]["failures"] == 4
    assert recovery.admit(NAVIGATE) is None  # another skill is not limited


def test_success_ends_the_stuck_point():
    recovery = Recovery(3, 3)
    recovery.settle(PICK, "NO_GRASP")
    recovery.coder_done("pick", "OK")
    assert PICK not in recovery.points
    recovery.settle(PICK, "NO_GRASP")
    recovery.admit(PICK)
    recovery.settle(PICK, "OK")
    assert PICK not in recovery.points


def test_bad_arguments_are_not_a_stuck_point():
    recovery = Recovery(3, 3)
    recovery.settle(PICK, "INVALID_ARGUMENT")
    assert not recovery.wants_coder(PICK, {"code": "INVALID_ARGUMENT"}) and "pick" not in recovery.points


def test_the_automatic_request_carries_the_failed_call():
    decision = Decision("The cup is on the shelf.", "pick", {"hand": "right"})
    ask = request("Put the cup away.", decision, {"code": "NO_GRASP", "message": "no grasp passed"})
    assert ask["skill"] == "pick" and "NO_GRASP: no grasp passed" in ask["tried"]
    assert "Put the cup away." in ask["goal"] and ask["success"].startswith("pick returns OK")


def test_a_final_refusal_is_left_to_the_agent():
    recovery = Recovery(3, 3)
    recovery.settle(NAVIGATE, "CARRY_CLEARED")
    assert not recovery.wants_coder(NAVIGATE, {"code": "CARRY_CLEARED", "details": {"final": True}})
    assert recovery.wants_coder(NAVIGATE, {"code": "CARRY_CLEARED", "details": {}})


def test_one_rescue_per_stuck_point_still_allows_three_tries():
    recovery = Recovery(3, 1)
    recovery.settle(PICK, "NO_GRASP")
    assert recovery.wants_coder(PICK, {"code": "NO_GRASP"})
    recovery.coder_done("pick", "CODER_FAILED")
    for _ in range(3):
        assert recovery.admit(PICK) is None
        recovery.settle(PICK, "NO_GRASP")
        assert not recovery.wants_coder(PICK, {"code": "NO_GRASP"})
    assert recovery.admit(PICK)["code"] == "RECOVERY_LIMIT"


def test_failures_on_one_target_do_not_block_the_skill_on_another():
    recovery = Recovery(3, 1)
    jar = stuck_key("pick", {"hand": "right", "point_normalized_1000": [420, 610]})
    can = stuck_key("pick", {"hand": "right", "point_normalized_1000": [780, 590]})
    assert stuck_key("pick", {"hand": "right", "point_normalized_1000": [445, 640]}) == jar  # the same object
    for _ in range(4):
        recovery.admit(jar)
        recovery.settle(jar, "NO_GRASP")
    assert recovery.admit(jar)["code"] == "RECOVERY_LIMIT"
    assert recovery.admit(can) is None
    recovery.settle(can, "OK")  # a success changes the situation: every stuck pick point ends
    assert recovery.admit(jar) is None


def test_an_automatic_coder_failure_counts_only_at_its_own_stuck_point():
    recovery = Recovery(3, 1)
    jar = stuck_key("pick", {"hand": "right", "point_normalized_1000": [420, 610]})
    can = stuck_key("pick", {"hand": "left", "point_normalized_1000": [780, 590]})
    recovery.settle(jar, "NO_GRASP")
    recovery.settle(can, "NO_GRASP")
    recovery.coder_done("pick", "CODER_FAILED", jar)
    assert not recovery.wants_coder(jar, {"code": "NO_GRASP"})
    assert recovery.wants_coder(can, {"code": "NO_GRASP"})
