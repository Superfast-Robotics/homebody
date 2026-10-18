"""Behavior trees over skills: composition, and the pick_and_carry template's reactive fallbacks."""
from homebody.tree import Action, Board, Condition, Fallback, Retry, Sequence, pick_and_carry


class Robot:
    """Skill results from a script, keyed by (skill, side); a pick that returns OK puts the object in that hand."""

    def __init__(self, script):
        self.script, self.calls, self.held = {k: list(v) for k, v in script.items()}, [], set()

    def call(self, skill, arguments):
        side = arguments["hand"]
        code = self.script[(skill, side)].pop(0)
        self.calls.append((skill, side, code))
        if skill == "pick" and code == "OK":
            self.held.add(side)
        if code == "CARRY_CLEARED":
            self.held.discard(side)
        return {"skill": skill, "code": code, "message": f"{skill} {code}"}


def tree():
    return pick_and_carry(click=lambda board: {"hand": board.state["side"]},
                          holding=lambda board: board.state["side"] in board.state["robot"].held,
                          legs=[lambda board: {"hand": board.state["side"]}], other_hand=True)


def board_for(robot):
    return Board(call=robot.call, state={"side": "right", "robot": robot})


def test_sequence_fallback_and_retry_compose():
    calls = []
    board = Board(call=lambda skill, arguments: calls.append(skill) or {"code": "OK" if len(calls) > 1 else "NO"})
    act = Action("a", "pick", lambda board: {})
    assert Retry("r", act, 3).run(board) and calls == ["pick", "pick"]
    assert Fallback("f", Condition("no", lambda board: False), Condition("yes", lambda board: True)).run(board)
    assert board.failure()["node"] == "a"  # the failed first pick: a failed skill call is the feedback
    checks = Board(call=board.call)
    assert not Sequence("s", Condition("yes", lambda board: True), Condition("no", lambda board: False)).run(checks)
    assert checks.failure()["node"] == "no"


def test_a_failed_pick_retries_then_switches_hands_without_a_model():
    robot = Robot({("pick", "right"): ["NO_GRASP", "NO_GRASP"], ("pick", "left"): ["OK"], ("navigate", "left"): ["OK"]})
    board = board_for(robot)
    assert tree().run(board)
    assert [c[:2] for c in robot.calls] == [("pick", "right"), ("pick", "right"), ("pick", "left"), ("navigate", "left")]
    assert board.state["side"] == "left"


def test_an_uncertain_carry_rechecks_the_grasp_and_walks_again():
    robot = Robot({("pick", "right"): ["OK", "OK"], ("navigate", "right"): ["HOLD_UNCERTAIN", "OK"]})
    assert tree().run(board_for(robot))
    assert [c[0] for c in robot.calls] == ["pick", "navigate", "pick", "navigate"]


def test_a_drop_on_the_walk_fails_with_the_skill_result_as_feedback():
    robot = Robot({("pick", "right"): ["OK"], ("navigate", "right"): ["CARRY_CLEARED"]})
    board = board_for(robot)
    assert not tree().run(board)
    assert board.failure()["code"] == "CARRY_CLEARED" and board.failure()["skill"] == "navigate"


def test_an_action_without_arguments_fails_without_calling():
    board = Board(call=lambda skill, arguments: (_ for _ in ()).throw(AssertionError("called")))
    assert not Action("a", "pick", lambda board: None).run(board)
    assert board.log[-1]["code"] == "NOT_STARTED"
