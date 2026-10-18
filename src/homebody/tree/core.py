"""The tree: Sequence and Fallback compose, Action calls one skill, Condition checks the board, Retry repeats.

A node's run(board) returns True (success) or False (failure); every action and failed condition is logged on
the board, so a caller (the agent, the coder) gets structured feedback: which node failed and the skill's result.
"""
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Board:
    """What the tree reads and writes. CALL runs a skill: call(skill, arguments) -> result dict (code, message).
    STATE holds what conditions read (for example "side", the hand in use); LOG the nodes run, oldest first."""
    call: Callable[[str, dict], dict]
    state: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    last: dict | None = None  # the newest skill result

    def failure(self):
        """What failed, for feedback: the newest failed skill call, else the newest failed check, or None. A recovery
        branch's own guard (for example "grasp uncertain") failing after a drop must not hide the drop itself."""
        failed = [entry for entry in self.log if not entry["ok"]]
        return next((e for e in reversed(failed) if e["kind"] == "action"), failed[-1] if failed else None)


class Node:
    name = "node"

    def run(self, board: Board) -> bool:
        raise NotImplementedError


class Action(Node):
    """One skill call; ARGUMENTS(board) builds its arguments (None: the action cannot start, a failure)."""

    def __init__(self, name, skill, arguments: Callable[[Board], dict | None]):
        self.name, self.skill, self.arguments = name, skill, arguments

    def run(self, board):
        arguments = self.arguments(board)
        if arguments is None:
            board.log.append({"node": self.name, "kind": "action", "ok": False, "code": "NOT_STARTED",
                              "message": "its arguments could not be built"})
            return False
        result = board.call(self.skill, arguments)
        board.last = result
        ok = result.get("code") == "OK"
        board.log.append({"node": self.name, "kind": "action", "skill": self.skill, "ok": ok,
                          "code": result.get("code"), "message": str(result.get("message", ""))[:200]})
        return ok


class Condition(Node):
    def __init__(self, name, check: Callable[[Board], bool]):
        self.name, self.check = name, check

    def run(self, board):
        ok = bool(self.check(board))
        if not ok:
            board.log.append({"node": self.name, "kind": "condition", "ok": False})
        return ok


class Sequence(Node):
    """Each child in turn; fails at the first child that fails."""

    def __init__(self, name, *children):
        self.name, self.children = name, children

    def run(self, board):
        return all(child.run(board) for child in self.children)


class Fallback(Node):
    """Each child in turn until one succeeds; fails when all do."""

    def __init__(self, name, *children):
        self.name, self.children = name, children

    def run(self, board):
        return any(child.run(board) for child in self.children)


class Retry(Node):
    """CHILD up to TIMES times until it succeeds."""

    def __init__(self, name, child, times):
        self.name, self.child, self.times = name, child, times

    def run(self, board):
        return any(self.child.run(board) for _ in range(self.times))
