"""The agent's task plan as a behavior tree of goals: validated before it runs, walked goal by goal.

    {"type": "sequence", "children": [
        {"type": "goal", "id": "g1", "goal": "pick the yellow bottle", "check": {"name": "holding", "hand": "right"}},
        {"type": "fallback", "children": [{"type": "goal", ...}, {"type": "goal", ...}]}, ...]}

A goal is worked on by the agent (any skills, the coder) until its check holds, the agent closes it, or its budget
of steps runs out. A sequence succeeds when every child does and fails at the first child that fails; a fallback
succeeds at the first child that succeeds and fails when all do.
"""
from dataclasses import dataclass, field

CODER = "call_coder"  # session.coder.CALL: a coder call's result names the skill it changed

TYPES = ("sequence", "fallback", "goal")
MAX_GOALS, MAX_DEPTH = 12, 4
BUDGET_DEFAULT, BUDGET_MAX = 8, 15
# What a goal's check may name: robot state a skill result or a hand reports, never simulator ground truth.
CHECKS = {
    "holding": "the hand holds the object its last pick verified",
    "hand_free": "the hand holds nothing",
    "placed": "a place call in this goal returned OK",
    "arrived": "a navigate call in this goal returned OK",
}
HANDS = ("left", "right")


class PlanError(ValueError):
    pass


def validate(tree):
    """TREE checked against the plan contract: PlanError naming the first problem, else the goal ids in order."""
    ids = []

    def walk(node, depth, where):
        if not isinstance(node, dict) or node.get("type") not in TYPES:
            raise PlanError(f"{where}: type must be one of {', '.join(TYPES)}")
        if depth > MAX_DEPTH:
            raise PlanError(f"{where}: deeper than {MAX_DEPTH} levels")
        if node["type"] != "goal":
            children = node.get("children")
            if not isinstance(children, list) or not children:
                raise PlanError(f"{where}: a {node['type']} needs a non-empty children list")
            for index, child in enumerate(children):
                walk(child, depth + 1, f"{where}.children[{index}]")
            return
        if not isinstance(node.get("id"), str) or not node["id"] or node["id"] in ids:
            raise PlanError(f"{where}: a goal needs a unique id")
        if not isinstance(node.get("goal"), str) or not node["goal"].strip():
            raise PlanError(f"{where}: a goal needs its goal text")
        check = node.get("check")
        if check is not None:
            if not isinstance(check, dict) or check.get("name") not in CHECKS:
                raise PlanError(f"{where}: check must be null or one of {', '.join(CHECKS)}")
            if check["name"] in ("holding", "hand_free") and check.get("hand") not in HANDS:
                raise PlanError(f"{where}: check {check['name']} needs hand left or right")
        budget = node.get("budget", BUDGET_DEFAULT)
        if type(budget) is not int or not 1 <= budget <= BUDGET_MAX:
            raise PlanError(f"{where}: budget must be an integer from 1 to {BUDGET_MAX}")
        ids.append(node["id"])
        if len(ids) > MAX_GOALS:
            raise PlanError(f"more than {MAX_GOALS} goals")

    walk(tree, 1, "plan")
    return ids


@dataclass
class Plan:
    """A validated tree and each goal's status: pending, active, done or failed (with the reason), and the live
    monitor's note on a goal until it closes."""
    tree: dict
    status: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

    def __post_init__(self):
        for goal_id in validate(self.tree):
            self.status.setdefault(goal_id, "pending")

    def next(self):
        """("goal", node) for the goal to work on, or ("done", None) / ("failed", node) when the tree has finished."""
        state, node = self._evaluate(self.tree)
        return ("goal", node) if state == "run" else (state, node)

    def _evaluate(self, node):
        if node["type"] == "goal":
            status = self.status[node["id"]]
            return ("done", node) if status == "done" else ("failed", node) if status == "failed" else ("run", node)
        if node["type"] == "sequence":
            for child in node["children"]:
                state, found = self._evaluate(child)
                if state != "done":
                    return state, found
            return "done", node
        failed = None
        for child in node["children"]:  # fallback
            state, found = self._evaluate(child)
            if state != "failed":
                return state, found
            failed = found
        return "failed", failed

    def close(self, goal_id, done, reason=""):
        self.status[goal_id] = "done" if done else "failed"
        self.flags.pop(goal_id, None)
        if reason:
            self.reasons[goal_id] = reason[:200]

    def flag(self, goal_id, text):
        """The monitor's note on GOAL_ID, shown until the goal closes."""
        if goal_id in self.status:
            self.flags[goal_id] = text[:200]

    def reopen(self, goal_id, reason):
        """A done goal back to pending (the monitor saw its result undone); whether it was done."""
        if self.status.get(goal_id) != "done":
            return False
        self.status[goal_id] = "pending"
        self.reasons.pop(goal_id, None)
        self.flags[goal_id] = f"monitor reopened: {reason}"[:200]
        return True

    def render(self, active=None):
        """The tree as indented text with each goal's status, for the agent's prompt."""
        lines = []

        def walk(node, depth):
            pad = "  " * depth
            if node["type"] != "goal":
                lines.append(f"{pad}{node['type']}")
                for child in node["children"]:
                    walk(child, depth + 1)
                return
            check = node.get("check")
            check_text = "agent closes it" if not check else " ".join(f"{k}={v}" for k, v in check.items())
            marker = "ACTIVE" if node["id"] == active else self.status[node["id"]]
            flag = self.flags.get(node["id"])
            if flag:
                marker += f" ⚠ {flag}" if flag.startswith("monitor") else f" ⚠ monitor: {flag}"
            reason = f" ({self.reasons[node['id']]})" if node["id"] in self.reasons else ""
            lines.append(f"{pad}[{marker}] {node['id']}: {node['goal']} | check: {check_text}{reason}")

        walk(self.tree, 0)
        return "\n".join(lines)


def check_holds(check, carried, results):
    """Whether CHECK holds, from CARRIED ({side: carry record}) and RESULTS (this goal's skill results, oldest
    first); a goal without a check never closes by itself."""
    if not check:
        return False
    name = check["name"]
    if name == "holding":
        return (carried.get(check["hand"]) or {}).get("state") == "verified"
    if name == "hand_free":
        return check["hand"] not in carried
    skill = {"placed": "place", "arrived": "navigate"}[name]
    return any(r.get("code") == "OK" and (r.get("skill") == skill or did_by_coder(r, skill)) for r in results)


def did_by_coder(result, skill):
    """Whether RESULT is a coder call that ran SKILL on the robot and got OK (the coder did the goal's work)."""
    details = result.get("details") or {}
    return (result.get("skill") == CODER and details.get("target") == skill
            and any(run.get("code") == "OK" for run in details.get("executes") or ()))
