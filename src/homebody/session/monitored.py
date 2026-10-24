"""What the session tells the live monitor (homebody/monitor) and what it does with the monitor's verdicts.

The monitor reads the goal the agent works on (the plan's active goal, or the task itself without a plan or before
one is written), the done goals, the hands and the skills; its verdicts flag a goal, reopen a done one, and reach
the agent as one line of its next prompt.
"""
from homebody.monitor.targets import NodeInfo, Verdict, VocabularyRequest
from homebody.tree.plan import BUDGET_DEFAULT

TASK_GOAL = "task"  # the goal id while no plan goal is active


def node_path(tree: dict, goal_id: str) -> str:
    """The node types above GOAL_ID in TREE, outermost first, e.g. "sequence > fallback"."""
    def walk(node, path):
        if node["type"] == "goal":
            return path if node["id"] == goal_id else None
        for child in node["children"]:
            found = walk(child, (*path, node["type"]))
            if found is not None:
                return found
        return None
    return " > ".join(walk(tree, ()) or ())


def goal_info(node: dict, plan, results: int = 0) -> NodeInfo:
    return NodeInfo(node["id"], node["goal"], node.get("check"), plan.status[node["id"]],
                    node.get("budget", BUDGET_DEFAULT) - results, node_path(plan.tree, node["id"]))


def goals(tree: dict):
    if tree["type"] == "goal":
        yield tree
        return
    for child in tree["children"]:
        yield from goals(child)


def still_true(node: dict, carried) -> bool:
    """Whether a done goal's result can still hold: a holding goal only while that hand still carries."""
    check = node.get("check") or {}
    return check.get("name") != "holding" or carried is None or check.get("hand") in carried


def current(planner, task: str, carried=None) -> tuple[NodeInfo, tuple[NodeInfo, ...], str]:
    """The goal the agent works on, the done goals whose result still holds (CARRIED: the hands that hold an
    object) and the rendered plan."""
    if planner is None or planner.plan is None or planner.abandoned:
        return NodeInfo(TASK_GOAL, task, None, "active"), (), ""
    plan = planner.plan
    done = tuple(goal_info(node, plan) for node in goals(plan.tree)
                 if plan.status[node["id"]] == "done" and still_true(node, carried))
    if planner.goal is None:
        return NodeInfo(TASK_GOAL, task, None, "between goals"), done, plan.render()
    node = goal_info(planner.goal, plan, len(planner.results))
    return NodeInfo(node.goal_id, node.goal, node.check, "active", node.budget_left, node.context), done, \
        plan.render(node.goal_id)


def plan_snapshot(planner) -> dict | None:
    """The plan as the console draws it: the tree, each goal's status, reason and monitor flag, and the active goal."""
    if planner is None or planner.plan is None or planner.abandoned:
        return None
    plan = planner.plan
    return {"tree": plan.tree, "status": dict(plan.status), "reasons": dict(plan.reasons), "flags": dict(plan.flags),
            "active": planner.goal["id"] if planner.goal is not None else None,
            "used": len(planner.results) if planner.goal is not None else 0,
            "results": [{"skill": r.get("skill"), "code": r.get("code"), "message": str(r.get("message", ""))[:160]}
                        for r in planner.results] if planner.goal is not None else []}


def request(planner, task: str, skills: tuple[str, ...], hands: str, images=(), *, stages=None,
            findings=(), skill=None, arguments=None, carried=None) -> VocabularyRequest:
    node, done, tree = current(planner, task, carried)
    return VocabularyRequest(task=task, tree=tree, node=node, done=done, skills=skills, hands=hands,
                             images=tuple(images), stages=dict(stages or {}), skill=skill, arguments=arguments,
                             findings=tuple(findings))


def apply(verdicts: list[Verdict], planner) -> str:
    """Put VERDICTS on the plan; the line the agent's next prompt carries ("" when there are none)."""
    lines = []
    for verdict in verdicts:
        if verdict.action == "reopen_goal":
            reopened = planner is not None and planner.reopen(verdict.goal_id, verdict.reason)
            lines.append(f"reopened goal {verdict.goal_id} ({verdict.reason})" if reopened else
                         f"saw {verdict.text()}")
            continue
        if planner is not None and verdict.goal_id != TASK_GOAL:
            planner.flag(verdict.goal_id, verdict.text())
        what = "stopped the skill" if verdict.action == "abort_skill" else "flagged"
        lines.append(f"{what}: {verdict.text()}")
    if not lines:
        return ""
    return ("Monitor (a model watching the camera; it can be wrong, so check before acting on it): "
            + "; ".join(dict.fromkeys(lines)) + ".")
