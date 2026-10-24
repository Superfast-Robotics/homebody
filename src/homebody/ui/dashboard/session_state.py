"""What the dashboard reads from a session besides its callbacks: the operator's task, the plan and the hands.

Read on the session thread only. The dashboard beside the session reads it directly; a harness server sends it with
each status (server/links.UiLink), so a dashboard elsewhere shows the same.
"""


def goal_texts(tree):
    texts = {}

    def walk(node):
        if node.get("type") == "goal":
            texts[node["id"]] = node.get("goal", node["id"])
        for child in node.get("children", ()):
            walk(child)
    walk(tree)
    return texts


def task(session):
    history = getattr(session, "history", None) or ()
    return next((entry["operator_task"] for entry in reversed(history) if "operator_task" in entry), None)


def plan(session):
    planner = getattr(session, "planner", None)
    if planner is None:
        return None
    current = planner.plan
    if current is None:
        return {"tree": None, "abandoned": planner.abandoned}
    goal = planner.goal
    return {"tree": current.tree, "goals": goal_texts(current.tree), "status": dict(current.status),
            "reasons": dict(current.reasons), "active": None if goal is None else goal["id"],
            "replans": planner.replans, "replans_allowed": planner.replans_allowed, "feedback": planner.feedback}


def robot(session):
    context = getattr(session, "context", None)
    if context is None:
        return {}
    carried = dict(context.carried)
    return {"hands": {side: "free" if side not in carried else carried[side].get("state", "held")
                      for side in ("left", "right")}}


def session_state(session) -> dict:
    """SESSION's task, plan and hands, as Store.update takes them."""
    return {"task": task(session), "plan": plan(session), "robot": robot(session)}
