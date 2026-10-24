"""The planner's behavior tree as flat, keyed nodes for the browser, with each goal's executed tool calls.

Node ids stay stable between snapshots ("goal:<id>", "tool:<id>", and child paths for composites), so the page can
update rows in place. MODE says where the task is: "plan" (a tree runs), "replanning" (the tree failed and the agent
is rewriting it), "planning" (plan mode, before the first tree), or "free" (no tree: the task's actions under one
root, labelled as such).

A goal is the tree's retry boundary: its skill calls are its attempts, up to its budget. When a goal has made more
than one, the finished earlier ones sit under an "attempts" node ("attempts:<goal id>"), a grouping for reading only
(detail.visual): the goal and its calls are unchanged, and the newest call stays directly under the goal.
"""
GOAL_STATUS = {"pending": "pending", "active": "running", "done": "success", "failed": "failure"}
TOOL_STATUS = {"running": "running", "succeeded": "success", "failed": "failure"}
LABELS = {"sequence": "Sequence", "fallback": "Fallback"}


def composite_status(kind, statuses):
    """A sequence or fallback's status from its children's, as homebody.tree.plan evaluates them."""
    if "running" in statuses:
        return "running"
    if kind == "sequence":
        if all(s == "success" for s in statuses):
            return "success"
        return "failure" if "failure" in statuses else "pending"
    if "success" in statuses:
        return "success"
    return "failure" if all(s == "failure" for s in statuses) else "pending"


def node_status(node, status, active=None):
    """NODE's status from the planner's goal STATUS, with ACTIVE the goal being worked on."""
    if node["type"] == "goal":
        return GOAL_STATUS["active" if node["id"] == active else status.get(node["id"], "pending")]
    return composite_status(node["type"], [node_status(child, status, active) for child in node["children"]])


def goal_ids(node):
    if node["type"] == "goal":
        return [node["id"]]
    return [goal for child in node["children"] for goal in goal_ids(child)]


def recovery_of(tree, status, goal_id):
    """The goal whose failure GOAL_ID recovers from: the failed sibling before it in a fallback, else None."""
    if tree["type"] == "goal":
        return None
    for index, child in enumerate(tree["children"]):
        if goal_id not in goal_ids(child):
            continue
        if tree["type"] == "fallback" and index > 0 and goal_id == goal_ids(child)[0]:
            failed = [g for g in goal_ids(tree["children"][index - 1]) if status.get(g) == "failed"]
            if failed:
                return failed[-1]
        return recovery_of(child, status, goal_id)
    return None


def check_text(check):
    """A goal's check in words, e.g. "holding (right hand)"; a goal without one is closed by the agent."""
    if not check:
        return "the agent closes it"
    name = str(check.get("name", "")).replace("_", " ")
    hand = check.get("hand")
    return f"{name} ({hand} hand)" if hand else name


def action_nodes(tools, interventions=()):
    """One node per tool call, with the requests to stop it (their states, as the store derives them): a request
    never changes the node's status, which only the call's own result sets."""
    on = {}
    for row in interventions:
        if row.get("tool"):
            on.setdefault(row["tool"], []).append({"id": row["id"], "state": row["state"],
                                                   "initiator": row["initiator"]})
    return [{"id": f"tool:{tool.id}", "type": "action", "label": tool.skill,
             "status": TOOL_STATUS.get(tool.state, "skipped"), "children": [],
             "detail": {"tool": tool.id, "code": tool.code, "message": tool.message, "start": tool.start,
                        "end": tool.end, "attempt": tool.attempt, "recovery": tool.recovery,
                        "number": tool.number, "execution": tool.execution,  # the store's identity, as the timeline shows it
                        "interventions": on.get(tool.id, [])}}
            for tool in tools]


def grouped(goal_id, actions, nodes):
    """ACTIONS' ids under the goal, the finished earlier ones in one attempts node when there are several."""
    if len(actions) < 2:
        return [n["id"] for n in actions]
    earlier, newest = actions[:-1], actions[-1]
    failed = [n for n in earlier if n["status"] == "failure"]
    key = f"attempts:{goal_id}"
    nodes[key] = {"id": key, "type": "attempts", "label": "Earlier attempts",
                  "status": "failure" if len(failed) == len(earlier) else composite_status("fallback", [
                      n["status"] for n in earlier]),
                  "children": [n["id"] for n in earlier],
                  "detail": {"visual": True, "count": len(earlier), "failed": len(failed),
                             "codes": [n["detail"]["code"] for n in earlier]}}
    return [key, newest["id"]]


def free_note(plan, task):
    if plan is None:
        return "No behavior tree: the agent chooses one step at a time (plan mode off)."
    if plan.get("abandoned"):
        return "No behavior tree: the agent's plans were invalid, so it continues one step at a time."
    if task["end"] is not None:
        return "The task ended before the agent wrote a behavior tree."
    return "Generating the execution plan: the tree appears when the agent's plan is accepted."


def free_nodes(plan, tools, task, interventions=()):
    actions = action_nodes(tools, interventions)
    nodes = {n["id"]: n for n in actions}
    status = ("running" if task["end"] is None else "success" if task["outcome"] == "model_done"
              else "skipped" if task["outcome"] in ("cancelled", "Stopped") else "failure")
    nodes["root"] = {"id": "root", "type": "root", "label": task["text"], "status": status,
                     "children": [n["id"] for n in actions], "detail": {"note": free_note(plan, task)}}
    planning = plan is not None and not plan.get("abandoned") and task["end"] is None
    return {"mode": "planning" if planning else "free", "root": "root", "nodes": nodes}


def plan_nodes(plan, tools, task, interventions=()):
    """{"mode", "root": id, "nodes": {id: node}} for the task's tree, or None before any task."""
    if task is None:
        return None
    tools = [t for t in tools if t.start >= task["start"]]
    if plan is None or plan["tree"] is None:
        return free_nodes(plan, tools, task, interventions)
    nodes, status, active = {}, plan["status"], plan.get("active")

    def walk(node, path):
        if node["type"] == "goal":
            return goal(node)
        children = [walk(child, f"{path}.{index}") for index, child in enumerate(node["children"])]
        statuses = [nodes[c]["status"] for c in children]
        selected = next((i for i, s in enumerate(statuses) if s != "failure"), None)
        detail = {"selected": None}
        if node["type"] == "fallback":
            last = len(children) - 1 if selected is None else selected
            detail = {"selected": selected,
                      "path": [{"label": nodes[c]["label"], "status": nodes[c]["status"]} for c in children[:last + 1]]}
        nodes[path] = {"id": path, "type": node["type"], "label": LABELS[node["type"]],
                       "status": composite_status(node["type"], statuses), "children": children, "detail": detail}
        return path

    def goal(node):
        goal_id, key = node["id"], f"goal:{node['id']}"
        actions = action_nodes([t for t in tools if t.goal == goal_id], interventions)
        nodes.update((n["id"], n) for n in actions)
        recovers = recovery_of(plan["tree"], status, goal_id)
        nodes[key] = {"id": key, "type": "goal", "label": node["goal"], "status": node_status(node, status, active),
                      "children": grouped(goal_id, actions, nodes),
                      "detail": {"goal": goal_id, "check": check_text(node.get("check")),
                                 "budget": node.get("budget", 8), "used": len(actions),
                                 "reason": plan["reasons"].get(goal_id, ""),
                                 "recovers": recovers, "recovers_label": plan["goals"].get(recovers)}}
        return key

    top = walk(plan["tree"], "p")
    root, note, mode = nodes[top]["status"], "", "plan"
    if task["end"] is None and root == "failure":
        mode, note = "replanning", "The tree failed; the agent is rewriting it. Goals already done stay done."
    elif task["end"] is not None and root in ("running", "pending"):
        root, note = "skipped", f"Task ended ({task['outcome']}) before the tree finished"
    nodes["root"] = {"id": "root", "type": "root", "label": task["text"], "status": root,
                     "children": [top], "detail": {"note": note} if note else {}}
    return {"mode": mode, "root": "root", "nodes": nodes}
