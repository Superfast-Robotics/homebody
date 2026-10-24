"""A task's outcome and the executor's lifecycle, kept apart: finishing a task is not succeeding at it.

The objectives are the plan's goals on its success path: in a fallback that succeeded only the child that succeeded
counts (the failed siblings it recovered from are history, not failed objectives); in one that did not, the child it
reached. A goal is verified when it closed with a check, which holds on robot state reported by skill results; a goal
the agent closed without a check is the agent's word only. Without a plan nothing verifies the agent's "done".
"""
from homebody.ui.dashboard.tree import node_status

BUSY = ("Thinking", "Executing")
ERRORS = ("error", "provider_error", "physics_invalid", "Reset required")
INTERRUPTED = ("cancelled", "Stopped", "interrupted")
LABELS = {
    "not_evaluated": "Not evaluated", "success": "Success", "success_with_recoveries": "Success with recovery",
    "partial": "Partial success", "failed": "Failed", "cancelled": "Cancelled", "unverified": "Outcome unverified",
}


def lifecycle(status, controls, tree_mode=None):
    """The executor's lifecycle: Loading, Ready, Planning, Running, Stopping, Paused, Resetting, Finished,
    Interrupted or Error."""
    if status == "Loading":
        return "Loading"
    if controls.get("paused") or status == "Paused":  # the operator's pause, or one the monitor asked for
        return "Paused"
    if status in BUSY:
        if controls.get("stop_pending"):
            return "Stopping"
        return "Planning" if tree_mode == "planning" else "Running"
    if status == "Resetting":
        return "Resetting"
    if status in ERRORS:
        return "Error"
    if status in INTERRUPTED:
        return "Interrupted"
    if status in ("model_done", "step_limit"):
        return "Finished"
    return "Ready"


def objectives(node, status):
    """The goals that count for the task: every goal of a sequence, the deciding child of a fallback."""
    if node["type"] == "goal":
        return [node]
    if node["type"] == "sequence":
        return [goal for child in node["children"] for goal in objectives(child, status)]
    statuses = [node_status(child, status) for child in node["children"]]
    chosen = statuses.index("success") if "success" in statuses else next(
        (index for index, s in enumerate(statuses) if s != "failure"), len(statuses) - 1)
    return objectives(node["children"][chosen], status)


def counts(plan):
    goals = objectives(plan["tree"], plan["status"])
    state = {goal["id"]: plan["status"].get(goal["id"], "pending") for goal in goals}
    done = [g for g in goals if state[g["id"]] == "done"]
    return {
        "total": len(goals),
        "verified": sum(1 for g in done if g.get("check")),
        "agent_closed": sum(1 for g in done if not g.get("check")),
        "failed": sum(1 for g in goals if state[g["id"]] == "failed"),
        "pending": sum(1 for g in goals if state[g["id"]] == "pending"),
    }, [g["goal"] for g in goals if state[g["id"]] == "failed"]


def evaluate(task, plan, recoveries):
    """The outcome of TASK from the planner's final PLAN and the task's RECOVERIES ({"started", "succeeded",
    "failed"}), as {"state", "label", "reason", "objectives", "failed_objectives", "recoveries"}."""
    planned = plan is not None and plan.get("tree") is not None
    objective, failed = counts(plan) if planned else (None, [])
    result = {"objectives": objective, "failed_objectives": failed, "recoveries": recoveries}

    def outcome(state, reason):
        return {**result, "state": state, "label": LABELS[state], "reason": reason}
    status = task["outcome"]
    if task["end"] is None:
        return outcome("not_evaluated", "The task is running.")
    if status in INTERRUPTED:
        return outcome("cancelled", "The task was stopped before it finished.")
    if not planned:
        if status == "model_done":
            return outcome("unverified", "The agent reports the task done; without a plan nothing verified it.")
        return outcome("failed", f"The task ended ({status}) before the agent reported it done.")
    if objective["verified"] == objective["total"]:
        ended = "" if status == "model_done" else f" The session then ended with {status}."
        if recoveries["succeeded"]:
            return outcome("success_with_recoveries", f"Every objective was verified by its check, after recovery.{ended}")
        return outcome("success", f"Every objective was verified by its check.{ended}")
    if objective["failed"] == 0 and objective["pending"] == 0:
        return outcome("unverified", f"{objective['agent_closed']} objective(s) were closed by the agent without a "
                                     "check, so nothing verified them.")
    if objective["verified"]:
        return outcome("partial", f"{objective['verified']} of {objective['total']} objectives were verified.")
    if objective["agent_closed"]:
        return outcome("unverified", "No objective was verified; the agent closed some without a check.")
    return outcome("failed", "No objective was verified.")
