"""How the store follows the planner: goal intervals, failed goals, recoveries through fallbacks, and replans.

A recovery is the planner moving to a fallback's next child after an earlier one failed; a replan is the agent
rewriting a failed tree. Both are actions the session took, so they go on the intervention track; the failures that
led to them are detections, whose OUTCOME says what followed.
"""
import json
from dataclasses import replace

from homebody.ui.dashboard.tree import plan_nodes, recovery_of

SUPERSEDED_LIMIT = 5  # replaced trees kept for the operator
FALLBACK = "Behavior Tree fallback"
SKILL_CALLS = "Agent-directed skill calls (robot skill executor)"
PLANNER = "Agent planner"


class PlanTracking:
    """Mixed into Store; every method runs under the store's lock."""

    def _plan(self, plan):
        if plan is None:
            return
        if plan["tree"] is None:  # plan mode, before the agent's plan is accepted
            self.plan = plan
            return
        previous = self.plan if self.plan is not None and self.plan.get("tree") is not None else None
        if previous is not None and plan["tree"] is not previous["tree"] and \
                json.dumps(plan["tree"], sort_keys=True) != json.dumps(previous["tree"], sort_keys=True):
            self._replanned(previous, plan)
            previous = None
        before = (previous or {}).get("status", {})
        for goal_id, status in plan["status"].items():
            if status == before.get(goal_id, "pending") or status not in ("done", "failed"):
                continue
            if status == "failed":
                self.incident("goal", "warning", f"Goal {goal_id} failed", plan["reasons"].get(goal_id, ""),
                              node=goal_id, detection=self._closed_by(goal_id))
            if goal_id in self._recoveries:
                self._recovery_closed(goal_id, status == "done", plan)
        self._follow(plan, (previous or {}).get("active"))
        self.plan = plan
        self._touch()

    def _follow(self, plan, previous):
        """Goal intervals follow the active goal; a goal entered after its fallback sibling failed is a recovery."""
        active = plan.get("active")
        if active == previous:
            return
        if previous is not None:
            final = plan["status"].get(previous)
            self._end("goal", {"done": "success", "failed": "failure"}.get(final, "ended"))
        if active is None:
            return
        failed = recovery_of(plan["tree"], plan["status"], active)
        self._begin("goal", plan["goals"].get(active, active), {"goal": active, "recovery": failed is not None})
        if failed is not None and active not in self._recoveries:
            label = plan["goals"].get(active, active)
            started = self.incident("recovery", "info", f"Recovery started: {label}",
                                    f"after goal {failed} failed: {plan['reasons'].get(failed, 'no reason given')}",
                                    source="planner", node=active, tag="recovery_start", initiator=FALLBACK,
                                    executor=SKILL_CALLS, caused_by=self._goal_failure(failed),
                                    action=f"The plan's fallback moved from goal {failed} to goal {active}.")
            self._recoveries[active] = {"incident": started.id, "failed": failed}
            self._outcome_for_goal(failed, f"Recovery started: goal {active} ({label})")

    def _recovery_closed(self, goal_id, done, plan):
        recovery = self._recoveries.pop(goal_id)
        label = plan["goals"].get(goal_id, goal_id)
        outcome = f"Recovery {'succeeded' if done else 'failed'}: goal {goal_id} ({label})"
        self.incident("recovery", "info", outcome, plan["reasons"].get(goal_id, ""), source="planner",
                      node=goal_id, tag="recovery_ok" if done else "recovery_fail", initiator=FALLBACK,
                      executor=SKILL_CALLS, caused_by=recovery["incident"], detection=self._closed_by(goal_id))
        self._outcome(recovery["incident"], "Succeeded" if done else "Failed")
        self._outcome_for_goal(recovery["failed"], outcome)

    def _replanned(self, previous, plan):
        """The agent rewrote the tree: keep the old one, marked superseded, and mark the replan."""
        self._superseded = [*self._superseded, {"t": self.clock(), "replan": plan.get("replans", 0),
                                                "tree": plan_nodes(previous, self.tools.values(), self.task)}]
        self._superseded = self._superseded[-SUPERSEDED_LIMIT:]
        self._end("goal", "ended")
        self._recoveries = {}
        failure = next((i for i in reversed(self.incidents.values()) if i.kind == "goal" and not i.outcome), None)
        self.incident("recovery", "info", f"Replanned (rewrite {plan.get('replans', '?')})",
                      plan.get("feedback", ""), source="agent", tag="replan", initiator=PLANNER, executor=PLANNER,
                      caused_by=failure and failure.id,
                      action="The agent rewrote the behavior tree; goals already done stay done.")
        if failure is not None:
            self._outcome(failure.id, f"Replanned (rewrite {plan.get('replans', '?')})")

    def _closed_by(self, goal_id):
        """Who closed GOAL_ID: the agent (a goal_done or goal_failed decision for it) or the plan runner itself."""
        start = self.task["start"] if self.task else float("-inf")
        agent = any(d.goal == goal_id and d.skill in ("goal_done", "goal_failed") and d.t >= start
                    for d in self.decisions.values())
        return "Agent planner (closed the goal)" if agent else "Plan runner (the goal's check or skill budget)"

    def _goal_failure(self, goal_id):
        return next((i.id for i in reversed(self.incidents.values()) if i.kind == "goal" and i.node == goal_id), None)

    def _outcome(self, incident_id, text):
        incident = self.incidents.get(incident_id)
        if incident is not None:
            self.incidents[incident_id] = replace(incident, outcome=text)

    def _outcome_for_goal(self, goal_id, text):
        """Fill in what followed the newest failure of GOAL_ID."""
        for incident in reversed(self.incidents.values()):
            if incident.kind == "goal" and incident.node == goal_id:
                self._outcome(incident.id, text)
                return
