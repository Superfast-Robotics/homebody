"""Plan mode: the agent first writes the task as a behavior tree of goals (homebody.tree.plan), then works through
it goal by goal with any skill calls and the coder. A goal closes when its check holds, when the agent closes it,
or when its budget of skill calls runs out; a failed tree may be rewritten at most `plan.replans` times.

The session asks this runner, every turn, which actions the agent may choose and what to add to the prompt, hands
it plan / goal_done / goal_failed decisions, and reports each skill result to it.
"""
import json

from homebody.tree.plan import Plan, PlanError, check_holds

PLAN, GOAL_DONE, GOAL_FAILED = "plan", "goal_done", "goal_failed"
OWN = (PLAN, GOAL_DONE, GOAL_FAILED)
INVALID_PLANS = 2  # invalid trees in a row before the task continues without a plan


class PlanRunner:
    def __init__(self, replans):
        self.replans_allowed = replans
        self.plan: Plan | None = None
        self.replans, self.invalid, self.abandoned = 0, 0, False
        self.goal, self.results, self.feedback = None, [], ""
        self.operator_goals, self._operator_closed = 0, None  # the operator's commands added as goals

    # --- what the agent may do this turn ---
    def phase(self):
        """'plan' (write or rewrite the tree), 'goal' (work on the active goal) or 'finish' (only done)."""
        if self.abandoned:
            return "free"
        if self.plan is None:
            return "plan"
        state, node = self.plan.next()
        if state == "goal":
            if self.goal is None or self.goal["id"] != node["id"]:
                self.goal, self.results = node, []
            return "goal"
        if state == "failed" and self.replans < self.replans_allowed:
            self.feedback = f"goal {node['id']} failed: {self.plan.reasons.get(node['id'], 'no reason given')}"
            return "plan"
        return "finish"

    def names(self, base):
        phase = self.phase()
        if phase == "plan":
            return (PLAN,)
        if phase == "finish":
            return ("done",)
        if phase == "goal":
            return (*[n for n in base if n != "done"], GOAL_DONE, GOAL_FAILED, "done")
        return base

    def note(self):
        """The plan section of the prompt."""
        phase = self.phase()
        if phase == "free":
            return ""
        if phase == "plan" and self.plan is None:
            return "This turn: write the plan (skill plan)."
        tree = self.plan.render(self.goal["id"] if phase == "goal" else None)
        if phase == "plan":
            left = self.replans_allowed - self.replans
            return (f"The plan failed ({self.feedback}); rewrite the rest of it (skill plan), {left} rewrite(s) "
                    f"left. Goals already done stay done:\n{tree}")
        if phase == "finish":
            state, _ = self.plan.next()
            return f"The plan is {'done' if state == 'done' else 'failed with no rewrites left'}; finish with done.\n{tree}"
        left = self.goal.get("budget", 8) - len(self.results)
        return (f"Plan:\n{tree}\nActive goal {self.goal['id']}: {self.goal['goal']} "
                f"({left} skill call{'s' if left != 1 else ''} left for it).")

    # --- decisions the runner itself answers ---
    def handle(self, decision, carried):
        """A plan, goal_done or goal_failed decision: its result dict (as a skill result)."""
        if decision.skill == PLAN:
            return self._plan(decision.arguments.get("tree"))
        if self.goal is None:
            return {"skill": decision.skill, "code": "INVALID_ARGUMENT", "message": "no goal is active"}
        reason = str(decision.arguments.get("reason", decision.text))
        if decision.skill == GOAL_DONE:
            check = self.goal.get("check")
            if check and not check_holds(check, carried, self.results):
                return {"skill": GOAL_DONE, "code": "CHECK_FAILED",
                        "message": f"goal {self.goal['id']} is not done: its check {json.dumps(check)} does not hold"}
            return self._close(True, reason)
        return self._close(False, reason)

    def _plan(self, tree):
        try:
            plan = Plan(tree)
        except PlanError as error:
            self.invalid += 1
            if self.invalid >= INVALID_PLANS:
                self.abandoned = True  # the task goes on without a plan
                return {"skill": PLAN, "code": "PLAN_INVALID", "message": f"{error}; continuing without a plan"}
            return {"skill": PLAN, "code": "PLAN_INVALID", "message": str(error)}
        if self.plan is not None:  # a rewrite: goals already done stay done
            self.replans += 1
            for goal_id, status in self.plan.status.items():
                if status == "done" and goal_id in plan.status:
                    plan.status[goal_id] = "done"
        self.plan, self.invalid, self.goal, self.results = plan, 0, None, []
        return {"skill": PLAN, "code": "OK", "message": f"plan accepted: {len(plan.status)} goals"}

    def _close(self, done, reason):
        goal_id = self.goal["id"]
        if self.goal.get("operator"):
            self._operator_closed = goal_id
        self.plan.close(goal_id, done, reason)
        self.goal, self.results = None, []
        return {"skill": GOAL_DONE if done else GOAL_FAILED, "code": "OK",
                "message": f"goal {goal_id} {'done' if done else 'failed'}"}

    # --- the operator's commands ---
    def insert_operator(self, text):
        """The operator's command TEXT as the plan's next goal: its id, or None without a plan to add it to."""
        if self.plan is None or self.abandoned:
            return None
        self.operator_goals += 1
        goal_id = f"op{self.operator_goals}"
        self.plan.insert_operator(goal_id, text)
        return goal_id

    def operator_pending(self):
        """Whether an operator goal is still to do."""
        return self.plan is not None and any(self.plan.status.get(f"op{n}") == "pending"
                                             for n in range(1, self.operator_goals + 1))

    def take_operator_closed(self):
        """The operator goal that closed since the last call, or None."""
        closed, self._operator_closed = self._operator_closed, None
        return closed

    # --- the live monitor ---
    def flag(self, goal_id, text):
        if self.plan is not None:
            self.plan.flag(goal_id, text)

    def reopen(self, goal_id, reason):
        """Reopen a done goal the monitor saw undone; the walk makes it active when it comes first."""
        return self.plan is not None and not self.abandoned and self.plan.reopen(goal_id, reason)

    # --- skill results ---
    def observe(self, result, carried):
        """A skill (or coder) result while a goal is active: close the goal when its check holds or its budget is spent."""
        if self.abandoned or self.plan is None or self.goal is None:
            return
        self.results.append(result)
        if check_holds(self.goal.get("check"), carried, self.results):
            self._close(True, f"check {self.goal['check']['name']} holds")
        elif len(self.results) >= self.goal.get("budget", 8):
            self._close(False, f"budget of {self.goal.get('budget', 8)} skill calls spent; last: "
                               f"{result.get('code')} {str(result.get('message'))[:80]}")
