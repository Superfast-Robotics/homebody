"""Stuck points for the coder's auto mode: a skill failing until one of its calls succeeds.

At a stuck point each failure goes to the coder automatically, at most TRIES times, and the agent may call
the skill at most TRIES more times; a call of it that succeeds (or a coder call that does) ends the point.
"""
import json

NOT_A_FAILURE = frozenset({"INVALID_ARGUMENT", "PHYSICS_INVALID", "RECOVERY_LIMIT"})


class Recovery:
    def __init__(self, tries: int):
        self.tries = tries
        self.points: dict[str, dict] = {}

    def reset(self):
        self.points = {}

    def admit(self, skill: str) -> dict | None:
        """Before a call of SKILL: None to run it (counting a retry at its stuck point), else the refusal."""
        point = self.points.get(skill)
        if point is None:
            return None
        if point["retries"] >= self.tries:
            return {"skill": skill, "code": "RECOVERY_LIMIT", "details": {"failures": point["failures"]},
                    "message": f"{skill} failed and was tried {self.tries} more times since; it is not run again "
                               "until another step changes the situation: choose another skill, or done"}
        self.points[skill] = {**point, "retries": point["retries"] + 1}
        return None

    def settle(self, skill: str, code: str):
        """After a call of SKILL ended with CODE."""
        if code == "OK":
            self.points.pop(skill, None)
        elif code not in NOT_A_FAILURE:
            point = self.points.get(skill, {"retries": 0, "coder": 0, "failures": 0})
            self.points[skill] = {**point, "failures": point["failures"] + 1}

    def wants_coder(self, skill: str, code: str) -> bool:
        point = self.points.get(skill)
        return code not in NOT_A_FAILURE and code != "OK" and point is not None and point["coder"] < self.tries

    def coder_done(self, skill: str, code: str):
        """After a coder call for SKILL ended with CODE: an OK version ends the stuck point."""
        if code == "OK":
            self.points.pop(skill, None)
        elif skill in self.points:
            point = self.points[skill]
            self.points[skill] = {**point, "coder": point["coder"] + 1}


def request(task: str, decision, result: dict) -> dict:
    """The coder's arguments for an automatic call after DECISION's failed RESULT."""
    call = f"{decision.skill} {json.dumps(decision.arguments)}"
    return {"skill": decision.skill,
            "goal": f"This step of the task '{task}': {decision.text}",
            "tried": f"{call} returned {result['code']}: {result['message']}",
            "success": f"{decision.skill} returns OK and achieves the step: {decision.text}",
            "constraints": ""}
