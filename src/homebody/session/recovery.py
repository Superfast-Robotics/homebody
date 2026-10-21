"""Stuck points for the coder's auto mode: a skill failing on one target until one of its calls succeeds.

A stuck point is a skill on one target (its hand and its target point, bucketed), so failures on one object do not
block the skill on another. At a stuck point a failure goes to the coder automatically, at most RESCUES times, and
the agent may call the skill there at most TRIES more times; a call of the skill that succeeds (or a coder call that
does) ends every stuck point of that skill: the situation has changed.
"""
import json

NOT_A_FAILURE = frozenset({"INVALID_ARGUMENT", "PHYSICS_INVALID", "RECOVERY_LIMIT"})
POINT_ARGUMENTS = ("point_normalized_1000", "release_point_1000", "goal_map_1000", "point_map_1000")
BUCKET = 100  # of 1000: target points this close count as one target


def stuck_key(skill: str, arguments: dict | None) -> tuple:
    """The stuck point of a call of SKILL with ARGUMENTS: the skill, its hand and its bucketed target point."""
    arguments = arguments or {}
    point = next((arguments[name] for name in POINT_ARGUMENTS if isinstance(arguments.get(name), (list, tuple))), None)
    cell = tuple(int(v) // BUCKET for v in point[:2]) if point is not None and len(point) >= 2 and all(
        isinstance(v, (int, float)) for v in point[:2]) else None
    return (skill, arguments.get("hand"), cell)


class Recovery:
    def __init__(self, tries: int, rescues: int):
        self.tries, self.rescues = tries, rescues
        self.points: dict[str, dict] = {}

    def reset(self):
        self.points = {}

    def admit(self, key: tuple) -> dict | None:
        """Before a call at stuck point KEY (stuck_key): None to run it (counting a retry there), else the refusal."""
        point, skill = self.points.get(key), key[0]
        if point is None:
            return None
        if point["retries"] >= self.tries:
            return {"skill": skill, "code": "RECOVERY_LIMIT", "details": {"failures": point["failures"]},
                    "message": f"{skill} failed on this target and was tried {self.tries} more times since; it is not "
                               "run on it again until another step changes the situation: choose another target or "
                               "skill, or done"}
        self.points[key] = {**point, "retries": point["retries"] + 1}
        return None

    def settle(self, key: tuple, code: str):
        """After a call at stuck point KEY ended with CODE: a success ends every stuck point of its skill."""
        if code == "OK":
            self.clear(key[0])
        elif code not in NOT_A_FAILURE:
            point = self.points.get(key, {"retries": 0, "coder": 0, "failures": 0})
            self.points[key] = {**point, "failures": point["failures"] + 1}

    def clear(self, skill: str):
        self.points = {key: point for key, point in self.points.items() if key[0] != skill}

    def wants_coder(self, key: tuple, result: dict) -> bool:
        """Whether RESULT at stuck point KEY goes to the coder: a counted failure, coder calls left there, and not
        one its skill marked final (no code change can succeed until the situation changes)."""
        point, code = self.points.get(key), result["code"]
        final = bool((result.get("details") or {}).get("final"))
        return (code not in NOT_A_FAILURE and code != "OK" and not final and point is not None
                and point["coder"] < self.rescues)

    def coder_done(self, skill: str, code: str, key: tuple | None = None):
        """After a coder call for SKILL ended with CODE: an OK version ends every stuck point of the skill; a failed
        one counts at KEY (an automatic call's stuck point), or at every stuck point of the skill without one."""
        if code == "OK":
            self.clear(skill)
            return
        for point_key, point in list(self.points.items()):
            if point_key == key or (key is None and point_key[0] == skill):
                self.points[point_key] = {**point, "coder": point["coder"] + 1}


def request(task: str, decision, result: dict) -> dict:
    """The coder's arguments for an automatic call after DECISION's failed RESULT."""
    call = f"{decision.skill} {json.dumps(decision.arguments)}"
    return {"skill": decision.skill,
            "goal": f"This step of the task '{task}': {decision.text}",
            "tried": f"{call} returned {result['code']}: {result['message']}",
            "success": f"{decision.skill} returns OK and achieves the step: {decision.text}",
            "constraints": ""}
