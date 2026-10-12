"""Skills the agent writes during a task: `customize_skill` compiles `def run(env, **arguments)`
against the harness primitives, and the new version replaces (or adds) that skill until the task
ends. Its runs are sandboxed (no imports or builtins beyond plain computation), bounded in wall
time, and reported back as ordinary skill results.
"""
import dataclasses
import time
import traceback

import numpy as np

from homebody import harness
from homebody.primitives.actions import PhysicsInvalid

CUSTOMIZE = "customize_skill"
RUN_SECONDS = 300.0  # wall time one customized-skill run may take
MAX_CODE_CHARS = 20000
RESERVED = frozenset({CUSTOMIZE, "done"})
NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
SAFE_BUILTINS = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                 for name in ("abs", "min", "max", "range", "len", "enumerate", "zip", "float", "int", "list", "dict",
                              "tuple", "sorted", "sum", "round", "isinstance", "any", "all", "reversed", "map", "filter",
                              "bool", "str", "set", "frozenset", "divmod", "pow", "repr", "hasattr", "getattr", "next",
                              "iter", "Exception", "ValueError", "RuntimeError", "KeyError", "IndexError",
                              "ZeroDivisionError", "TypeError", "AttributeError", "StopIteration")}


class SkillTimeout(Exception):
    pass


class Deadline:
    """The robot as a skill sees it, refusing to advance once the run's wall time is spent."""

    def __init__(self, robot, deadline):
        self._robot, self._deadline = robot, deadline

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def advance(self, epoch, seconds):
        if time.monotonic() > self._deadline:
            raise SkillTimeout(f"the skill ran longer than {RUN_SECONDS:g} s")
        self._robot.advance(epoch, seconds)


class CustomSkills:
    """This task's customized skills: name -> {description, code, version, run}."""

    def __init__(self):
        self.skills = {}

    def reset(self):
        self.skills.clear()

    def names(self):
        return tuple(self.skills)

    def prompt_section(self):
        return harness.custom_section({name: {k: item[k] for k in ("description", "code", "version")}
                                       for name, item in self.skills.items()})

    def define(self, arguments):
        """Compile ARGUMENTS {name, description, code}; a skill result dict."""
        name, code = arguments.get("name"), arguments.get("code")
        description = arguments.get("description", "")
        if not isinstance(name, str) or not name or set(name) - NAME_CHARS or name in RESERVED:
            return {"skill": CUSTOMIZE, "code": "INVALID_ARGUMENT", "details": {},
                    "message": "name must be lowercase letters, digits or _, and not customize_skill or done"}
        if not isinstance(code, str) or not code.strip() or len(code) > MAX_CODE_CHARS or not isinstance(description, str):
            return {"skill": CUSTOMIZE, "code": "INVALID_ARGUMENT", "details": {},
                    "message": f"code must be non-empty Python under {MAX_CODE_CHARS} characters; description text"}
        namespace = {"__builtins__": dict(SAFE_BUILTINS), "np": np, "math": __import__("math")}
        try:
            exec(compile(code, f"<{name}>", "exec"), namespace)  # noqa: S102 -- the agent's skill, sandboxed names
        except Exception as error:  # noqa: BLE001 -- a broken definition is the agent's feedback
            return {"skill": CUSTOMIZE, "code": "INVALID_CODE", "details": {},
                    "message": f"{type(error).__name__}: {error}"[:500]}
        if not callable(namespace.get("run")):
            return {"skill": CUSTOMIZE, "code": "INVALID_CODE", "details": {}, "message": "the code must define run(env, **arguments)"}
        version = self.skills.get(name, {}).get("version", 0) + 1
        self.skills[name] = {"description": description[:300], "code": code, "version": version,
                             "run": namespace["run"], "namespace": namespace}
        return {"skill": CUSTOMIZE, "code": "OK", "details": {"name": name, "version": version},
                "message": f"{name} version {version} is ready; call it as a skill"}

    def run(self, name, arguments, context, frame):
        """Run customized skill NAME on the decision's FRAME through CONTEXT; a skill result dict."""
        item = self.skills[name]
        deadline = Deadline(context.observations, time.monotonic() + RUN_SECONDS)
        bounded = dataclasses.replace(context, observations=deadline, actions=deadline)
        bounded._skill_name = name
        env = harness.build_env(bounded, frame)
        item["namespace"]["__builtins__"]["print"] = env.log
        returned, error = None, None
        try:
            returned = item["run"](env, **arguments)
        except PhysicsInvalid as exc:
            return {"skill": name, "code": "PHYSICS_INVALID", "message": str(exc), "details": {}}
        except Exception as exc:  # noqa: BLE001 -- the skill's failure is its result
            frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == f"<{name}>"]
            where = f" (line {frames[-1].lineno})" if frames else ""
            error = f"{type(exc).__name__}: {exc}{where}"
        result = harness.custom_result(env, returned, error)
        return {"skill": name, **result}
