"""The harness reference agrees with the env programs call, and the prompt assembles."""
import inspect
import re
from types import SimpleNamespace

import numpy as np

from homebody import harness
from homebody.harness.env import Env

CALL = re.compile(r"`env\.(\w+)\(([^`]*)\)")


def test_every_documented_call_exists_with_its_parameters():
    for name, arguments in CALL.findall(harness.api_doc()):
        assert hasattr(Env, name), f"API.md documents env.{name}, which env.py lacks"
        parameters = inspect.signature(getattr(Env, name)).parameters
        for argument in filter(None, (a.strip() for a in arguments.split(","))):
            key = argument.split("=")[0].strip().strip("*").strip()
            if key and key.isidentifier() and key not in ("side",):
                assert key in parameters, f"env.{name} has no parameter {key!r} that API.md documents"


def test_prompt_assembles_with_memory_and_history():
    memory = [{"id": "cup", "label": "Red cup", "location": "On the counter.", "anchor_xyz": [1.0, 2.0, 0.9],
               "portable": True}]
    text = harness.render_prompt("Return the red cup.", memory, [{"number": 1, "report": "Error: none"}], 4)
    assert "Return the red cup." in text and "cup: Red cup" in text and "Attempt 1" in text and "attempt 2" in text


def test_skill_pixels_are_plain_floats():
    frame = SimpleNamespace(labels=np.full((20, 30), 5), depth=np.ones((20, 30)), rgb=np.zeros((20, 30, 3)))
    env = Env(ctx=None, memory=[], labels={"cup": 5})
    pixel = env._pixel(frame, "cup")
    assert all(type(value) is float and 0 <= value <= 1000 for value in pixel)
