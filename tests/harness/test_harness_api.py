"""The harness reference agrees with the primitives, and the prompt pieces render."""
import inspect
import re
from types import SimpleNamespace

import numpy as np

from homebody import harness
from homebody.harness.env import Env, Target
from homebody.session.settings import Settings

CALL = re.compile(r"`env\.(\w+)\(([^`]*)\)")


def test_every_documented_call_exists_with_its_parameters():
    for name, arguments in CALL.findall(harness.api_doc()):
        assert hasattr(Env, name), f"API.md documents env.{name}, which env.py lacks"
        parameters = inspect.signature(getattr(Env, name)).parameters
        for argument in filter(None, (a.strip() for a in arguments.split(","))):
            key = argument.split("=")[0].strip().strip("*").strip()
            if key.isidentifier():
                assert key in parameters, f"env.{name} has no parameter {key!r} that API.md documents"


def test_instructions_render_settings_and_explain_call_coder():
    text = harness.agent_instructions(vars(Settings()))
    assert "call_coder" in text and "{" not in text


def test_coder_request_carries_the_request_and_its_limits():
    text = harness.coder_request(target="pick", origin="the built-in skill",
                                 request={"goal": "lift the cup", "success": "cup held", "tried": "", "constraints": ""},
                                 task="Clear the table", recent=['{"decision": 1}'], source_root="/src", python="/py",
                                 limits={"seconds": 300, "checks": 6, "executes": 2}, library=[])
    assert "skill `pick`" in text and "lift the cup" in text and "cup held" in text and "2 executes" in text
    assert "/src/homebody" in text and '- {"decision": 1}' in text
    assert "check" in harness.coder_instructions() and "execute" in harness.coder_instructions()


def test_click_selects_the_labelled_thing_under_it():
    labels = np.zeros((20, 30), int)
    labels[5:10, 10:20] = 7
    frame = SimpleNamespace(labels=labels)
    env = Env(ctx=None, frame=frame)
    assert env.target([500, 350]) == Target(7)
    assert env.target([10, 10]) is None
