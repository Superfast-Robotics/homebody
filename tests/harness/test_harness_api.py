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


def test_instructions_render_settings_and_include_the_primitives():
    text = harness.agent_instructions(vars(Settings()))
    assert "customize_skill" in text and "env.target(click)" in text


def test_custom_section_shows_each_current_version():
    assert harness.custom_section({}) == ""
    text = harness.custom_section({"pick": {"description": "deeper grasp", "code": "def run(env):\n    pass\n", "version": 2}})
    assert "pick (version 2): deeper grasp" in text and "def run(env):" in text


def test_click_selects_the_labelled_thing_under_it():
    labels = np.zeros((20, 30), int)
    labels[5:10, 10:20] = 7
    frame = SimpleNamespace(labels=labels)
    env = Env(ctx=None, frame=frame)
    assert env.target([500, 350]) == Target(7)
    assert env.target([10, 10]) is None


def test_result_reports_code_message_and_prints():
    env = Env(ctx=None, frame=None)
    env.log("tried", 3)
    assert harness.custom_result(env, {"code": "NO_GRASP", "message": "walls"}, None) == {
        "code": "NO_GRASP", "message": "walls", "details": {"log": ["tried 3"]}}
    assert harness.custom_result(env, None, "KeyError: x (line 2)")["code"] == "SKILL_ERROR"
