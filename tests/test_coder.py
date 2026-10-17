"""Customized skill modules load under the import allowlist, and the MCP bridge forwards tool calls."""
import inspect
import io
import json
import socket
import threading

import pytest

from homebody.session import coder_mcp
from homebody.session.skill_module import Loader, SkillLoadError, digest
from homebody.skills.registry import REGISTRY


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_a_built_in_skills_source_loads_as_a_customized_version(tmp_path):
    for name in ("pick", "place", "navigate"):
        path = write(tmp_path / "skills" / f"{name}.py", inspect.getsource(REGISTRY[name]))
        module = Loader(tmp_path / "lib").load(name, path, 1)
        assert module.PROMPT == REGISTRY[name].PROMPT and module is not REGISTRY[name]


def test_lib_modules_import_under_the_same_rules(tmp_path):
    write(tmp_path / "lib" / "shapes.py", "import math\n\ndef area(r):\n    return math.pi * r * r\n")
    skill = write(tmp_path / "skills" / "measure.py", (
        "from lib.shapes import area\nimport lib.shapes\nfrom .contract import Result\n"
        "NEEDS = ()\nFAILURES = ()\nPROMPT = 'measure: r.'\n"
        "def prepare(arguments, selected, current, planning):\n    return area(arguments['r']) + lib.shapes.area(0)\n"
        "def execute(plan, ctx):\n    return Result('measure', 'OK', str(plan))\n"))
    module = Loader(tmp_path / "lib").load("measure", skill, 1)
    assert module.prepare({"r": 1.0}, None, None, None) == pytest.approx(3.14159, abs=1e-4)


@pytest.mark.parametrize("line, reason", [
    ("import os", "not allowed"), ("import subprocess", "not allowed"),
    ("from homebody.backends import mujoco", "not allowed"), ("from homebody.session import session", "not allowed"),
    ("open('/etc/passwd')", "open"), ("from lib.missing import x", "no lib/missing.py"),
    ("PROMPT = 'other: x.'", "PROMPT must start with"),
])
def test_forbidden_code_is_refused_with_its_reason(tmp_path, line, reason):
    base = ("from .contract import Result\nNEEDS = ()\nFAILURES = ()\nPROMPT = 'probe: x.'\n"
            "def prepare(arguments, selected, current, planning):\n    return None\n"
            "def execute(plan, ctx):\n    return Result('probe', 'OK', '')\n")
    path = write(tmp_path / "skills" / "probe.py", base + line + "\n")
    with pytest.raises(SkillLoadError, match=reason):
        Loader(tmp_path / "lib").load("probe", path, 1)


def test_digest_changes_with_any_file(tmp_path):
    a, b = write(tmp_path / "a.py", "x = 1\n"), write(tmp_path / "b.py", "y = 1\n")
    before = digest([a, b])
    b.write_text("y = 2\n")
    assert digest([a, b]) != before and digest([b, a]) == digest([a, b])


def test_mcp_bridge_lists_tools_and_forwards_calls(tmp_path, monkeypatch):
    path = str(tmp_path / "robot.sock")
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen(1)
    received = []

    def answer():
        connection, _ = server.accept()
        with connection:
            received.append(json.loads(connection.recv(65536)))
            connection.sendall(b'{"prepare": {"code": "OK"}}\n')

    thread = threading.Thread(target=answer)
    thread.start()
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    coder_mcp.handle(path, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "x"}})
    coder_mcp.handle(path, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    coder_mcp.handle(path, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    coder_mcp.handle(path, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "check", "arguments": {"arguments": {"hand": "left"}}}})
    thread.join(5)
    server.close()
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2, 3]
    assert replies[0]["result"]["protocolVersion"] == "x"
    assert [tool["name"] for tool in replies[1]["result"]["tools"]] == ["check", "execute"]
    assert received == [{"tool": "check", "arguments": {"hand": "left"}}]
    assert json.loads(replies[2]["result"]["content"][0]["text"]) == {"prepare": {"code": "OK"}}


def test_a_final_answer_wrapped_in_prose_is_still_read():
    from homebody.session.coder import answer_json
    assert answer_json('{"status": "failed", "summary": "s"}')["status"] == "failed"
    wrapped = 'Done.\n```json\n{"status": "success", "summary": "a {b}", "state": "", "usage": ""}\n```'
    assert answer_json(wrapped)["status"] == "success"
    assert answer_json("no answer here") is None


def test_a_final_answer_written_as_markdown_fields_is_still_read():
    from homebody.session.coder import answer_json
    text = ("\n\n**status:** `success`\n\n**summary:** Fixed `_admit_hidden` (line 451).\nIt now holds.\n\n"
            "**state:** Right hand holds the cup.\n\n**usage:** `pick {\"hand\": \"right\"}`")
    answer = answer_json(text)
    assert answer["status"] == "success"
    assert answer["summary"] == "Fixed `_admit_hidden` (line 451).\nIt now holds."
    assert answer["usage"] == 'pick {"hand": "right"}'
    assert answer_json("**summary:** no status given") is None


def test_a_final_answer_with_the_value_inside_the_bold_label_is_still_read():
    from homebody.session.coder import answer_json
    text = "\n\nHere's my final answer:\n\n**Status: failed**\n\n**Summary:**\n- Widened the grasp range.\n"
    answer = answer_json(text)
    assert answer["status"] == "failed"
    assert answer["summary"] == "- Widened the grasp range."
    assert answer_json("## Final Result\n\n**Status: success**\n\nHeld.")["status"] == "success"
    assert answer_json("**status**: `failed`\n\n**summary**: none")["status"] == "failed"
    assert answer_json("**Status:** failed (ran out of checks)")["status"] == "failed"


def test_a_skill_result_with_unusual_details_still_converts():
    from homebody.skills.contract import Result

    class Plan:  # a class object in the details: deep-copying it fails (mappingproxy)
        pass
    result = Result("probe", "OK", "done", {"plan": Plan, "rise": 0.1}).as_dict()
    assert result["code"] == "OK" and result["details"]["rise"] == 0.1 and result["details"]["plan"] is Plan


def test_the_coder_sees_the_head_image_in_its_first_prompt(tmp_path):
    from homebody.harness import CODER_START
    from homebody.session.coder import CoderCall
    from homebody.session.config import Coder as CoderSettings
    call = CoderCall.__new__(CoderCall)
    call.settings, call.workspace, call.call_dir = CoderSettings(), tmp_path / "workspace", tmp_path
    (call.workspace / "observation").mkdir(parents=True)
    command = call.command(str(tmp_path / "robot.sock"))
    assert "--image" not in command
    (call.workspace / "observation" / "image-1.png").write_bytes(b"png")
    command = call.command(str(tmp_path / "robot.sock"))
    index = command.index("--image")
    assert command[index + 1].endswith("observation/image-1.png") and command[index + 2].startswith("--")
    assert command[-1] == CODER_START
