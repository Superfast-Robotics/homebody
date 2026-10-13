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
