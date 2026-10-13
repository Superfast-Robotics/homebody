"""call_coder: the agent asks a coding sub-agent (Codex) for a skill. The sub-agent works in a fresh
workspace holding the skill's current source, the task's library and the current observation; it edits
the skill and runs it on the live robot through two tools served from this process (check: load and
dry-run prepare; execute: run for real). A skill that executed with OK becomes the task's version of
that skill. Codex is sandboxed to its workspace; the call is bounded in wall time, checks and executes.
"""
import dataclasses
import inspect
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from homebody import harness
from homebody.primitives.actions import PhysicsInvalid
from homebody.session.skill_module import Loader, SkillLoadError, digest
from homebody.skills.contract import SkillFailure
from homebody.skills.registry import REGISTRY
from homebody.vlm.observations import model_observation, rounded

CALL = "call_coder"
NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
RESERVED = frozenset({CALL, "done"})
STATUSES = ("success", "failed", "infeasible")
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(STATUSES)},
        "summary": {"type": "string"},
        "state": {"type": "string"},
        "usage": {"type": "string"},
    },
    "required": ["status", "summary", "state", "usage"],
    "additionalProperties": False,
}
DISABLED_FEATURES = ("apps", "plugins", "remote_plugin", "multi_agent", "image_generation", "browser_use",
                     "browser_use_external", "computer_use", "in_app_browser", "tool_suggest", "skill_search",
                     "goals", "hooks", "worktrees")
MIN_EXECUTE_SECONDS = 20.0  # an execute needs at least this much of the call's time left
PLAN_CHARS = 1500
TEMPLATE = '''"""{name}: a new skill."""
from .contract import Result, SkillFailure

NEEDS = ("base_velocity", "arm_target", "grip", "advance", "stop")
FAILURES = ()
PROMPT = "{name}: <its arguments and what it does>."


def prepare(arguments, selected, current, planning):
    raise SkillFailure("INVALID_ARGUMENT", "not written yet")


def execute(plan, ctx):
    return Result("{name}", "OK", "done")
'''


def source_root() -> Path:
    return Path(inspect.getfile(harness)).resolve().parents[2]  # src/homebody/harness/__init__.py -> src


def refusal(code, message, **details):
    return {"skill": CALL, "code": code, "message": message, "details": details}


def short(value, limit=PLAN_CHARS):
    text = repr(brief(value))
    return text if len(text) <= limit else text[:limit] + "..."


def brief(value, depth=0):
    """VALUE with arrays shown by shape and long sequences by length, for the coder's reports."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"array{tuple(value.shape)}" if value.size > 8 else value.round(3).tolist()
    if depth > 2:
        return type(value).__name__
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: brief(getattr(value, f.name), depth + 1) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: brief(item, depth + 1) for key, item in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [brief(item, depth + 1) for item in value] if len(value) <= 6 else f"{len(value)} items"
    if isinstance(value, float):
        return round(value, 4)
    return value


class Coder:
    """This task's coder calls and the skill versions they produced."""

    def __init__(self, settings):
        self.settings = settings
        self.skills: dict[str, dict] = {}  # name -> {module, version, path}
        self.calls = 0
        self.task_dir: Path | None = None

    def reset(self, task_dir: Path | None = None):
        self.skills, self.calls, self.task_dir = {}, 0, task_dir

    def names(self):
        return tuple(self.skills)

    def module(self, name):
        return self.skills[name]["module"]

    def library(self) -> Path:
        path = self.task_dir / "coder" / "lib"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def current_source(self, name):
        if name in self.skills:
            return self.skills[name]["path"].read_text(), f"your task's version {self.skills[name]['version']}"
        if name in REGISTRY:
            return inspect.getsource(REGISTRY[name]), "the built-in skill"
        return TEMPLATE.format(name=name), "a new skill (template)"

    def call(self, arguments, session):
        """One coder call for the agent's ARGUMENTS on SESSION's robot; a skill result dict."""
        target = arguments.get("skill")
        if not isinstance(target, str) or not target or set(target) - NAME_CHARS or target in RESERVED:
            return refusal("INVALID_ARGUMENT", "skill must be a name of lowercase letters, digits or _")
        if not all(isinstance(arguments.get(key, ""), str) for key in ("goal", "tried", "success", "constraints")):
            return refusal("INVALID_ARGUMENT", "goal, tried, success and constraints are text")
        if not arguments.get("goal") or not arguments.get("success"):
            return refusal("INVALID_ARGUMENT", "give goal and success")
        if self.calls >= self.settings.calls:
            return refusal("CODER_LIMIT", f"this task already used its {self.settings.calls} coder calls")
        self.calls += 1
        call_dir = self.task_dir / "coder" / f"call-{self.calls}"
        return CoderCall(self, session, target, arguments, call_dir).run()

    def register(self, target, module, version, source):
        path = self.task_dir / "coder" / "skills" / f"{target}_v{version}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        self.skills[target] = {"module": module, "version": version, "path": path}


class CoderCall:
    """One sub-agent session: workspace, Codex process and the robot tools it calls."""

    def __init__(self, coder, session, target, arguments, call_dir):
        self.coder, self.session, self.settings = coder, session, coder.settings
        self.target, self.arguments, self.call_dir = target, arguments, call_dir
        self.workspace = call_dir / "workspace"
        self.skill_path = self.workspace / "skills" / f"{target}.py"
        self.lib_dir = self.workspace / "lib"
        self.loader = Loader(self.lib_dir)
        self.frozen_lib: dict[str, str] = {}
        self.checked: dict | None = None  # {digest, module, source, version}
        self.checks, self.executes, self.versions = 0, [], 0
        self.registered = None
        self.physics_error = None
        self.frame = None
        self.deadline = 0.0

    # --- workspace ---
    def build(self):
        self.skill_path.parent.mkdir(parents=True)
        source, origin = self.coder.current_source(self.target)
        self.skill_path.write_text(source)
        (self.call_dir / f"start_{self.target}.py").write_text(source)
        shutil.copytree(self.coder.library(), self.lib_dir)
        self.frozen_lib = {path.name: digest([path]) for path in self.lib_dir.glob("*.py")}
        (self.workspace / "AGENTS.md").write_text(harness.coder_instructions())
        self.observe()
        recent = [entry for entry in self.session.history if "decision" in entry][-harness.CODER_HISTORY_STEPS:]
        (self.workspace / "REQUEST.md").write_text(harness.coder_request(
            target=self.target, origin=origin, request=self.arguments,
            task=next((entry["operator_task"] for entry in reversed(self.session.history) if "operator_task" in entry), ""),
            recent=[json.dumps(rounded({k: entry[k] for k in ("decision", "result") if k in entry})) for entry in recent],
            source_root=str(source_root()), python=sys.executable,
            limits={"seconds": self.settings.seconds, "checks": self.settings.checks, "executes": self.settings.executes},
            library=sorted(self.frozen_lib)))
        (self.call_dir / "schema.json").write_text(json.dumps(OUTPUT_SCHEMA))

    def observe(self):
        """Capture the current frame and show it in observation/ (what clicks in arguments refer to)."""
        self.frame = self.session.observed.snapshot()
        folder = self.workspace / "observation"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        for index, image in enumerate(self.session.recorder.agent_images(self.frame), 1):
            shutil.copyfile(image, folder / f"image-{index}{image.suffix}")
        hands = {side: self.session.context.carried.get(side) for side in ("left", "right")}
        (folder / "state.json").write_text(json.dumps(rounded({
            "observation": model_observation(self.frame),
            "hands": {side: (carry or {}).get("state", "free") for side, carry in hands.items()}}), indent=1))

    # --- the robot tools ---
    def lib_problem(self):
        for name, value in self.frozen_lib.items():
            path = self.lib_dir / name
            if not path.is_file() or digest([path]) != value:
                return f"lib/{name} existed before this call and is read-only; add a new lib file instead"
        return None

    def code_digest(self):
        return digest([self.skill_path, *self.lib_dir.glob("*.py")])

    def check(self, arguments):
        if self.checks >= self.settings.checks:
            return {"error": f"no checks left ({self.settings.checks} per call)"}
        self.checks += 1
        problem = self.lib_problem()
        if problem:
            return {"error": problem}
        self.versions += 1
        self.loader = Loader(self.lib_dir)
        try:
            module = self.loader.load(self.target, self.skill_path, self.versions)
        except SkillLoadError as error:
            return {"loaded": False, "error": str(error)}
        planning = self.session.context.planning()
        try:
            current = self.session.context.observe(self.frame.epoch)
            plan = module.prepare(arguments, self.frame, current, planning)
        except SkillFailure as failure:
            outcome = {"code": failure.code, "message": str(failure)}
        except Exception as error:  # noqa: BLE001 -- the skill's own failure is the report
            outcome = {"code": "ERROR", "message": f"{type(error).__name__}: {error}", "where": self.where(error)}
        else:
            outcome = {"code": "OK", "plan": short(plan)}
        self.checked = {"digest": self.code_digest(), "module": module, "source": self.skill_path.read_text(),
                        "version": self.versions}
        return harness.check_report({"loaded": True, "version": self.versions, "prepare": outcome,
                                     "checks_left": self.settings.checks - self.checks})

    def execute(self, arguments):
        if len(self.executes) >= self.settings.executes:
            return {"error": f"no executes left ({self.settings.executes} per call)"}
        if self.checked is None or self.checked["digest"] != self.code_digest():
            return {"error": "call check on this exact code first (the files changed since the last check)"}
        if self.deadline - time.monotonic() < MIN_EXECUTE_SECONDS:
            return {"error": "too little of this call's time is left to run the skill"}
        module, version = self.checked["module"], self.checked["version"]
        try:
            result = asdict(self.session.runner.run(self.target, arguments, self.frame, skill=module))
        except PhysicsInvalid as error:
            self.physics_error = str(error)
            result = {"code": "PHYSICS_INVALID", "message": str(error), "details": {}}
        except Exception as error:  # noqa: BLE001 -- a crash in the skill is its result
            result = {"code": "ERROR", "message": f"{type(error).__name__}: {error}", "details": {"where": self.where(error)}}
        if result["code"] == "PHYSICS_INVALID":
            self.physics_error = result["message"]
        self.executes.append({"version": version, "arguments": arguments, "code": result["code"],
                              "message": result["message"][:300]})
        if result["code"] == "OK":
            self.registered = (module, version, self.checked["source"])
        if self.physics_error is None:
            self.observe()
        details = {key: value for key, value in result.get("details", {}).items() if key != "traceback"}
        return harness.execute_report({"version": version, "code": result["code"], "message": result["message"],
                                       "details": short(details, 800),
                                       "executes_left": self.settings.executes - len(self.executes)})

    def where(self, error):
        names = {str(self.skill_path), *(str(path) for path in self.lib_dir.glob("*.py"))}
        frames = [f for f in traceback.extract_tb(error.__traceback__) if f.filename in names]
        return [f"{Path(f.filename).name}:{f.lineno} {f.line}" for f in frames[-3:]]

    def serve(self, request):
        """The tool's answer, with the call's remaining wall time."""
        tool, arguments = request.get("tool"), request.get("arguments")
        handler = {"check": self.check, "execute": self.execute}.get(tool)
        if not isinstance(arguments, dict):
            answer = {"error": "arguments must be an object"}
        elif handler is None:
            answer = {"error": f"unknown tool {tool!r}"}
        elif self.physics_error is not None:
            answer = {"error": "the simulation became invalid; stop and report failed"}
        else:
            answer = handler(arguments)
        return {**answer, "seconds_left": max(0, round(self.deadline - time.monotonic()))}

    # --- the Codex process ---
    def command(self, socket_path):
        s = self.settings
        provider = f'{{name="coder", base_url="{s.base_url}", env_key="LITELLM_API_KEY", wire_api="responses"}}'
        server = (f'{{command={json.dumps(sys.executable)}, args=["-m", "homebody.session.coder_mcp", '
                  f'{json.dumps(socket_path)}], env={{PYTHONPATH={json.dumps(str(source_root()))}}}, '
                  f'tool_timeout_sec={int(s.seconds)}, startup_timeout_sec=30, default_tools_approval_mode="approve"}}')
        command = [os.environ.get("HOMEBODY_CODER_CODEX", s.codex), "exec", "--ignore-user-config", "--ignore-rules",
                   "--skip-git-repo-check", "--ephemeral", "--json", "-s", "workspace-write",
                   "-C", str(self.workspace), "-m", s.model, "-c", f"model_reasoning_effort={s.effort}",
                   "-c", "model_provider=coder", "-c", f"model_providers.coder={provider}",
                   "-c", f"mcp_servers.robot={server}",
                   "-c", "sandbox_workspace_write.exclude_slash_tmp=true",
                   "-c", "sandbox_workspace_write.exclude_tmpdir_env_var=true",
                   "--output-schema", str(self.call_dir / "schema.json"),
                   "-o", str(self.call_dir / "last_message.json")]
        for feature in DISABLED_FEATURES:
            command += ["-c", f"features.{feature}=false"]
        return [*command, harness.CODER_START]

    def run(self):
        started = time.monotonic()
        self.build()
        root = Path.home() / ".homebody-coder"  # Codex refuses a home under /tmp
        root.mkdir(exist_ok=True)
        home = Path(tempfile.mkdtemp(prefix="c-", dir=root))
        socket_path = str(home / "robot.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)
        self.deadline = started + self.settings.seconds
        status, process = None, None
        try:
            with open(self.call_dir / "events.jsonl", "wb") as events, open(self.call_dir / "stderr.log", "wb") as errors:
                env = {**os.environ, "CODEX_HOME": str(home)}
                process = subprocess.Popen(self.command(socket_path), stdin=subprocess.DEVNULL, stdout=events,
                                           stderr=errors, env=env, start_new_session=True)
                status = self.loop(process, server)
        except OSError as error:
            status = f"error: {error}"
        finally:
            server.close()
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            shutil.rmtree(home, ignore_errors=True)
        return self.finish(status, time.monotonic() - started)

    def loop(self, process, server):
        while process.poll() is None:
            if self.physics_error is not None:
                return "physics"
            if time.monotonic() > self.deadline:
                return "timeout"
            ready, _, _ = select.select([server], [], [], 0.0)
            if not ready:
                self.session.heartbeat()
                continue
            connection, _ = server.accept()
            with connection:
                connection.settimeout(10)
                data = b""
                while not data.endswith(b"\n"):
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                try:
                    request = json.loads(data or b"{}")
                except json.JSONDecodeError:
                    request = {}
                answer = self.serve(request) if request else {"error": "malformed request"}
                reply = json.dumps(answer, default=str)
                with open(self.call_dir / "tools.jsonl", "a") as log:
                    log.write(json.dumps({"request": request, "answer": json.loads(reply)}, default=str) + "\n")
                connection.sendall((reply + "\n").encode())
        return "exited"

    def finish(self, how, seconds):
        report = self.final_message()
        if self.registered is not None:
            module, version, source = self.registered
            self.coder.register(self.target, module, version, source)
            for path in self.lib_dir.glob("*.py"):
                if path.name not in self.frozen_lib:
                    shutil.copyfile(path, self.coder.library() / path.name)
        ok_runs = [run for run in self.executes if run["code"] == "OK"]
        if how == "physics":
            code = "PHYSICS_INVALID"
        elif how == "timeout":
            code = "CODER_TIMEOUT"
        elif report is None:
            code = "CODER_ERROR"
        elif report["status"] == "success" and ok_runs:
            code = "OK"
        elif report["status"] == "infeasible":
            code = "CODER_INFEASIBLE"
        else:
            code = "CODER_FAILED"
        summary = (report or {}).get("summary") or {"timeout": f"no answer within {self.settings.seconds:g} s",
                                                     "physics": self.physics_error}.get(how, f"the coder ended without an answer ({how})")
        if report and report["status"] == "success" and not ok_runs:
            summary = "claimed success without an execute that returned OK: " + summary
        details = {"status": (report or {}).get("status", how), "state": (report or {}).get("state", ""),
                   "usage": (report or {}).get("usage", ""), "executes": self.executes, "checks": self.checks,
                   "seconds": round(seconds, 1),
                   "skill_version": self.coder.skills[self.target]["version"] if self.target in self.coder.skills else None}
        (self.call_dir / "result.json").write_text(json.dumps({"code": code, "message": summary, "details": details},
                                                              indent=1, default=str))
        return {"skill": CALL, "code": code, "message": summary[:600], "details": details}

    def final_message(self):
        path = self.call_dir / "last_message.json"
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(report, dict) or report.get("status") not in STATUSES:
            return None
        return {key: str(report.get(key, ""))[:1500] for key in ("status", "summary", "state", "usage")}
