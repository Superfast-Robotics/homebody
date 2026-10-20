"""End-to-end optimisation: Astra improves the code an agent session runs on (a profile's editable files, e.g. the
Dex1 skills and the agent harness), judged by full HomeBody agent sessions on a task suite scored from physics.

    python3 -m evolve.e2e_loop [--profile P] baseline COMMIT [--id b000]
    python3 -m evolve.e2e_loop [--profile P] run --rounds N [--strategy greedy|evolve]
    python3 -m evolve.e2e_loop [--profile P] test CID       the version on the test tasks (never used by a round)
    python3 -m evolve.e2e_loop [--profile P] legacy CID     the legacy tasks, reported apart from test
    python3 -m evolve.e2e_loop [--profile P] status

A profile is an [e2e.<P>] table of config.toml: the suite (train, val, test and legacy globs of scene packages, each
with a sequence.json of scoring targets), sessions per task, the session settings, the agent's model, the services
every host starts and checks, the editable files, the Astra models, the search, the credit caps and the preregistered
acceptance margin. It runs the skill loops' machinery (skill_loop.py, evo_loop.py) with sessions in place of skill
episodes: an episode is one session of one task (TASK__rN). Its value is the share of the task's target objects
placed (settled, in a valid recording), less `disturbance_penalty` of one object per other product it disturbed,
floored at 0; its stage value (mean object stage / 3) is reported. Sessions of one task, and every task of one
shared layout (the profile's `clusters` prefixes), count as one independent unit in the standard errors.

Validation is development data, as in the skill loops: it decides acceptance (so the accepted version, the next
parent, follows it), but its numbers, tasks and sessions never reach the proposer; round winners are exploratory
until the untouched test confirms them. Test and legacy run only on their explicit commands.

Every task-repeat runs on the same host for every version ((task index + repeat) mod hosts), so compared sessions
share their host, and each version's sessions split evenly over the hosts. Before sessions start, the profile's
services are checked ready on every host and running the planner this checkout ships. A session that could not be
judged on its merits (not started, skipped after a stop, the model provider failing, a result that cannot be read
or scored) is an infrastructure failure: the evaluation stops without a verdict, and running the command again
resumes it, re-running only those sessions (results already written are kept and re-scored if needed).
"""
import argparse
import dataclasses
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import astra, score, skill_score, tiers
from .evo_loop import EvoLoop
from .loop import REPO, QuotaExhausted, git
from .remote import rsync, suite_tasks
from .skill_loop import COMPLETE, HERE, SPLITS, SkillLoop, coder_digest, measured, public_reason
from .workspace import copy_runs

WATCH_SECONDS = 60  # planner servers are watched (and restarted) this often while a batch runs
SESSION_TIMEOUT = 7200  # seconds run_jobs.sh allows a session: a timed-out or crashed session is charged all of it
# the planner's Python environment, listed without pip (it has none)
PACKAGES = ("import importlib.metadata as m, sys; print(sys.version.split()[0]); "
            "[print(d.metadata['Name'], d.version) for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower())]")
SESSION_CREDIT_CAP = 8.0  # a session's charge when its usage is unknown, where no profile gives the worst case
# Credit ceilings (the profile may override): about 1.5 times the largest seen in 811 recorded sessions (2026-10-20:
# 7,484 decisions, at most 0.67 credits each; 1,019 coder calls, at most 8.7) and in 182 Astra calls (at most 96).
DECISION_CEILING, CODER_CEILING, ASTRA_CALL_CEILING = 1.0, 13.0, 120.0
KEYFRAMES = 40           # head and wrist keyframes kept per failed train session in the workspace
COLLECTED = ("result.json", "usage.json", "score.out", "score.err", "skipped", "run.log", "exit_code", "events.jsonl",
             "outcome.json", "trace.json", "REQUEST.md", "tools.jsonl", "start_*.py", "last_message.json")
FINISHED = ("model_done", "step_limit")  # session statuses of a session that ran its course
CRASHED = ("error", "physics_invalid", "cancelled", "interrupted")  # session statuses its own code caused
INFRA_STATUS = ("provider_error",)  # the model provider failed: not the candidate's doing
STATUSES = FINISHED + CRASHED + INFRA_STATUS  # any other status is a record the loop cannot judge
RESCORABLE = ("unscored", "invalid result", "unmetered")  # a session ran but its record could not be read
ASSIGNMENT = "host = (task index + repeat) mod hosts; versions interleaved per host"  # part of the experiment
# what the proposer's source export leaves out: how held-out tasks are made, the loop itself, the original targets
EXPORT_EXCLUDES = ("evolve", "tools/warehouse_variants.py", "tools/scenegen.py", "tests/test_warehouse_variants.py",
                   "tests/test_scenegen_warehouse.py", "configs/warehouse_*.json")
# what a patch's added source lines may not do in the evaluated process
UNSAFE = {r"\bimport\s+(os|subprocess|socket|importlib|shutil|ctypes|multiprocessing|signal)\b": "imports a system module",
          r"\bfrom\s+(os|subprocess|socket|importlib|shutil|ctypes|multiprocessing|signal)\b": "imports a system module",
          r"\b(os|sys)\.(system|popen|exec\w*|spawn\w*|remove|unlink|rename|replace|rmdir|makedirs|mkdir|environ|kill)\b":
              "touches processes, files or the environment",
          r"\bopen\([^)]*['\"][wax]\+?b?['\"]": "opens a file for writing",
          r"\.(write_text|write_bytes|unlink|rmdir|mkdir|touch|rename|replace)\(": "writes the filesystem",
          r"\b(__import__|eval|exec|compile)\(": "runs dynamic code",
          r"\b(urllib|requests|http\.client|socket)\b": "reaches the network",
          r"\bsettings\.\w+(\.\w+)*\s*=(?!=)|\bsetattr\([^,]*settings": "writes the settings",
          r"\bobject\.__setattr__\(": "bypasses frozen settings"}


class InfrastructureFailure(RuntimeError):
    """Sessions failed for reasons outside the candidate's code: no verdict; resume after fixing the cause."""


BudgetExceeded = astra.BudgetExceeded  # the evaluation or the round reached its credit cap: no verdict
# reserve ARGV[3] credits as ARGV[2] in the ledger ARGV[1] (under its lock) when the ledger stays within ARGV[4]
LEDGER_RESERVE = """import fcntl, sys
path, key, amount, budget = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
with open(path + ".lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    last = {}
    for line in open(path):
        name, _, value = line.strip().rpartition(" ")
        try:
            if name:
                last[name] = float(value)
        except ValueError:
            continue
    total = sum(last.values())
    if total + amount > budget:
        print(f"refused {total:.3f}")
    else:
        with open(path, "a") as ledger:
            ledger.write(f"{key} {amount}\\n")
        print(f"reserved {total:.3f}")
"""


def episode_key(task, repeat):
    return f"{task}__{repeat}"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def finite(value, low=None, high=None):
    """VALUE as a finite float within [LOW, HIGH], else None: booleans, strings, NaN and infinities are no numbers."""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
    except OverflowError:  # an integer too large for a float
        return None
    if (low is not None and value < low) or (high is not None and value > high):
        return None
    return float(value)


def count(value):
    """VALUE as a count (an int >= 0, not a boolean), else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def session_credits(folder, cap):
    """(credits, known, readable) of one session's usage.json. A missing or unreadable record, one the runner marked
    failed, or credits that are not a finite number >= 0 is unreadable: charged CAP, and the session is scored again.
    A readable record with unmetered parts (a decision or coder call that reported no usage) is charged CAP (the
    worst case a session can cost) and marked unknown; it is judged on its outcome."""
    usage = read_json(folder / "usage.json")
    if not isinstance(usage, dict) or usage.get("failed") or finite(usage.get("credits"), 0) is None:
        return float(cap), False, False
    if usage.get("unmetered"):
        return float(cap), False, True
    return finite(usage["credits"], 0), True, True


def invalid_fields(result, objects, targets):
    """Why a scoring record cannot be judged (every field the loop reads, type-checked), or None."""
    if not isinstance(objects, dict) or (targets is not None and sorted(objects) != sorted(targets)):
        return f"it does not score the task's targets {sorted(targets or [])}: {sorted(objects) if isinstance(objects, dict) else objects!r}"
    if any(not isinstance(o, dict) or count(o.get("stage")) not in (0, 1, 2, 3) for o in objects.values()):
        return "an object has no stage 0-3"
    checks = {"stage": finite(result.get("stage", 0.0), 0, 3) is not None,
              "wall_seconds": result.get("wall_seconds") is None or finite(result["wall_seconds"], 0) is not None,
              "success": isinstance(result.get("success", False), bool),
              "disturbed": isinstance(result.get("disturbed", []), list)
              and all(isinstance(n, str) for n in result.get("disturbed", [])),
              "invalid": isinstance(result.get("invalid", []), list),
              "steps": count(result.get("steps", 0)) is not None,
              "customizations": count(result.get("customizations", 0)) is not None}
    bad = sorted(name for name, ok in checks.items() if not ok)
    return f"malformed {', '.join(bad)}" if bad else None


def session_row(task, repeat, folder, targets=None, penalty=0.0, cap=SESSION_CREDIT_CAP):
    """One collected session as an episode row: value (placed share less the disturbance penalty), stage, success,
    credits and seconds. "infra" marks a session that could not be judged on its merits (never credit, scored again
    or run again); "crashed" one its own code ended (an error status, a nonzero exit, a timeout): a failure that
    counts toward the crash share, charged its credits and, without a measured time, the whole timeout. A session
    whose usage could not be read is charged CAP and is infrastructure; one with unmetered parts is charged CAP.
    Never raises: any malformed record is an infrastructure row."""
    try:
        return judged_row(task, repeat, folder, targets, penalty, cap)
    except (TypeError, ValueError, AttributeError, KeyError, OSError) as error:
        return {"episode": episode_key(task, repeat), "task": task, "stage": 0.0, "placed": 0.0, "success": False,
                "credits": float(cap), "seconds": float(SESSION_TIMEOUT), "objects": {}, "disturbed": [],
                "invalid": [], "infra": True, "crashed": False, "status": "invalid result", "metered": False,
                "error": f"the session's record could not be read: {type(error).__name__}: {error}"}


def judged_row(task, repeat, folder, targets, penalty, cap):
    key = episode_key(task, repeat)
    exit_code = (folder / "exit_code").read_text().strip() if (folder / "exit_code").exists() else None
    credits, metered, readable = session_credits(folder, cap) if exit_code is not None else (0.0, True, True)
    log = folder / "run.log"
    tail = " | ".join(log.read_text(errors="replace").strip().splitlines()[-3:])[:300] if log.exists() else "no output"
    base = {"episode": key, "task": task, "stage": 0.0, "placed": 0.0, "success": False, "credits": credits,
            "seconds": float(SESSION_TIMEOUT), "objects": {}, "disturbed": [], "invalid": [], "infra": False,
            "crashed": False, "status": "?", "metered": metered}
    if (folder / "skipped").exists() or exit_code is None:
        why = (folder / "skipped").read_text().strip() if (folder / "skipped").exists() else "never ran"
        return {**base, "infra": True, "status": "not run", "error": f"session not run: {why}"}
    if not readable:
        return {**base, "infra": True, "status": "unmetered",
                "error": f"the session's usage could not be read; charged {cap:g} credits: {tail}"}
    result = read_json(folder / "result.json")
    if result is not None and not isinstance(result, dict):
        return {**base, "infra": True, "status": "invalid result", "error": "result.json is not a scoring record"}
    if result is None:
        if exit_code == "0":  # the session ended normally but its scoring failed: the scorer's problem
            err = (folder / "score.err").read_text(errors="replace")[-300:] if (folder / "score.err").exists() else ""
            return {**base, "infra": True, "status": "unscored", "error": f"scoring failed: {err}"}
        return {**base, "crashed": True, "status": "crashed" if exit_code != "124" else "timed out",
                "error": f"session exited {exit_code} without a scorable record: {tail}"}
    objects = result.get("objects")
    why = invalid_fields(result, objects, targets)
    if why:
        return {**base, "infra": True, "status": "invalid result", "error": f"the result cannot be judged: {why}"}
    status = result.get("status")
    if status is None or status == "":  # no outcome record: a crash when the session failed, else unjudgeable
        if exit_code != "0":
            return {**base, "crashed": True, "status": "crashed" if exit_code != "124" else "timed out",
                    "error": f"session exited {exit_code} without an outcome: {tail}"}
        return {**base, "infra": True, "status": "invalid result", "error": "the result records no session status"}
    if status not in STATUSES:
        return {**base, "infra": True, "status": "invalid result", "error": f"unknown session status {status!r}"}
    if status in INFRA_STATUS:
        return {**base, "infra": True, "status": status, "error": f"the model provider failed: {tail}"}
    disturbed = list(result.get("disturbed", []))
    placed = sum(o["stage"] == 3 for o in objects.values())
    crashed = status in CRASHED or exit_code != "0"
    seconds = finite(result.get("wall_seconds"), 0) or 0.0
    return {**base, "stage": float(result.get("stage", 0.0)), "placed_objects": placed,
            "placed": max(0.0, placed - penalty * len(disturbed)) / max(1, len(objects)),
            "success": result.get("success", False) and not crashed, "crashed": crashed, "status": status,
            "seconds": seconds if seconds > 0 and exit_code != "124" else float(SESSION_TIMEOUT),
            "steps": result.get("steps", 0), "coder_calls": result.get("customizations", 0), "objects": objects,
            "disturbed": disturbed, "invalid": list(result.get("invalid", [])),
            **({"error": f"exit {exit_code}, status {status}: {tail}"} if crashed else {})}


def read_split(split_dir, expected=(), targets=None, penalty=0.0, cap=SESSION_CREDIT_CAP):
    """episode key -> row for every EXPECTED (task, repeat) under SPLIT_DIR/<task>/<repeat>/ (a session that left
    nothing is an infrastructure row, never left out); TARGETS: {task: target ids} to validate results against."""
    split_dir = Path(split_dir)
    return {episode_key(task, repeat): session_row(task, repeat, split_dir / task / repeat,
                                                   (targets or {}).get(task), penalty, cap)
            for task, repeat in expected}


def summarise(rows, cluster):
    """A split's summary in the skill loops' shape (per-episode values for the paired comparison) plus placed shares
    by layout cluster."""
    if not rows:
        return {"episodes": 0, "errors": 0, "success_rate": 0.0, "value": 0.0, "placed": 0.0}
    n = len(rows)
    outcomes, layouts, by_task = {}, {}, {}
    for key, row in rows.items():
        outcome = ("infrastructure" if row["infra"] else "crashed" if row["crashed"] else
                   f"{row['status']}, placed {row['placed']:.0%}")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        placed, total = layouts.get(cluster(key), (0, 0))
        layouts[cluster(key)] = (placed + row.get("placed_objects", 0), total + max(1, len(row["objects"])))
        by_task.setdefault(row["task"], []).append(round(row["placed"], 3))
    return {"episodes": n, "errors": sum(r["crashed"] for r in rows.values()),
            "infra": sorted(k for k, r in rows.items() if r["infra"]),
            "success_rate": round(sum(r["success"] for r in rows.values()) / n, 3),
            "value": round(sum(r["stage"] for r in rows.values()) / 3 / n, 3),
            "placed": round(sum(r["placed"] for r in rows.values()) / n, 3),
            "disturbed": sum(len(r["disturbed"]) for r in rows.values()),
            "invalid_recordings": sum(bool(r["invalid"]) for r in rows.values()),
            "credits": round(sum(r["credits"] for r in rows.values()) / n, 3),
            "minutes": round(sum(r["seconds"] for r in rows.values()) / 60 / n, 1),
            "outcomes": dict(sorted(outcomes.items(), key=lambda item: -item[1])),
            "placed_by_layout": {c: f"{p}/{t}" for c, (p, t) in sorted(layouts.items())}, "by_task": by_task,
            "values": {k: r["stage"] / 3 for k, r in rows.items()},
            "placed_values": {k: r["placed"] for k, r in rows.items()},
            "held_values": {k: float(r["success"]) for k, r in rows.items()},
            "seconds": {k: r["seconds"] for k, r in rows.items()},
            "credit_values": {k: r["credits"] for k, r in rows.items()},
            "untimed": sorted(k for k, r in rows.items() if r["crashed"] or r["infra"])}


def layout_cluster(episode, prefixes):
    """The independent unit of a session: its task (repeats together), or the shared layout its name's first
    matching PREFIXES entry names (variants of one scanned scene are not independent samples)."""
    task = re.sub(r"__r\d+$", "", episode.split("/")[-1])
    return next((unit for prefix, unit in prefixes.items() if task.startswith(prefix)), task)


def absolute(path, base):
    return path if path.startswith("/") else f"{base}/{path}"


def host_of(task_index, repeat, hosts):
    """The host a task-repeat runs on, the same for every version (ASSIGNMENT)."""
    return (task_index + repeat) % hosts


def unsafe(added):
    """What the ADDED source lines do that a patch may not (UNSAFE), as {reason: first offending line}."""
    found = {}
    for line in added:
        code = line.split("#", 1)[0]
        for pattern, reason in UNSAFE.items():
            if reason not in found and re.search(pattern, code):
                found[reason] = line.strip()[:120]
    return found


def md5(*paths):
    return hashlib.md5(b"".join(Path(p).read_bytes() for p in paths)).hexdigest()[:12]


class E2EMixin:
    """The skill loops' bookkeeping, search and judging (SkillLoop, EvoLoop) over agent sessions."""
    TABLE = "e2e"
    FOLDER = "e2e"

    def __init__(self, profile=None):
        super().__init__(profile)
        if self.cfg.get("session_sandbox", True):  # sessions and candidate tests in bubblewrap (remote.SANDBOX)
            self.loop.remotes = [dataclasses.replace(r, sandbox=True) for r in self.loop.remotes]
            self.remote = self.loop.remotes[0]

    # --- the suite ---
    @property
    def repeats(self):
        return int(self.cfg.get("repeats", 2))

    @property
    def penalty(self):
        return float(self.cfg.get("disturbance_penalty", 0.25))

    @property
    def cap(self):
        """What a session reserves before it starts and is charged when its usage is unknown: the profile's
        session_credit_cap, else the worst case a session can cost (session_worst_case)."""
        return float(self.cfg.get("session_credit_cap") or self.session_worst_case())

    def session_worst_case(self):
        """The most a session can cost under its settings: every decision it may make (the step limit, plus the
        plan and its replans) at decision_credit_ceiling, and every coder call it may make at coder_credit_ceiling.
        A retry of a busy service is refused before generation and costs nothing."""
        settings = tomllib.loads((HERE / self.cfg["settings"]).read_text())
        try:
            steps, calls = int(settings["agent"]["max_steps"]), int(settings["coder"]["calls"])
        except KeyError as error:
            raise SystemExit(f"{self.cfg['settings']} must set [agent] max_steps and [coder] calls: the worst case a "
                             f"session can cost is computed from them") from error
        plan = settings.get("plan", {})
        decisions = steps + (1 + int(plan.get("replans", 0)) if plan.get("enabled") else 0)
        return round(decisions * float(self.cfg.get("decision_credit_ceiling", DECISION_CEILING))
                     + calls * float(self.cfg.get("coder_credit_ceiling", CODER_CEILING)), 2)

    def split_rows(self, folder, split):
        """The rows of SPLIT's expected sessions collected under FOLDER."""
        return read_split(folder, self.expected(split), self.targets(split), self.penalty, self.cap)

    def tasks(self, split):
        """The split's tasks, listed from the volume once per process or fingerprint refresh (never cached across
        runs: a changed suite is a changed experiment)."""
        cache = self.__dict__.setdefault("_tasks", {})
        if split not in cache:
            cache[split] = suite_tasks(self.remote, self.cfg["suite"][split])
        return cache[split]

    def expected(self, split):
        return [(t["name"], f"r{r}") for t in self.tasks(split) for r in range(self.repeats)]

    def episodes(self, split):
        return [episode_key(task, repeat) for task, repeat in self.expected(split)]

    def task_info(self, splits=("train",)):
        """{task: {"instruction", "names", "targets"}} of SPLITS' tasks, cached by the listing AND the content of
        every task's task_scene.json and targets (a task replaced at the same path is read afresh)."""
        listing = {t["name"]: (absolute(t["scene"], self.remote.base), absolute(t["targets"], self.remote.base))
                   for split in splits for t in self.tasks(split)}
        files = " ".join(shlex.quote(f) for scene, targets in sorted(listing.values())
                         for f in (f"{scene}/task_scene.json", targets))
        content = self.remote.ssh(f"md5sum {files} | md5sum | cut -c1-12").strip() if files else ""
        key = hashlib.md5((json.dumps(listing, sort_keys=True) + content).encode()).hexdigest()[:10]
        cache = self.work / f"task_info_{key}.json"
        if not cache.exists():
            script = ("import json,sys\nout={}\nfor name,(scene,targets) in json.loads(sys.argv[1]).items():\n"
                      " s=json.load(open(scene+'/task_scene.json'));q=json.load(open(targets))\n"
                      " steps=q.get('steps',[])\n"
                      " names=[str(x[k]) for x in steps for k in ('fixture_id','label','destination_label') if x.get(k)]\n"
                      " out[name]={'instruction':s.get('default_task',''),'names':names+list(s.get('movable_entities',[])),"
                      "'targets':sorted({str(x['fixture_id']) for x in steps} if steps else q.get('objects',{}))}\n"
                      "print(json.dumps(out))")
            cache.write_text(self.remote.ssh(f"python3 - {shlex.quote(json.dumps(listing))} << 'EOF'\n{script}\nEOF"))
        return json.loads(cache.read_text())

    def targets(self, split):
        return {task: info["targets"] for task, info in self.task_info((split,)).items()}

    def benchmark_names(self):
        """Train task, object and place names only: a screen fed held-out names could quote them back."""
        names = set()
        for task, info in self.task_info().items():
            names.update([task, *info["names"]])
        return names

    def cluster(self, episode):
        return layout_cluster(episode, self.cfg.get("clusters", {}))

    def judge(self, parent, child):
        return skill_score.judge(parent, child, self.objective, self.weights, self.speed, self.cluster,
                                 val_margin=self.cfg.get("val_margin"), val_z=float(self.cfg.get("val_z", 1.0)))

    def archived_rounds(self):
        """No round is taken as finished from the archive: only rounds.json says a round finished (an end-to-end
        work folder always has it from its first round on)."""
        return []

    def barren(self, entries):
        """The user's rule for end-to-end rounds: a first round with no accepted candidate ends the run."""
        return not any(e.get("status") == "accepted" for e in entries)

    def allowed(self, files):
        """Files outside the editable globs, and any the tier rules freeze (the agent loop, scorer, scenes)."""
        outside = super().allowed(files)
        return outside + [f for f in files if f not in outside and tiers.tier_of(f) == "frozen"]

    def screen(self, base, commit, files):
        """The skill loops' screens, then what the added code may not do in the evaluated process (UNSAFE)."""
        reason = super().screen(base, commit, files)
        if reason:
            return reason
        added = [line[1:] for line in git("diff", base, commit, "--", "src").splitlines()
                 if line.startswith("+") and not line.startswith("+++")]
        found = unsafe(added)
        return ("unsafe in the session process: " + "; ".join(f"{why} ({line})" for why, line in found.items())
                if found else None)

    def objective_text(self):
        return (f"placed objects: a session scores the share of its task's target objects the physics scorer finds "
                f"placed and settled (stage 3, in a valid recording), less {self.penalty:g} of an object for every "
                f"other product it disturbed (floored at 0); the stage value (0 untouched, 1 lifted, 2 in the region, "
                f"3 placed; mean / 3) is reported but does not decide")

    # --- the experiment's identity ---
    def services_check(self, start=True):
        """Every host's services started (START) and watched: a server that is down or fails its CUDA check is
        restarted (counted in self.restarts and logged), then each must be ready, or busy with a session, on this
        checkout's planner. {host: [(port, planner md5)]}, the same whatever the servers are doing."""
        expected, out = md5(REPO / "tools/curobo_server.py"), {}
        for host in self.hosts():
            for command in self.cfg.get("services", []) if start else ():
                host.ssh(command, timeout=900, check=False)
            for command in self.cfg.get("service_checks", []):
                lines = [line for line in host.ssh(command, timeout=900, check=False).split("\n") if line.strip()]
                restarted = [line for line in lines if " restarted" in line]
                if restarted:
                    self.restarts = getattr(self, "restarts", 0) + len(restarted)
                    self.log(f"{host.host}: planner servers restarted: {restarted}")
                state = [line for line in lines if " restarted" not in line]
                bad = [line for line in state if line.split()[1:2] not in (["ready"], ["busy"]) or not line.endswith(expected)]
                if not state or bad:
                    raise InfrastructureFailure(f"{host.host}: services not ready or not running planner {expected}: "
                                                f"{bad or 'no output'}")
                out[host.host] = sorted((line.split()[0], expected) for line in state)
        return out

    def watch_services(self, done, every=WATCH_SECONDS):
        """While a batch runs (until DONE is set), restart dead or broken planner servers every EVERY seconds."""
        while not done.wait(every):
            try:
                self.services_check(start=False)
            except InfrastructureFailure as error:  # sessions then fail as planner-unavailable; logged, not fatal
                self.log(f"planner watch: {error}")

    def policy(self):
        """The scheduling and statistical policy a verdict depends on."""
        return {"agent_model": self.cfg.get("agent_model", self.loop.cfg["agent"]["model"]), "repeats": self.repeats,
                "parallel": int(self.cfg.get("parallel", self.remote.parallel)), "hosts": [h.host for h in self.hosts()],
                "assignment": ASSIGNMENT, "objective": self.objective, "clusters": self.cfg.get("clusters", {}),
                "disturbance_penalty": self.penalty, "val_margin": self.cfg.get("val_margin"),
                "val_z": self.cfg.get("val_z", 1.0), "barren": "round 1 without an accepted candidate",
                "judge": {k: getattr(skill_score, k) for k in ("Z_DROP", "Z_RISE", "Z_RISE_SLOWER", "SLOWER",
                                                                 "CRASH_SHARE")}, "speed": self.speed}

    def setup_fingerprint(self, refresh=False):
        """The experiment id: every input a verdict assumes fixed. The runner, scorer, evaluator and acceptance code,
        the settings and model, the suite's membership and content, the robot, every host's Python environment, the
        planner's environment and services, the scheduling and statistical policy and the profile. Computed once per
        process; REFRESH measures everything again (the after-check of a baseline, the frozen check of a round)."""
        if refresh:
            self.__dict__.pop("_experiment", None)
            self.__dict__.pop("_tasks", None)
        if "_experiment" in self.__dict__:
            return self._experiment
        listing = {s: sorted(t["name"] for t in self.tasks(s)) for s in self.cfg["suite"]}
        folders = " ".join(sorted({absolute(t["scene"], self.remote.base) for s in self.cfg["suite"]
                                   for t in self.tasks(s)}))
        content = self.remote.ssh(
            f"(cd {self.remote.base}/assets && find -L robot -type f | sort | xargs md5sum; "
            f"find -L {folders} -type f | sort | xargs md5sum) | awk '{{print $1}}' | md5sum | cut -c1-12",
            timeout=1200).strip()
        environments = {h.host: h.ssh(f"({h.base}/.venv/bin/python -V; {h.base}/.venv/bin/pip freeze 2>/dev/null) "
                                      f"| md5sum | cut -c1-12").strip() for h in self.hosts()}
        planner = {h.host: h.ssh(f"cd {Path(h.base).parent}/curobo && (.venv/bin/python -c {shlex.quote(PACKAGES)}; "
                                 f"git rev-parse HEAD 2>/dev/null) | md5sum | cut -c1-12").strip() for h in self.hosts()}
        profile = measured(self.cfg)  # raising a credit cap is no new experiment
        parts = {"local": md5(HERE / "run_jobs.sh", HERE / "score_episode.py", REPO / "evaluation/evaluate.py",
                              HERE / self.cfg["settings"]),
                 "acceptance": md5(HERE / "skill_score.py"), "suite": listing, "content": content,
                 "environments": environments, "planner_environment": planner,
                 "services": self.services_check(), "policy": self.policy(), "profile": profile}
        self._experiment = hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]
        (self.work / f"experiment_{self._experiment}.json").write_text(json.dumps(parts, indent=1, default=str))
        return self._experiment

    def lineage(self, population):
        """The versions built on the latest baseline AND measured in today's experiment: an older measurement
        (another suite, settings, planner or policy) never becomes a parent or a crossover donor."""
        experiment = self.setup_fingerprint()
        return {i: e for i, e in super().lineage(population).items()
                if (e.get("manifest") or {}).get("setup", e.get("setup")) == experiment}

    # --- running sessions ---
    def results_root(self, cid):
        return f"{self.remote.work}/e2e/{self.cfg.get('workdir', self.skill)}/{cid}"

    def summaries(self, cid, splits=SPLITS):
        return {s: summarise(self.split_rows(self.work / "results" / cid / s, s), self.cluster) for s in splits}

    def round(self, number):
        self.round_dir = self.work / "rounds" / f"{'e' if isinstance(self, EvoLoop) else 's'}{number:03d}"
        return super().round(number)

    # --- credits: one persistent ledger per round (or per evaluation outside a round) ---
    def ledger(self, cids, splits):
        """The remote ledger file this evaluation charges: the round's when in a round, else the commands' own
        (the splits and versions it measures). It outlives batches and resumes."""
        round_dir = getattr(self, "round_dir", None)
        scope = round_dir.name if round_dir is not None else f"{'+'.join(splits)}-{'+'.join(sorted(cids))}"
        return f"{self.remote.work}/e2e/{self.cfg.get('workdir', self.skill)}/ledger-{scope}.txt"

    def committed(self, ledger):
        return ledger_total(self.remote.ssh(f"cat {ledger} 2>/dev/null; true"))

    def budget(self, sessions):
        """The cap of the ledger being charged: in a round, round_credit_budget (sessions and Astra calls alike);
        outside one, the evaluation's own allowance: evaluation_credit_budget, else every one of its SESSIONS at the
        worst case a session can cost."""
        if getattr(self, "round_dir", None) is not None:
            return float(self.cfg["round_credit_budget"])
        return float(self.cfg.get("evaluation_credit_budget") or sessions * self.cap)

    def check_budget(self, sessions, ledger, budget):
        """Refuse an evaluation the cap cannot pay for: the ledger's credits (earlier sessions and attempts, Astra)
        plus SESSIONS at the expected credits per session."""
        estimate = self.committed(ledger) + sessions * float(self.cfg.get("expected_session_credits", 3.0))
        if estimate > budget:
            raise BudgetExceeded(f"{sessions} sessions would bring {Path(ledger).name} to about {estimate:.0f} credits, "
                                 f"over its cap of {budget:g}; raise the cap deliberately to go on (results are kept)")

    def reserve(self, ledger, key, amount, budget):
        """Write KEY's AMOUNT into LEDGER (under its lock) when the ledger stays within BUDGET; else BudgetExceeded."""
        self.remote.ssh(f"mkdir -p {Path(ledger).parent} && touch {ledger}")
        verdict, _, total = self.remote.ssh(f"python3 - {shlex.quote(ledger)} {shlex.quote(key)} {amount} {budget} "
                                            f"<< 'EOF'\n{LEDGER_RESERVE}EOF").strip().partition(" ")
        if verdict != "reserved":
            raise BudgetExceeded(f"{key} ({amount:g} credits) would take {Path(ledger).name} past its cap of "
                                 f"{budget:g} ({total} committed); raise the cap deliberately to go on")

    def settle_charge(self, ledger, key, amount):
        """KEY's reservation back to 0 and KEY at the AMOUNT it cost, in one locked append."""
        self.remote.ssh(f"( flock 9; printf 'reserve:%s 0\\n%s %s\\n' {shlex.quote(key)} {shlex.quote(key)} "
                        f"{amount} >> {ledger} ) 9>> {ledger}.lock")

    def astra_call(self, prompt, cd, sandbox, out, **model):
        """Every Codex call of a round reserves astra_call_credit_ceiling in the round's ledger before it starts (a
        call that would take the round past round_credit_budget is not started: BudgetExceeded), and settles at the
        credits it recorded, failed and timed-out calls included."""
        if getattr(self, "round_dir", None) is None:
            return astra.codex(prompt, cd, sandbox, out, jail=self.jail(), **model)
        ledger, ceiling = self.ledger((), ()), float(self.cfg.get("astra_call_credit_ceiling", ASTRA_CALL_CEILING))
        key = f"astra:{Path(out).name}@{time.time_ns()}"
        self.reserve(ledger, f"reserve:{key}", ceiling, self.budget(0))
        try:
            return astra.codex(prompt, cd, sandbox, out, jail=self.jail(), **model)
        finally:
            self.settle_charge(ledger, key, astra.call_credits(out, ceiling))

    # --- collecting, re-scoring and re-running ---
    def collect(self, root, split, local):
        """The remote SPLIT under ROOT, copied afresh to LOCAL (the remote keeps every attempt)."""
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        rsync("-a", "--include=*/", *(f"--include={name}" for name in COLLECTED),
              "--include=**/workspace/skills/*.py", "--include=**/workspace/lib/*.py", "--exclude=*",
              "--prune-empty-dirs", f"{self.remote.host}:{root}/{split}/", f"{local}/")

    def rescore(self, root, split, sessions):
        """Measure the usage of, and score again on the remote, SESSIONS ('task/rN') that ran but whose usage or
        result could not be read; what still fails is re-run."""
        if not sessions:
            return
        model = self.cfg.get("agent_model", self.loop.cfg["agent"]["model"])
        targets = {t["name"]: absolute(t["targets"], self.remote.base) for t in self.tasks(split)}
        python, work = f"{self.remote.base}/.venv/bin/python", self.remote.work
        commands = [f"(cd {root}/{split}/{s} && {python} {work}/score_episode.py --usage session {model} > usage.new "
                    f"2>> score.err && mv usage.new usage.json; rm -f result.json; {python} {work}/score_episode.py "
                    f"session {targets[s.split('/')[0]]} {work}/evaluate.py {model} > score.out 2>> score.err "
                    f"&& cp score.out result.json)" for s in sorted(sessions)]
        self.remote.ssh("; ".join(commands) + "; true", timeout=1800)

    def retire(self, root, split, sessions):
        """Keep SESSIONS' earlier attempts beside them (task/rN.attempt-K) before they run again: their evidence and
        their charges stay."""
        if sessions:
            self.remote.ssh(f"cd {root}/{split} && for s in {' '.join(sorted(sessions))}; do n=1; "
                            f"while [ -e $s.attempt-$n ]; do n=$((n+1)); done; mv $s $s.attempt-$n; done")

    def settle(self, root, split, local):
        """Collect SPLIT, score again what ran but could not be read, collect again. The rows by state:
        {"rerun": sessions that ran and still cannot be judged, "run": sessions never run}."""
        self.collect(root, split, local)
        rows = self.split_rows(local, split)
        path = {key: f"{row['task']}/{key.rsplit('__', 1)[1]}" for key, row in rows.items()}
        rescore = [path[k] for k, r in rows.items() if r["infra"] and r["status"] in RESCORABLE]
        if rescore:
            self.rescore(root, split, rescore)
            self.collect(root, split, local)
            rows = self.split_rows(local, split)
        infra = {k: r for k, r in rows.items() if r["infra"]}
        return {"run": sorted(path[k] for k, r in infra.items() if r["status"] == "not run"),
                "rerun": sorted(path[k] for k, r in infra.items() if r["status"] != "not run")}

    def evaluate_many(self, items, splits=SPLITS, videos=False):
        """Every pending session of every version in ITEMS ((cid, checkout, only, cand)) in one queue per host, each
        task-repeat on its fixed host, versions interleaved; each version's splits are collected, checked (an
        infrastructure failure stops without a verdict) and summarised. A split already complete under the same
        commit and experiment is reused; an interrupted one resumes: sessions never run are run, sessions whose
        usage or result could not be read are scored again, and those that still cannot be judged (or whose model
        provider failed) run again with their earlier attempt kept. Every session reserves its cap in the ledger
        before it starts. {cid: summaries}."""
        experiment, results, pending = self.setup_fingerprint(), {}, []
        hosts = self.hosts()
        per_host = [[] for _ in hosts]
        settings = self.cfg["settings"]
        rsync("-a", str(HERE / "run_jobs.sh"), str(HERE / "score_episode.py"), str(REPO / "evaluation/evaluate.py"),
              str(HERE / "episode_settings.toml"), str(HERE / settings), f"{self.remote.host}:{self.remote.work}/")
        for cid, checkout, _, cand in items:
            local, root = self.work / "results" / cid, self.results_root(cid)
            identity = {"commit": git("rev-parse", "HEAD", cwd=checkout), "experiment": experiment,
                        "repeats": self.repeats}
            todo = [s for s in splits if not ((local / s / COMPLETE).exists()
                                              and read_json(local / s / "identity.json") == identity)]
            if not todo:
                results[cid] = self.summaries(cid, splits)
                continue
            cand = cand or self.cand(cid)
            self.loop.remote.push_candidate(cand, checkout)
            for split in todo:  # a split measured under another identity starts over; one interrupted resumes
                marker = f"{root}/{split}/identity.json"
                stale = self.remote.ssh(f"cat {marker} 2>/dev/null; true").strip()
                if stale and read_json_text(stale) != identity:
                    self.remote.ssh(f"rm -rf {root}/{split}")
                self.remote.ssh(f"mkdir -p {root}/{split} && echo {shlex.quote(json.dumps(identity))} > {marker}")
                state = self.settle(root, split, local / split)
                self.retire(root, split, state["rerun"])
                waiting = set(state["run"] + state["rerun"])
                for index, task in enumerate(self.tasks(split)):
                    for r in range(self.repeats):
                        if f"{task['name']}/r{r}" not in waiting:
                            continue
                        per_host[host_of(index, r, len(hosts))].append(
                            (index, r, f"{self.remote.work}/cands/{cand} {absolute(task['scene'], self.remote.base)} "
                                       f"{absolute(task['targets'], self.remote.base)} {root}/{split}/{task['name']}/r{r}"))
            pending.append((cid, local, root, identity, todo))
        if not pending:
            return results
        sessions = sum(map(len, per_host))
        ledger = self.ledger([cid for cid, *_ in pending], splits)
        self.remote.ssh(f"mkdir -p {Path(ledger).parent} && touch {ledger}")
        budget = self.budget(len(items) * sum(len(self.expected(s)) for s in splits))  # all of it, resumes alike
        self.check_budget(sessions, ledger, budget)
        started = time.monotonic()
        batch = f"{self.remote.work}/e2e/{self.cfg.get('workdir', self.skill)}/batch-{pending[0][0]}-{len(pending)}"
        stop = f"{batch}/STOP"
        self.remote.ssh(f"mkdir -p {batch} && rm -f {stop}")
        parallel, model = int(self.cfg.get("parallel", self.remote.parallel)), self.cfg.get(
            "agent_model", self.loop.cfg["agent"]["model"])

        def run_share(index):
            host, share = hosts[index], [line for *_, line in sorted(per_host[index], key=lambda x: (x[0], x[1]))]
            if not share:
                return
            jobs = f"{batch}/jobs{index}.txt"
            host.ssh(f"cat > {jobs} << 'EOF'\n" + "\n".join(share) + "\nEOF")
            waves = -(-len(share) // parallel)
            host.ssh(f"bash {host.work}/run_jobs.sh {host.base} {parallel} {jobs} {host.work} {model} "
                     f"{host.work}/{settings} {stop} {budget} {self.cap} {ledger} {int(host.sandbox)}",
                     timeout=waves * (SESSION_TIMEOUT + 300) + 600)

        if sessions:
            self.log(f"dispatching {sessions} sessions ({', '.join(f'{h.host} {len(q)}' for h, q in zip(hosts, per_host))}), "
                     f"{self.committed(ledger):.1f} credits committed in {Path(ledger).name}, cap {budget:g}")
            self.restarts, done = 0, threading.Event()
            watcher = threading.Thread(target=self.watch_services, args=(done,), daemon=True)
            watcher.start()
            try:
                with ThreadPoolExecutor(len(hosts)) as pool:
                    list(pool.map(run_share, range(len(hosts))))
            finally:
                done.set()
                watcher.join()
            self.remote.ssh(f"echo {self.restarts} > {batch}/planner_restarts")
            self.log(f"planner servers restarted {self.restarts} times during the batch")
        seconds = time.monotonic() - started
        stopped = self.remote.ssh(f"cat {stop} 2>/dev/null; true").strip()
        for cid, local, root, identity, todo in pending:
            unjudged = {split: self.settle(root, split, local / split) for split in todo}
            if score.quota_hit(local) or "usage limit" in stopped:
                raise QuotaExhausted(f"the model's usage limit was hit while evaluating {cid}; resume once it resets")
            left = {s: state["run"] + state["rerun"] for s, state in unjudged.items() if state["run"] or state["rerun"]}
            if stopped and any(state["run"] for state in unjudged.values()):
                raise BudgetExceeded(f"evaluation stopped ({stopped}); sessions of {cid} not run: {left}; resume "
                                     f"deliberately")
            if left:
                raise InfrastructureFailure(f"{cid}: sessions that could not be judged: {left}; fix the cause and run "
                                            f"the command again (they run again, earlier attempts kept)")
            local.mkdir(parents=True, exist_ok=True)
            with (local / "planner_restarts.txt").open("a") as record:  # the batch's planner restarts, all versions
                record.write(f"{Path(batch).name} {getattr(self, 'restarts', 0)}\n")
            for split in todo:
                (local / split / "identity.json").write_text(json.dumps(identity))
                (local / split / COMPLETE).write_text("done\n")
            summaries = results[cid] = self.summaries(cid, splits)
            spent = sum(sum(s["credit_values"].values()) for s in summaries.values() if s.get("credit_values"))
            self.log(f"{cid}: " + ", ".join(f"{s} placed {v['placed']:.3f} value {v['value']:.3f} success "
                                             f"{v['success_rate']:.2f} disturbed {v['disturbed']} by layout "
                                             f"{v.get('placed_by_layout', {})}" for s, v in summaries.items())
                     + f"; {spent:.1f} session credits ({seconds:.0f} s for {len(pending)} version"
                     + f"{'s' if len(pending) > 1 else ''})")
        self.log(f"{Path(ledger).name}: {self.committed(ledger):.1f} credits committed (sessions, earlier attempts, Astra)")
        return results

    def audit(self, cid, summaries):
        for split in SPLITS:
            s = summaries.get(split, {})
            self.log(f"{cid} {split}: {s.get('episodes', 0)} sessions of {len(self.expected(split))}, "
                     f"{s.get('errors', 0)} crashed, {len(s.get('infra', []))} infrastructure, "
                     f"{sum((s.get('credit_values') or {}).values()):.1f} credits")

    def smoke(self, cid, path):
        """No trial session: it would cost credits, and a crashing candidate is rejected by its crash share."""
        return None

    def init(self, commit, sample=8):
        raise SystemExit("init is for a skill benchmark; an end-to-end profile starts from a baseline")

    def remeasure(self, ids):
        raise SystemExit("remeasure is for a skill benchmark")

    def test(self, cid, split="test"):
        """CID on the test (or legacy) tasks, in their own result folders: train and val evidence is untouched."""
        entry = next(e for e in reversed(self.archive()) if e["id"] == cid)
        path = self.loop.worktree(self.cand(cid), entry["commit"])
        result = self.evaluate(cid, path, splits=(split,))[split]
        self.log(f"{cid} on {split}: placed {result['placed']:.3f}, value {result['value']:.3f}, success "
                 f"{result['success_rate']:.2f}, disturbed {result['disturbed']}, by layout "
                 f"{result.get('placed_by_layout')}, {result['credits']} credits and {result['minutes']} min per session")
        return result

    # --- what the proposer sees (train only, from a source export) ---
    def export(self, path, out):
        """A plain copy of the checkout PATH for Codex to work in: no git history, none of EXPORT_EXCLUDES, with a
        throwaway repository of its own so its changes can be listed. Returns the copy's folder."""
        folder = out.parent / f"{out.stem}.src"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", "HEAD"], cwd=path, check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(folder)], input=archive, check=True)
        for pattern in EXPORT_EXCLUDES:
            for match in folder.glob(pattern):
                shutil.rmtree(match) if match.is_dir() else match.unlink()
        for command in (["init", "-q"], ["add", "-A"], ["-c", "user.name=export", "-c", "user.email=export@localhost",
                                                         "commit", "-q", "-m", "export"]):
            git(*command, cwd=folder)
        return folder

    def jail(self):
        """Codex runs in a bubblewrap jail (astra.jailed) unless the profile sets proposer_jail = false: it sees its
        source export and the round's folder (the train workspace, its own notes), never other rounds, results, the
        archive, the repository or its Codex history."""
        return [] if self.cfg.get("proposer_jail", True) else None

    def codex_write(self, prompt, path, out, **model):
        """The proposer writes in a source export; its changes are copied back into the worktree PATH, where the
        screens judge them like any patch."""
        folder = self.export(path, out)
        note = self.astra_call(prompt.replace(str(path), str(folder)), folder, "workspace-write", out, **model)
        git("add", "-A", cwd=folder)
        for line in git("diff", "--cached", "--name-status", "--no-renames", cwd=folder).splitlines():
            status, name = line.split("\t", 1)
            target = Path(path) / name
            if status == "D":
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder / name, target)
        return note

    def codex_read(self, prompt, path, out, **model):
        folder = self.export(path, out)
        return self.astra_call(prompt.replace(str(path), str(folder)), folder, "read-only", out, **model)

    def run_table(self, cid):
        rows = self.split_rows(self.work / "results" / cid / "train", "train")
        lines = ["| session | placed | stage | status | object outcomes | disturbed | coder calls | credits | min |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for key, row in sorted(rows.items()):
            objects = "; ".join(f"{n}: {o.get('stage')}{' dropped' if o.get('dropped') else ''}"
                                f"{' (' + str(o['reason']) + ')' if o.get('reason') else ''}" for n, o in row["objects"].items())
            lines.append(f"| {key} | {row['placed']:.2f} | {row['stage']:.2f} | {row['status']} | "
                         f"{objects or row.get('error', '-')[:200]} | {', '.join(row['disturbed']) or '-'} | "
                         f"{row.get('coder_calls', 0)} | {row['credits']:.1f} | {row['seconds'] / 60:.1f} |")
        return lines, rows

    def workspace(self, round_dir, best):
        ws = round_dir / "workspace"
        if ws.exists():
            shutil.rmtree(ws)
        (ws / "runs").mkdir(parents=True)
        editable = ", ".join(p.format(skill=self.skill) for p in self.editable)
        brief = (HERE / "prompts" / f"{self.prompts}_brief.md").read_text().format(
            objective=self.objective_text(), editable=editable, repeats=self.repeats)
        (ws / "BRIEF.md").write_text(brief + (f"\n## This robot\n\n{self.brief_note.strip()}\n" if self.brief_note else ""))
        info, summary = self.task_info(), self.summaries(best["id"], ("train",))["train"]
        table, rows = self.run_table(best["id"])
        lines = [f"# Train sessions of version {best['id']}", "",
                 f"Placed share {summary['placed']:.3f}, stage value {summary['value']:.3f}, task success "
                 f"{summary['success_rate']:.2f}, {summary['disturbed']} other products disturbed, "
                 f"{summary['credits']} credits and {summary['minutes']} min per session.",
                 f"Placed objects by layout: {json.dumps(summary.get('placed_by_layout', {}))}",
                 f"Outcomes: {json.dumps(summary['outcomes'])}", "", "## Tasks", ""]
        lines += [f"- **{t['name']}**: {info.get(t['name'], {}).get('instruction', '')}" for t in self.tasks("train")]
        (ws / "results.md").write_text("\n".join(lines + ["", "## Sessions", ""] + table) + "\n")
        train = self.work / "results" / best["id"] / "train"
        if train.exists():
            copy_runs(train, ws / "runs")
            (ws / "coder_edits.md").write_text(coder_digest(train))
        for key, row in rows.items():
            out = ws / "runs" / row["task"] / key.rsplit("__", 1)[1]
            out.mkdir(parents=True, exist_ok=True)
            self.skill_calls(train / row["task"] / key.rsplit("__", 1)[1], out)
            if row["placed"] < 1:
                self.session_keyframes(best["id"], row["task"], key.rsplit("__", 1)[1], out)
        history = [f"## {e['id']} (round {e.get('round')}): {e['status']}\n{public_reason(e)}\n{(e.get('notes') or '')[:1500]}"
                   for e in self.archive()]
        (ws / "history.md").write_text("# Earlier versions\n\n" + "\n\n".join(history) + "\n")
        return ws

    @staticmethod
    def skill_calls(source, out):
        """Every skill call's trace (stages, plan outline, result) of one session as skill_calls.md."""
        lines = []
        for path in sorted(source.rglob("trace/call-*/trace.json")):
            trace = json.loads(path.read_text())
            result = trace.get("result") or {}
            lines.append(f"## {path.parent.name}: {trace.get('skill')} ({trace.get('side') or 'no hand'}): "
                         f"{result.get('code')}: {str(result.get('message'))[:300]}")
            lines.append(f"plan: {trace.get('plan') or 'none recorded'}")
            lines += [f"- t={s['time']:.2f} s {s['stage']}: holding {s.get('carried') or 'nothing'}; "
                      f"{', '.join(x for x in (s.get('head'), s.get('wrist')) if x)}" for s in trace.get("stages", [])]
        if lines:
            (out / "skill_calls.md").write_text("# Skill calls (keyframes in keyframes/)\n\n" + "\n".join(lines) + "\n")

    def session_keyframes(self, cid, task, repeat, out):
        """The session's skill-stage keyframes from the remote results, the last KEYFRAMES of them."""
        frames = out / "keyframes"
        frames.mkdir(exist_ok=True)
        try:
            rsync("-a", "--prune-empty-dirs", "--include=*/", "--include=*.jpg", "--exclude=*",
                  f"{self.remote.host}:{self.results_root(cid)}/train/{task}/{repeat}/session/", f"{frames}/")
        except Exception:  # noqa: BLE001 -- keyframes are extra evidence
            return
        for path in sorted(frames.rglob("*.jpg")):
            path.rename(frames / f"{path.parent.name}-{path.name}")
        for path in sorted(frames.glob("*.jpg"))[:-KEYFRAMES]:
            path.unlink()
        for folder in sorted((p for p in frames.rglob("*") if p.is_dir()), reverse=True):
            folder.rmdir()

    def parent_results(self, ws, parent, summaries):
        path = ws / "parents" / f"{parent}.md"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            table, _ = self.run_table(parent)
            path.write_text(f"# {parent} on the train sessions: placed {summaries[parent]['train']['placed']:.3f}\n\n"
                            + "\n".join(table) + "\n")
        return path

    def skill_diff(self, a, b):
        roots = sorted({p.split("*")[0].rstrip("/") for p in self.editable if not p.startswith("tests")})
        return git("diff", a, b, "--", *roots)

    def activation(self, entry):
        """Where the candidate's new code reported acting (mechanism_active skill events its parent's sessions lack)
        and where a session's placed share changed: diagnosis evidence, not judged."""
        cid, parent = entry["id"], entry["parent"]

        def mechanisms(version, task, repeat):
            names = []
            for path in (self.work / "results" / version / "train" / task / repeat).rglob("events.jsonl"):
                if "coder" in path.parts:
                    continue
                for line in path.read_text(errors="replace").splitlines():
                    if '"mechanism_active"' in line:
                        names.append(str((json.loads(line).get("event") or {}).get("mechanism")))
            return names
        child = self.split_rows(self.work / "results" / cid / "train", "train")
        before = self.split_rows(self.work / "results" / parent / "train", "train")
        new = {}
        for key, row in child.items():
            repeat = key.rsplit("__", 1)[1]
            new[key] = [m for m in mechanisms(cid, row["task"], repeat)
                        if m not in set(mechanisms(parent, row["task"], repeat))]
        changed = sorted(k for k, r in child.items() if r["placed"] != before.get(k, {}).get("placed"))
        record = {"episodes": len(child), "new_activations": sum(map(len, new.values())),
                  "activated_in": sorted(k for k, v in new.items() if v), "changed": changed}
        self.log(f"{cid}: new code acted in {len(record['activated_in'])} train sessions, placed share changed in "
                 f"{len(changed)}")
        return record


def ledger_total(text):
    """A ledger's credits: the last amount of every key (a reservation is replaced by its settlement)."""
    last = {}
    for line in text.splitlines():
        key, _, amount = line.strip().rpartition(" ")
        if key:
            try:
                last[key] = float(amount)
            except ValueError:
                continue
    return round(sum(last.values()), 3)


def read_json_text(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


class E2ELoop(E2EMixin, SkillLoop):
    """Greedy: one patch per diagnosed mechanism from the best version."""


class E2EEvoLoop(E2EMixin, EvoLoop):
    """Evolutionary: the population search of evo_loop.py."""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="an [e2e.<profile>] of config.toml (default: [e2e] default)")
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("baseline")
    base.add_argument("commit")
    base.add_argument("--id", default="b000", help="archive id (b001, ... for a re-baseline after a pipeline change)")
    run = sub.add_parser("run")
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--strategy", choices=("greedy", "evolve"), help="the round's search (default: profile strategy)")
    run.add_argument("--no-stop", action="store_true", help="run every round even if round 1 is barren")
    run.add_argument("--candidates", type=int, help="versions a round proposes, this run only (default: evo_candidates)")
    ext = sub.add_parser("extend", help="more candidates for a finished round (evolutionary strategy)")
    ext.add_argument("round", type=int)
    ext.add_argument("--candidates", type=int, required=True)
    ext.add_argument("--wait", action="store_true", help="write now; evaluate once the round is recorded finished")
    sub.add_parser("test").add_argument("cid")
    sub.add_parser("legacy").add_argument("cid")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    loop = E2ELoop(args.profile)
    if (args.command == "run" and (args.strategy or loop.cfg.get("strategy", "greedy")) == "evolve"
            or args.command == "extend"):
        loop = E2EEvoLoop(args.profile)
    if args.command == "baseline":
        loop.baseline(args.commit, args.id)
    elif args.command in ("test", "legacy"):
        loop.test(args.cid, args.command)
    elif args.command == "extend":
        loop.extend(args.round, args.candidates, wait=args.wait)
    elif args.command == "run":
        if args.candidates:  # the run's search width, outside the profile and so outside the experiment identity
            loop.candidates = args.candidates
        loop.run(args.rounds, stop_rule=not args.no_stop)
    else:
        for entry in loop.archive():
            print(entry["id"], entry["status"], {s: (entry[s].get("placed"), entry[s].get("value")) for s in SPLITS
                                                 if s in entry}, entry.get("reason", "")[:120])


if __name__ == "__main__":
    main()
