"""The remote machine: candidate checkouts, scripted suite runs and their results."""
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
# What a candidate checkout needs to run; assets come from the base by symlink.
SYNCED = ("src", "tools", "tests", "configs", "evaluation", "vendor", "hardware", "pyproject.toml")
RETRIES, RETRY_WAIT = 5, 30  # a dropped network: attempts and seconds between them
NOT_CONNECTED = ("connect to host", "Could not resolve", "Network is unreachable", "Connection refused")


def run(command, **kwargs):
    """subprocess.run, retried while the host cannot be reached at all (ssh and rsync exit 255 and say so):
    nothing ran remotely, so a retry repeats nothing."""
    for attempt in range(RETRIES):
        result = subprocess.run(command, check=False, capture_output=True, text=True, **kwargs)
        if result.returncode != 255 or not any(m in result.stderr for m in NOT_CONNECTED) or attempt == RETRIES - 1:
            return result
        time.sleep(RETRY_WAIT)
    raise AssertionError("unreachable")


def rsync(*args):
    result = run(["rsync", *args])
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ["rsync", *args], result.stdout, result.stderr)


# Candidate code on a host runs with the whole filesystem read-only (the seed, the scorer, settings, scenes, other
# candidates and results), /dev and /tmp as they are (GPU devices; the planners' admission locks), and the network
# shared (the proxy, relay and planner tunnels on localhost), and its own process and IPC namespaces (a /proc of its
# own: no outside process is visible or signalled); run_jobs.sh adds the session's own folder writable.
SANDBOX = ("bwrap --die-with-parent --unshare-pid --unshare-ipc --ro-bind / / --dev-bind /dev /dev --proc /proc "
           "--bind /tmp /tmp --")


@dataclass(frozen=True)
class Remote:
    host: str
    root: str
    base: str
    work: str
    parallel: int
    sandbox: bool = False  # sessions and candidate tests run in bubblewrap (SANDBOX): a read-only root

    @classmethod
    def from_config(cls, path=HERE / "config.toml", host=None):
        r = tomllib.loads(path.read_text())["remote"]
        return cls(host or r["host"], r["root"], f"{r['root']}/{r['base']}", f"{r['root']}/{r['work']}", r["parallel"])

    @classmethod
    def all_from_config(cls, path=HERE / "config.toml"):
        """One Remote per evaluation host: [remote] hosts, or just host. They share the group volume, so a
        candidate's files and results are the same paths on each."""
        r = tomllib.loads(path.read_text())["remote"]
        return [cls.from_config(path, host) for host in r.get("hosts", [r["host"]])]

    def ssh(self, command, timeout=None, check=True):
        result = run(["ssh", "-o", "ServerAliveInterval=30", self.host, command], timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(f"remote command failed ({result.returncode}): {command[:200]}\n{result.stderr[-2000:]}")
        return result.stdout

    def push_candidate(self, cid, checkout):
        """Copy a local checkout's runnable code to the remote as candidate CID."""
        target = f"{self.work}/cands/{cid}"
        self.ssh(f"mkdir -p {target} && ln -sfn {self.base}/assets {target}/assets")
        sources = [str(Path(checkout) / name) for name in SYNCED if (Path(checkout) / name).exists()]
        rsync("-a", "--delete", "--exclude", "__pycache__", *sources, f"{self.host}:{target}/")
        return target

    def static_checks(self, cid):
        """Lint and the unit tests with the candidate's code. Returns (passed, output tail)."""
        cand = f"{self.work}/cands/{cid}"
        jail = f"{SANDBOX} " if self.sandbox else ""
        command = (f"cd {cand} && PYTHONPATH={cand}/src:{cand}/vendor PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl timeout 900 "
                   f"{jail}{self.base}/.venv/bin/python -m pytest -q -x -p no:cacheprovider tests 2>&1 | tail -n 15; "
                   f"echo PYTEST=${{PIPESTATUS[0]}}; {self.base}/.venv/bin/ruff check --no-cache src tests 2>&1 | tail -n 15; "
                   f"echo RUFF=${{PIPESTATUS[0]}}")
        out = self.ssh(command, timeout=1200, check=False)
        passed = "PYTEST=0" in out and "RUFF=0" in out
        return passed, out[-3000:]

    def run_suite(self, cid, tasks, repeats, label, model, settings=None):
        """Run one HomeBody session per task (dicts with name, scene, targets) REPEATS times with
        candidate CID's code (SETTINGS: a settings file name in the evolve folder, default episode_settings.toml).
        Returns the remote results directory."""
        cand = f"{self.work}/cands/{cid}"
        results = f"{self.work}/results/{cid}/{label}"
        lines = [f"{cand} {self.base}/{t['scene']} {self.base}/{t['targets']} {results}/{t['name']}/r{r}"
                 for t in tasks for r in range(repeats)]
        jobs = f"{results}/jobs.txt"
        self.ssh(f"rm -rf {results} && mkdir -p {results} && cat > {jobs} << 'EOF'\n" + "\n".join(lines) + "\nEOF")
        rsync("-a", str(HERE / "run_jobs.sh"), str(HERE / "score_episode.py"), str(HERE.parent / "evaluation/evaluate.py"),
              str(HERE / "episode_settings.toml"), *([str(HERE / settings)] if settings else []), f"{self.host}:{self.work}/")
        waves = -(-len(lines) // self.parallel)
        extra = f" {self.work}/{settings}" if settings else ""
        self.ssh(f"bash {self.work}/run_jobs.sh {self.base} {self.parallel} {jobs} {self.work} {model}{extra}",
                 timeout=waves * 7500 + 600)
        return results

    def collect(self, results, local):
        """Copy each run's result, log, session events and coder sessions to LOCAL."""
        if local.exists():  # a re-run must not inherit an earlier run's files
            shutil.rmtree(local)
        local.mkdir(parents=True)
        rsync("-a", "--include=*/", "--include=result.json", "--include=run.log", "--include=exit_code",
              "--include=events.jsonl", "--include=outcome.json", "--include=REQUEST.md", "--include=tools.jsonl",
              "--include=start_*.py", "--include=last_message.json", "--include=trace.json",
              "--include=**/workspace/skills/*.py", "--include=**/workspace/lib/*.py",
              "--exclude=*", "--prune-empty-dirs", f"{self.host}:{results}/", f"{local}/")
        return local


def suite_tasks(remote, patterns):
    """Expand suite patterns ("scene_glob" or "scene_glob::targets") on the remote base."""
    tasks = []
    for pattern in patterns:
        scene_glob, _, sequence = pattern.partition("::")
        listing = remote.ssh(f"cd {remote.base} && ls -d {scene_glob}").split()
        for scene in listing:
            name = Path(scene).name
            tasks.append({"name": name, "scene": scene, "targets": sequence or f"{scene}/sequence.json"})
    return tasks
