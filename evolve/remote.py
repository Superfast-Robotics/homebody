"""The remote machine: candidate checkouts, scripted suite runs and their results."""
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
# What a candidate checkout needs to run; assets come from the base by symlink.
SYNCED = ("src", "tools", "tests", "configs", "evaluation", "vendor", "pyproject.toml")


@dataclass(frozen=True)
class Remote:
    host: str
    root: str
    base: str
    work: str
    parallel: int

    @classmethod
    def from_config(cls, path=HERE / "config.toml"):
        r = tomllib.loads(path.read_text())["remote"]
        return cls(r["host"], r["root"], f"{r['root']}/{r['base']}", f"{r['root']}/{r['work']}", r["parallel"])

    def ssh(self, command, timeout=None, check=True):
        result = subprocess.run(["ssh", "-o", "ServerAliveInterval=30", self.host, command], check=False,
                                capture_output=True,
                                text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(f"remote command failed ({result.returncode}): {command[:200]}\n{result.stderr[-2000:]}")
        return result.stdout

    def push_candidate(self, cid, checkout):
        """Copy a local checkout's runnable code to the remote as candidate CID."""
        target = f"{self.work}/cands/{cid}"
        self.ssh(f"mkdir -p {target} && ln -sfn {self.base}/assets {target}/assets")
        sources = [str(Path(checkout) / name) for name in SYNCED if (Path(checkout) / name).exists()]
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", *sources, f"{self.host}:{target}/"],
                       check=True, capture_output=True)
        return target

    def static_checks(self, cid):
        """Lint and the unit tests with the candidate's code. Returns (passed, output tail)."""
        cand = f"{self.work}/cands/{cid}"
        command = (f"cd {cand} && PYTHONPATH={cand}/src:{cand}/vendor MUJOCO_GL=egl timeout 900 "
                   f"{self.base}/.venv/bin/python -m pytest -q -x -p no:cacheprovider tests 2>&1 | tail -n 15; "
                   f"echo PYTEST=${{PIPESTATUS[0]}}; {self.base}/.venv/bin/ruff check --no-cache src tests 2>&1 | tail -n 15; "
                   f"echo RUFF=${{PIPESTATUS[0]}}")
        out = self.ssh(command, timeout=1200, check=False)
        passed = "PYTEST=0" in out and "RUFF=0" in out
        return passed, out[-3000:]

    def run_suite(self, cid, tasks, repeats, label):
        """Run one HomeBody session per task (dicts with name, scene, targets) REPEATS times with
        candidate CID's code. Returns the remote results directory."""
        cand = f"{self.work}/cands/{cid}"
        results = f"{self.work}/results/{cid}/{label}"
        lines = [f"{cand} {self.base}/{t['scene']} {self.base}/{t['targets']} {results}/{t['name']}/r{r}"
                 for t in tasks for r in range(repeats)]
        jobs = f"{results}/jobs.txt"
        self.ssh(f"rm -rf {results} && mkdir -p {results} && cat > {jobs} << 'EOF'\n" + "\n".join(lines) + "\nEOF")
        subprocess.run(["rsync", "-a", str(HERE / "run_jobs.sh"), str(HERE / "score_episode.py"),
                        str(HERE / "episode_settings.toml"), f"{self.host}:{self.work}/"], check=True)
        waves = -(-len(lines) // self.parallel)
        self.ssh(f"bash {self.work}/run_jobs.sh {self.base} {self.parallel} {jobs} {self.work}",
                 timeout=waves * 7500 + 600)
        return results

    def collect(self, results, local):
        """Copy each run's result, log and session events to LOCAL."""
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--include=*/", "--include=result.json", "--include=run.log",
                        "--include=exit_code", "--include=events.jsonl", "--include=outcome.json",
                        "--exclude=*", "--prune-empty-dirs",
                        f"{self.host}:{results}/", f"{local}/"], check=True, capture_output=True)
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
