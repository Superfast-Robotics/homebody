"""The skill-evolution loop: diagnose the best version's train failures, let astra write K
patches in their own git worktrees, gate each by tier, evaluate on train and held-out splits, and
accept by the non-regression rule into an archive of versions.

    python3 -m evolve.loop baseline         # evaluate the current HEAD as v000
    python3 -m evolve.loop round            # one round from the best version
    python3 -m evolve.loop run --rounds 5
    python3 -m evolve.loop status
"""
import argparse
import json
import re
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import astra, score, tiers, workspace
from .remote import Remote, suite_tasks

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MECHANISM = re.compile(r"^## Mechanism (\d+):", re.MULTILINE)


def config():
    return tomllib.loads((HERE / "config.toml").read_text())


def git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class Loop:
    def __init__(self):
        self.cfg = config()
        self.work = Path(self.cfg["local"]["work"])
        self.work.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.work / "archive.jsonl"
        self.remote = Remote.from_config()

    # --- archive ---
    def archive(self):
        if not self.archive_path.exists():
            return []
        return [json.loads(line) for line in self.archive_path.read_text().splitlines() if line.strip()]

    def record(self, entry):
        with self.archive_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self.log(f"archive: {entry['id']} {entry['status']} {entry.get('reason', '')}")

    def best(self):
        accepted = [e for e in self.archive() if e["status"] in ("baseline", "accepted")]
        if not accepted:
            raise SystemExit("no evaluated version yet: run `python3 -m evolve.loop baseline`")
        return accepted[-1]

    def log(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.work / "loop.log").open("a") as f:
            f.write(line + "\n")

    # --- worktrees and evaluation ---
    def worktree(self, cid, commit):
        path = self.work / "worktrees" / cid
        if not path.exists():
            git("worktree", "add", "-f", "-B", f"evolve/{cid}", str(path), commit)
        return path

    def tasks(self, split):
        cache = self.work / f"tasks_{split}.json"
        if not cache.exists():
            cache.write_text(json.dumps(suite_tasks(self.remote, self.cfg["suite"][split]), indent=1))
        return json.loads(cache.read_text())

    def instructions(self):
        cache = self.work / "train_instructions.json"
        if not cache.exists():
            text = {}
            for task in self.tasks("train"):
                raw = self.remote.ssh(f"cat {self.remote.base}/{task['scene']}/task_scene.json")
                text[task["name"]] = json.loads(raw)["default_task"]
            cache.write_text(json.dumps(text, indent=1))
        return json.loads(cache.read_text())

    def evaluate(self, cid, checkout, tier):
        """Push, run both splits at the tier's repeats, collect; returns {train, val} summaries."""
        self.remote.push_candidate(cid, checkout)
        repeats = self.cfg["gates"][str(tier)]["repeats"]
        summaries = {}
        for split in ("train", "val"):
            started = time.monotonic()
            results = self.remote.run_suite(cid, self.tasks(split), repeats, self.cfg["suite"]["wall_timeout"], split)
            local = self.remote.collect(results, self.work / "results" / cid / split)
            summaries[split] = score.summary(score.read_split(local))
            self.log(f"{cid} {split}: score {summaries[split]['score']:.3f}, success "
                     f"{summaries[split]['success_rate']:.2f} ({time.monotonic() - started:.0f} s)")
        return summaries

    # --- commands ---
    def baseline(self):
        commit = git("rev-parse", "HEAD")
        checkout = self.worktree("v000", commit)
        passed, output = self.remote_checks("v000", checkout)
        if not passed:
            raise SystemExit(f"the baseline fails its own checks:\n{output}")
        summaries = self.evaluate("v000", checkout, 0)
        self.record({"id": "v000", "parent": None, "round": 0, "commit": commit, "status": "baseline", **summaries})

    def remote_checks(self, cid, checkout):
        self.remote.push_candidate(cid, checkout)
        return self.remote.static_checks(cid)

    def round(self):
        best = self.best()
        number = 1 + max((e.get("round", 0) for e in self.archive()), default=0)
        round_dir = self.work / "rounds" / f"r{number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"round {number}: improving {best['id']} (train {best['train']['score']:.3f}, "
                 f"val {best['val']['score']:.3f})")
        history = [e for e in self.archive() if e["status"] != "baseline"]
        ws = workspace.build(round_dir, best, self.work / "results" / best["id"] / "train", history, self.instructions())
        diag_tree = self.worktree(f"r{number:03d}d", best["commit"])
        diagnosis = astra.diagnose(diag_tree, ws, round_dir / "diagnosis.md")
        mechanisms = [int(n) for n in MECHANISM.findall(diagnosis)][: self.cfg["astra"]["proposals"]] or [1]
        self.log(f"round {number}: diagnosis names mechanisms {MECHANISM.findall(diagnosis)}; proposing {mechanisms}")
        cids = [f"r{number:03d}c{n}" for n in mechanisms]
        with ThreadPoolExecutor(len(cids)) as pool:
            notes = list(pool.map(lambda pair: self.propose(pair[0], pair[1], best, ws, diagnosis, round_dir),
                                  zip(cids, mechanisms)))
        candidates = [c for c in notes if c is not None]
        accepted = []
        for cand in candidates:
            entry = {"id": cand["cid"], "parent": best["id"], "round": number, **cand["meta"]}
            if cand["meta"].get("status"):
                self.record(entry)
                continue
            passed, output = self.remote_checks(cand["cid"], cand["path"])
            if not passed:
                (round_dir / f"{cand['cid']}_checks.txt").write_text(output)
                self.record({**entry, "status": "rejected", "reason": "lint or unit tests failed: " + output[-600:]})
                continue
            summaries = self.evaluate(cand["cid"], cand["path"], entry["tier"])
            ok, d_train, d_val = score.accept(best, summaries)
            verdict = (f"train {d_train:+.3f}, held-out {'did not drop' if d_val >= 0 else 'dropped'}"
                       f"{' and rose' if d_val > 0 else ''}")
            entry = {**entry, **summaries, "status": "accepted" if ok else "rejected", "reason": verdict,
                     "d_train": d_train, "d_val": d_val}
            (accepted if ok else []).append(entry)
            if not ok:
                self.record(entry)
        if accepted:
            winner = max(accepted, key=lambda e: (e["d_train"] + e["d_val"], -e.get("lines_changed", 0)))
            for entry in accepted:
                if entry is not winner:
                    entry = {**entry, "status": "rejected", "reason": entry["reason"] + "; passed, but another "
                             "candidate this round scored higher (or tied with a smaller diff)"}
                self.record(entry)
        self.log(f"round {number}: done; best is now {self.best()['id']}")

    def propose(self, cid, mechanism, best, ws, diagnosis, round_dir):
        path = self.worktree(cid, best["commit"])
        try:
            note = astra.propose(path, ws, diagnosis, mechanism, round_dir / f"{cid}_note.md")
        except Exception as error:  # noqa: BLE001 -- a failed proposal is recorded, not fatal
            return {"cid": cid, "path": path, "meta": {"status": "failed", "reason": f"proposer error: {error}"}}
        git("add", "-A", cwd=path)
        if not git("status", "--porcelain", cwd=path):
            return {"cid": cid, "path": path, "meta": {"status": "rejected", "reason": "no change", "notes": note}}
        git("-c", "user.name=astra", "-c", "user.email=astra@localhost", "commit", "-q", "-m",
            f"evolve {cid}: mechanism {mechanism}\n\n{note[:3000]}", cwd=path)
        commit = git("rev-parse", "HEAD", cwd=path)
        files = git("diff", "--name-only", best["commit"], commit).splitlines()
        numstat = git("diff", "--numstat", best["commit"], commit).splitlines()
        lines = sum(int(a) + int(b) for a, b, _ in (row.split("\t") for row in numstat) if a != "-")
        classified = tiers.classify(files)
        meta = {"commit": commit, "files": files, "lines_changed": lines, "tier": classified.tier, "notes": note,
                "mechanism": mechanism}
        if classified.frozen:
            meta.update(status="rejected", reason=f"touches frozen files {list(classified.frozen)}")
        return {"cid": cid, "path": path, "meta": meta}

    def status(self):
        for e in self.archive():
            train, val = e.get("train", {}), e.get("val", {})
            print(f"{e['id']:8} {e['status']:9} tier {e.get('tier', '-')!s:2} lines {e.get('lines_changed', '-')!s:5} "
                  f"train {train.get('score', float('nan')):.3f} val {val.get('score', float('nan')):.3f}  "
                  f"{e.get('reason', '')[:90]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline")
    sub.add_parser("round")
    run = sub.add_parser("run")
    run.add_argument("--rounds", type=int, required=True)
    sub.add_parser("status")
    args = parser.parse_args(argv)
    loop = Loop()
    if args.command == "baseline":
        loop.baseline()
    elif args.command == "round":
        loop.round()
    elif args.command == "run":
        for _ in range(args.rounds):
            loop.round()
    else:
        loop.status()


if __name__ == "__main__":
    main()
