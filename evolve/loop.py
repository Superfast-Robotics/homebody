"""The skill-evolution loop: diagnose the best version's train failures, let astra write K
patches in their own git worktrees, gate each by tier, evaluate on train and held-out splits, and
accept by the non-regression rule into an archive of versions.

    python3 -m evolve.loop baseline         # evaluate the current HEAD as v000
    python3 -m evolve.loop round            # one round from the best version
    python3 -m evolve.loop run --rounds 5
    python3 -m evolve.loop status
    python3 -m evolve.loop rebaseline       # pool the baseline with its own confirmation run
    python3 -m evolve.loop rejudge 1        # re-apply the rule to round 1 against the current best
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
COMPLETE = ".complete"  # written once a split has finished and been collected
DONE = ".done"  # written once a round has given every verdict
MECHANISM = re.compile(r"^## Mechanism (\d+):", re.MULTILINE)


class QuotaExhausted(RuntimeError):
    pass


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

    def run_split(self, cid, split, repeats, suffix=""):
        local = self.work / "results" / cid / (split + suffix)
        if (local / COMPLETE).exists():  # resuming: this split already finished and was collected
            runs = score.read_split(local)
            self.log(f"{cid} {split}{suffix}: reusing its completed runs (score {score.summary(runs)['score']:.3f})")
            return runs
        started = time.monotonic()
        results = self.remote.run_suite(cid, self.tasks(split), repeats, split + suffix, self.cfg["agent"]["model"])
        self.remote.collect(results, local)
        if score.quota_hit(local):  # infrastructure, not the harness: stop without judging anything
            self.log(f"{cid} {split}{suffix}: the model's usage limit was hit; stopping without a verdict")
            raise QuotaExhausted(f"usage limit hit while evaluating {cid}; resume the round once it resets")
        (local / COMPLETE).write_text(f"{len(score.read_split(local))} runs\n")
        runs = score.read_split(local)
        summary = score.summary(runs)
        self.log(f"{cid} {split}{suffix}: score {summary['score']:.3f}, success {summary['success_rate']:.2f} "
                 f"({time.monotonic() - started:.0f} s)")
        return runs

    def both_splits(self, cid, repeats, suffix=""):
        """Train and held-out at once: an evaluation waits for its slowest session, not two in turn."""
        with ThreadPoolExecutor(2) as pool:
            futures = {split: pool.submit(self.run_split, cid, split, repeats, suffix) for split in ("train", "val")}
            return {split: future.result() for split, future in futures.items()}

    def evaluate(self, cid, checkout, tier):
        """Push, run both splits at the tier's repeats, collect; returns {train, val} summaries."""
        self.remote.push_candidate(cid, checkout)
        repeats = self.cfg["gates"][str(tier)]["repeats"]
        return {split: score.summary(runs) for split, runs in self.both_splits(cid, repeats).items()}

    def confirm(self, cid, tier):
        """A passing candidate's second, independent evaluation, pooled with its first."""
        repeats = self.cfg["gates"][str(tier)]["repeats"]
        second = self.both_splits(cid, repeats, "_confirm")
        return {split: score.summary(score.read_split(self.work / "results" / cid / split) + runs)
                for split, runs in second.items()}

    # --- commands ---
    def baseline(self, passes=3):
        commit = git("rev-parse", "HEAD")
        checkout = self.worktree("v000", commit)
        passed, output = self.remote_checks("v000", checkout)
        if not passed:
            raise SystemExit(f"the baseline fails its own checks:\n{output}")
        self.evaluate("v000", checkout, 0)
        repeats = self.cfg["gates"]["0"]["repeats"]
        runs = {s: score.read_split(self.work / "results" / "v000" / s) for s in ("train", "val")}
        for index in range(2, passes + 1):  # more passes than a candidate gets: a steadier reference
            for split, extra in self.both_splits("v000", repeats, f"_pass{index}").items():
                runs[split] = runs[split] + extra
        summaries = {split: score.summary(r) for split, r in runs.items()}
        self.record({"id": "v000", "parent": None, "round": 0, "commit": commit, "status": "baseline",
                     "passes": passes, **summaries})

    def rebaseline(self):
        """Pool the baseline with its own confirmation run, so it is measured like a confirmed candidate."""
        entry = next(e for e in self.archive() if e["status"] == "baseline")
        pooled = self.confirm(entry["id"], 0)
        self.record({**entry, **pooled, "status": "baseline", "reason": "re-measured: pooled with a confirmation run"})

    def rejudge(self, number):
        """Re-apply the acceptance rule to round NUMBER's evaluated candidates against the current best,
        from their stored runs; one that newly passes its first evaluation gets its confirmation."""
        best = self.best()
        latest = {e["id"]: e for e in self.archive() if e.get("round") == number and e.get("train")}
        accepted = []
        for entry in latest.values():
            cid, tier = entry["id"], entry["tier"]
            first = {s: score.summary(score.read_split(self.work / "results" / cid / s)) for s in ("train", "val")}
            if all((self.work / "results" / cid / f"{s}_confirm" / COMPLETE).exists() for s in ("train", "val")):
                pooled = {s: score.summary(score.read_split(self.work / "results" / cid / s) +
                                           score.read_split(self.work / "results" / cid / f"{s}_confirm"))
                          for s in ("train", "val")}
            elif score.accept(best, first)[0]:
                self.log(f"{cid} now passes its first evaluation against {best['id']}; confirming")
                pooled = self.confirm(cid, tier)
            else:
                pooled = first
            ok, d_train, d_val = score.accept(best, pooled)
            verdict = (f"re-judged against re-measured {best['id']}: train {d_train:+.3f}, held-out "
                       f"{'did not drop' if d_val >= 0 else 'dropped'}{' and rose' if d_val > 0 else ''}")
            judged = {**entry, **pooled, "status": "accepted" if ok else "rejected", "reason": verdict,
                      "d_train": d_train, "d_val": d_val}
            (accepted.append(judged) if ok else self.record(judged))
        if accepted:
            winner = max(accepted, key=lambda e: (e["d_train"] + e["d_val"], -e.get("lines_changed", 0)))
            for entry in accepted:
                self.record(entry if entry is winner else {**entry, "status": "rejected",
                            "reason": entry["reason"] + "; passed, but another candidate scored higher"})
        self.log(f"round {number} re-judged; best is now {self.best()['id']}")

    def remote_checks(self, cid, checkout):
        self.remote.push_candidate(cid, checkout)
        return self.remote.static_checks(cid)

    def round(self):
        best = self.best()
        unfinished = sorted(p for p in (self.work / "rounds").glob("r[0-9][0-9][0-9]") if not (p / DONE).exists())
        number = int(unfinished[0].name[1:]) if unfinished else 1 + max((e.get("round", 0) for e in self.archive()), default=0)
        round_dir = self.work / "rounds" / f"r{number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"round {number}: improving {best['id']} (train {best['train']['score']:.3f}, "
                 f"val {best['val']['score']:.3f})")
        history = list({e["id"]: e for e in self.archive() if e["status"] != "baseline"}.values())  # latest verdicts
        ws = workspace.build(round_dir, best, self.work / "results" / best["id"] / "train", history, self.instructions())
        saved = round_dir / "diagnosis.md"
        if saved.exists() and saved.read_text().strip():  # resuming an interrupted round
            diagnosis = saved.read_text()
            self.log(f"round {number}: resuming with its saved diagnosis")
        else:
            diag_tree = self.worktree(f"r{number:03d}d", best["commit"])
            diagnosis = astra.diagnose(diag_tree, ws, saved)
        mechanisms = [int(n) for n in MECHANISM.findall(diagnosis)][: self.cfg["astra"]["proposals"]] or [1]
        self.log(f"round {number}: diagnosis names mechanisms {MECHANISM.findall(diagnosis)}; proposing {mechanisms}")
        cids = [f"r{number:03d}c{n}" for n in mechanisms]
        with ThreadPoolExecutor(len(cids)) as pool:
            notes = list(pool.map(lambda pair: self.propose(pair[0], pair[1], best, ws, diagnosis, round_dir),
                                  zip(cids, mechanisms)))
        judged = {e["id"] for e in self.archive()}
        candidates = [c for c in notes if c is not None and c["cid"] not in judged]  # resuming: skip verdicts given
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
            if ok:  # noise check: accept only if the pooled first and confirming runs still pass
                self.log(f"{cand['cid']} passed ({d_train:+.3f} train, {d_val:+.3f} held-out); confirming")
                summaries = self.confirm(cand["cid"], entry["tier"])
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
        (round_dir / DONE).write_text("done\n")
        self.log(f"round {number}: done; best is now {self.best()['id']}")

    def propose(self, cid, mechanism, best, ws, diagnosis, round_dir):
        path = self.worktree(cid, best["commit"])
        note_path = round_dir / f"{cid}_note.md"
        if git("rev-parse", "HEAD", cwd=path) != best["commit"] and note_path.exists():  # resuming: already committed
            self.log(f"{cid}: resuming with its committed patch")
            return {"cid": cid, "path": path, "meta": self.patch_meta(path, best, note_path.read_text(), mechanism)}
        try:
            note = astra.propose(path, ws, diagnosis, mechanism, note_path)
        except Exception as error:  # noqa: BLE001 -- a failed proposal is recorded, not fatal
            return {"cid": cid, "path": path, "meta": {"status": "failed", "reason": f"proposer error: {error}"}}
        # Safe lint fixes first, so a style rule does not cost a whole proposal.
        subprocess.run(["uvx", "--quiet", "ruff", "check", "--fix", "--quiet", "src", "tests"], cwd=path,
                       check=False, capture_output=True)
        git("add", "-A", cwd=path)
        if not git("status", "--porcelain", cwd=path):
            return {"cid": cid, "path": path, "meta": {"status": "rejected", "reason": "no change", "notes": note}}
        git("-c", "user.name=astra", "-c", "user.email=astra@localhost", "commit", "-q", "-m",
            f"evolve {cid}: mechanism {mechanism}\n\n{note[:3000]}", cwd=path)
        return {"cid": cid, "path": path, "meta": self.patch_meta(path, best, note, mechanism)}

    def patch_meta(self, path, best, note, mechanism):
        """Files, size, tier and frozen check of the worktree's committed patch against BEST."""
        commit = git("rev-parse", "HEAD", cwd=path)
        files = git("diff", "--name-only", best["commit"], commit).splitlines()
        numstat = git("diff", "--numstat", best["commit"], commit).splitlines()
        lines = sum(int(a) + int(b) for a, b, _ in (row.split("\t") for row in numstat) if a != "-")
        classified = tiers.classify(files)
        meta = {"commit": commit, "files": files, "lines_changed": lines, "tier": classified.tier, "notes": note,
                "mechanism": mechanism}
        if classified.frozen:
            meta.update(status="rejected", reason=f"touches frozen files {list(classified.frozen)}")
        return meta

    def status(self):
        for e in self.archive():
            train, val = e.get("train", {}), e.get("val", {})
            print(f"{e['id']:8} {e['status']:9} tier {e.get('tier', '-')!s:2} lines {e.get('lines_changed', '-')!s:5} "
                  f"train {train.get('score', float('nan')):.3f} val {val.get('score', float('nan')):.3f}  "
                  f"{e.get('reason', '')[:90]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("--passes", type=int, default=3, help="evaluations pooled into the baseline")
    sub.add_parser("round")
    run = sub.add_parser("run")
    run.add_argument("--rounds", type=int, required=True)
    sub.add_parser("status")
    sub.add_parser("rebaseline")
    rejudge = sub.add_parser("rejudge")
    rejudge.add_argument("round", type=int)
    args = parser.parse_args(argv)
    loop = Loop()
    if args.command == "baseline":
        loop.baseline(args.passes)
    elif args.command == "round":
        loop.round()
    elif args.command == "rebaseline":
        loop.rebaseline()
    elif args.command == "rejudge":
        loop.rejudge(args.round)
    elif args.command == "run":
        for _ in range(args.rounds):
            loop.round()
    else:
        loop.status()


if __name__ == "__main__":
    main()
