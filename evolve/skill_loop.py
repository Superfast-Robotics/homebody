"""Skill-level optimisation: Astra improves one built-in skill (config [skill] name) on the skill
benchmark (skill_bench.py episodes, no model at run time), round by round, from the version recorded
as the skill baseline.

    python3 -m evolve.skill_loop [--profile P] init COMMIT       Astra writes the skill's first version from COMMIT
    python3 -m evolve.skill_loop [--profile P] baseline COMMIT   measure COMMIT on every episode
    python3 -m evolve.skill_loop [--profile P] run --rounds N    diagnose, patch, evaluate, accept
    python3 -m evolve.skill_loop [--profile P] test CID          the version on the test episodes
    python3 -m evolve.skill_loop [--profile P] status

A profile ([skills.<P>] in config.toml) names the skill, its episodes, the coder switch, the editable
files and the prompts, so another skill needs only a driver in skill_bench.py, an episode generator
and a profile.

A patch may change only the files [skill] editable names (default EDITABLE); everything else is
rejected unseen, and so is a patch whose source mentions the benchmark's scenes, objects or episodes
(the generality screen). With [skill] coder = true the skill is dynamic: a failed call gets one coder
rescue in every episode, and the coder's work is shown to the diagnosis. Val episodes decide acceptance but are never shown to the proposer.
"""
import argparse
import fnmatch
import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import astra, skill_score
from .loop import REPO, Loop, git
from .remote import rsync
from .workspace import coder_call

RECORDING = threading.Lock()
SPLIT_PREFIXES = {f"{family}{split}" for family in ("pk", "pl", "pl2", "mv") for split in ("train", "val", "test")}


def in_source(commit, word):
    """Whether WORD already occurs (as a whole word) in COMMIT's src tree."""
    return subprocess.run(["git", "grep", "-q", "-w", word, commit, "--", "src"], cwd=REPO,
                          capture_output=True).returncode == 0

HERE = Path(__file__).resolve().parent
COMPLETE = ".complete"
EDITABLE = ("src/homebody/skills/{skill}.py", "src/homebody/motion/*", "src/homebody/helpers/*",
            "src/homebody/toolkit/*", "tests/*")
SPLITS = ("train", "val")


def coder_digest(split_dir):
    """What the coder changed across a split's episodes: per edited function of the skill, how often and
    how often that call ended OK, the diffs of the calls that did, and the recurring fixes: functions changed
    in two or more episodes that ended successfully (the call OK or the object held), as helper candidates."""
    import difflib
    import re
    split_dir = Path(split_dir)
    edits, wins, diffs, fixed = {}, {}, [], {}
    for call_dir in sorted(split_dir.rglob("coder/call-*")):
        start = next(call_dir.glob("start_*.py"), None)
        final = call_dir / "workspace" / "skills" / start.name.removeprefix("start_") if start else None
        if not start or not final.exists():
            continue
        before, after = start.read_text().splitlines(), final.read_text().splitlines()
        result = json.loads((call_dir / "result.json").read_text()) if (call_dir / "result.json").exists() else {}
        episode = call_dir.relative_to(split_dir).parts[0]
        outcome = split_dir / episode / "result.json"
        held = outcome.exists() and bool(json.loads(outcome.read_text()).get("success"))
        defs = [(i, m.group(1)) for i, line in enumerate(before) if (m := re.match(r"\s*def (\w+)", line))]
        touched = set()
        for tag, i1, _, _, _ in difflib.SequenceMatcher(None, before, after).get_opcodes():
            if tag != "equal":  # the function the change falls in (the last def above it)
                touched.add(max(((i, n) for i, n in defs if i <= i1), default=(0, "<module>"))[1])
        for name in touched:
            edits[name] = edits.get(name, 0) + 1
            wins[name] = wins.get(name, 0) + (result.get("code") == "OK")
            if result.get("code") == "OK" or held:
                fixed.setdefault(name, []).append(episode)
        if result.get("code") == "OK":
            diffs.append(f"### {call_dir.relative_to(split_dir)}\n{result.get('message', '')}\n```diff\n"
                         + "\n".join(difflib.unified_diff(before, after, "before", "after", lineterm="", n=1))[:6000] + "\n```")
    recurring = sorted(((n, e) for n, e in fixed.items() if len(set(e)) >= 2), key=lambda kv: -len(set(kv[1])))
    lines = ["# What the coder changed in the skill, across the train episodes", "",
             "| function | calls that edited it | of those, ended OK |", "|---|---|---|"]
    lines += [f"| {name} | {count} | {wins[name]} |" for name, count in sorted(edits.items(), key=lambda kv: -kv[1])]
    lines += ["", "## Recurring fixes (helper candidates)", "",
              "Functions the coder changed in two or more episodes that ended successfully (its call OK, or the "
              "object held at the end). Compare their edits in the episodes' coder-call-1.md: a fix the coder keeps "
              "rewriting by hand is a candidate for a tested helper or an extension point.", ""]
    lines += [f"- `{name}`: {len(set(eps))} successful episodes ({', '.join(sorted(set(eps)))})" for name, eps in recurring] or ["none yet"]
    lines += ["", "## The edits of the calls that ended OK", ""] + (diffs or ["none"])
    return "\n".join(lines) + "\n"


class SkillLoop:
    def __init__(self, profile=None):
        self.loop = Loop()
        profiles = self.loop.cfg["skills"]
        self.profile = profile or profiles["default"]
        self.cfg = profiles[self.profile]
        self.skill = self.cfg["name"]
        self.work = self.loop.work / "skill" / self.cfg.get("workdir", self.skill)
        self.editable = tuple(self.cfg.get("editable", EDITABLE))
        # the skill's module (a hand's own version lives under skills/<hands>/) and what the brief adds about the robot
        self.source = self.cfg.get("source", "src/homebody/skills/{skill}.py").format(skill=self.skill)
        self.brief_note = self.cfg.get("brief_note", "")
        self.coder = bool(self.cfg.get("coder", False))
        self.prompts = self.cfg.get("prompts", "skill")
        self.proposals = int(self.cfg.get("proposals", self.loop.cfg["astra"]["proposals"]))  # per round, by mechanism rank
        self.work.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.work / "archive.jsonl"
        self.remote = self.loop.remote
        self.diagnose_model = None

    # --- bookkeeping ---
    def archive(self):
        if not self.archive_path.exists():
            return []
        return [json.loads(line) for line in self.archive_path.read_text().splitlines() if line.strip()]

    def record(self, entry):
        stored = {k: v for k, v in entry.items() if k not in ("train", "val")}
        stored.update({s: {k: v for k, v in entry[s].items() if k not in ("values", "seconds")} for s in SPLITS
                       if s in entry})
        with RECORDING, self.archive_path.open("a") as out:  # candidates of a round finish on parallel threads
            out.write(json.dumps(stored) + "\n")
        self.log(f"archive: {entry['id']} {entry['status']} {entry.get('reason', '')}")

    def best(self):
        archive = self.archive()
        latest = {e["id"]: e for e in archive}  # a later verdict on the same id (a correction) wins
        accepted = [e for e in archive if latest[e["id"]] is e and e["status"] in ("baseline", "accepted")]
        if not accepted:
            raise SystemExit("no skill baseline yet: run `python3 -m evolve.skill_loop baseline COMMIT`")
        return accepted[-1]

    def log(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.work / "loop.log").open("a") as out:
            out.write(line + "\n")

    def summaries(self, cid, splits=SPLITS):
        return {s: skill_score.summary(skill_score.read(self.work / "results" / cid / s)) for s in splits}

    # --- running episodes ---
    def episodes(self, split):
        listing = self.remote.ssh(f"cd {self.remote.work} && ls {self.cfg['episodes']}/{split}/*.json")
        return listing.split()

    def cand(self, cid):
        """The remote candidate name and local worktree name of CID, unique per profile."""
        return f"{self.cfg.get('workdir', self.skill)}-{cid}"

    def evaluate(self, cid, checkout, splits=SPLITS, only=None, cand=None):
        """Push CID's code and run every episode of SPLITS on the remote (or ONLY: split -> episode paths, under the
        candidate copy CAND); collect and summarise."""
        local = self.work / "results" / cid
        if all((local / s / COMPLETE).exists() for s in splits):
            return self.summaries(cid)
        cand = cand or self.cand(cid)
        self.loop.remote.push_candidate(cand, checkout)
        root = f"{self.remote.work}/skill/{self.cfg.get('workdir', self.skill)}/{cid}"
        lines = [f"{self.remote.work}/cands/{cand} {self.remote.work}/{episode} {root}/{split}/{Path(episode).stem}"
                 for split in splits for episode in (only[split] if only else self.episodes(split))]
        jobs = f"{root}/jobs.txt"
        self.remote.ssh(f"rm -rf {root} && mkdir -p {root} && cat > {jobs} << 'EOF'\n" + "\n".join(lines) + "\nEOF")
        rsync("-a", str(HERE / "skill_bench.py"), str(HERE / "skill_bench_tasks.py"), str(HERE / "run_skill_jobs.sh"),
              f"{self.remote.host}:{self.remote.work}/")
        started = time.monotonic()
        self.remote.ssh(f"bash {self.remote.work}/run_skill_jobs.sh {self.remote.base} {self.cfg['parallel']} {jobs} "
                        f"{self.remote.work} {int(self.coder)} '{self.cfg.get('coder_model', '')}' "
                        f"'{self.cfg.get('coder_rates', '')}' '{self.cfg.get('coder_seconds', '')}'", timeout=24 * 3600)
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        rsync("-a", "--include=*/", "--include=result.json", "--include=trace.json", "--include=run.log",
              "--include=REQUEST.md", "--include=tools.jsonl", "--include=start_*.py", "--include=last_message.json",
              "--include=**/coder/call-*/events.jsonl", "--include=**/workspace/skills/*.py",
              "--include=**/workspace/lib/*.py", "--exclude=*", "--prune-empty-dirs",
              f"{self.remote.host}:{root}/", f"{local}/")
        for split in splits:
            (local / split).mkdir(exist_ok=True)
            (local / split / COMPLETE).write_text("done\n")
        if splits != SPLITS:
            return self.summaries(cid, splits)
        summaries = self.summaries(cid)
        self.log(f"{cid}: train success {summaries['train']['success_rate']:.2f} value {summaries['train']['value']:.3f}, "
                 f"val success {summaries['val']['success_rate']:.2f} value {summaries['val']['value']:.3f} "
                 f"({time.monotonic() - started:.0f} s)")
        return summaries

    # --- commands ---
    def baseline(self, commit):
        cid = "p000"
        path = self.loop.worktree(self.cand(f"{cid}"), git("rev-parse", commit))
        summaries = self.evaluate(cid, path)
        self.record({"id": cid, "round": 0, "commit": git("rev-parse", "HEAD", cwd=path), "status": "baseline",
                     "reason": f"{self.skill} skill baseline", **summaries})

    def init(self, commit, sample=8):
        """Astra writes the skill's first version on top of COMMIT, for a robot the copied skill was not written
        for, from the copy's results on SAMPLE train episodes; screened like a candidate. Its commit, or None."""
        base = git("rev-parse", commit)
        path = self.loop.worktree(self.cand("i000"), base)
        round_dir = self.work / "rounds" / "i000"
        round_dir.mkdir(parents=True, exist_ok=True)
        train = sorted(self.episodes("train"))
        evidence = train[::max(1, len(train) // sample)][:sample]
        self.log(f"init: the copy on {len(evidence)} train episodes")
        self.evaluate("i000-copy", path, splits=("train",), only={"train": evidence}, cand=self.cand("i000-copy"))
        ws = self.workspace(round_dir, {"id": "i000-copy"})
        note_path = round_dir / "i000_note.md"
        entry = {"id": "i000", "round": 0, "parent": None, "status": "init"}
        if git("rev-parse", "HEAD", cwd=path) == base:
            prompt = (HERE / "prompts" / "skill_init.md").read_text().format(
                workspace=ws, skill=self.skill, source=self.source, ruff=astra.settings()["ruff"],
                python=astra.settings()["python"], editable=", ".join(p.format(skill=self.skill) for p in self.editable))
            note = astra.codex(prompt, path, "workspace-write", note_path, **self.proposer())
            git("add", "-A", cwd=path)
            changed = [f for f in git("diff", "--cached", "--name-only", "--diff-filter=AM", cwd=path).splitlines()
                       if f.endswith(".py")]
            if changed:
                subprocess.run([astra.settings()["ruff"], "check", "--fix", "--quiet", *changed], cwd=path,
                               check=False, capture_output=True)
            git("add", "-A", cwd=path)
            if not git("status", "--porcelain", cwd=path):
                self.record({**entry, "status": "rejected", "reason": "no change", "notes": note})
                return None
            git("-c", "user.name=astra", "-c", "user.email=astra@localhost", "commit", "-q", "-m",
                f"skill {self.skill} i000: first version\n\n{note[:3000]}", cwd=path)
        commit = git("rev-parse", "HEAD", cwd=path)
        files = git("diff", "--name-only", base, commit).splitlines()
        entry.update(commit=commit, files=files, notes=note_path.read_text() if note_path.exists() else "")
        outside, specific = self.allowed(files), self.specific(base, commit)
        reason = (f"edits outside the skill: {outside}" if outside else
                  f"not general: the source names {specific}" if specific else None)
        if reason is None:
            passed, output = self.loop.remote_checks(self.cand("i000"), path)
            reason = None if passed else "lint or unit tests failed: " + output[-600:]
        if reason is None:
            crash = self.smoke("i000", path)
            reason = f"crashed in a one-episode smoke run: {crash}" if crash else None
        self.record({**entry, "reason": reason or "first version written", **({"status": "rejected"} if reason else {})})
        self.log(f"init: {reason or 'ready'} ({commit[:10]}); optimisation credits {astra.credits_in(round_dir)}")
        return None if reason else commit

    def workspace(self, round_dir, best):
        ws = round_dir / "workspace"
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        minutes = f"{int(self.cfg.get('coder_seconds', 300)) / 60:g}"
        bench = HERE / "prompts" / f"bench_{self.skill}.md"
        benchmark = bench.read_text().format(skill=self.skill, coder_minutes=minutes) if bench.exists() else ""
        brief = (HERE / "prompts" / f"{self.prompts}_brief.md").read_text().format(
            skill=self.skill, source=self.source, coder_model=self.cfg.get("coder_model", "Sol"),
            coder_minutes=minutes, benchmark=benchmark)
        (ws / "BRIEF.md").write_text(brief + (f"\n## This robot\n\n{self.brief_note.strip()}\n" if self.brief_note else ""))
        train = skill_score.read(self.work / "results" / best["id"] / "train")
        summary = skill_score.summary(train)
        lines = [f"# {self.skill} on the train episodes with version {best['id']}", "",
                 (f"Success {summary['success_rate']:.2f} (the skill alone {summary['builtin_success_rate']:.2f}), "
                  f"value {summary['value']:.3f} (stage / 2), {summary['skill_seconds']} s and {summary['credits']} "
                  f"coder credits per episode; coder calls {summary['coder_calls']}, OK {summary['coder_ok']}."), "",
                 f"Outcomes of the skill call: {json.dumps(summary['outcomes'])}",
                 f"Success by family: {json.dumps(summary['success_by_family'])}", "",
                 "| episode | family | stage | skill result | message | coder | coder message | s |",
                 "|---|---|---|---|---|---|---|---|"]
        for episode, row in sorted(train.items()):
            result, coder = skill_score.call(row), row.get("coder") or {}
            lines.append(f"| {episode} | {row['family']} | {row['stage']} | {result.get('code', row.get('error', '?'))} | "
                         f"{(result.get('message') or '')[:120]} | {coder.get('code', '-')} | "
                         f"{(coder.get('message') or '')[:160]} | {skill_score.seconds(row)} |")
        (ws / "results.md").write_text("\n".join(lines) + "\n")
        for episode in train:
            source = self.work / "results" / best["id"] / "train" / episode
            target = ws / "episodes" / episode
            target.mkdir(parents=True)
            for name in ("result.json", "trace.json"):
                if (source / name).exists():
                    shutil.copy(source / name, target / name)
            for call_dir in sorted(source.rglob("coder/call-*")):
                (target / f"coder-{call_dir.name}.md").write_text(coder_call(call_dir))
        (ws / "coder_edits.md").write_text(coder_digest(self.work / "results" / best["id"] / "train"))
        history = [f"## {e['id']} (round {e.get('round')}): {e['status']}\n{e.get('reason', '')}\n{(e.get('notes') or '')[:1500]}"
                   for e in self.archive()]
        (ws / "history.md").write_text("# Earlier versions of the skill\n\n" + "\n\n".join(history) + "\n")
        return ws

    def proposer(self):
        """The model that patches the skill ([skill] model and effort)."""
        return {"model": self.cfg["model"], "effort": self.cfg["effort"]}

    def diagnoser(self):
        """The model that diagnoses the skill: [skill], unless a run named another (--diagnose-model)."""
        return {"model": self.diagnose_model or self.cfg["model"], "effort": self.cfg["effort"]}

    def benchmark_names(self):
        """Scene, object and episode names of the benchmark (from a local copy of its episode files): a
        general change never mentions them."""
        local = self.work / "episodes"
        if not local.exists():
            rsync("-a", f"{self.remote.host}:{self.remote.work}/{self.cfg['episodes']}/", f"{local}/")
        names = set()
        for path in local.glob("*/*.json"):
            episode = json.loads(path.read_text())
            names.update([episode["id"], Path(episode["scene"]).name, str(episode["step"]["label"]),
                          str(episode["step"]["fixture_id"])])
            if episode["step"].get("destination_label") is not None:
                names.add(str(episode["step"]["destination_label"]))
        return names

    def specific(self, base, commit):
        """Benchmark names (scenes, objects, episodes, split prefixes) in the source lines a patch adds."""
        added = [line[1:] for line in git("diff", base, commit, "--", "src").splitlines()
                 if line.startswith("+") and not line.startswith("+++")]
        names = {n for n in self.benchmark_names() if len(n) > 3}
        names = {n for n in names if not in_source(base, n)} | SPLIT_PREFIXES  # words the code already uses are not
        return sorted({n for n in names for line in added if n in line})          # benchmark-specific

    def allowed(self, files):
        patterns = [p.format(skill=self.skill) for p in self.editable]
        return [f for f in files if not any(fnmatch.fnmatch(f, p) for p in patterns)]

    def round(self, number):
        """Diagnose the best version, write and screen one candidate per ranked mechanism at once (they share
        one parent and one diagnosis), then evaluate them one by one; the largest accepted gain wins."""
        best = self.best()
        round_dir = self.work / "rounds" / f"s{number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"round {number}: improving {best['id']} (train {best['train']['value']:.3f}, val {best['val']['value']:.3f})")
        ws = self.workspace(round_dir, best)
        reader = self.loop.worktree(self.cand(f"s{number:03d}d"), best["commit"])
        prompt = (HERE / "prompts" / f"{self.prompts}_diagnose.md").read_text().format(workspace=ws, skill=self.skill,
                                                                                      source=self.source)
        diagnosis_path = round_dir / "diagnosis.md"
        if diagnosis_path.exists():  # resuming
            diagnosis = diagnosis_path.read_text()
        else:
            diagnosis = astra.codex(prompt, reader, "read-only", diagnosis_path, **self.diagnoser())
        indices = range(1, self.proposals + 1)
        paths = {index: self.loop.worktree(self.cand(f"s{number:03d}c{index}"), best["commit"]) for index in indices}
        with ThreadPoolExecutor(len(paths)) as pool:
            ready = [entry for entry in pool.map(
                lambda index: self.prepare(number, index, paths[index], best, ws, diagnosis, round_dir), indices) if entry]
        accepted = []
        for entry in ready:  # one at a time: the coder shares one GPU, and three evaluations at once starve it
            summaries = self.evaluate(entry["id"], paths[entry["mechanism"]])
            verdict = skill_score.judge(self.summaries(best["id"]), summaries)
            entry = {**entry, **summaries, "status": "accepted" if verdict["ok"] else "rejected",
                     "reason": skill_score.describe(verdict), "verdict": verdict}
            (accepted.append(entry) if verdict["ok"] else self.record(entry))
        if accepted:
            winner = max(accepted, key=lambda e: e["verdict"]["pooled"][0])
            for entry in accepted:
                self.record(entry if entry is winner else {**entry, "status": "rejected",
                            "reason": entry["reason"] + "; passed, but another candidate scored higher"})
        self.log(f"round {number}: done; best is now {self.best()['id']}; "
                 f"optimisation credits this round {astra.credits_in(round_dir)}")

    def prepare(self, number, index, path, best, ws, diagnosis, round_dir):
        """Write (or reuse) candidate INDEX of round NUMBER in PATH and screen it; its archive entry when it is
        ready to evaluate, else None (the rejection is recorded)."""
        cid = f"s{number:03d}c{index}"
        note_path = round_dir / f"{cid}_note.md"
        entry = {"id": cid, "round": number, "parent": best["id"], "mechanism": index}
        if git("rev-parse", "HEAD", cwd=path) == best["commit"]:
            prompt = (HERE / "prompts" / f"{self.prompts}_propose.md").read_text().format(
                workspace=ws, skill=self.skill, source=self.source, diagnosis=diagnosis, mechanism=index,
                ruff=astra.settings()["ruff"], python=astra.settings()["python"],
                editable=", ".join(p.format(skill=self.skill) for p in self.editable))
            try:
                note = astra.codex(prompt, path, "workspace-write", note_path, **self.proposer())
            except Exception as error:  # noqa: BLE001 -- a failed proposal is recorded, not fatal
                self.record({**entry, "status": "failed", "reason": f"proposer error: {error}"})
                return None
            git("add", "-A", cwd=path)
            changed = [f for f in git("diff", "--cached", "--name-only", "--diff-filter=AM", cwd=path).splitlines()
                       if f.endswith(".py")]
            if changed:  # tidy only what the proposal touched, so no other file is swept into its diff
                subprocess.run([astra.settings()["ruff"], "check", "--fix", "--quiet", *changed], cwd=path,
                               check=False, capture_output=True)
            git("add", "-A", cwd=path)
            if not git("status", "--porcelain", cwd=path):
                self.record({**entry, "status": "rejected", "reason": "no change", "notes": note})
                return None
            git("-c", "user.name=astra", "-c", "user.email=astra@localhost", "commit", "-q", "-m",
                f"skill {self.skill} {cid}: mechanism {index}\n\n{note[:3000]}", cwd=path)
        note = note_path.read_text() if note_path.exists() else ""
        commit = git("rev-parse", "HEAD", cwd=path)
        files = git("diff", "--name-only", best["commit"], commit).splitlines()
        entry.update(commit=commit, files=files, notes=note)
        outside = self.allowed(files)
        if outside:
            self.record({**entry, "status": "rejected", "reason": f"edits outside the skill: {outside}"})
            return None
        specific = self.specific(best["commit"], commit)
        if specific:
            self.record({**entry, "status": "rejected", "reason": f"not general: the source names {specific}"})
            return None
        passed, output = self.loop.remote_checks(self.cand(cid), path)
        if not passed:
            (round_dir / f"{cid}_checks.txt").write_text(output)
            self.record({**entry, "status": "rejected", "reason": "lint or unit tests failed: " + output[-600:]})
            return None
        crash = self.smoke(cid, path)
        if crash:
            self.record({**entry, "status": "rejected", "reason": f"crashed in a one-episode smoke run: {crash}"})
            return None
        return entry

    def smoke(self, cid, path):
        """Run CID on one train episode before the full evaluation; the crash it raised, or None. A candidate whose
        code breaks the episode itself (not the skill's own failure) is rejected in a minute, not a full run."""
        episode = sorted(self.episodes("train"))[0]
        self.evaluate(f"{cid}-smoke", path, splits=("train",), only={"train": [episode]}, cand=self.cand(cid))
        result = self.work / "results" / f"{cid}-smoke" / "train" / Path(episode).stem / "result.json"
        if not result.exists():
            return "the episode wrote no result"
        error = json.loads(result.read_text()).get("error")
        return error[:300] if error else None

    def test(self, cid):
        """CID on the test episodes, which no round ever sees or judges by."""
        entry = next(e for e in reversed(self.archive()) if e["id"] == cid)
        path = self.loop.worktree(self.cand(cid), entry["commit"])
        result = self.evaluate(cid, path, splits=("test",))["test"]
        self.log(f"{cid} on test: success {result['success_rate']:.2f} (skill alone {result['builtin_success_rate']:.2f}), "
                 f"value {result['value']:.3f}, {result['skill_seconds']} s and {result['credits']} credits per episode")
        return result

    def run(self, rounds):
        done = max((int(e["round"]) for e in self.archive() if isinstance(e.get("round"), int)), default=0)
        for number in range(done + 1, done + rounds + 1):
            self.round(number)
            if number == 1 and not skill_score.promising([e for e in self.archive() if e.get("round") == 1]):
                self.log("round 1 produced no reasonable candidate (none accepted or more than one standard "
                         "error up); stopping instead of running more rounds")
                return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="a [skills.<profile>] of config.toml (default: [skills] default)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init").add_argument("commit")
    sub.add_parser("baseline").add_argument("commit")
    run = sub.add_parser("run")
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--diagnose-model", help="Codex model for the diagnoses of this run (default: [skill] model)")
    sub.add_parser("test").add_argument("cid")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    loop = SkillLoop(args.profile)
    if args.command == "init":
        print(loop.init(args.commit))
    elif args.command == "baseline":
        loop.baseline(args.commit)
    elif args.command == "test":
        loop.test(args.cid)
    elif args.command == "run":
        loop.diagnose_model = args.diagnose_model
        loop.run(args.rounds)
    else:
        for entry in loop.archive():
            print(entry["id"], entry["status"], {s: (entry[s]["success_rate"], entry[s]["value"]) for s in SPLITS if s in entry},
                  entry.get("reason", "")[:120])


if __name__ == "__main__":
    main()
