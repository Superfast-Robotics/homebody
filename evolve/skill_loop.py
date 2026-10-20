"""Skill-level optimisation: Astra improves one built-in skill (config [skill] name) on the skill
benchmark (skill_bench.py episodes, no model at run time), round by round, from the version recorded
as the skill baseline.

    python3 -m evolve.skill_loop [--profile P] init COMMIT       Astra writes the skill's first version from COMMIT
    python3 -m evolve.skill_loop [--profile P] baseline COMMIT   measure COMMIT on every episode
    python3 -m evolve.skill_loop [--profile P] run --rounds N    diagnose, patch, evaluate, accept
        [--strategy greedy|evolve]   greedy (one patch per mechanism, from the best) or evolve (evo_loop.py:
                                     a population search); default: the profile's `strategy`, else greedy
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
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
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
EPISODE_TIMEOUT = 5400  # seconds run_skill_jobs.sh allows an episode: a missing result is charged all of it
EDITABLE = ("src/homebody/skills/{skill}.py", "src/homebody/motion/*", "src/homebody/helpers/*",
            "src/homebody/toolkit/*", "tests/*")
SPLITS = ("train", "val")
# what the operator may spend, not what is measured: kept out of every identity, so raising a cap keeps the results
SPENDING = ("round_credit_budget", "evaluation_credit_budget", "session_credit_cap", "expected_session_credits",
            "decision_credit_ceiling", "coder_credit_ceiling", "astra_call_credit_ceiling")


def measured(cfg):
    """A profile without its spending authorisation (SPENDING) and its prose note: what an identity hashes."""
    return {k: v for k, v in cfg.items() if k not in SPENDING and k != "brief_note"}


def touched(row):
    """The first things an episode's hand touched (result.json hand_contacts), for the results table."""
    return "; ".join(f"{c['with']} by {c['hand_part'].removeprefix('right_').removeprefix('left_')} from {c['first']} s"
                     for c in (row.get("hand_contacts") or [])[:3]) or "-"


def public_reason(entry):
    """An archive entry's verdict without validation numbers: what the proposer and diagnoser may read."""
    verdict = entry.get("verdict")
    if verdict:
        return skill_score.describe_train(verdict)
    reason = entry.get("reason", "")
    if reason.startswith("inert"):  # before round 5 this only meant unchanged outcomes on a few probe episodes
        return "outcomes unchanged on a few probe episodes (old probe: whether its code ran is unknown)"
    return reason


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
    TABLE = "skills"  # the config.toml table of this loop's profiles
    FOLDER = "skill"  # its work folder, under the loop's

    def __init__(self, profile=None):
        self.loop = Loop()
        profiles = self.loop.cfg[self.TABLE]
        self.profile = profile or profiles["default"]
        self.cfg = profiles[self.profile]
        self.skill = self.cfg["name"]
        self.work = self.loop.work / self.FOLDER / self.cfg.get("workdir", self.skill)
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
        # what accepts a candidate: "value" (stage / 2) or "held" (completed holds), episodes weighted by id prefix
        self.objective = {"held": "held_values", "placed": "placed_values"}.get(self.cfg.get("objective"), "values")
        self.weights = dict(self.cfg.get("weights", {}))
        self.speed = bool(self.cfg.get("speed_acceptance", True))  # whether a time or cost win alone may accept

    # --- bookkeeping ---
    def archive(self):
        if not self.archive_path.exists():
            return []
        return [json.loads(line) for line in self.archive_path.read_text().splitlines() if line.strip()]

    def record(self, entry):
        stored = {k: v for k, v in entry.items() if k not in ("train", "val")}
        stored.update({s: {k: v for k, v in entry[s].items() if k not in ("values", "seconds", "placed_values")}
                       for s in SPLITS if s in entry})
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

    def cluster(self, episode):
        """The independent unit EPISODE belongs to, for the standard errors."""
        return skill_score.cluster(episode)

    def judge(self, parent, child):
        """The acceptance rule with this profile's objective, weights, speed switch and clusters."""
        return skill_score.judge(parent, child, self.objective, self.weights, self.speed, self.cluster)

    def summaries(self, cid, splits=SPLITS):
        return {s: skill_score.summary(skill_score.read(self.work / "results" / cid / s)) for s in splits}

    # --- running episodes ---
    def episodes(self, split):
        listing = self.remote.ssh(f"cd {self.remote.work} && ls {self.cfg['episodes']}/{split}/*.json")
        return listing.split()

    def fill_missing(self, local, lines):
        """An episode that left no result (its process died or timed out) scores as a crashed failure, not as an
        episode the version did not run: every job line has a result.json."""
        for line in lines:
            *_, out = line.split()
            split, name = Path(out).parent.name, Path(out).name
            path = local / split / name / "result.json"
            if path.exists():
                continue
            local_episode = self.work / "episodes" / split / f"{name}.json"
            family = json.loads(local_episode.read_text()).get("family", "?") if local_episode.exists() else "?"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"episode": name, "skill": self.skill, "family": family, "stage": 0,
                                        "success": False, "crashed": True, "timing": {"skill_wall": EPISODE_TIMEOUT},
                                        "error": "no result: the episode process died or timed out"}))
            self.log(f"{name}: no result, scored as a crashed failure")

    def hosts(self):
        """The evaluation hosts an evaluation spreads its episodes over: all of them ([remote] hosts), or the one
        this thread is pinned to."""
        pinned = getattr(self.loop._local, "remote", None)
        return [pinned] if pinned is not None else list(self.loop.remotes)

    def cand(self, cid):
        """The remote candidate name and local worktree name of CID, unique per profile."""
        return f"{self.cfg.get('workdir', self.skill)}-{cid}"

    def evaluate(self, cid, checkout, splits=SPLITS, only=None, cand=None, videos=False):
        """Push CID's code and run every episode of SPLITS on the remote (or ONLY: split -> episode paths, under the
        candidate copy CAND); collect and summarise."""
        return self.evaluate_many([(cid, checkout, only, cand)], splits, videos)[cid]

    def evaluate_many(self, items, splits=SPLITS, videos=False):
        """Evaluate several versions at once: ITEMS are (cid, checkout, only, cand). Every episode of every version
        goes into one queue per host, so a version's slowest episodes overlap others' work instead of idling the
        hosts; each version is then collected and summarised as on its own. {cid: summaries}."""
        setup, results, pending, lines = self.setup_fingerprint(), {}, [], []
        for cid, checkout, only, cand in items:
            local = self.work / "results" / cid
            identity = {"commit": git("rev-parse", "HEAD", cwd=checkout), "setup": setup}
            recorded = local / "identity.json"
            if all((local / s / COMPLETE).exists() for s in splits) and recorded.exists() \
                    and json.loads(recorded.read_text()) == identity:  # reused only for the same code and setup
                results[cid] = self.summaries(cid, splits) if splits != SPLITS else self.summaries(cid)
                continue
            cand = cand or self.cand(cid)
            self.loop.remote.push_candidate(cand, checkout)
            root = f"{self.remote.work}/skill/{self.cfg.get('workdir', self.skill)}/{cid}"
            mine = [f"{self.remote.work}/cands/{cand} {self.remote.work}/{episode} {root}/{split}/{Path(episode).stem}"
                    for split in splits for episode in (only[split] if only else self.episodes(split))]
            # only the splits run now are replaced: a test run never deletes the train and val evidence
            self.remote.ssh(" && ".join(f"rm -rf {root}/{s}" for s in splits) + f" && mkdir -p {root}")
            pending.append((cid, local, root, mine, identity))
            lines.append(mine)
        if not pending:
            return results
        queue = [line for batch in zip_longest(*lines) for line in batch if line]  # versions interleaved
        rsync("-a", str(HERE / "skill_bench.py"), str(HERE / "skill_bench_tasks.py"), str(HERE / "run_skill_jobs.sh"),
              f"{self.remote.host}:{self.remote.work}/")
        started = time.monotonic()
        hosts = self.hosts()
        for host in hosts:  # the profile's services (e.g. a planning server) up on every host the episodes run on
            for command in self.cfg.get("services", []):
                host.ssh(command, timeout=600)
        batch = f"{self.remote.work}/skill/{self.cfg.get('workdir', self.skill)}/batch-{pending[0][0]}-{len(pending)}"
        self.remote.ssh(f"rm -rf {batch} && mkdir -p {batch}")

        def run_share(index):  # the hosts share the group volume: one copy per version, one results folder each
            host, share = hosts[index], queue[index::len(hosts)]
            if not share:
                return
            jobs = f"{batch}/jobs{index}.txt"
            host.ssh(f"cat > {jobs} << 'EOF'\n" + "\n".join(share) + "\nEOF")
            host.ssh(f"bash {host.work}/run_skill_jobs.sh {host.base} {self.cfg['parallel']} {jobs} "
                     f"{host.work} {int(self.coder)} '{self.cfg.get('coder_model', '')}' "
                     f"'{self.cfg.get('coder_rates', '')}' '{self.cfg.get('coder_seconds', '')}' {int(videos)}", timeout=24 * 3600)

        with ThreadPoolExecutor(len(hosts)) as pool:
            list(pool.map(run_share, range(len(hosts))))
        seconds = time.monotonic() - started
        for cid, local, root, mine, identity in pending:
            for split in splits:
                if (local / split).exists():
                    shutil.rmtree(local / split)
            local.mkdir(parents=True, exist_ok=True)
            rsync("-a", "--include=*/", "--include=result.json", "--include=trace.json", "--include=run.log",
                  "--include=arm_track.json", "--include=REQUEST.md", "--include=tools.jsonl", "--include=start_*.py",
                  "--include=last_message.json", "--include=**/coder/call-*/events.jsonl",
                  "--include=**/workspace/skills/*.py", "--include=**/workspace/lib/*.py", "--exclude=*",
                  "--prune-empty-dirs", f"{self.remote.host}:{root}/", f"{local}/")
            for split in splits:
                (local / split).mkdir(exist_ok=True)
            self.fill_missing(local, mine)
            (local / "identity.json").write_text(json.dumps(identity))
            for split in splits:
                (local / split / COMPLETE).write_text("done\n")
            if splits != SPLITS:
                results[cid] = self.summaries(cid, splits)
                continue
            summaries = results[cid] = self.summaries(cid)
            self.log(f"{cid}: train success {summaries['train']['success_rate']:.2f} value {summaries['train']['value']:.3f} "
                     f"held {summaries['train']['held_by_group']}, val success {summaries['val']['success_rate']:.2f} value "
                     f"{summaries['val']['value']:.3f} held {summaries['val']['held_by_group']} "
                     f"({seconds:.0f} s for {len(pending)} version{'s' if len(pending) > 1 else ''})")
        return results

    # --- commands ---
    def baseline(self, commit, cid="p000", videos=False):
        """Measure COMMIT as CID; a later baseline (a pipeline change under the skill) becomes the version rounds build
        on. It records the scenes' and robot's fingerprint, which later rounds check (the frozen setup)."""
        commit = git("rev-parse", commit)
        path = self.loop.worktree(self.cand(f"{cid}"), commit)
        before = self.manifest(path, commit)
        summaries = self.evaluate(cid, path, videos=videos)
        after = self.manifest(path, commit, refresh=True)
        if after != before:
            raise SystemExit(f"{cid}: the experiment changed while it ran: {before} -> {after}; not recorded")
        self.audit(cid, summaries)
        self.record({"id": cid, "round": 0, "commit": commit, "status": "baseline",
                     "reason": f"{self.skill} skill baseline", "setup": before["setup"], "manifest": before, **summaries})

    def verify_checkout(self, path, commit):
        """PATH must be exactly COMMIT with no local changes: what is recorded is what runs."""
        head, dirty = git("rev-parse", "HEAD", cwd=path), git("status", "--porcelain", cwd=path)
        if head != commit or dirty:
            raise SystemExit(f"{path} is {head[:10]}{' with local changes' if dirty else ''}, not {commit[:10]}")

    def manifest(self, path, commit, refresh=False):
        """What an evaluation depends on: the candidate's tree, the profile (episodes, objective, weights, services,
        prompts), and the setup fingerprint (scenes, robot, episodes, driver and scorer, planner, the seed's Python
        environment). Checked before and after a baseline runs."""
        self.verify_checkout(path, commit)
        profile = hashlib.md5(json.dumps(measured(self.cfg), sort_keys=True).encode()).hexdigest()[:12]
        return {"tree": git("rev-parse", f"{commit}^{{tree}}"), "profile": profile,
                "setup": self.setup_fingerprint(refresh=refresh)}

    def audit(self, cid, summaries):
        """Log a baseline's reconciliation: episodes scored, crashed and set up wrongly per split."""
        for split in SPLITS:
            s = summaries.get(split, {})
            expected = len(self.episodes(split))
            self.log(f"{cid} {split}: {s.get('episodes', 0)} scored of {expected} episodes, {s.get('errors', 0)} crashed, "
                     f"{len(s.get('untimed', []))} untimed")

    def setup_fingerprint(self, refresh=False):
        """A hash of the benchmark's scene packages and robot profiles on the remote seed: what a round assumes fixed
        (measured every time; REFRESH is for loops that keep it)."""
        episodes = f"{self.remote.work}/{self.cfg['episodes']}"
        local = hashlib.md5(b"".join((HERE / name).read_bytes() for name in  # the episode driver and the scoring
                                     ("skill_bench.py", "skill_score.py", "run_skill_jobs.sh"))).hexdigest()[:6]
        return local + "-" + self.remote.ssh(
            f"cd {self.remote.base}/assets && (find -L robot real2sim skillbench -type f -not -path '*.bak*' | sort | xargs md5sum; "
            f"find {episodes} -type f | sort | xargs md5sum; "
            f"md5sum {Path(self.remote.root)}/curobo_dex1/*.py {Path(self.remote.root)}/curobo_dex1/*.sh 2>/dev/null; "
            f"{self.remote.base}/.venv/bin/pip freeze 2>/dev/null | md5sum) "
            f"| awk '{{print $1}}' | md5sum | cut -c1-12", timeout=600).strip()

    def check_frozen(self):
        """Stop when the scenes or robot changed since the latest baseline: its results no longer compare."""
        base = [e for e in self.archive() if e.get("status") == "baseline"][-1]
        if base.get("setup") and base["setup"] != self.setup_fingerprint(refresh=True):
            raise SystemExit(f"the scenes or robot changed since baseline {base['id']}: re-baseline before a round")

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
            coder_minutes=minutes, benchmark=benchmark, objective=self.objective_text(),
            editable=", ".join(p.format(skill=self.skill) for p in self.editable))
        (ws / "BRIEF.md").write_text(brief + (f"\n## This robot\n\n{self.brief_note.strip()}\n" if self.brief_note else ""))
        train = skill_score.read(self.work / "results" / best["id"] / "train")
        summary = skill_score.summary(train)
        lines = [f"# {self.skill} on the train episodes with version {best['id']}", "",
                 f"Held after the carry, by group: {json.dumps(summary.get('held_by_group', {}))}", "",
                 (f"Success {summary['success_rate']:.2f} (the skill alone {summary['builtin_success_rate']:.2f}), "
                  f"value {summary['value']:.3f} (stage / 2), {summary['skill_seconds']} s and {summary['credits']} "
                  f"coder credits per episode; coder calls {summary['coder_calls']}, OK {summary['coder_ok']}."), "",
                 f"Outcomes of the skill call: {json.dumps(summary['outcomes'])}",
                 f"Success by family: {json.dumps(summary['success_by_family'])}", "",
                 "| episode | family | stage | skill result | message | coder | coder message | s | hand touched |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for episode, row in sorted(train.items()):
            result, coder = skill_score.call(row), row.get("coder") or {}
            lines.append(f"| {episode} | {row['family']} | {row['stage']} | {result.get('code', row.get('error', '?'))} | "
                         f"{(result.get('message') or '')[:120]} | {coder.get('code', '-')} | "
                         f"{(coder.get('message') or '')[:160]} | {skill_score.seconds(row)} | {touched(row)} |")
        (ws / "results.md").write_text("\n".join(lines) + "\n")
        for episode in train:
            source = self.work / "results" / best["id"] / "train" / episode
            target = ws / "episodes" / episode
            target.mkdir(parents=True)
            for name in ("result.json", "trace.json", "arm_track.json"):
                if (source / name).exists():
                    shutil.copy(source / name, target / name)
            if train[episode]["stage"] < 2:
                self.keyframes(best["id"], "train", episode, target)
            for call_dir in sorted(source.rglob("coder/call-*")):
                (target / f"coder-{call_dir.name}.md").write_text(coder_call(call_dir))
        (ws / "coder_edits.md").write_text(coder_digest(self.work / "results" / best["id"] / "train"))
        history = [f"## {e['id']} (round {e.get('round')}): {e['status']}\n{public_reason(e)}\n{(e.get('notes') or '')[:1500]}"
                   for e in self.archive()]
        (ws / "history.md").write_text("# Earlier versions of the skill\n\n" + "\n\n".join(history) + "\n")
        return ws

    def objective_text(self):
        weighted = ", ".join(f"episodes starting {p!r} count {w:g}x" for p, w in self.weights.items())
        if self.objective == "held_values":
            return ("completed holds: an episode scores 1 when the object is still held after the whole carry "
                    "(every leg arrived), else 0; stage value (lifted = 0.5) is reported but does not decide"
                    + (f"; {weighted}" if weighted else ""))
        return "stage value: 0 not lifted, 0.5 lifted but not held at the end, 1 held" + (f"; {weighted}" if weighted else "")

    def keyframes(self, cid, split, episode, target, limit=40):
        """The episode's per-stage head and wrist keyframes (trace/*.jpg) from the remote results, at most LIMIT."""
        source = f"{self.remote.work}/skill/{self.cfg.get('workdir', self.skill)}/{cid}/{split}/{episode}/session/"
        frames = target / "keyframes"
        frames.mkdir(exist_ok=True)
        try:
            rsync("-a", "--prune-empty-dirs", "--include=*/", "--include=*.jpg", "--exclude=*",
                  f"{self.remote.host}:{source}", f"{frames}/")
        except Exception:  # noqa: BLE001 -- keyframes are extra evidence; a version without them is still diagnosed
            return
        jpgs = sorted(frames.rglob("*.jpg"))
        for path in jpgs:
            call = path.parent.name  # call-NNN: the episode's skill calls in order
            path.rename(frames / f"{call}-{path.name}")
        for path in sorted(frames.glob("*.jpg"))[:-limit]:  # the last stages, where an episode fails, are kept
            path.unlink()
        for folder in sorted((p for p in frames.rglob("*") if p.is_dir()), reverse=True):
            folder.rmdir()

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
        self.check_frozen()
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
            diagnosis = self.codex_read(prompt, reader, diagnosis_path, **self.diagnoser())
        indices = range(1, self.proposals + 1)
        paths = {index: self.loop.worktree(self.cand(f"s{number:03d}c{index}"), best["commit"]) for index in indices}
        with ThreadPoolExecutor(len(paths)) as pool:
            ready = [entry for entry in pool.map(
                lambda index: self.prepare(number, index, paths[index], best, ws, diagnosis, round_dir), indices) if entry]
        accepted = []
        for entry in ready:  # one at a time: the coder shares one GPU, and three evaluations at once starve it
            summaries = self.evaluate(entry["id"], paths[entry["mechanism"]])
            verdict = self.judge(self.summaries(best["id"]), summaries)
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
                note = self.codex_write(prompt, path, note_path, **self.proposer())
            except astra.BudgetExceeded:  # the round's credit cap, not the proposal: stop the round
                raise
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
        reason = self.screen(best["commit"], commit, files)
        if reason:
            self.record({**entry, "status": "rejected", "reason": reason})
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

    def archived_rounds(self):
        """The rounds of an older work folder that has no rounds.json: every round in its archive."""
        return sorted({int(e["round"]) for e in self.archive() if isinstance(e.get("round"), int) and e["round"] > 0})

    def round_state(self):
        """rounds.json: {"finished": rounds that finished, "started": the round under way (resumed, never skipped),
        "stop": the stop decision}. Written before a round's first proposal and, with the stop decision, after its
        last verdict; an older list-only record, or none, reads as finished rounds (archived_rounds)."""
        path = self.work / "rounds.json"
        state = json.loads(path.read_text()) if path.exists() else {"finished": self.archived_rounds()}
        if isinstance(state, list):
            state = {"finished": state}
        legacy = self.work / "STOP.json"
        stop = state.get("stop") or (json.loads(legacy.read_text()) if legacy.exists() else None)
        return {"finished": sorted(state.get("finished", [])), "started": state.get("started"), "stop": stop}

    def save_round_state(self, state):
        """Replace rounds.json in one step: completion and the stop decision are never written apart."""
        path = self.work / "rounds.json"
        path.with_suffix(".tmp").write_text(json.dumps(state))
        os.replace(path.with_suffix(".tmp"), path)

    def completed_rounds(self):
        return self.round_state()["finished"]

    def barren(self, entries):
        """Whether a round's ENTRIES hold no reasonable candidate (none accepted or more than one standard error up)."""
        return not skill_score.promising(entries)

    def run(self, rounds, stop_rule=True):
        """ROUNDS more rounds; with STOP_RULE, a barren round 1 stops the run (and later runs)."""
        state = self.round_state()
        if state["stop"]:
            self.log(f"stopped earlier: {state['stop']['reason']}; remove \"stop\" from {self.work / 'rounds.json'} "
                     f"(and {self.work / 'STOP.json'}) to run more rounds")
            return
        done = max(state["finished"], default=0)
        for number in range(done + 1, done + rounds + 1):
            self.save_round_state({**state, "started": number})
            self.round(number)
            stop = None
            if stop_rule and number == 1 and self.barren([e for e in self.archive() if e.get("round") == 1]):
                stop = {"round": number, "reason": "round 1 produced no reasonable candidate; not worth more rounds"}
            state = {"finished": sorted({*state["finished"], number}), "started": None, "stop": stop}
            self.save_round_state(state)
            if stop:
                (self.work / "STOP.json").write_text(json.dumps(stop))  # a readable copy of the decision
                self.log(f"{stop['reason']}; stopping (recorded in {self.work / 'rounds.json'})")
                return

    # --- what a profile may redirect ---
    def jail(self):
        """The folders a jailed proposer may also read (astra.jailed), or None to run Codex without a jail."""
        return None

    def astra_call(self, prompt, cd, sandbox, out, **model):
        """Every Codex call of a round (diagnosis, proposals, crossover, rewrite, lessons) goes through here."""
        return astra.codex(prompt, cd, sandbox, out, jail=self.jail(), **model)

    def codex_write(self, prompt, path, out, **model):
        """A proposer writing a patch in the worktree PATH."""
        return self.astra_call(prompt, path, "workspace-write", out, **model)

    def codex_read(self, prompt, path, out, **model):
        """A read-only diagnosis of the checkout PATH."""
        return self.astra_call(prompt, path, "read-only", out, **model)

    def screen(self, base, commit, files):
        """Why a patch from BASE to COMMIT (changing FILES) is refused before it runs, or None."""
        outside = self.allowed(files)
        if outside:
            return f"edits outside the skill: {outside}"
        specific = self.specific(base, commit)
        return f"not general: the source names {specific}" if specific else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="a [skills.<profile>] of config.toml (default: [skills] default)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init").add_argument("commit")
    base = sub.add_parser("baseline")
    base.add_argument("commit")
    base.add_argument("--id", default="p000", help="archive id (p001, ... for a re-baseline after a pipeline change)")
    base.add_argument("--videos", action="store_true", help="also record each episode's room and shoulder videos (remote)")
    run = sub.add_parser("run")
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--diagnose-model", help="Codex model for the diagnoses of this run (default: [skill] model)")
    run.add_argument("--strategy", choices=("greedy", "evolve"), help="the round's search (default: profile strategy)")
    sub.add_parser("test").add_argument("cid")
    sub.add_parser("remeasure").add_argument("ids", nargs="+", help="older versions to re-score on the current best")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    loop = SkillLoop(args.profile)
    if args.command == "run" and (args.strategy or loop.cfg.get("strategy", "greedy")) == "evolve":
        from .evo_loop import EvoLoop
        loop = EvoLoop(args.profile)
    if args.command == "init":
        print(loop.init(args.commit))
    elif args.command == "baseline":
        loop.baseline(args.commit, args.id, args.videos)
    elif args.command == "test":
        loop.test(args.cid)
    elif args.command == "remeasure":
        from .evo_loop import EvoLoop
        EvoLoop(args.profile).remeasure(args.ids)
    elif args.command == "run":
        loop.diagnose_model = args.diagnose_model
        loop.run(args.rounds)
    else:
        for entry in loop.archive():
            print(entry["id"], entry["status"], {s: (entry[s]["success_rate"], entry[s]["value"]) for s in SPLITS if s in entry},
                  entry.get("reason", "")[:120])


if __name__ == "__main__":
    main()
