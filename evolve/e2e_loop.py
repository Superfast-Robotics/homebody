"""End-to-end optimisation: Astra improves the code an agent session runs on (a profile's editable files, e.g. the
Dex1 skills and the agent harness), judged by full HomeBody agent sessions on a task suite scored from physics.

    python3 -m evolve.e2e_loop [--profile P] baseline COMMIT [--id b000]
    python3 -m evolve.e2e_loop [--profile P] run --rounds N [--strategy greedy|evolve]
    python3 -m evolve.e2e_loop [--profile P] test CID       the version on the test tasks (never used by a round)
    python3 -m evolve.e2e_loop [--profile P] status

A profile is an [e2e.<P>] table of config.toml: the suite (train, val and test globs of scene packages, each with a
sequence.json of scoring targets), sessions per task, the session settings, the agent's model, the services every
host starts, the editable files, the Astra models and the search. It runs the skill loops' machinery (skill_loop.py,
evo_loop.py) with sessions in place of skill episodes: an episode is one session of one task (TASK__rN); its value
is the share of the task's objects placed (objective "placed"), its stage value (mean object stage / 3) is reported.
Sessions of one task, and every task of one shared layout (the profile's `clusters` prefixes), count as one
independent unit in the standard errors. Val decides acceptance but never reaches the proposer; test runs only on
the explicit command.
"""
import argparse
import hashlib
import json
import re
import shlex
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

from . import score, tiers
from .evo_loop import EvoLoop
from .loop import QuotaExhausted, git
from .remote import rsync, suite_tasks
from .skill_loop import COMPLETE, HERE, SPLITS, SkillLoop, coder_digest, public_reason
from .workspace import copy_runs

SESSION_TIMEOUT = 7200  # seconds run_jobs.sh allows a session: a missing result is charged all of it
KEYFRAMES = 40           # head and wrist keyframes kept per failed train session in the workspace
COLLECTED = ("result.json", "run.log", "exit_code", "events.jsonl", "outcome.json", "trace.json", "REQUEST.md",
             "tools.jsonl", "start_*.py", "last_message.json")


def episode_key(task, repeat):
    return f"{task}__{repeat}"


def session_row(task, repeat, folder):
    """One collected session as an episode row: placed share, stage, success, credits and seconds; a session that
    wrote no result (crashed or timed out) is a failure charged the whole timeout."""
    path = folder / "result.json"
    key = episode_key(task, repeat)
    if not path.exists():
        log = folder / "run.log"
        tail = " | ".join(log.read_text(errors="replace").strip().splitlines()[-3:])[:300] if log.exists() else "no output"
        return {"episode": key, "task": task, "stage": 0.0, "placed": 0.0, "success": False, "crashed": True,
                "status": "crashed", "credits": 0.0, "seconds": float(SESSION_TIMEOUT), "objects": {},
                "error": f"session crashed: {tail}"}
    result = json.loads(path.read_text())
    objects = result.get("objects") or {}
    return {"episode": key, "task": task, "stage": float(result.get("stage", 0.0)),
            "placed": sum(o.get("stage") == 3 for o in objects.values()) / max(1, len(objects)),
            "success": bool(result.get("success")), "crashed": False, "status": result.get("status") or "?",
            "credits": float((result.get("cost") or {}).get("credits", 0.0)),
            "seconds": float(result.get("wall_seconds") or 0.0), "steps": result.get("steps", 0),
            "coder_calls": result.get("customizations", 0), "objects": objects}


def read_split(split_dir, expected=()):
    """episode key -> row for every session under SPLIT_DIR/<task>/<repeat>/, and a crashed row for every EXPECTED
    (task, repeat) that left nothing, so a version never gains by losing sessions."""
    split_dir = Path(split_dir)
    rows = {}
    if split_dir.exists():
        for task_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for repeat_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                rows[episode_key(task_dir.name, repeat_dir.name)] = session_row(task_dir.name, repeat_dir.name, repeat_dir)
    for task, repeat in expected:
        if episode_key(task, repeat) not in rows:
            rows[episode_key(task, repeat)] = session_row(task, repeat, split_dir / task / repeat)
    return rows


def summarise(rows, cluster):
    """A split's summary in the skill loops' shape (per-episode values for the paired comparison) plus placed shares
    by layout cluster."""
    if not rows:
        return {"episodes": 0, "errors": 0, "success_rate": 0.0, "value": 0.0, "placed": 0.0}
    n = len(rows)
    outcomes, layouts, by_task = {}, {}, {}
    for key, row in rows.items():
        outcome = "crashed" if row["crashed"] else f"{row['status']}, placed {row['placed']:.0%}"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        placed, total = layouts.get(cluster(key), (0, 0))
        layouts[cluster(key)] = (placed + sum(o.get("stage") == 3 for o in row["objects"].values()),
                                 total + max(1, len(row["objects"])))
        by_task.setdefault(row["task"], []).append(round(row["placed"], 3))
    return {"episodes": n, "errors": sum(r["crashed"] for r in rows.values()),
            "success_rate": round(sum(r["success"] for r in rows.values()) / n, 3),
            "value": round(sum(r["stage"] for r in rows.values()) / 3 / n, 3),
            "placed": round(sum(r["placed"] for r in rows.values()) / n, 3),
            "credits": round(sum(r["credits"] for r in rows.values()) / n, 3),
            "minutes": round(sum(r["seconds"] for r in rows.values()) / 60 / n, 1),
            "outcomes": dict(sorted(outcomes.items(), key=lambda item: -item[1])),
            "placed_by_layout": {c: f"{p}/{t}" for c, (p, t) in sorted(layouts.items())}, "by_task": by_task,
            "values": {k: r["stage"] / 3 for k, r in rows.items()},
            "placed_values": {k: r["placed"] for k, r in rows.items()},
            "held_values": {k: float(r["success"]) for k, r in rows.items()},
            "seconds": {k: r["seconds"] for k, r in rows.items()},
            "credit_values": {k: r["credits"] for k, r in rows.items()},
            "untimed": sorted(k for k, r in rows.items() if r["crashed"])}


def layout_cluster(episode, prefixes):
    """The independent unit of a session: its task (repeats together), or the shared layout its name's first
    matching PREFIXES entry names (variants of one scanned scene are not independent samples)."""
    task = re.sub(r"__r\d+$", "", episode.split("/")[-1])
    return next((unit for prefix, unit in prefixes.items() if task.startswith(prefix)), task)


def absolute(path, base):
    return path if path.startswith("/") else f"{base}/{path}"


class E2EMixin:
    """The skill loops' bookkeeping, search and judging (SkillLoop, EvoLoop) over agent sessions."""
    TABLE = "e2e"
    FOLDER = "e2e"

    # --- the suite ---
    @property
    def repeats(self):
        return int(self.cfg.get("repeats", 2))

    def tasks(self, split):
        cache = self.work / f"tasks_{split}.json"
        if not cache.exists():
            cache.write_text(json.dumps(suite_tasks(self.remote, self.cfg["suite"][split]), indent=1))
        return json.loads(cache.read_text())

    def expected(self, split):
        return [(t["name"], f"r{r}") for t in self.tasks(split) for r in range(self.repeats)]

    def episodes(self, split):
        return [episode_key(task, repeat) for task, repeat in self.expected(split)]

    def task_info(self):
        """{task: {"instruction", "names"}} of every suite task (instruction and entity names), fetched once."""
        cache = self.work / "task_info.json"
        if not cache.exists():
            folders = {t["name"]: (absolute(t["scene"], self.remote.base), absolute(t["targets"], self.remote.base))
                       for split in self.cfg["suite"] for t in self.tasks(split)}
            script = ("import json,sys\nout={}\nfor name,(scene,targets) in json.loads(sys.argv[1]).items():\n"
                      " s=json.load(open(scene+'/task_scene.json'));q=json.load(open(targets))\n"
                      " steps=q.get('steps',[])\n"
                      " names=[str(x[k]) for x in steps for k in ('fixture_id','label','destination_label') if x.get(k)]\n"
                      " out[name]={'instruction':s.get('default_task',''),'names':names+list(s.get('movable_entities',[]))}\n"
                      "print(json.dumps(out))")
            raw = self.remote.ssh(f"python3 - {shlex.quote(json.dumps(folders))} << 'EOF'\n{script}\nEOF")
            cache.write_text(raw)
        return json.loads(cache.read_text())

    def benchmark_names(self):
        names = set()
        for task, info in self.task_info().items():
            names.update([task, *info["names"]])
        return names

    def cluster(self, episode):
        return layout_cluster(episode, self.cfg.get("clusters", {}))

    def allowed(self, files):
        """Files outside the editable globs, and any the tier rules freeze (the agent loop, scorer, scenes)."""
        outside = super().allowed(files)
        return outside + [f for f in files if f not in outside and tiers.tier_of(f) == "frozen"]

    def objective_text(self):
        return ("placed objects: a session scores the share of its task's target objects the physics scorer finds "
                "placed (stage 3); the stage value (0 untouched, 1 lifted, 2 in the region, 3 placed; mean / 3) is "
                "reported but does not decide")

    # --- running sessions ---
    def results_root(self, cid):
        return f"{self.remote.work}/e2e/{self.cfg.get('workdir', self.skill)}/{cid}"

    def summaries(self, cid, splits=SPLITS):
        return {s: summarise(read_split(self.work / "results" / cid / s, self.expected(s)), self.cluster) for s in splits}

    def setup_fingerprint(self):
        """A hash of what every session assumes fixed: the session runner and scorer, the settings, the suite's scene
        packages, the robot, the planner and the seed's Python environment."""
        local = hashlib.md5(b"".join((HERE / name).read_bytes() for name in (
            "run_jobs.sh", "score_episode.py", self.cfg["settings"]))).hexdigest()[:6]
        folders = " ".join(sorted({absolute(t["scene"], self.remote.base) for split in self.cfg["suite"]
                                   for t in self.tasks(split)}))
        return local + "-" + self.remote.ssh(
            f"(cd {self.remote.base}/assets && find -L robot -type f | sort | xargs md5sum; "
            f"find -L {folders} -type f | sort | xargs md5sum; "
            f"md5sum {Path(self.remote.root)}/curobo_dex1/*.py {Path(self.remote.root)}/curobo_dex1/*.sh 2>/dev/null; "
            f"{self.remote.base}/.venv/bin/pip freeze 2>/dev/null | md5sum) "
            f"| awk '{{print $1}}' | md5sum | cut -c1-12", timeout=1200).strip()

    def evaluate_many(self, items, splits=SPLITS, videos=False):
        """Every session of every version in ITEMS ((cid, checkout, only, cand)) in one queue per host, versions
        interleaved; each version is collected, filled (a missing session is a crash) and summarised. {cid: summaries}."""
        setup, results, pending, queues = self.setup_fingerprint(), {}, [], []
        settings = self.cfg["settings"]
        for cid, checkout, _, cand in items:
            local = self.work / "results" / cid
            identity = {"commit": git("rev-parse", "HEAD", cwd=checkout), "setup": setup, "repeats": self.repeats}
            recorded = local / "identity.json"
            if all((local / s / COMPLETE).exists() for s in splits) and recorded.exists() \
                    and json.loads(recorded.read_text()) == identity:  # reused only for the same code and setup
                results[cid] = self.summaries(cid, splits)
                continue
            cand = cand or self.cand(cid)
            self.loop.remote.push_candidate(cand, checkout)
            root = self.results_root(cid)
            lines = [f"{self.remote.work}/cands/{cand} {absolute(t['scene'], self.remote.base)} "
                     f"{absolute(t['targets'], self.remote.base)} {root}/{split}/{t['name']}/r{r}"
                     for split in splits for t in self.tasks(split) for r in range(self.repeats)]
            self.remote.ssh(f"rm -rf {root} && mkdir -p {root}")
            pending.append((cid, local, root, identity))
            queues.append(lines)
        if not pending:
            return results
        queue = [line for batch in zip_longest(*queues) for line in batch if line]
        rsync("-a", str(HERE / "run_jobs.sh"), str(HERE / "score_episode.py"), str(HERE / "episode_settings.toml"),
              str(HERE / settings), f"{self.remote.host}:{self.remote.work}/")
        hosts, started = self.hosts(), time.monotonic()
        for host in hosts:
            for command in self.cfg.get("services", []):
                host.ssh(command, timeout=600)
        batch = f"{self.remote.work}/e2e/{self.cfg.get('workdir', self.skill)}/batch-{pending[0][0]}-{len(pending)}"
        self.remote.ssh(f"rm -rf {batch} && mkdir -p {batch}")
        parallel, model = int(self.cfg.get("parallel", self.remote.parallel)), self.cfg.get("agent_model",
                                                                                            self.loop.cfg["agent"]["model"])

        def run_share(index):
            host, share = hosts[index], queue[index::len(hosts)]
            if not share:
                return
            jobs = f"{batch}/jobs{index}.txt"
            host.ssh(f"cat > {jobs} << 'EOF'\n" + "\n".join(share) + "\nEOF")
            waves = -(-len(share) // parallel)
            host.ssh(f"bash {host.work}/run_jobs.sh {host.base} {parallel} {jobs} {host.work} {model} "
                     f"{host.work}/{settings}", timeout=waves * (SESSION_TIMEOUT + 300) + 600)

        with ThreadPoolExecutor(len(hosts)) as pool:
            list(pool.map(run_share, range(len(hosts))))
        seconds = time.monotonic() - started
        for cid, local, root, identity in pending:
            if local.exists():
                shutil.rmtree(local)
            local.mkdir(parents=True)
            rsync("-a", "--include=*/", *(f"--include={name}" for name in COLLECTED),
                  "--include=**/workspace/skills/*.py", "--include=**/workspace/lib/*.py", "--exclude=*",
                  "--prune-empty-dirs", f"{self.remote.host}:{root}/", f"{local}/")
            if score.quota_hit(local):  # infrastructure, not the code: stop without a verdict
                raise QuotaExhausted(f"the model's usage limit was hit while evaluating {cid}; resume once it resets")
            for split in splits:
                (local / split).mkdir(exist_ok=True)
                (local / split / COMPLETE).write_text("done\n")
            (local / "identity.json").write_text(json.dumps(identity))
            summaries = results[cid] = self.summaries(cid, splits)
            spent = sum(sum(s["credit_values"].values()) for s in summaries.values() if s.get("credit_values"))
            self.log(f"{cid}: " + ", ".join(f"{s} placed {v['placed']:.3f} value {v['value']:.3f} success "
                                             f"{v['success_rate']:.2f} by layout {v.get('placed_by_layout', {})}"
                                             for s, v in summaries.items())
                     + f"; {spent:.1f} session credits ({seconds:.0f} s for {len(pending)} version"
                     + f"{'s' if len(pending) > 1 else ''})")
        return results

    def audit(self, cid, summaries):
        for split in SPLITS:
            s = summaries.get(split, {})
            self.log(f"{cid} {split}: {s.get('episodes', 0)} sessions of {len(self.expected(split))}, "
                     f"{s.get('errors', 0)} crashed, {sum((s.get('credit_values') or {}).values()):.1f} credits")

    def smoke(self, cid, path):
        """No trial session: it would cost credits, and a crashing candidate is rejected by its crash share."""
        return None

    def init(self, commit, sample=8):
        raise SystemExit("init is for a skill benchmark; an end-to-end profile starts from a baseline")

    def remeasure(self, ids):
        raise SystemExit("remeasure is for a skill benchmark")

    def test(self, cid):
        entry = next(e for e in reversed(self.archive()) if e["id"] == cid)
        path = self.loop.worktree(self.cand(cid), entry["commit"])
        result = self.evaluate(cid, path, splits=("test",))["test"]
        self.log(f"{cid} on test: placed {result['placed']:.3f}, value {result['value']:.3f}, success "
                 f"{result['success_rate']:.2f}, by layout {result.get('placed_by_layout')}, {result['credits']} credits "
                 f"and {result['minutes']} min per session")
        return result

    # --- what the proposer sees (train only) ---
    def run_table(self, cid):
        rows = read_split(self.work / "results" / cid / "train", self.expected("train"))
        lines = ["| session | placed | stage | status | object outcomes | coder calls | credits | min |",
                 "|---|---|---|---|---|---|---|---|"]
        for key, row in sorted(rows.items()):
            objects = "; ".join(f"{n}: {o.get('stage')}{' dropped' if o.get('dropped') else ''}"
                                f"{' (' + str(o['reason']) + ')' if o.get('reason') else ''}" for n, o in row["objects"].items())
            lines.append(f"| {key} | {row['placed']:.2f} | {row['stage']:.2f} | {row['status']} | "
                         f"{objects or row.get('error', '-')[:200]} | {row.get('coder_calls', 0)} | {row['credits']:.1f} | "
                         f"{row['seconds'] / 60:.1f} |")
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
                 f"{summary['success_rate']:.2f}, {summary['credits']} credits and {summary['minutes']} min per session.",
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
        child = read_split(self.work / "results" / cid / "train", self.expected("train"))
        before = read_split(self.work / "results" / parent / "train", self.expected("train"))
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
    sub.add_parser("test").add_argument("cid")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    loop = E2ELoop(args.profile)
    if args.command == "run" and (args.strategy or loop.cfg.get("strategy", "greedy")) == "evolve":
        loop = E2EEvoLoop(args.profile)
    if args.command == "baseline":
        loop.baseline(args.commit, args.id)
    elif args.command == "test":
        loop.test(args.cid)
    elif args.command == "run":
        loop.run(args.rounds)
    else:
        for entry in loop.archive():
            print(entry["id"], entry["status"], {s: (entry[s].get("placed"), entry[s].get("value")) for s in SPLITS
                                                 if s in entry}, entry.get("reason", "")[:120])


if __name__ == "__main__":
    main()
