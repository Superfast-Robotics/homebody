"""Evolutionary skill optimisation: the skill loop's benchmark, screens and acceptance, with a population
search in place of one greedy patch per mechanism (ideas from ShinkaEvolve, ASPIRE's evosearch and GEPA).

    python3 -m evolve.skill_loop [--profile P] run --rounds N --strategy evolve
    (or `strategy = "evolve"` in the profile)

Each round:
1. Archive: every scored version built on the latest baseline (baselines, accepted and rejected candidates)
   can be a parent. Parents are sampled by train value and by how few offspring they already have (weighted
   sampling), from the top versions and the train Pareto front (versions best on some episode); val never
   picks a parent. Older versions (before a re-baseline) still serve as inspirations and crossover partners.
2. Diagnosis of the best version, as in the greedy loop, with the lessons file and inspiration programs.
3. K candidates (profile `evo_candidates`), each a focused diff on one mechanism from a sampled parent, plus
   a crossover of two complementary parents and, after barren rounds, one larger rewrite.
4. Each diff's model is picked by a UCB1 bandit over `evo_models`, rewarded by its train gain over its
   parent, so a cheaper model writes more of them when it does as well; crossovers and rewrites take the
   first (strongest) model.
5. Probe: the candidate runs a few train episodes (one per outcome of its parent); a crash is rejected, and
   so is a candidate that behaves exactly like its parent there (its change never ran).
6. Survivors are evaluated in full one after another, each spread over all evaluation hosts, and judged
   against the current best with the greedy loop's rule (skill_score.judge); the largest accepted gain
   becomes the best.
7. Lessons: a cheap model rewrites lessons.md from the whole history, for the next round's prompts.
"""
import json
import math
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import astra, skill_score
from .loop import git
from .skill_loop import HERE, SPLITS, SkillLoop

DEFAULT_MODELS = ("gpt-6-astra", "gpt-6.1-sol")
TOP_PARENTS = 3          # the top versions by train value are always in the parent pool
SIGMOID_SHARPNESS = 10.0  # weighted parent sampling: sigmoid of (value - median) * this
INSPIRATIONS = 3          # top other versions shown to every proposal, plus one random
STALLED_ROUNDS = 2        # consecutive rounds without an accepted candidate before a rewrite slot
PROBE_EPISODES = 8
PARTNER_MARGIN = 0.05   # a crossover partner's train value is at most this far below its parent's
UCB_EXPLORATION = math.sqrt(2.0)


def fitness(summaries):
    """A version's value for parent selection: its train value (val never steers the search)."""
    return summaries["train"]["value"]


def dominates(a, b):
    """Whether per-episode values A are at least B's everywhere and higher somewhere (shared episodes)."""
    shared = a.keys() & b.keys()
    return all(a[e] >= b[e] for e in shared) and any(a[e] > b[e] for e in shared)


def pareto_front(values):
    """The ids of VALUES ({id: {episode: value}}) that no other version dominates."""
    return [i for i, v in values.items() if not any(dominates(w, v) for j, w in values.items() if j != i)]


def parent_weights(scores, offspring):
    """Weighted sampling: a sigmoid of value over the median, times 1 / (1 + offspring)."""
    median = sorted(scores.values())[len(scores) // 2]
    return {i: 1 / (1 + math.exp(-SIGMOID_SHARPNESS * (s - median))) / (1 + offspring.get(i, 0))
            for i, s in scores.items()}


def ucb1(stats, models):
    """The model with the highest UCB1 score; one never tried comes first. STATS: {model: [pulls, reward sum]}."""
    untried = [m for m in models if stats.get(m, [0, 0.0])[0] == 0]
    if untried:
        return untried[0]
    total = sum(stats[m][0] for m in models)
    return max(models, key=lambda m: stats[m][1] / stats[m][0] + UCB_EXPLORATION * math.sqrt(math.log(total) / stats[m][0]))


def reward(gain):
    """ShinkaEvolve's bandit reward: exp(max(gain, 0)) - 1."""
    return math.exp(max(gain, 0.0)) - 1


def fingerprint(row):
    """What an episode did, to tell a candidate that changed nothing: stage, the call's result and timing."""
    call = skill_score.call(row)
    return (row.get("stage"), call.get("code"), call.get("message"), round(row.get("simulation_seconds") or 0, 1),
            row.get("error"))


class EvoLoop(SkillLoop):
    def __init__(self, profile=None):
        super().__init__(profile)
        self.models = tuple(self.cfg.get("evo_models", DEFAULT_MODELS))
        self.candidates = int(self.cfg.get("evo_candidates", 4))
        self.lesson_model = self.cfg.get("evo_lesson_model", "gpt-6.1-sol")
        self.rng = random.Random(self.cfg.get("evo_seed", 0))

    def on_host(self, index, function, *args):
        """FUNCTION(*ARGS) with this thread pinned to evaluation host INDEX (round robin), then unpinned."""
        self.loop._local.remote = self.loop.remotes[index % len(self.loop.remotes)]
        try:
            return function(*args)
        finally:
            del self.loop._local.remote

    # --- the archive as a population ---
    def population(self):
        """Every scored version still on record, latest verdict per id: {id: entry}."""
        latest = {}
        for entry in self.archive():
            if all(s in entry for s in SPLITS) and entry.get("commit") and (self.work / "results" / entry["id"]).exists():
                latest[entry["id"]] = entry
        return latest

    def lineage(self, population):
        """The versions built on the latest baseline (a re-baseline after a pipeline change starts a new lineage)."""
        base = [e for e in self.archive() if e.get("status") == "baseline"][-1]["commit"]
        return {i: e for i, e in population.items()
                if subprocess.run(["git", "merge-base", "--is-ancestor", base, e["commit"]], cwd=HERE.parent, check=False,
                                  capture_output=True).returncode == 0}

    def skill_diff(self, a, b):
        """The skill folder's diff from commit A to commit B (what one version changed against another)."""
        return git("diff", a, b, "--", str(Path(self.source).parent))

    def scored(self, ids):
        return {i: self.summaries(i) for i in ids}

    def offspring(self):
        counts = {}
        for entry in self.archive():
            for parent in entry.get("parents") or [entry.get("parent")]:
                if parent:
                    counts[parent] = counts.get(parent, 0) + 1
        return counts

    def parent_pool(self, population, summaries):
        """The top versions by train value and the train Pareto front."""
        ranked = sorted(population, key=lambda i: -fitness(summaries[i]))
        front = pareto_front({i: summaries[i]["train"]["values"] for i in population})
        return list(dict.fromkeys(ranked[:TOP_PARENTS] + front))

    def sample_parent(self, pool, summaries, offspring):
        weights = parent_weights({i: fitness(summaries[i]) for i in pool}, offspring)
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def complementary(self, pool, others, summaries):
        """A POOL version and another scored version (any lineage) that each win the most train episodes the
        other loses, or None; the first is the parent the crossover is written on."""
        best, pair = 0, None
        for a in pool:
            for b in others:
                if a != b and fitness(summaries[b]) >= fitness(summaries[a]) - PARTNER_MARGIN:
                    va, vb = summaries[a]["train"]["values"], summaries[b]["train"]["values"]
                    wins = [sum(va[e] > vb.get(e, 0) for e in va), sum(vb[e] > va.get(e, 0) for e in vb)]
                    if min(wins) > best:
                        best, pair = min(wins), (a, b)
        return pair

    def bandit(self):
        """{model: [pulls, reward sum]} from the candidates on record (train gain over their parent)."""
        population = self.population()
        stats = {m: [0, 0.0] for m in self.models}
        for entry in self.archive():
            model, parent = entry.get("model"), entry.get("parent")
            if model not in stats:
                continue
            stats[model][0] += 1
            if entry["id"] in population and parent in population:
                stats[model][1] += reward(fitness(self.summaries(entry["id"])) - fitness(self.summaries(parent)))
        return stats

    def stalled(self):
        rounds = sorted({e["round"] for e in self.archive() if isinstance(e.get("round"), int) and e["round"] > 0})
        recent = rounds[-STALLED_ROUNDS:]
        return len(recent) == STALLED_ROUNDS and not any(
            e.get("status") == "accepted" for e in self.archive() if e.get("round") in recent)

    # --- what every prompt sees ---
    def evo_workspace(self, round_dir, best, population, summaries):
        ws = self.workspace(round_dir, best)
        lessons = self.work / "lessons.md"
        (ws / "lessons.md").write_text(lessons.read_text() if lessons.exists() else "# Lessons so far\n\nNone yet.\n")
        others = sorted((i for i in population if i != best["id"]), key=lambda i: -fitness(summaries[i]))
        chosen = others[:INSPIRATIONS] + self.rng.sample(others[INSPIRATIONS:], min(1, len(others[INSPIRATIONS:])))
        folder = ws / "inspirations"
        folder.mkdir()
        for i in chosen:
            entry, s = population[i], summaries[i]
            diff = self.skill_diff(best["commit"], entry["commit"])[:12000]
            (folder / f"{i}.md").write_text(
                f"# {i} ({entry['status']}): train value {s['train']['value']:.3f}, success {s['train']['success_rate']:.2f}\n\n"
                f"Outcomes on train: {json.dumps(s['train']['outcomes'])}\n\nVerdict: {entry.get('reason', '')}\n\n"
                f"## Its note\n\n{(entry.get('notes') or '')[:2500]}\n\n## Its code against {best['id']}\n\n```diff\n{diff}\n```\n")
        return ws

    def parent_results(self, ws, parent, summaries):
        """PARENT's own train results, when it is not the version the workspace describes."""
        path = ws / "parents" / f"{parent}.md"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            rows = skill_score.read(self.work / "results" / parent / "train")
            lines = [f"# {parent} on the train episodes: value {summaries[parent]['train']['value']:.3f}", "",
                     "| episode | stage | skill result | message |", "|---|---|---|---|"]
            lines += [f"| {e} | {r['stage']} | {skill_score.call(r).get('code', r.get('error', '?'))} | "
                      f"{(skill_score.call(r).get('message') or '')[:120]} |" for e, r in sorted(rows.items())]
            path.write_text("\n".join(lines) + "\n")
        return path

    # --- a round ---
    def plan(self, best, population, summaries, mechanisms):
        """The round's candidates: [{operator, parents, mechanism, model}]."""
        pool = self.parent_pool(self.lineage(population), summaries)
        offspring, stats, slots = self.offspring(), self.bandit(), []
        pair = self.complementary(pool, list(population), summaries) if len(population) > 1 else None
        extra = (["crossover"] if pair else []) + (["rewrite"] if self.stalled() else [])
        diffs = max(1, self.candidates - len(extra))
        for index in range(diffs):
            parent = best["id"] if index == 0 else self.sample_parent(pool, summaries, offspring)
            offspring[parent] = offspring.get(parent, 0) + 1
            slots.append({"operator": "diff", "parents": [parent], "mechanism": index % max(1, mechanisms) + 1})
        if pair:
            slots.append({"operator": "crossover", "parents": list(pair), "mechanism": None})
        if "rewrite" in extra:
            slots.append({"operator": "rewrite", "parents": [best["id"]], "mechanism": None})
        for slot in slots:  # the bandit picks each diff's model as if the earlier picks already paid their mean
            if slot["operator"] != "diff":  # combining or restructuring takes the strongest model
                slot["model"] = self.models[0]
                continue
            slot["model"] = ucb1(stats, self.models)
            pulls, total = stats[slot["model"]]
            stats[slot["model"]] = [pulls + 1, total + (total / pulls if pulls else 0.0)]
        return slots

    def round(self, number):
        best = self.best()
        population = self.population()
        population.setdefault(best["id"], best)
        summaries = self.scored(population)
        round_dir = self.work / "rounds" / f"e{number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"evolve round {number}: best {best['id']} (train {best['train']['value']:.3f}, val "
                 f"{best['val']['value']:.3f}); population {len(population)}")
        ws = self.evo_workspace(round_dir, best, population, summaries)
        diagnosis_path = round_dir / "diagnosis.md"
        if diagnosis_path.exists():
            diagnosis = diagnosis_path.read_text()
        else:
            reader = self.loop.worktree(self.cand(f"e{number:03d}d"), best["commit"])
            prompt = (HERE / "prompts" / "evo_diagnose.md").read_text().format(workspace=ws, skill=self.skill,
                                                                              source=self.source)
            diagnosis = astra.codex(prompt, reader, "read-only", diagnosis_path, **self.diagnoser())
        mechanisms = max(1, diagnosis.count("## Mechanism "))
        plan_path = round_dir / "plan.json"
        slots = json.loads(plan_path.read_text()) if plan_path.exists() else self.plan(best, population, summaries, mechanisms)
        plan_path.write_text(json.dumps(slots, indent=1))
        self.log(f"evolve round {number}: plan " + "; ".join(
            f"c{i} {s['operator']} {'+'.join(s['parents'])} m{s['mechanism'] or '-'} {s['model']}" for i, s in enumerate(slots, 1)))
        with ThreadPoolExecutor(len(slots)) as pool:
            ready = [e for e in pool.map(lambda item: self.write(number, item[0], item[1], population, summaries, ws,
                                                                  diagnosis, round_dir), enumerate(slots, 1)) if e]
        probed = [e for e in map(self.probe, ready) if e]  # each spread over every host, so one at a time
        judged = [self.judge_candidate(entry, best) for entry in probed]
        accepted = [e for e in judged if e["status"] == "accepted"]
        for entry in judged:
            if entry["status"] != "accepted":
                self.record(entry)
        if accepted:
            winner = max(accepted, key=lambda e: e["verdict"]["pooled"][0])
            for entry in accepted:
                self.record(entry if entry is winner else {**entry, "status": "rejected",
                            "reason": entry["reason"] + "; passed, but another candidate scored higher"})
        self.lessons(round_dir)
        self.log(f"evolve round {number}: done; best is now {self.best()['id']}; "
                 f"optimisation credits this round {astra.credits_in(round_dir)}")

    def write(self, number, index, slot, population, summaries, ws, diagnosis, round_dir):
        """Write candidate INDEX from SLOT and screen it like a greedy candidate; its entry, or None."""
        cid = f"e{number:03d}c{index}"
        parent = population[slot["parents"][0]]
        path = self.loop.worktree(self.cand(cid), parent["commit"])
        note_path = round_dir / f"{cid}_note.md"
        entry = {"id": cid, "round": number, "parent": parent["id"], "parents": slot["parents"],
                 "operator": slot["operator"], "mechanism": slot["mechanism"], "model": slot["model"], "strategy": "evolve"}
        if git("rev-parse", "HEAD", cwd=path) == parent["commit"]:
            other = population[slot["parents"][1]] if slot["operator"] == "crossover" else None
            prompt = (HERE / "prompts" / f"evo_{slot['operator']}.md").read_text().format(
                workspace=ws, skill=self.skill, source=self.source, diagnosis=diagnosis, mechanism=slot["mechanism"],
                parent=parent["id"], parent_results=self.parent_results(ws, parent["id"], summaries),
                other=other["id"] if other else "", other_results=self.parent_results(ws, other["id"], summaries) if other else "",
                other_diff=self.skill_diff(parent["commit"], other["commit"])[:20000] if other else "",
                ruff=astra.settings()["ruff"], python=astra.settings()["python"],
                editable=", ".join(p.format(skill=self.skill) for p in self.editable))
            try:
                note = astra.codex(prompt, path, "workspace-write", note_path, model=slot["model"], effort=self.cfg["effort"])
            except Exception as error:  # noqa: BLE001 -- a failed proposal is recorded, not fatal
                self.record({**entry, "status": "failed", "reason": f"proposer error: {error}"})
                return None
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
                f"skill {self.skill} {cid}: {slot['operator']} of {'+'.join(slot['parents'])}\n\n{note[:3000]}", cwd=path)
        commit = git("rev-parse", "HEAD", cwd=path)
        files = git("diff", "--name-only", parent["commit"], commit).splitlines()
        entry.update(commit=commit, files=files, notes=note_path.read_text() if note_path.exists() else "", path=str(path))
        outside, specific = self.allowed(files), self.specific(parent["commit"], commit)
        reason = (f"edits outside the skill: {outside}" if outside else
                  f"not general: the source names {specific}" if specific else None)
        if reason is None:  # the candidates' lint and tests alternate between the hosts
            passed, output = self.on_host(index, self.loop.remote_checks, self.cand(cid), path)
            if not passed:
                (round_dir / f"{cid}_checks.txt").write_text(output)
                reason = "lint or unit tests failed: " + output[-600:]
        if reason:
            self.record({**entry, "status": "rejected", "reason": reason})
            return None
        return entry

    def probe_episodes(self, parent):
        """One train episode per outcome of PARENT (its stage and result code), up to PROBE_EPISODES; the
        benchmark's priority episodes (profile `evo_probe_prefix`, e.g. the warehouse's) first."""
        rows = skill_score.read(self.work / "results" / parent / "train")
        prefix = self.cfg.get("evo_probe_prefix", "")
        chosen = {}
        for episode, row in sorted(rows.items(), key=lambda item: (not item[0].startswith(prefix), item[0])):
            chosen.setdefault((row["stage"], skill_score.call(row).get("code")), episode)
        episodes = {Path(e).stem: e for e in self.episodes("train")}
        return [episodes[e] for e in list(chosen.values())[:PROBE_EPISODES] if e in episodes]

    def probe(self, entry):
        """Run ENTRY on its probe episodes: a crash or behaviour identical to its parent there rejects it."""
        cid, parent = entry["id"], entry["parent"]
        episodes = self.probe_episodes(parent)
        self.evaluate(f"{cid}-probe", Path(entry["path"]), splits=("train",), only={"train": episodes}, cand=self.cand(cid))
        child = skill_score.read(self.work / "results" / f"{cid}-probe" / "train")
        before = skill_score.read(self.work / "results" / parent / "train")
        crashed = [e for e, r in child.items() if r.get("error")]
        if crashed or not child:
            self.record({**entry, "status": "rejected", "reason": "crashed on probe episodes: "
                         + (str(child[crashed[0]].get("error"))[:300] if crashed else "no results")})
            return None
        if all(fingerprint(r) == fingerprint(before.get(e, {})) for e, r in child.items()):
            self.record({**entry, "status": "rejected", "reason": f"inert: behaved exactly like {parent} on "
                         f"{len(child)} probe episodes (one per outcome), so its change never ran"})
            return None
        return entry

    def judge_candidate(self, entry, best):
        summaries = self.evaluate(entry["id"], Path(entry["path"]))
        verdict = skill_score.judge(self.summaries(best["id"]), summaries)
        return {**entry, **summaries, "status": "accepted" if verdict["ok"] else "rejected",
                "reason": skill_score.describe(verdict), "verdict": verdict}

    def lessons(self, round_dir):
        """A cheap model rewrites lessons.md from the whole history (the meta-scratchpad)."""
        history = "\n\n".join(
            f"## {e['id']} (round {e.get('round')}, {e.get('operator', 'greedy')}, parent {e.get('parent')}, "
            f"model {e.get('model', self.cfg['model'])}): {e['status']}\n{e.get('reason', '')}\n{(e.get('notes') or '')[:1200]}"
            for e in self.archive())
        current = self.work / "lessons.md"
        prompt = (HERE / "prompts" / "evo_lessons.md").read_text().format(
            skill=self.skill, history=history, lessons=current.read_text() if current.exists() else "none yet")
        scratch = round_dir / "lessons_workdir"
        scratch.mkdir(exist_ok=True)
        try:
            text = astra.codex(prompt, scratch, "read-only", round_dir / "lessons.md", model=self.lesson_model,
                               effort=self.cfg["effort"])
            shutil.copy(round_dir / "lessons.md", current)
            self.log(f"lessons updated ({len(text.splitlines())} lines)")
        except Exception as error:  # noqa: BLE001 -- lessons are advice; the round's verdicts stand without them
            self.log(f"lessons not updated: {error}")
