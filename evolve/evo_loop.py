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
5. Every written candidate is evaluated in full in one pooled run (all their episodes in one queue across the
   hosts) and judged against the current best with the greedy loop's rule (skill_score.judge); the largest
   accepted gain becomes the best. Where its new code reported acting (mechanism_active events its parent's
   runs lack) and where outcomes changed are recorded from those results for the diagnosis, not judged.
6. A cheap model rewrites lessons.md from the whole history, each version's activation record included.
"""
import json
import math
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import astra, skill_score
from .loop import git
from .skill_loop import HERE, SPLITS, SkillLoop, public_reason

DEFAULT_MODELS = ("gpt-6-astra", "gpt-6.1-sol")
TOP_PARENTS = 3          # the top versions by train value are always in the parent pool
SIGMOID_SHARPNESS = 10.0  # weighted parent sampling: sigmoid of (value - median) * this
INSPIRATIONS = 3          # top other versions shown to every proposal, plus one random
STALLED_ROUNDS = 2        # consecutive rounds without an accepted candidate before a rewrite slot
PARTNER_MARGIN = 0.05   # a crossover partner's train value is at most this far below its parent's
UCB_EXPLORATION = math.sqrt(2.0)
CREDIT_UNIT = 10.0       # the bandit's reward is per this many credits a candidate cost to write
ACTIVATION_EVENT = "mechanism_active"  # what a candidate's new code emits when it runs (evo_diff.md)


def fitness(summaries, key="values", weights=None):
    """A version's score for parent selection: its train mean of KEY ("values" or "held_values"), episodes
    weighted by prefix (val never steers the search)."""
    values = summaries["train"].get(key) or {}
    if not values:
        return 0.0
    w = {e: skill_score.weight(e, weights) for e in values}
    return sum(w[e] * v for e, v in values.items()) / sum(w.values())


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
        self.evo_prompts = self.cfg.get("evo_prompts", "evo")  # prompts/<this>_diagnose.md, _diff.md, ... _lessons.md

    def fit(self, summaries):
        return fitness(summaries, self.objective, self.weights)

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
        ranked = sorted(population, key=lambda i: -self.fit(summaries[i]))
        front = pareto_front({i: summaries[i]["train"][self.objective] for i in population})
        return list(dict.fromkeys(ranked[:TOP_PARENTS] + front))

    def sample_parent(self, pool, summaries, offspring):
        weights = parent_weights({i: self.fit(summaries[i]) for i in pool}, offspring)
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def complementary(self, pool, others, summaries):
        """A POOL version and another scored version (any lineage) that each win the most train episodes the
        other loses, or None; the first is the parent the crossover is written on."""
        best, pair = 0, None
        for a in pool:
            for b in others:
                if a != b and self.fit(summaries[b]) >= self.fit(summaries[a]) - PARTNER_MARGIN:
                    va, vb = summaries[a]["train"][self.objective], summaries[b]["train"][self.objective]
                    wins = [sum(va[e] > vb.get(e, 0) for e in va), sum(vb[e] > va.get(e, 0) for e in vb)]
                    if min(wins) > best:
                        best, pair = min(wins), (a, b)
        return pair

    def bandit(self):
        """{model: [pulls, reward sum]} from the candidates on record: the objective's train gain over the parent,
        per CREDIT_UNIT credits the candidate cost to write (a cheaper model that does as well earns more)."""
        population = self.population()
        stats = {m: [0, 0.0] for m in self.models}
        for entry in self.archive():
            model, parent = entry.get("model"), entry.get("parent")
            if model not in stats:
                continue
            stats[model][0] += 1
            if entry["id"] in population and parent in population:
                usage = self.work / "rounds" / f"e{int(entry['round']):03d}" / f"{entry['id']}_note.usage.json"
                credits = json.loads(usage.read_text()).get("credits", CREDIT_UNIT) if usage.exists() else CREDIT_UNIT
                gain = self.fit(self.summaries(entry["id"])) - self.fit(self.summaries(parent))
                stats[model][1] += reward(gain) * CREDIT_UNIT / max(credits, 1.0)
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
        others = sorted((i for i in population if i != best["id"]), key=lambda i: -self.fit(summaries[i]))
        chosen = others[:INSPIRATIONS] + self.rng.sample(others[INSPIRATIONS:], min(1, len(others[INSPIRATIONS:])))
        folder = ws / "inspirations"
        folder.mkdir()
        for i in chosen:
            entry, s = population[i], summaries[i]
            diff = self.skill_diff(best["commit"], entry["commit"])[:12000]
            (folder / f"{i}.md").write_text(
                f"# {i} ({entry['status']}): train value {s['train']['value']:.3f}, success {s['train']['success_rate']:.2f}\n\n"
                f"Outcomes on train: {json.dumps(s['train']['outcomes'])}\n\nVerdict: {public_reason(entry)}\n\n"
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
        lineage = self.lineage(population)  # parents and crossover donors share today's scenes and sensors
        pool = self.parent_pool(lineage, summaries)
        offspring, stats, slots = self.offspring(), self.bandit(), []
        pair = self.complementary(pool, list(lineage), summaries) if len(lineage) > 1 else None
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

    def remeasure(self, ids):
        """Re-apply each older version's own skill change (against its parent) on top of the current best and
        score it there, so it can be a parent or crossover donor measured in today's scenes and sensors."""
        self.check_frozen()
        best, archive = self.best(), {e["id"]: e for e in self.archive()}
        folder = str(Path(self.source).parent)
        for cid in ids:
            old = archive[cid]
            parent = archive.get(old.get("parent")) or {}
            new = f"r{best['id']}-{cid}"
            path = self.loop.worktree(self.cand(new), best["commit"])
            if git("rev-parse", "HEAD", cwd=path) == best["commit"]:
                patch = git("diff", parent.get("commit", old["commit"] + "^"), old["commit"], "--", folder, "tests")
                applied = subprocess.run(["git", "apply", "--3way", "-"], input=patch + "\n", text=True, cwd=path,
                                         capture_output=True, check=False)
                if applied.returncode != 0:  # leave no half-applied state for a later attempt
                    git("reset", "-q", "--hard", best["commit"], cwd=path)
                    git("clean", "-q", "-fd", cwd=path)
                    self.log(f"remeasure {cid}: its change does not apply on {best['id']}: {applied.stderr[-300:]}")
                    continue
                git("add", "-A", cwd=path)
                git("-c", "user.name=astra", "-c", "user.email=astra@localhost", "commit", "-q", "-m",
                    f"skill {self.skill} {new}: {cid}'s change on {best['id']}", cwd=path)
            files = git("diff", "--name-only", best["commit"], "HEAD", cwd=path).splitlines()
            outside, specific = self.allowed(files), self.specific(best["commit"], git("rev-parse", "HEAD", cwd=path))
            if outside or specific:
                self.log(f"remeasure {cid}: screened out on {best['id']}: "
                         + (f"edits outside the skill {outside}" if outside else f"names {specific}"))
                continue
            passed, output = self.loop.remote_checks(self.cand(new), path)
            if not passed:
                self.log(f"remeasure {cid}: lint or unit tests fail on {best['id']}: {output[-300:]}")
                continue
            commit = git("rev-parse", "HEAD", cwd=path)
            manifest = self.manifest(path, commit)
            summaries = self.evaluate(new, path)
            verdict = self.judge(self.summaries(best["id"]), summaries)
            if self.manifest(path, commit) != manifest:
                self.log(f"remeasure {cid}: the experiment changed while it ran; not recorded")
                continue
            self.record({"id": new, "round": old.get("round"), "parent": best["id"], "parents": [best["id"], cid],
                         "operator": "remeasure", "commit": commit, "status": "remeasured", "manifest": manifest,
                         "notes": old.get("notes", ""), **summaries, "verdict": verdict,
                         "reason": "remeasured on " + best["id"] + ": " + skill_score.describe(verdict)})

    def round(self, number):
        best = self.best()
        self.check_frozen()
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
            prompt = (HERE / "prompts" / f"{self.evo_prompts}_diagnose.md").read_text().format(workspace=ws, skill=self.skill,
                                                                              source=self.source)
            diagnosis = self.codex_read(prompt, reader, diagnosis_path, **self.diagnoser())
        mechanisms = max(1, diagnosis.count("## Mechanism "))
        plan_path = round_dir / "plan.json"
        slots = json.loads(plan_path.read_text()) if plan_path.exists() else self.plan(best, population, summaries, mechanisms)
        plan_path.write_text(json.dumps(slots, indent=1))
        self.log(f"evolve round {number}: plan " + "; ".join(
            f"c{i} {s['operator']} {'+'.join(s['parents'])} m{s['mechanism'] or '-'} {s['model']}" for i, s in enumerate(slots, 1)))
        with ThreadPoolExecutor(len(slots)) as pool:
            ready = [e for e in pool.map(lambda item: self.write(number, item[0], item[1], population, summaries, ws,
                                                                  diagnosis, round_dir), enumerate(slots, 1)) if e]
        judged = self.judge_candidates(ready, best)  # one pooled evaluation of every written candidate
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

    def measured_before(self, cid, commit):
        """Whether CID has evaluation results for COMMIT (it passed its checks then); profiles that know redirect."""
        return False

    def extend(self, number, extra=0, wait=False, only=()):
        """EXTRA more candidates for round NUMBER, from the same parent pool and diagnosis, judged against the round's
        own starting best; then one winner across all of the round's candidates (a beaten earlier winner gets a later
        "rejected" verdict, the archive's correction rule). With WAIT, the candidates are written while the round is
        still evaluating, and their sessions wait until rounds.json records the round finished. ONLY (slot numbers of
        plan.json) resumes those planned candidates instead of planning EXTRA new ones; a written one is reused."""
        round_dir = self.work / "rounds" / f"e{number:03d}"
        self.round_dir = round_dir  # the round's ledger (end-to-end loops)
        plan_path, diagnosis_path = round_dir / "plan.json", round_dir / "diagnosis.md"
        if not (plan_path.exists() and diagnosis_path.exists()):
            raise SystemExit(f"round {number} has no plan and diagnosis to extend")
        extended_path = round_dir / "plan_extend.json"  # kept apart: a resumed round reads plan.json alone
        self.check_frozen()
        prefix = f"e{number:03d}c"
        slots = json.loads(plan_path.read_text())
        slots += json.loads(extended_path.read_text()) if extended_path.exists() else []
        population = {i: e for i, e in self.population().items() if not i.startswith(prefix)}
        best = population[slots[0]["parents"][0]]  # the round's starting best: its first diff's parent
        summaries = self.scored(population)
        diagnosis = diagnosis_path.read_text()
        mechanisms = max(1, diagnosis.count("## Mechanism "))
        if only:
            chosen = [(index, slots[index - 1]) for index in only]
        else:
            width, self.candidates = self.candidates, extra
            added = self.plan(best, population, summaries, mechanisms)
            self.candidates = width
            diffs = sum(s["operator"] == "diff" for s in slots)
            for slot in added:  # the next mechanisms in turn, after the ones the round already tried
                if slot["operator"] == "diff":
                    slot["mechanism"] = diffs % mechanisms + 1
                    diffs += 1
            chosen = list(enumerate(added, len(slots) + 1))
            base = json.loads(plan_path.read_text())
            extended_path.write_text(json.dumps(slots[len(base):] + added, indent=1))
        self.log(f"evolve round {number}: extend with " + "; ".join(
            f"c{i} {s['operator']} {'+'.join(s['parents'])} m{s['mechanism'] or '-'} {s['model']}" for i, s in chosen))
        ws = self.evo_workspace(round_dir, best, population, summaries)
        with ThreadPoolExecutor(len(chosen)) as pool:
            ready = [e for e in pool.map(lambda item: self.write(number, item[0], item[1], population, summaries, ws,
                                                                  diagnosis, round_dir), chosen) if e]
        while wait and number not in self.round_state()["finished"]:
            time.sleep(30)  # the round's own candidates still hold the hosts; their verdicts come first
        if wait:
            self.log(f"evolve round {number}: finished; evaluating the extension")
        judged = self.judge_candidates(ready, best)
        latest = {e["id"]: e for e in self.archive()}
        earlier = [e for i, e in latest.items() if i.startswith(prefix) and e["status"] == "accepted"]
        accepted = earlier + [e for e in judged if e["status"] == "accepted"]
        winner = max(accepted, key=lambda e: e["verdict"]["pooled"][0]) if accepted else None
        for entry in judged:
            if entry is not winner:
                self.record(entry if entry["status"] != "accepted" else {**entry, "status": "rejected",
                            "reason": entry["reason"] + "; passed, but another candidate of the round scored higher"})
        for entry in earlier:
            if entry is not winner:
                self.record({**entry, "status": "rejected",
                             "reason": entry["reason"] + "; passed, but a later candidate of the round scored higher"})
        if winner is not None and winner not in earlier:
            self.record(winner)
        self.lessons(round_dir)
        self.log(f"evolve round {number}: extended; best is now {self.best()['id']}; "
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
            prompt = (HERE / "prompts" / f"{self.evo_prompts}_{slot['operator']}.md").read_text().format(
                workspace=ws, skill=self.skill, source=self.source, diagnosis=diagnosis,
                mechanism=slot.get("focus") or slot["mechanism"],
                parent=parent["id"], parent_results=self.parent_results(ws, parent["id"], summaries),
                other=other["id"] if other else "", other_results=self.parent_results(ws, other["id"], summaries) if other else "",
                other_diff=self.skill_diff(parent["commit"], other["commit"])[:20000] if other else "",
                ruff=astra.settings()["ruff"], python=astra.settings()["python"],
                editable=", ".join(p.format(skill=self.skill) for p in self.editable))
            try:
                note = self.codex_write(prompt, path, note_path, model=slot["model"], effort=self.cfg["effort"])
            except astra.BudgetExceeded:  # the round's credit cap, not the proposal: stop the round
                raise
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
        reason = self.screen(parent["commit"], commit, files)
        if reason is None and self.measured_before(cid, commit):  # a resumed round: it passed before it was measured
            self.log(f"{cid}: already measured at {commit[:10]}; not screened again")
            return entry
        if reason is None:  # the candidates' lint and tests alternate between the hosts
            passed, output = self.on_host(index, self.loop.remote_checks, self.cand(cid), path)
            if not passed:  # once more on the other host: some tests are timing-sensitive under a loaded host
                passed, again = self.on_host(index + 1, self.loop.remote_checks, self.cand(cid), path)
                if passed:
                    (round_dir / f"{cid}_checks_flaky.txt").write_text(output)
                    self.log(f"{cid}: checks failed once and passed on a second run (flaky; see {cid}_checks_flaky.txt)")
                output = output if passed else again
            if not passed:
                (round_dir / f"{cid}_checks.txt").write_text(output)
                reason = "lint or unit tests failed: " + output[-600:]
        if reason:
            self.record({**entry, "status": "rejected", "reason": reason})
            return None
        return entry

    def judge_candidates(self, entries, best):
        """Evaluate ENTRIES together (one episode queue across the hosts), each with its manifest checked before and
        after, its activation record read from its own full results, and judge each against BEST."""
        if not entries:
            return []
        before = {e["id"]: self.manifest(Path(e["path"]), e["commit"]) for e in entries}
        everything = self.evaluate_many([(e["id"], Path(e["path"]), None, None) for e in entries])
        reference = self.summaries(best["id"])
        judged = []
        for index, entry in enumerate(entries):  # the setup measured afresh once after the evaluation, then reused
            after = self.manifest(Path(entry["path"]), entry["commit"], refresh=index == 0)
            summaries = everything[entry["id"]]
            verdict = self.judge(reference, summaries)
            if after != before[entry["id"]]:
                verdict = {**verdict, "ok": False, "basis": f"the experiment changed while it ran ({before[entry['id']]} -> {after})"}
            judged.append({**entry, **summaries, "probe": self.activation(entry), "status": "accepted" if verdict["ok"]
                           else "rejected", "manifest": before[entry["id"]], "reason": skill_score.describe(verdict),
                           "verdict": verdict})
        return judged

    def activation(self, entry):
        """Where the candidate's new code reported acting (mechanism_active events its parent's runs lack) and where
        its outcomes changed, over the train episodes, the ones its note expects first: diagnosis evidence."""
        cid, parent = entry["id"], entry["parent"]
        child = skill_score.read(self.work / "results" / cid / "train")
        before = skill_score.read(self.work / "results" / parent / "train")

        def mechanisms(version, episode):
            path = self.work / "results" / version / "train" / episode / "trace.json"
            events = json.loads(path.read_text()) if path.exists() else []
            return [str(e.get("mechanism")) for e in events if e.get("type") == ACTIVATION_EVENT]
        new = {e: [m for m in mechanisms(cid, e) if m not in set(mechanisms(parent, e))] for e in child}
        changed = sorted(e for e, r in child.items() if fingerprint(r) != fingerprint(before.get(e, {})))
        notes = entry.get("notes", "")
        expected = notes.split("Expected effect", 1)[-1].split("Risks", 1)[0] if "Expected effect" in notes else ""
        named = sorted(e for e in child if e in expected)
        record = {"episodes": len(child), "new_activations": sum(len(f) for f in new.values()),
                  "activated_in": sorted(e for e, f in new.items() if f), "changed": changed,
                  "named_activated": sorted(e for e in named if new.get(e))}
        self.log(f"{cid}: new code acted in {len(record['activated_in'])} train episodes ({len(record['named_activated'])} "
                 f"of the {len(named)} its note named), outcomes changed in {len(changed)}")
        return record

    def lessons(self, round_dir):
        """A cheap model rewrites lessons.md from the whole history (the meta-scratchpad)."""
        history = "\n\n".join(
            f"## {e['id']} (round {e.get('round')}, {e.get('operator', 'greedy')}, parent {e.get('parent')}, "
            f"model {e.get('model', self.cfg['model'])}): {e['status']}\n{public_reason(e)}\n"
            f"{'probe: ' + json.dumps(e['probe']) + chr(10) if e.get('probe') else ''}{(e.get('notes') or '')[:1200]}"
            for e in self.archive())
        current = self.work / "lessons.md"
        prompt = (HERE / "prompts" / f"{self.evo_prompts}_lessons.md").read_text().format(
            skill=self.skill, history=history, lessons=current.read_text() if current.exists() else "none yet")
        scratch = round_dir / "lessons_workdir"
        scratch.mkdir(exist_ok=True)
        try:
            text = self.astra_call(prompt, scratch, "read-only", round_dir / "lessons.md", model=self.lesson_model,
                                   effort=self.cfg["effort"])
            shutil.copy(round_dir / "lessons.md", current)
            self.log(f"lessons updated ({len(text.splitlines())} lines)")
        except Exception as error:  # noqa: BLE001 -- lessons are advice; the round's verdicts stand without them
            self.log(f"lessons not updated: {error}")
