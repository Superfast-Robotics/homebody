"""GEPA prompt optimisation of the open setup's own texts (astra-evolve/e2e/gepa_open_models_design.md): the
open-model guidance, the planner's target note and the pointer's pick and place notes of SPR3 (Sol's recorded first
plan, Qwen3.8-27B with the RoboBrain2.5 pointer and every open-model fix). Sol's own prompt is untouched.

    python3 -m evolve.gepa_open init NAME              index the bench cases, split them, write the seed
    python3 -m evolve.gepa_open run NAME [--iterations N] [--credits C] [--e2e-every K]
    python3 -m evolve.gepa_open status NAME            the candidates, their scores and the credits spent

The loop (GEPA: reflective prompt evolution with a Pareto front, Agrawal et al. 2025) runs here, beside the LiteLLM
proxy that reaches Sol. Each iteration takes a parent from the Pareto front over the validation cases (a candidate is
on it while it is best on some case; picked in proportion to how many), one text in turn, a minibatch of train cases
testing what that text affects, and asks Sol to rewrite the text from the parent's failures there (with the latest
end-to-end sessions' failed steps for the guidance). The child is kept when it does better on the same minibatch and
is then scored on every validation case. Capability evaluations run on the GH200 (evolve/gepa_eval.py) against the
setup's own vLLM servers; every --e2e-every iterations the best untested candidate runs the demo-pp baseline (17
sessions, evolve/agent_experiments.py) and is dropped from the front if it places clearly fewer objects than the
seed's runs (SPR2, SPR3). Proposals naming the scenarios' objects or places, or numbers that are no robot fact, are
refused (the texts must hold in scenes never seen); each text has a word limit."""
import argparse
import json
import random
import re
import subprocess
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from homebody.vlm.composite import TARGET_NOTE
from homebody.vlm.provider import streamed

from .gepa_groups import GROUPS, evaluate_planning, evaluate_pointer

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "astra-evolve" / "gepa"
GPU_HOST, GPU_REPO, GPU_PY = "loki-gh200", "~/hb-bench", "~/vllm-venv/bin/python"
GPU_CASES, GPU_RUNS = "~/bench/gepa-cases", "~/bench/gepa"
# where capability turns are asked: the GH200 against its own servers, and loki-a100 against its Qwen3.8 replica
# (port 8100) with the GH200's pointer through a tunnel (9101). A case always goes to the same one (by its name), so
# two candidates are compared on the same hardware's numbers; three in four go to the GH200 (it is about 3 times
# faster: 32 batched requests against the replica's ~5).
EXECUTORS = (
    {"host": "loki-gh200", "repo": "~/hb-bench", "py": "~/vllm-venv/bin/python", "cases": "~/bench/gepa-cases-v2",
     "urls": "", "jobs": 24, "weight": 3},
    {"host": "loki-a100", "repo": "~/gepa-data-code",
     "py": "/lambda/nfs/group-volume/workcell-sim/homebody-baseline/homebody-astra/.venv/bin/python",
     "cases": "~/bench/gepa-cases-v2",
     "urls": "--local-url http://10.19.89.218:8100/v1 --second-url http://10.19.89.218:9101/v1", "jobs": 6, "weight": 1},
)
E2E_HOST, E2E_REPO = "loki-a100", "~/hb-agents"
SHARED = "/lambda/nfs/group-volume/workcell-sim/homebody-baseline/agent-experiments"
E2E_WORK = "~/hb-agents-work/e2e"
E2E_MARGIN = 0.06  # a candidate whose val placed share is this far below the seed's is dropped
PROXY = "http://127.0.0.1:4000/v1/responses"
CRITIC = "gpt-6.1-sol"
RATES = {"input": 50.0, "cached": 2.5, "output": 250.0}  # credits per 1M tokens (evolve/score_episode.py)
MINIBATCH, VAL_SIZE = 40, 240
PERFECT = 1.0  # a minibatch the parent gets all right is skipped (gepa's perfect_score)

FAMILIES = {"guidance": ("choose", "judge", "recover", "navigate", "pick", "place"), "plan_guidance": ("plan",),
            "target_note": ("pick", "place"), "pick_note": ("pick",), "place_note": ("place",)}
# words per text: room for general rules (the scene-word lint and the val and end-to-end checks keep them general);
# the pointer notes stay short, as the pointing model is trained on short questions
WORD_LIMIT = {"guidance": 400, "target_note": 100, "pick_note": 60, "place_note": 60, "plan_guidance": 350}
ROLES = {
    "plan_guidance": ("Guidance appended to the planner's prompt on plan turns only, when Qwen3.8 writes the task's "
                      "behaviour tree (goals with harness checks and skill-call budgets, in sequences and fallbacks) "
                      "that it then works through."),
    "guidance": ("Guidance appended (after 'Guidance:') to every decision prompt of the planner, Qwen3.8-27B, an "
                 "open-weight vision-language model that drives a humanoid robot with two grippers through skills "
                 "(navigate, pick, place, goal_done, goal_failed, done) while it works through a plan of goals. The "
                 "session's own prompt (skills, images, the plan, the last results) comes before it."),
    "target_note": ("A note appended to the planner's prompt: when its decision takes a click (point_normalized_1000), "
                    "it must name the click target in words; a pointing model (RoboBrain2.5-8B) then clicks those "
                    "words on the head-camera image, and its click replaces the planner's. The note decides how the "
                    "planner names the target."),
    "pick_note": ("A sentence added after the pointing model's own question when it clicks the object a pick grasps "
                  "(RoboBrain2.5-8B, asked '<target words>. Please provide its 2D coordinates ...'). It sees only "
                  "the head-camera image (pixels beyond the arm's reach greyed out) and the target words."),
    "place_note": ("A sentence added after the pointing model's own question when it clicks where a held object is "
                   "set down (RoboBrain2.5-8B, asked '<target words>. Please provide its 2D coordinates ...'). It "
                   "sees only the head-camera image (pixels beyond the arm's reach greyed out) and the target words; "
                   "a click beyond reach is moved toward the robot afterwards."),
}
ALLOWED_NUMBERS = {"0.4", "0.42", "0.45", "0.5", "1000", "0", "1", "2", "3"}


# --- remote ---
def ssh(host: str, command: str, stdin: str | None = None, timeout: float = 7200) -> str:
    done = None
    for attempt in range(5):
        done = subprocess.run(["ssh", host, command], input=stdin, capture_output=True, text=True, timeout=timeout)
        if done.returncode != 255:  # 255: the connection failed, not the command
            break
        time.sleep(10 * (attempt + 1))
    if done is None or done.returncode:
        raise RuntimeError(f"{host}: {command[:120]} failed: {done and done.stderr[-800:]}")
    return done.stdout


def gpu(command: str, **kwargs) -> str:
    return ssh(GPU_HOST, f"cd {GPU_REPO} && export VLLM_API_KEY=$(cat ~/vlm/api_key) && PYTHONPATH=src:. "
                         f"{GPU_PY} -m evolve.gepa_eval {command}", **kwargs)


# --- the run's state ---
def load(run: Path) -> dict:
    return json.loads((run / "state.json").read_text())


def save(run: Path, state: dict):
    (run / "state.json.tmp").write_text(json.dumps(state, indent=1))
    (run / "state.json.tmp").replace(run / "state.json")


def log(run: Path, message: str):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run / "log.txt").open("a") as out:
        out.write(line + "\n")


def records(run: Path, cid: str) -> dict:
    path = run / "records" / f"{cid}.jsonl"
    if not path.is_file():
        return {}
    return {r["case"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def executor_of(name: str, executors) -> dict:
    """The executor a case always goes to: by its name's hash, in proportion to the executors' weights."""
    slot = zlib.crc32(name.encode()) % sum(e["weight"] for e in executors)
    for executor in executors:
        if slot < executor["weight"]:
            return executor
        slot -= executor["weight"]
    raise AssertionError("weights cover every slot")


def ask_on(run: Path, executor: dict, cid: str, texts: dict, names: list[str]) -> str:
    """CID's records of NAMES asked on EXECUTOR, as JSON lines."""
    remote = f"{GPU_RUNS}/{run.name}"
    listed = f"{remote}/lists/{cid}-{zlib.crc32(''.join(names).encode()):08x}.txt"
    ssh(executor["host"], f"mkdir -p {remote}/lists {remote}/cands && cat > {remote}/cands/{cid}.json",
        json.dumps(texts))
    ssh(executor["host"], f"cat > {listed}", "\n".join(names))
    ssh(executor["host"], f"cd {executor['repo']} && export VLLM_API_KEY=$(cat ~/vlm/api_key) && PYTHONPATH=src:. "
                          f"{executor['py']} -m evolve.gepa_eval ask {remote}/cands/{cid}.json {executor['cases']} "
                          f"{remote}/out/{cid} --list {listed} --jobs {executor['jobs']} {executor['urls']}")
    return ssh(executor["host"], f"cat {remote}/out/{cid}/results.jsonl")


def evaluate(run: Path, state: dict, cid: str, names: list[str]) -> dict:
    """CID's records of NAMES, asked where not yet known, each case on its executor (all at once)."""
    known = records(run, cid)
    missing = [n for n in names if n not in known]
    if missing:
        texts = next(c for c in state["candidates"] if c["id"] == cid)["texts"]
        started = time.monotonic()
        group = state.get("group", "planner")
        if group == "pointer":
            fetched = [evaluate_pointer(run, cid, texts, missing, ssh)]
        elif group == "planning":
            fetched = [evaluate_planning(run, cid, texts, missing, ssh)]
        else:
            executors = [e for e in EXECUTORS if e["host"] in state.get("executors", [EXECUTORS[0]["host"]])]
            shares = {e["host"]: [n for n in missing if executor_of(n, executors) is e] for e in executors}
            with ThreadPoolExecutor(len(executors)) as pool:
                fetched = list(pool.map(lambda e: ask_on(run, e, cid, texts, shares[e["host"]])
                                        if shares[e["host"]] else "", executors))
        with (run / "records" / f"{cid}.jsonl").open("a") as out:
            out.write("".join(fetched))
        log(run, f"{cid}: {len(missing)} cases asked in {time.monotonic() - started:.0f} s")
        known = records(run, cid)
    return {n: known[n] for n in names}


POINTER_NOTES = {"pick_note": "pick", "place_note": "place"}  # a text read only by the pointer, for this skill


def inherit(run: Path, child: str, parent: str, component: str, names: list[str]):
    """A pointer note changes only the turns where the planner chose its skill (the planner's prompt is the same):
    the child takes the parent's records of NAMES' other turns, so the server's run-to-run jitter on them (about 8%
    of turns at temperature 0) does not decide whether the note helped."""
    skill = POINTER_NOTES.get(component)
    if skill is None:
        return
    mine, theirs = records(run, child), records(run, parent)
    same = [theirs[n] for n in names if n in theirs and n not in mine and "error" not in theirs[n]["decision"]
            and theirs[n]["decision"].get("skill") != skill]
    with (run / "records" / f"{child}.jsonl").open("a") as out:
        out.write("".join(json.dumps(r) + "\n" for r in same))


def mean(rows) -> float:
    scores = [r["score"] for r in rows if r["score"] is not None]
    return sum(scores) / len(scores) if scores else 0.0


# --- init ---
def seed_texts() -> dict:
    notes = REPO / "configs" / "prompts" / "open_models"
    return {"guidance": "\n".join((notes / f"{n}.md").read_text().strip() for n in ("reach", "claims", "recover")),
            "target_note": TARGET_NOTE, "pick_note": "", "place_note": "", "plan_guidance": ""}


def split_of(scenario: str) -> str | None:
    """Train or val by scenario family: the demo-pp train scenarios (pp_, pa_) and the warehouse suite's train
    scenarios train, the demo-pp harder ones (hd_) and the suite's val scenarios val (no test case exists)."""
    if scenario.startswith(("pp_", "pa_")) or "_train_" in scenario:
        return "train"
    if scenario.startswith("hd_") or "_val_" in scenario:
        return "val"
    return None


def sample_val(rows: list[dict], size: int, rng: random.Random) -> list[str]:
    """Every demo-pp val case, then warehouse val cases spread over the capabilities until SIZE."""
    chosen = [r["case"] for r in rows if r["scenario"].startswith("hd_")]
    rest = [r for r in rows if not r["scenario"].startswith("hd_")]
    rng.shuffle(rest)
    families = sorted({f for r in rest for f in r["families"]})
    while len(chosen) < size and rest:
        for family in families:
            pick = next((r for r in rest if family in r["families"]), None)
            if pick and len(chosen) < size:
                chosen.append(pick["case"])
                rest.remove(pick)
    return chosen


# words that name what a scene holds: colours, products, containers, furniture and its parts (a word of the
# scenarios' tasks among these is refused; "can" and "case" stay free as ordinary words)
SCENE_WORDS = frozenset("""
red orange yellow green blue teal purple pink brown black white grey gray silver gold
allulose basil blueberry corn jam sauce spice sweetener tea cereal coffee sugar salt food foods fruit snack snacks
bottle bottles box boxes cans jar jars mug mugs cup cups bowl bowls tin tins carton cartons packet packets lid lids
product products tray trays tote totes crate crates bin bins basket baskets
shelf shelves bay bays rack racks lane lanes slot slots aisle aisles counter counters cabinet cabinets drawer drawers
table tables kitchen warehouse restock fridge sink
""".split())


def vocabulary(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)}


def banned_words(tasks: list[str]) -> list[str]:
    """The scene words the scenarios' tasks use (SCENE_WORDS): objects, colours, containers and places."""
    return sorted(vocabulary(" ".join(tasks)) & SCENE_WORDS)


def init(args):
    """The run's cases (the executors' case folder, indexed once), its splits and the seed's scores: every train
    case, and for val every case of the open setup's own rollouts plus up to --sol-val of Sol's recorded turns."""
    run = RUNS / args.name
    (run / "records").mkdir(parents=True, exist_ok=True)
    executor = EXECUTORS[0]
    out = ssh(executor["host"], f"cd {executor['repo']} && PYTHONPATH=src:. {executor['py']} -m evolve.gepa_eval "
                                f"index {executor['cases']}", timeout=1800)
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    for row in rows:
        row["split"] = row.get("split") or split_of(row["scenario"])
    rows = [r for r in rows if r["families"] and r["split"] in ("train", "val")]
    rng = random.Random(7)
    train = [r for r in rows if r["split"] == "train"]
    own_val = [r["case"] for r in rows if r["split"] == "val" and r["source"] != "recorded"]
    sol_val = sample_val([r for r in rows if r["split"] == "val" and r["source"] == "recorded"], args.sol_val, rng)
    seed = {"id": "c000", "parent": None, "component": None, "texts": seed_texts(), "iteration": 0}
    state = {"train": {r["case"]: r["families"] for r in train}, "val": own_val + sol_val, "candidates": [seed],
             "iteration": 0, "credits": 0.0, "banned": [], "e2e_runs": 0, "episodes": {},
             "executors": [e["host"] for e in EXECUTORS]}
    save(run, state)
    log(run, f"{len(train)} train cases, {len(own_val)} + {len(sol_val)} val cases (own rollouts + Sol's)")
    seed_records = evaluate(run, state, "c000", state["val"])
    seed["val_mean"] = round(mean(seed_records.values()), 4)
    tasks = [r["task"] for r in seed_records.values()] + [r["goal"] or "" for r in seed_records.values()]
    state["banned"] = banned_words(sorted(set(tasks)))
    save(run, state)
    log(run, f"seed val {seed['val_mean']:.3f}; {len(state['banned'])} scenario words banned")
    if not args.no_e2e:
        end_to_end(run, state, seed)


PLAN_VAL = 24  # judged plan tasks per val pass: two Opus calls each


def init_group(args):
    """A group's run (evolve/gepa_groups.py) on the cases and scene-word list of SOURCE (an `init` run): planner
    keeps its train and val cases and the seed's records; pointer freezes the seed's planner turns that pick or place
    (the planner asked first where a train case has none) and asks only the pointer again; planning takes the plan
    tasks (astra-evolve/gepa-plan-cases) and judges PLAN_VAL val tasks a pass."""
    source, run = RUNS / args.source, RUNS / args.name
    src = load(source)
    (run / "records").mkdir(parents=True, exist_ok=True)
    seed = {"id": "c000", "parent": None, "component": None, "texts": seed_texts(), "iteration": 0}
    state = {"group": args.group, "candidates": [seed], "iteration": 0, "credits": 0.0, "e2e_runs": 0,
             "seed_shared": GROUPS[args.group]["e2e"],
             "episodes": {}, "banned": src["banned"], "executors": src.get("executors", [])}
    if args.group == "planner":
        state |= {"train": src["train"], "val": src["val"]}
        (run / "records" / "c000.jsonl").write_text((source / "records" / "c000.jsonl").read_text())
        (run / "plans_ready").touch()  # every end-to-end task's Sol plan is recorded (configs/plans)
    elif args.group == "pointer":
        (run / "plans_ready").touch()
        clicks = {c: [f for f in fams if f in ("pick", "place")] for c, fams in src["train"].items()}
        names = [c for c, f in clicks.items() if f] + [n for n in src["val"] if n not in clicks]
        frozen = {n: r for n, r in evaluate(source, src, "c000", names).items()
                  if r["decision"].get("skill") in ("pick", "place") and r["decision"]["skill"] in r["families"]
                  and (r.get("pointer") or {}).get("target")}
        ssh(GPU_HOST, f"mkdir -p {GPU_RUNS}/{run.name} && cat > {GPU_RUNS}/{run.name}/frozen.jsonl",
            "".join(json.dumps(r) + "\n" for r in frozen.values()))
        state |= {"train": {n: [r["decision"]["skill"]] for n, r in frozen.items() if n in src["train"]},
                  "val": [n for n in src["val"] if n in frozen]}
    else:
        cases = REPO.parent / "astra-evolve" / "gepa-plan-cases"
        metas = {p.parent.name: json.loads(p.read_text()) for p in sorted(cases.glob("*/case.json"))}
        val = sorted(n for n, m in metas.items() if m["split"] == "val")
        rng = random.Random(7)
        rng.shuffle(val)
        by_suite: dict[str, list[str]] = {}
        for n in val:
            by_suite.setdefault(n[:2], []).append(n)
        picked = []
        while len(picked) < PLAN_VAL and any(by_suite.values()):
            for group in by_suite.values():
                if group and len(picked) < PLAN_VAL:
                    picked.append(group.pop())
        state |= {"train": {n: ["plan"] for n, m in metas.items() if m["split"] == "train"}, "val": picked}
    save(run, state)
    log(run, f"{args.group} group: {len(state['train'])} train, {len(state['val'])} val cases")
    seed["val_mean"] = round(mean(evaluate(run, state, "c000", state["val"]).values()), 4)
    save(run, state)
    log(run, f"seed val {seed['val_mean']:.3f}")


def seed_e2e(args):
    """The seed end to end, when init ran without it (every test task's plan recorded first)."""
    run = RUNS / args.name
    state = load(run)
    end_to_end(run, state, state["candidates"][0])


def episodes(workdir: str) -> list[dict]:
    """An end-to-end run's train sessions (its failed steps), for the reflection; [] when it has none."""
    out = ssh(E2E_HOST, f"cd {E2E_REPO} && PYTHONPATH=src:. .venv-loop/bin/python -m evolve.gepa_eval episodes "
                        f"{E2E_WORK}/{workdir} || true")
    return [json.loads(line) for line in out.splitlines() if line.strip()]


# --- reflection ---
def critic(prompt: str) -> tuple[str, float]:
    """Sol's answer to PROMPT through the LiteLLM proxy and its cost in credits."""
    key = (REPO.parent / "litellm_proxy" / "master_key").read_text().strip()
    body = {"model": CRITIC, "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "reasoning": {"effort": "high"}}
    request = urllib.request.Request(PROXY, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        raw = response.read().decode()
    reply = streamed(raw) if raw.startswith("data:") else json.loads(raw)  # the ChatGPT backend streams
    text = "".join(part.get("text", "") for item in reply.get("output") or [] if item.get("type") == "message"
                   for part in item.get("content") or [] if part.get("type") == "output_text")
    usage = reply.get("usage") or {}
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    cost = ((usage.get("input_tokens", 0) - cached) * RATES["input"] + cached * RATES["cached"]
            + usage.get("output_tokens", 0) * RATES["output"]) / 1e6
    return text, cost


def example(record: dict) -> str:
    if "plan" in record:  # the planning group: the model's tree, the reference's, the judge's verdict
        return (f"Task: {record['task']}\nThe model's plan:\n{record['plan']['open'][:1800]}\nA strong reference "
                f"model's plan:\n{record['plan']['sol'][:1800]}\nScore {record['score'] or 0:.2f}. Judge: "
                f"{record['feedback']}")
    decision = record["decision"]
    lines = [f"Task: {record['task']}", f"Active goal: {record['goal'] or '(none)'}",
             f"Last result: {record['previous']}", f"Allowed decisions: {', '.join(record['allowed'])}",
             f"Model decided: {decision.get('skill')} {json.dumps(decision.get('arguments'))[:300]} "
             f"text: {str(decision.get('text', ''))[:300]}" if "error" not in decision
             else f"Model reply failed: {decision['error'][:300]}"]
    if record.get("pointer"):
        lines.append(f"Pointer was asked for: {record['pointer'].get('target')!r}; it answered "
                     f"{str(record['pointer'].get('answer', record['pointer'].get('error', '')))[:160]!r}")
    reference = record["reference"]
    lines += [f"A strong reference model decided: {reference['skill']} {json.dumps(reference['arguments'])[:200]} "
              f"text: {reference['text'][:200]}",
              f"Score {record['score']:.2f}. Feedback: {record['feedback']}"]
    return "\n".join(lines)


def episode_text(sessions: list[dict]) -> str:
    shown = sorted(sessions, key=lambda s: s["placed"])[:8]
    return "\n".join(f"- {s['scenario']}: placed {s['placed']}; steps: {'; '.join(s['steps'])[:900]}; final "
                     f"report: {s['done_summary'][:200]}" for s in shown)


def reflection_prompt(component: str, text: str, batch: list[dict], sessions: list[dict], banned: list[str]) -> str:
    failures = [r for r in batch if r["score"] is not None and r["score"] < 1][:12]
    passes = [r for r in batch if r["score"] == 1][:3]
    parts = [
        "You improve one text of a robot agent's prompt.",
        f"What the text is: {ROLES[component]}",
        f"The current text (between the lines; empty means none):\n---\n{text}\n---",
        ("Below are plans the model wrote with this text for real tasks, beside a strong reference model's plan "
         "for the same task, with an independent judge's rubric scores (coverage, feasibility, verifiable checks, "
         "recovery, no invention; 0-10) and reasons." if component == "plan_guidance" else
         "Below are decision turns from recorded sessions where the agent with this text was scored against a "
         "strong reference model and the scene's ground truth, with feedback.")
        + " Diagnose the general behaviour that causes the failures and write an improved text that fixes it while "
          "keeping what already works.",
        "Weak cases:\n\n" + "\n\n".join(example(r) for r in failures),
        "Cases it got right:\n\n" + "\n\n".join(example(r) for r in passes)]
    if component == "guidance" and sessions:
        parts.append("Whole sessions of the same agent in simulation (failed or refused steps with their messages; "
                     "'placed' counts the task's objects placed):\n" + episode_text(sessions))
    parts.append(
        "Rules for the new text: it must work in any scene, building or task the robot meets, not only these: "
        "describe the robot and how to decide, never these scenes' objects, products, colours, furniture, shelf "
        "levels or bays, and no numbers except the robot's own facts (the arm reaches about 0.4 m ahead of the "
        f"torso; clicks are normalised 0-1000). At most {WORD_LIMIT[component]} words; plain sentences, no "
        "markdown headings. Return only the new text inside one ``` block.")
    return "\n\n".join(parts)


def proposal(answer: str) -> str:
    block = re.search(r"```[a-zA-Z]*\n?(.*?)```", answer, re.DOTALL)
    return (block[1] if block else answer).strip()


def lint(component: str, text: str, banned: list[str]) -> str | None:
    """Why TEXT is refused, or None."""
    words = len(text.split())
    if words > WORD_LIMIT[component]:
        return f"{words} words, over the limit of {WORD_LIMIT[component]}"
    named = sorted(vocabulary(text) & set(banned))
    if named:
        return f"names words of these scenarios: {', '.join(named)}"
    numbers = sorted(set(re.findall(r"\d+(?:\.\d+)?", text)) - ALLOWED_NUMBERS)
    if numbers:
        return f"numbers that are no robot fact: {', '.join(numbers)}"
    return None


# --- the loop ---
def eligible(state: dict) -> list[dict]:
    """Candidates that may be parents: scored on val and neither dropped by an end-to-end check nor descended from
    one dropped without a passed check of their own (the nearest checked ancestor-or-self decides)."""
    by_id = {c["id"]: c for c in state["candidates"]}

    def cleared(candidate) -> bool:
        while candidate is not None:
            if candidate.get("dropped"):
                return False
            if "val" in candidate.get("e2e", {}):
                return True
            candidate = by_id.get(candidate["parent"])
        return True

    return [c for c in state["candidates"] if c.get("val_mean") is not None and cleared(c)]


def dominated_removed(fronts: dict[str, set], means: dict[str, float]) -> dict[str, set]:
    """gepa's remove_dominated_programs (gepa_utils.py): a candidate is dominated when every case it is best on
    also has another surviving best candidate; checked from the lowest val mean up, one at a time."""
    programs = sorted({p for front in fronts.values() for p in front}, key=lambda p: means[p])
    dominated: set[str] = set()
    found = True
    while found:
        found = False
        for y in programs:
            if y in dominated:
                continue
            others = set(programs) - {y} - dominated
            if all(any(o in front for o in others) for front in fronts.values() if y in front):
                dominated.add(y)
                found = True
                break
    return {case: {p for p in front if p not in dominated} for case, front in fronts.items()}


def front(state: dict, run: Path) -> dict[str, int]:
    """The Pareto front over the val cases (gepa's select_program_candidate_from_pareto_front): the eligible
    candidates best on each case, dominated ones removed, and how many cases each survivor is best on."""
    alive = eligible(state)
    scores = {c["id"]: {k: r["score"] for k, r in records(run, c["id"]).items()} for c in alive}
    fronts: dict[str, set] = {}
    for case in state["val"]:
        values = {cid: s.get(case) for cid, s in scores.items() if s.get(case) is not None}
        if values:
            best = max(values.values())
            fronts[case] = {cid for cid, value in values.items() if value == best}
    counts: dict[str, int] = {}
    for kept in dominated_removed(fronts, {c["id"]: c["val_mean"] for c in alive}).values():
        for cid in kept:
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def minibatch(state: dict, component: str, rng: random.Random) -> list[str]:
    wanted = FAMILIES[component]
    pool = [case for case, fams in state["train"].items() if set(fams) & set(wanted)]
    rng.shuffle(pool)
    chosen, per = [], max(1, MINIBATCH // len(wanted))
    for family in wanted:
        chosen += [c for c in pool if family in state["train"][c] and c not in chosen][:per]
    return chosen[:MINIBATCH]


def iterate(run: Path, state: dict, rng: random.Random):
    counts = front(state, run)
    ids = sorted(counts)
    parent_id = rng.choices(ids, weights=[counts[i] for i in ids])[0]
    parent = next(c for c in state["candidates"] if c["id"] == parent_id)
    texts = GROUPS[state.get("group", "planner")]["texts"]
    component = texts[parent.get("next_text", 0) % len(texts)]  # gepa's round robin: a pointer per candidate
    state["iteration"] += 1
    names = minibatch(state, component, rng)
    batch = list(evaluate(run, state, parent_id, names).values())
    if all(r["score"] is None or r["score"] >= PERFECT for r in batch):  # gepa's skip_perfect_score
        log(run, f"iteration {state['iteration']}: {component} of {parent_id} perfect on its minibatch, skipped")
        save(run, state)
        return
    parent["next_text"] = (parent.get("next_text", 0) + 1) % len(texts)
    sessions = state["episodes"].get(parent_id) or state["episodes"].get("c000") or []  # none before a check
    prompt = reflection_prompt(component, parent["texts"][component], batch, sessions, state["banned"])
    refused, text = "no proposal", ""
    for _ in range(2):
        answer, cost = critic(prompt)
        state["credits"] += cost
        text = proposal(answer)
        refused = lint(component, text, state["banned"])
        if not refused:
            break
        log(run, f"iteration {state['iteration']}: proposal refused ({refused})")
        prompt += f"\n\nYour previous text was refused: {refused}. Write it again within the rules."
    save(run, state)
    if refused:
        return
    cid = f"c{len(state['candidates']):03d}"
    child = {"id": cid, "parent": parent_id, "component": component, "iteration": state["iteration"],
             "texts": {**parent["texts"], component: text}, "next_text": parent["next_text"]}
    state["candidates"].append(child)
    save(run, state)
    inherit(run, cid, parent_id, component, names)
    before, after = mean(batch), mean(evaluate(run, state, cid, names).values())
    child["minibatch"] = [round(before, 4), round(after, 4)]
    if after <= before:
        child["val_mean"] = None
        log(run, f"{cid} ({component} of {parent_id}): minibatch {before:.3f} -> {after:.3f}, not kept")
    else:
        evaluate(run, state, parent_id, state["val"])  # known for every kept candidate; inherit reads them
        inherit(run, cid, parent_id, component, state["val"])
        child["val_mean"] = round(mean(evaluate(run, state, cid, state["val"]).values()), 4)
        log(run, f"{cid} ({component} of {parent_id}): minibatch {before:.3f} -> {after:.3f}, val "
                 f"{child['val_mean']:.3f}")
    save(run, state)


# a candidate end to end: the Dex1 profile (demo-pp and warehouse tasks) and the Dex3 one (G1 and move-variety
# tasks) at once, each train, val and test with Sol's recorded first plans and live rewrites; val decides, test is
# only reported
E2E_BASES = {"A": ("GD1", "gepa-dex1"), "B": ("GD3", "gepa-dex3")}


def run_e2e(run: Path, state: dict, candidate: dict) -> dict:
    eid = f"G{run.name[:4].upper()}{candidate['id'][1:]}"
    texts = candidate["texts"]
    options = {"local_notes": "", "local_guidance": texts["guidance"], "local_target_note": texts["target_note"],
               "local_pick_note": texts["pick_note"], "local_place_note": texts["place_note"]}
    ids = {suffix: f"{eid}{suffix}" for suffix in E2E_BASES}
    for suffix, (base, _) in E2E_BASES.items():
        ssh(E2E_HOST, f"mkdir -p {SHARED}/gepa && cat > {SHARED}/gepa/{ids[suffix]}.json",
            json.dumps({"id": ids[suffix], "base": base, "options": options}))
    log(run, f"{candidate['id']}: end to end as {', '.join(ids.values())}")
    server = GROUPS[state.get("group", "planner")].get("server", GPU_HOST)
    ssh(E2E_HOST, f"cd {E2E_REPO} && setsid nohup .venv-loop/bin/python -m evolve.agent_experiments worker --server "
                  f"{server} --keep --concurrent {' '.join(ids.values())} > ~/agent-logs/gepa-{eid}.log 2>&1 "
                  f"< /dev/null &")
    rows = wait_rows(ids)
    return e2e_result(state, candidate, eid, rows, {s: f"{profile}-{ids[s]}" for s, (_, profile) in E2E_BASES.items()})


def wait_rows(ids: dict[str, str]) -> dict:
    rows = {}
    while len(rows) < len(ids):
        time.sleep(120)
        for suffix, xid in ids.items():
            found = ssh(E2E_HOST, f"cat {SHARED}/rows/{xid}.json 2>/dev/null || cat {SHARED}/failed/{xid}.txt "
                                  f"2>/dev/null || true").strip()
            if found and suffix not in rows:
                rows[suffix] = json.loads(found) if found.startswith("{") else {"failed": found[:300]}
    return rows


def e2e_result(state: dict, candidate: dict, eid: str, rows: dict, workdirs: dict[str, str]) -> dict:
    """A candidate's end-to-end result from its experiments' rows (the mean of the two profiles' placed shares),
    its train sessions kept as reflection material."""
    failed = [r["failed"] for r in rows.values() if "failed" in r]
    if failed:
        return {"id": eid, "failed": "; ".join(failed)}
    result = {"id": eid, "val": round(sum(r["val_placed"] for r in rows.values()) / len(rows), 4),
              "test": round(sum(r.get("test_placed", 0.0) for r in rows.values()) / len(rows), 4),
              "train": round(sum(r["train_placed"] for r in rows.values()) / len(rows), 4),
              "rows": {s: {k: r.get(k) for k in ("train_placed", "val_placed", "test_placed")} for s, r in rows.items()}}
    state["episodes"][candidate["id"]] = [e for workdir in workdirs.values() for e in episodes(workdir)]
    return result


def seed_from(args):
    """The seed's end-to-end check from experiments already run (the base checks GD1 and GD3 with the seed's
    texts), for a group that shares it."""
    state = load(RUNS / args.names[0])
    ids = dict(zip(E2E_BASES, args.ids))
    rows = wait_rows(ids)
    workdirs = {s: f"{E2E_BASES[s][1]}-{ids[s]}" for s in ids}
    episodes_of: dict = {"episodes": {}}
    result = e2e_result(episodes_of, state["candidates"][0], "+".join(args.ids), rows, workdirs)
    for name in args.names:  # a file beside the state: the running loop adopts it (never two writers of state)
        (RUNS / name / "seed_e2e.json").write_text(json.dumps({"e2e": result, "episodes": episodes_of["episodes"].get(
            "c000", [])}))
        log(RUNS / name, f"seed end to end (from {', '.join(args.ids)}): {result}")


def end_to_end(run: Path, state: dict, candidate: dict | None = None):
    """CANDIDATE (else the best not yet run, if it beats the seed on val) end to end; dropped from the front when
    its val placed share is clearly below the seed's. The test share is logged, never used to choose."""
    seed = state["candidates"][0]
    shared = run / "seed_e2e.json"
    if candidate is not seed and "e2e" not in seed and state.get("seed_shared"):  # seed-from writes it
        while not shared.exists():
            log(run, "waiting for the shared seed check (seed_e2e.json)")
            time.sleep(300)
        adopted = json.loads(shared.read_text())
        seed["e2e"], state["episodes"]["c000"] = adopted["e2e"], adopted["episodes"]
        save(run, state)
    if candidate is not seed and "e2e" not in seed:  # the seed's check first, once every test plan is recorded
        while not (run / "plans_ready").exists():
            log(run, "waiting for plans_ready (every end-to-end task's Sol plan recorded) before the seed's check")
            time.sleep(300)
        end_to_end(run, state, seed)
    if candidate is None:
        untested = [c for c in eligible(state) if "e2e" not in c and c["val_mean"] > seed["val_mean"]]
        if not untested:
            return
        candidate = max(untested, key=lambda c: c["val_mean"])
    state["e2e_runs"] += 1
    candidate["e2e"] = run_e2e(run, state, candidate)
    result = candidate["e2e"]
    if "failed" in result:
        log(run, f"{candidate['id']} end to end failed: {result['failed']}")
    elif candidate is not seed:
        candidate["dropped"] = result["val"] < seed["e2e"]["val"] - E2E_MARGIN
        log(run, f"{candidate['id']} end to end: val {result['val']} (seed {seed['e2e']['val']}), test "
                 f"{result['test']} (seed {seed['e2e']['test']}){', dropped' if candidate['dropped'] else ''}")
    else:
        log(run, f"seed end to end: train {result['train']}, val {result['val']}, test {result['test']}")
    save(run, state)


def run_loop(args):
    run = RUNS / args.name
    state = load(run)
    rng = random.Random(1000 + state["iteration"])
    last = state["iteration"] + args.iterations
    while state["iteration"] < last and state["credits"] < args.credits:
        iterate(run, state, rng)
        if GROUPS[state.get("group", "planner")]["e2e"] and state["iteration"] % args.e2e_every == 0:
            end_to_end(run, state)
    if GROUPS[state.get("group", "planner")]["e2e"]:
        end_to_end(run, state)
    log(run, f"stopped at iteration {state['iteration']}, {state['credits']:.1f} credits")


def status(args):
    run = RUNS / args.name
    state = load(run)
    counts = front(state, run)
    print(f"iteration {state['iteration']}, {state['credits']:.1f} credits")
    print("| id | parent | text | minibatch | val | front | e2e |\n| --- | --- | --- | --- | --- | --- | --- |")
    for c in state["candidates"]:
        e2e = c.get("e2e", {})
        print(f"| {c['id']} | {c['parent'] or '-'} | {c['component'] or 'seed'} | {c.get('minibatch', '-')} | "
              f"{c.get('val_mean', '-')} | {counts.get(c['id'], '')} | "
              f"{('val ' + str(e2e['val']) + ' test ' + str(e2e['test'])) if 'val' in e2e else e2e.get('failed', '')}"
              f"{' dropped' if c.get('dropped') else ''} |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("name")
    i.add_argument("--sol-val", type=int, default=100, help="Sol's recorded val turns beside the own rollouts'")
    i.add_argument("--no-e2e", action="store_true", help="leave the seed's end-to-end check to `seed-e2e`")
    sub.add_parser("seed-e2e").add_argument("name")
    f = sub.add_parser("seed-from")
    f.add_argument("ids", nargs=2, help="the Dex1 and Dex3 experiments whose rows are the seed's check (GD1 GD3)")
    f.add_argument("--names", nargs="+", required=True, help="the group runs that share it")
    g = sub.add_parser("init-group")
    g.add_argument("name")
    g.add_argument("--group", required=True, choices=sorted(GROUPS))
    g.add_argument("--source", default="own1", help="the `init` run whose cases and seed records it starts from")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--iterations", type=int, default=40)
    r.add_argument("--credits", type=float, default=400.0, help="stop before Sol's reflections pass this")
    r.add_argument("--e2e-every", type=int, default=10)
    sub.add_parser("status").add_argument("name")
    args = parser.parse_args(argv)
    {"init": init, "run": run_loop, "status": status, "seed-e2e": seed_e2e, "init-group": init_group,
     "seed-from": seed_from}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
