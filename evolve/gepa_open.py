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
from pathlib import Path

from homebody.vlm.composite import TARGET_NOTE
from homebody.vlm.provider import streamed

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "astra-evolve" / "gepa"
GPU_HOST, GPU_REPO, GPU_PY = "loki-gh200", "~/hb-bench", "~/vllm-venv/bin/python"
GPU_CASES, GPU_RUNS = "~/bench/gepa-cases", "~/bench/gepa"
E2E_HOST, E2E_REPO = "loki-a100", "~/hb-agents"
SHARED = "/lambda/nfs/group-volume/workcell-sim/homebody-baseline/agent-experiments"
E2E_WORK = "~/hb-agents-work/e2e"
BASE_EXPERIMENT, SEED_RUNS = "SPR3", ("SPR2", "SPR3")
E2E_MARGIN = 3 / 36  # a candidate placing three objects (of 36) fewer than the seed's mean is dropped
PROXY = "http://127.0.0.1:4000/v1/responses"
CRITIC = "gpt-6.1-sol"
RATES = {"input": 50.0, "cached": 2.5, "output": 250.0}  # credits per 1M tokens (evolve/score_episode.py)
MINIBATCH, VAL_SIZE = 40, 240

TEXTS = ("guidance", "target_note", "pick_note", "place_note")
FAMILIES = {"guidance": ("choose", "judge", "recover", "navigate", "pick", "place"),
            "target_note": ("pick", "place"), "pick_note": ("pick",), "place_note": ("place",)}
WORD_LIMIT = {"guidance": 230, "target_note": 70, "pick_note": 60, "place_note": 60}
ROLES = {
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


def evaluate(run: Path, state: dict, cid: str, names: list[str]) -> dict:
    """CID's records of NAMES, asked on the GH200 where not yet known."""
    known = records(run, cid)
    missing = [n for n in names if n not in known]
    if missing:
        remote = f"{GPU_RUNS}/{run.name}"
        candidate = next(c for c in state["candidates"] if c["id"] == cid)
        listed = f"{remote}/lists/{cid}-{len(known)}.txt"
        ssh(GPU_HOST, f"mkdir -p {remote}/lists {remote}/cands && cat > {remote}/cands/{cid}.json",
            json.dumps(candidate["texts"]))
        ssh(GPU_HOST, f"cat > {listed}", "\n".join(missing))
        started = time.monotonic()
        gpu(f"ask {remote}/cands/{cid}.json {GPU_CASES} {remote}/out/{cid} --list {listed} --jobs 16")
        fetched = ssh(GPU_HOST, f"cat {remote}/out/{cid}/results.jsonl")
        with (run / "records" / f"{cid}.jsonl").open("a") as out:
            out.write(fetched)
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
            "target_note": TARGET_NOTE, "pick_note": "", "place_note": ""}


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
product products item items tray trays tote totes crate crates bin bins basket baskets
shelf shelves bay bays rack racks lane lanes slot slots aisle aisles counter counters cabinet cabinets drawer drawers
table tables kitchen warehouse restock fridge sink
""".split())


def vocabulary(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)}


def banned_words(tasks: list[str]) -> list[str]:
    """The scene words the scenarios' tasks use (SCENE_WORDS): objects, colours, containers and places."""
    return sorted(vocabulary(" ".join(tasks)) & SCENE_WORDS)


def init(args):
    run = RUNS / args.name
    (run / "records").mkdir(parents=True, exist_ok=True)
    ssh(GPU_HOST, f"mkdir -p {GPU_CASES} && for d in ~/bench/cases-v3-pp/* ~/bench/cases-v3-wh/*; do "
                  f"ln -sfn $d {GPU_CASES}/$(basename $d); done")
    rows = [json.loads(line) for line in gpu(f"index {GPU_CASES}").splitlines() if line.strip()]
    rows = [r for r in rows if r["families"] and split_of(r["scenario"])]
    rng = random.Random(7)
    train = [r for r in rows if split_of(r["scenario"]) == "train"]
    val = sample_val([r for r in rows if split_of(r["scenario"]) == "val"], VAL_SIZE, rng)
    seed = {"id": "c000", "parent": None, "component": None, "texts": seed_texts(), "iteration": 0}
    state = {"train": {r["case"]: r["families"] for r in train}, "val": val, "candidates": [seed],
             "iteration": 0, "credits": 0.0, "banned": [], "seed_e2e": [], "e2e_runs": 0}
    save(run, state)
    log(run, f"{len(train)} train cases, {len(val)} val cases")
    seed_records = evaluate(run, state, "c000", val)
    seed["val_mean"] = round(mean(seed_records.values()), 4)
    state["banned"] = banned_words(sorted({r["task"] for r in seed_records.values()}
                                          | {r["goal"] or "" for r in seed_records.values()}))
    state["seed_e2e"] = [json.loads(ssh(E2E_HOST, f"cat {SHARED}/rows/{eid}.json"))["all_placed"]
                         for eid in SEED_RUNS]
    state["episodes"] = {"c000": episodes(BASE_EXPERIMENT) + episodes("SPR2")}
    save(run, state)
    log(run, f"seed val {mean(seed_records.values()):.3f}; {len(state['banned'])} scenario words banned")


def episodes(eid: str) -> list[dict]:
    out = ssh(E2E_HOST, f"cd {E2E_REPO} && PYTHONPATH=src:. .venv-loop/bin/python -m evolve.gepa_eval episodes "
                        f"{E2E_WORK}/demo-pp-{eid}")
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
        "Below are decision turns from recorded sessions where the agent with this text was scored against a strong "
        "reference model and the scene's ground truth, with feedback. Diagnose the general behaviour that causes the "
        "failures and write an improved text that fixes it while keeping what already works.",
        "Failed turns:\n\n" + "\n\n".join(example(r) for r in failures),
        "Turns it got right:\n\n" + "\n\n".join(example(r) for r in passes)]
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
def front(state: dict, run: Path) -> dict[str, int]:
    """The Pareto front over the val cases: each kept candidate and how many cases it is (jointly) best on."""
    alive = [c for c in state["candidates"] if c.get("val_mean") is not None and not c.get("dropped")]
    scores = {c["id"]: {k: r["score"] for k, r in records(run, c["id"]).items()} for c in alive}
    counts: dict[str, int] = {}
    for case in state["val"]:
        values = {cid: s.get(case) for cid, s in scores.items() if s.get(case) is not None}
        if not values:
            continue
        best = max(values.values())
        for cid, value in values.items():
            if value == best:
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
    component = TEXTS[state["iteration"] % len(TEXTS)]
    state["iteration"] += 1
    names = minibatch(state, component, rng)
    batch = list(evaluate(run, state, parent_id, names).values())
    sessions = state["episodes"].get(parent_id) or state["episodes"]["c000"]
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
             "texts": {**parent["texts"], component: text}}
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


def end_to_end(run: Path, state: dict):
    """The best candidate not yet run end to end, if it beats the seed on val: its demo-pp baseline."""
    seed_val = state["candidates"][0]["val_mean"]
    untested = [c for c in state["candidates"] if c.get("val_mean") and "e2e" not in c and not c.get("dropped")
                and c["val_mean"] > seed_val]
    if not untested:
        return
    best = max(untested, key=lambda c: c["val_mean"])
    state["e2e_runs"] += 1
    eid = f"G{run.name[:4].upper()}{best['id'][1:]}"
    options = {"local_notes": "", "local_guidance": best["texts"]["guidance"],
               "local_target_note": best["texts"]["target_note"], "local_pick_note": best["texts"]["pick_note"],
               "local_place_note": best["texts"]["place_note"]}
    ssh(E2E_HOST, f"mkdir -p {SHARED}/gepa && cat > {SHARED}/gepa/{eid}.json",
        json.dumps({"id": eid, "base": BASE_EXPERIMENT, "options": options}))
    log(run, f"{best['id']}: end to end as {eid}")
    ssh(E2E_HOST, f"cd {E2E_REPO} && setsid nohup .venv-loop/bin/python -m evolve.agent_experiments worker --server "
                  f"{GPU_HOST} --keep {eid} > ~/agent-logs/gepa-{eid}.log 2>&1 < /dev/null &")
    while True:
        time.sleep(120)
        found = ssh(E2E_HOST, f"cat {SHARED}/rows/{eid}.json 2>/dev/null || cat {SHARED}/failed/{eid}.txt "
                              f"2>/dev/null || true")
        if found.strip():
            break
    if not found.lstrip().startswith("{"):
        log(run, f"{eid} failed: {found.strip()[:300]}")
        best["e2e"] = {"failed": found.strip()[:300]}
        save(run, state)
        return
    row = json.loads(found)
    seed = sum(state["seed_e2e"]) / len(state["seed_e2e"])
    best["e2e"] = {"id": eid, "all_placed": row["all_placed"], "train_placed": row["train_placed"],
                   "val_placed": row["val_placed"]}
    best["dropped"] = row["all_placed"] < seed - E2E_MARGIN
    state["episodes"][best["id"]] = episodes(eid)
    log(run, f"{eid}: placed {row['all_placed']} (seed runs {state['seed_e2e']})"
             f"{', dropped' if best['dropped'] else ''}")
    save(run, state)


def run_loop(args):
    run = RUNS / args.name
    state = load(run)
    rng = random.Random(1000 + state["iteration"])
    last = state["iteration"] + args.iterations
    while state["iteration"] < last and state["credits"] < args.credits:
        iterate(run, state, rng)
        if state["iteration"] % args.e2e_every == 0:
            end_to_end(run, state)
    end_to_end(run, state)
    log(run, f"stopped at iteration {state['iteration']}, {state['credits']:.1f} credits")


def status(args):
    run = RUNS / args.name
    state = load(run)
    counts = front(state, run)
    print(f"iteration {state['iteration']}, {state['credits']:.1f} credits, seed e2e {state['seed_e2e']}")
    print("| id | parent | text | minibatch | val | front | e2e |\n| --- | --- | --- | --- | --- | --- | --- |")
    for c in state["candidates"]:
        e2e = c.get("e2e", {})
        print(f"| {c['id']} | {c['parent'] or '-'} | {c['component'] or 'seed'} | {c.get('minibatch', '-')} | "
              f"{c.get('val_mean', '-')} | {counts.get(c['id'], '')} | "
              f"{e2e.get('all_placed', e2e.get('failed', ''))}{' dropped' if c.get('dropped') else ''} |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init").add_argument("name")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--iterations", type=int, default=40)
    r.add_argument("--credits", type=float, default=400.0, help="stop before Sol's reflections pass this")
    r.add_argument("--e2e-every", type=int, default=10)
    sub.add_parser("status").add_argument("name")
    args = parser.parse_args(argv)
    {"init": init, "run": run_loop, "status": status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
