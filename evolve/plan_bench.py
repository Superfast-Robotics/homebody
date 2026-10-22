"""Plan bench: how good the open models' own plans are next to Sol's, judged by Claude Opus 5.5 (2026-10-22). Each
case is a task's first plan turn (its prompt and first frame, Sol's recorded plan beside it); a model writes its plan
for the same turn, the plan check is applied, and the judge compares the two plans blind, in both orders.

    python3 -m evolve.plan_bench ask CASES OUT --model M [--local-url U] [--jobs N]   the model's plan per case
    python3 -m evolve.plan_bench judge CASES OUT [--jobs N] [--limit N]               Opus compares, both orders
    python3 -m evolve.plan_bench score CASES OUT                                      the table

The judge runs through the local Claude Code CLI (the operator's login, as vlm/claude.py): its only tool is Read on
the case's head camera and map images, and its reply follows JSON_SCHEMA. A preference counts only when it holds in
both orders (LLM judges favour a position); otherwise it is a tie."""
import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from homebody.session.config import Agent
from homebody.session.planned import Plan, PlanError
from homebody.vlm.catalog import make
from homebody.vlm.provider import ProviderError

from .agent_bench import IMAGES

NOTES = "reach,claims,recover"
JUDGE_MODEL = "claude-opus-5-5"
CRITERIA = ("coverage", "feasibility", "verifiable_checks", "recovery", "no_invention")
JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["A", "B", "preferred", "reason"],
    "properties": {
        **{side: {"type": "object", "additionalProperties": False, "required": list(CRITERIA),
                  "properties": {c: {"type": "integer", "minimum": 0, "maximum": 10} for c in CRITERIA}}
           for side in ("A", "B")},
        "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"}}}
JUDGE_PROMPT = """You judge two behaviour-tree plans for the same robot task. A humanoid robot with two grippers
works through a plan's goals in order with skills (navigate, pick, place, goal_done, goal_failed); a "sequence" runs
its children in order, a "fallback" tries its first child and runs the next only if it fails; each goal has a check
the harness verifies (placed, arrived, holding; or none: the robot itself declares it done) and a budget of skill
calls.

Task: {task}

Read the attached images: image 1 is the robot's head camera now, image 3 the floor map (light: free floor; dark:
furniture; the blue dot and arrow: the robot and its heading).

Plan A:
{plan_a}

Plan B:
{plan_b}

Score each plan 0-10 on: coverage (every object the task names gets its own pick, carry and place, in the order the
task asks), feasibility (stances and destinations make sense for this scene), verifiable_checks (goals use the
harness checks where they can rather than self-declared ones), recovery (fallbacks that record a failure and move
on, budgets that allow the retries the task permits), no_invention (no steps, objects or places the task and scene
do not have). Then say which plan you would rather the robot run (or tie) and why in one sentence."""


def tree_text(node, depth=0) -> str:
    pad = "  " * depth
    if node.get("type") == "goal":
        check = node.get("check") or {}
        name = check.get("name", "none (self-declared)")
        extra = ", ".join(f"{k}={v}" for k, v in check.items() if k != "name")
        return f"{pad}- goal {node.get('id')}: {node.get('goal')} [check: {name}{' ' + extra if extra else ''}; " \
               f"budget {node.get('budget', '-')}]"
    lines = [f"{pad}{str(node.get('type', '?')).upper()}"]
    return "\n".join(lines + [tree_text(child, depth + 1) for child in node.get("children", [])])


def accepted(tree) -> bool:
    try:
        Plan(tree)
        return True
    except (PlanError, TypeError, KeyError, AttributeError):
        return False


# --- ask ---
def ask_one(case: Path, out: Path, settings: Agent) -> dict:
    done = out / case.name / "plan.json"
    if done.is_file():
        return json.loads(done.read_text())
    provider = make(settings)
    if hasattr(provider, "turn"):
        provider.turn(("plan",))
    attempt = out / case.name / "request"
    shutil.rmtree(attempt, ignore_errors=True)
    try:
        decision = provider.generate((case / "prompt.txt").read_text(), images=tuple(case / n for n in IMAGES),
                                     names=("plan",), attempt=attempt, tick=lambda: None, cancelled=lambda: False)
        row = {"text": decision.text, "tree": decision.arguments.get("tree"),
               "accepted": accepted(decision.arguments.get("tree"))}
    except ProviderError as error:
        row = {"error": str(error)[:300], "accepted": False}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({"case": case.name, **row}, indent=1))
    return row


def run_ask(args):
    settings = Agent(model=args.model, local_url=args.local_url, local_notes=NOTES)
    cases, out = sorted(p.parent for p in Path(args.cases).glob("*/case.json")), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"model": args.model}))
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda c: ask_one(c, out, settings), cases))
    print(f"{args.model}: {sum(r['accepted'] for r in rows)} of {len(rows)} plans accepted by the plan check")


# --- judge ---
def judge_once(case: Path, plan_a, plan_b, record: Path) -> dict:
    """One Opus comparison through the Claude Code CLI, in a throwaway folder holding only the images."""
    meta = json.loads((case / "case.json").read_text())
    with tempfile.TemporaryDirectory(prefix="plan-judge-") as work:
        for i, name in ((1, "rgb.png"), (3, "map.png")):
            shutil.copy(case / name, Path(work) / f"observation-{i}.png")
        prompt = JUDGE_PROMPT.format(task=meta["task"], plan_a=tree_text(plan_a), plan_b=tree_text(plan_b))
        prompt += "\nAttachment 1: read ./observation-1.png\nAttachment 3: read ./observation-3.png"
        command = [shutil.which("claude") or "claude", "--print", "--output-format", "json", "--strict-mcp-config",
                   "--json-schema", json.dumps(JSON_SCHEMA), "--tools", "Read", "--allowedTools",
                   "Read(./observation-*)", "--permission-mode", "dontAsk", "--no-session-persistence",
                   "--model", JUDGE_MODEL]
        done = subprocess.run(command, input=prompt, capture_output=True, text=True, cwd=work, timeout=900)
    record.write_text(done.stdout or done.stderr)
    try:
        return json.loads(done.stdout)["structured_output"]
    except (ValueError, KeyError, TypeError):
        return {"error": (done.stderr or done.stdout)[-300:]}


def judge_case(case: Path, out: Path) -> dict | None:
    done = out / case.name / "judge.json"
    if done.is_file():
        return json.loads(done.read_text())
    mine = json.loads((out / case.name / "plan.json").read_text())
    if not mine.get("accepted"):
        return None  # a plan the check refuses never runs: scored as such, not judged
    sol = json.loads((case / "case.json").read_text())["sol"]["tree"]
    rng = random.Random(case.name)
    first_open = rng.random() < 0.5  # the first order is random, the second the swap
    verdicts = []
    for k, open_is_a in enumerate((first_open, not first_open)):
        a, b = (mine["tree"], sol) if open_is_a else (sol, mine["tree"])
        verdict = judge_once(case, a, b, out / case.name / f"judge-{k}.txt")
        if "error" not in verdict:
            own, theirs = ("A", "B") if open_is_a else ("B", "A")
            verdicts.append({"open": verdict[own], "sol": verdict[theirs], "reason": verdict["reason"],
                             "preferred": {"tie": "tie", own: "open", theirs: "sol"}[verdict["preferred"]]})
    result = {"case": case.name, "verdicts": verdicts,
              "preferred": (verdicts[0]["preferred"] if len(verdicts) == 2
                            and verdicts[0]["preferred"] == verdicts[1]["preferred"] else "tie")}
    done.write_text(json.dumps(result, indent=1))
    return result


def run_judge(args):
    cases = sorted(p.parent for p in Path(args.cases).glob("*/case.json"))[:args.limit]
    out = Path(args.out)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = [r for r in pool.map(lambda c: judge_case(c, out), cases) if r]
    print(f"{len(results)} judged")


def run_score(args):
    out, rows = Path(args.out), []
    for case in sorted(p.parent for p in Path(args.cases).glob("*/case.json")):
        meta = json.loads((case / "case.json").read_text())
        plan = out / case.name / "plan.json"
        if not plan.is_file():
            continue
        mine = json.loads(plan.read_text())
        judged = out / case.name / "judge.json"
        rows.append((meta["split"], mine, json.loads(judged.read_text()) if judged.is_file() else None))
    print("| split | cases | plan check accepts | judged | open preferred | Sol preferred | tie | "
          + " | ".join(f"{c} open/Sol" for c in CRITERIA) + " |\n|" + " --- |" * (7 + len(CRITERIA)))
    for split in ("train", "val", "all"):
        chosen = [r for r in rows if split in ("all", r[0])]
        judged = [j for _, _, j in chosen if j and j["verdicts"]]
        means = []
        for c in CRITERIA:
            scores = [(v["open"][c], v["sol"][c]) for j in judged for v in j["verdicts"]]
            means.append(f"{sum(o for o, _ in scores) / len(scores):.1f}/{sum(s for _, s in scores) / len(scores):.1f}"
                         if scores else "-")
        print(f"| {split} | {len(chosen)} | {sum(m['accepted'] for _, m, _ in chosen)} | {len(judged)} | "
              + " | ".join(str(sum(j['preferred'] == p for j in judged)) for p in ("open", "sol", "tie"))
              + " | " + " | ".join(means) + " |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask")
    a.add_argument("cases")
    a.add_argument("out")
    a.add_argument("--model", required=True)
    a.add_argument("--local-url", default="http://127.0.0.1:8000/v1")
    a.add_argument("--jobs", type=int, default=8)
    j = sub.add_parser("judge")
    j.add_argument("cases")
    j.add_argument("out")
    j.add_argument("--jobs", type=int, default=4)
    j.add_argument("--limit", type=int)
    s = sub.add_parser("score")
    s.add_argument("cases")
    s.add_argument("out")
    args = parser.parse_args(argv)
    {"ask": run_ask, "judge": run_judge, "score": run_score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
