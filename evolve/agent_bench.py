"""Offline capability bench for the robot agent's models: real decision turns of recorded sessions replayed without
the simulator, so each capability the demo needs is measured per model in minutes (evolve/agent_capabilities.py
scores them).

    python3 -m evolve.agent_bench collect OUT SESSION_DIRS...   turns of recorded sessions (e.g. Sol's b002) as cases
    python3 -m evolve.agent_bench turns CASES OUT --model M [--local-url U] [--layout L] [--no-labels] [--limit N]
    python3 -m evolve.agent_bench point CASES OUT --served S --format F --url U   object pointing asked alone
    python3 -m evolve.agent_bench verify CASES OUT --served S --url U            goal checks asked as yes/no
    python3 -m evolve.agent_bench score OUT...                  one row per run, one column per capability
    python3 -m evolve.agent_bench template CASES --model M --local-url U   the text the server renders for one case

A case is one turn a session decided: its prompt, the four images it sent, the label image of that frame, the
decisions the turn allowed (the plan phase, read from the prompt's plan note), the active goal, the previous turn's
decision and result, and the reference model's decision and result. `turns` asks a model through the production
provider (vlm/catalog.make) as a session would; `point` and `verify` ask a model in its own words through the
composite agents' endpoint (vlm/composite.Endpoint, to_1000, verdict), so pointers and verifiers are measured
with the code that will use them."""
import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from homebody.session.config import Agent
from homebody.session.planned import GOAL_DONE, GOAL_FAILED, PLAN
from homebody.vlm.catalog import make
from homebody.vlm.composite import POINTER_PROMPTS, VERIFY_PROMPT, Endpoint, to_1000, verdict
from homebody.vlm.provider import ProviderError

from .agent_capabilities import pixel, score_run

IMAGES = ("rgb.png", "depth_preview.png", "map.png", "topdown.png")
PLAN_DECISIONS = (PLAN, GOAL_DONE, GOAL_FAILED)
ACTIVE_GOAL = re.compile(r"^Active goal [^:\n]+:\s*(.+?)(?: \(\d+ skill calls? left for it\)\.)?$", re.MULTILINE)
OBSERVATION = re.compile(r"^Current observation:\n(\{.*\})$", re.MULTILINE)


# --- collect ---
def allowed(prompt: str, offered: tuple[str, ...]) -> tuple[str, ...]:
    """The turn's allowed decisions, as planned.PlanRunner.names gave them, from the plan note ending the prompt."""
    base = tuple(n for n in offered if n not in PLAN_DECISIONS)
    if "This turn: write the plan (skill plan)." in prompt or "rewrite the rest of it (skill plan)" in prompt:
        return (PLAN,)
    if "; finish with done." in prompt:
        return ("done",)
    if "\nActive goal " in prompt:
        return (*[n for n in base if n != "done"], GOAL_DONE, GOAL_FAILED, "done")
    return base if PLAN not in offered else offered


def turns(session: Path):
    """(step, decision event or None for a refused reply, result event) of each step the session took."""
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines() if line.strip()]
    decisions = {e["step"]: e for e in events if e["kind"] == "decision"}
    for result in (e for e in events if e["kind"] == "result"):
        yield result["step"], decisions.get(result["step"]), result


def find_enum(value):
    """The skill enum of a saved request (any provider's schema shape)."""
    if isinstance(value, dict):
        if value.get("type") == "string" and "enum" in value and "done" in value["enum"]:
            return value["enum"]
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            if (found := find_enum(item)) is not None:
                return found
    return None


def target(decision, result, labels):
    """The label the reference clicked when its pick or place succeeded (the object to hit), else None."""
    if decision["skill"] not in ("pick", "place") or result.get("code") != "OK":
        return None
    uv = pixel(decision["arguments"].get("point_normalized_1000"), labels.shape)
    return int(labels[uv[1], uv[0]]) if uv else None


def collect_session(session: Path, scenario: str, out: Path) -> int:
    count, previous = 0, None
    for step, decision, result in turns(session):
        folder = session / "tasks" / "task-000" / "decisions" / f"step-{step:03d}"
        here = {"decision": decision and decision["decision"], "code": result["result"].get("code"),
                "message": result["result"].get("message", "")[:300]}
        frame = decision and session / "frames" / f"{decision['epoch']:03d}-{decision['sequence']:08d}"
        if frame is None or not (folder / "prompt.txt").is_file() or not all(
                (frame / name).is_file() for name in IMAGES):
            previous = here
            continue
        prompt = (folder / "prompt.txt").read_text()
        offered = tuple(find_enum(json.loads((folder / "request.json").read_text())) or ())
        labels = np.load(frame / "sensors.npz")["labels"]
        case = out / f"{scenario}-{session.name[-8:]}-{step:03d}"
        case.mkdir(parents=True, exist_ok=True)
        for name in IMAGES:
            shutil.copy(frame / name, case / name)
        np.savez_compressed(case / "labels.npz", labels=labels)
        (case / "prompt.txt").write_text(prompt)
        goal, observation = ACTIVE_GOAL.search(prompt), OBSERVATION.search(prompt)
        (case / "case.json").write_text(json.dumps({
            "scenario": scenario, "step": step, "offered": offered, "allowed": allowed(prompt, offered),
            "goal": goal[1].strip() if goal else None,
            "observation": json.loads(observation[1]) if observation else None,
            "previous": previous, "reference": decision["decision"], "result": here["code"],
            "message": here["message"], "target": target(decision["decision"], result["result"], labels)}, indent=1))
        previous, count = here, count + 1
    return count


def collect(args):
    out, total = Path(args.out), 0
    for root in args.sessions:
        for events in sorted(Path(root).glob("**/session/*/events.jsonl")):
            session = events.parent
            total += collect_session(session, session.parents[2].name, out)  # <scenario>/r<k>/session/<session>
    print(f"{total} cases in {out}")


# --- asking ---
def cases_of(args):
    return sorted(p.parent for p in Path(args.cases).glob("*/case.json"))[:args.limit]


def save(out: Path, case: Path, record: dict, started: float) -> dict:
    record = {"case": case.name, **record, "seconds": round(time.monotonic() - started, 2)}
    (out / case.name).mkdir(parents=True, exist_ok=True)
    (out / case.name / "bench.json").write_text(json.dumps(record, indent=1))
    return record


def ask_turn(settings: Agent, case: Path, out: Path) -> dict:
    meta, started = json.loads((case / "case.json").read_text()), time.monotonic()
    provider = make(settings)
    if hasattr(provider, "turn"):
        provider.turn(tuple(meta["allowed"]))
    try:
        decision = provider.generate((case / "prompt.txt").read_text(), images=tuple(case / n for n in IMAGES),
                                     names=tuple(meta["offered"]), attempt=out / case.name, tick=lambda: None,
                                     cancelled=lambda: False)
        record = {"skill": decision.skill, "arguments": decision.arguments, "text": decision.text}
    except ProviderError as error:
        record = {"error": str(error)[:500]}
    return save(out, case, record, started)


def point_target(meta) -> str | None:
    """The object of a pick the reference made successfully, as the pointer is told it: the reference's own words."""
    if meta["target"] is None or meta["reference"]["skill"] != "pick":
        return None
    return meta["reference"]["text"].rstrip(".")


def ask_point(endpoint: Endpoint, case: Path, out: Path) -> dict | None:
    meta = json.loads((case / "case.json").read_text())
    if (wanted := point_target(meta)) is None:
        return None
    started = time.monotonic()
    prompt = POINTER_PROMPTS[endpoint.format].format(target=f"the object in this robot step: {wanted}")
    try:
        answer = endpoint.ask(prompt, (case / "rgb.png",), attempt=out / case.name, tick=lambda: None,
                              cancelled=lambda: False)
    except ProviderError as error:
        return save(out, case, {"error": str(error)[:300]}, started)
    try:
        record = {"answer": answer[:500], "point": to_1000(answer, endpoint.format, case / "rgb.png")}
    except ProviderError as error:
        record = {"answer": answer[:500], "error": str(error)[:300]}
    return save(out, case, record, started)


def verify_label(meta) -> bool | None:
    """True for a goal the reference claimed done and the plan's check accepted; False for one it was still acting
    on (a navigate, pick or place inside the goal); None otherwise."""
    if meta["goal"] is None:
        return None
    if meta["reference"]["skill"] == GOAL_DONE:
        return True if meta["result"] == "OK" else None
    return False if meta["reference"]["skill"] in ("navigate", "pick", "place") else None


def ask_verify(endpoint: Endpoint, case: Path, out: Path) -> dict | None:
    meta = json.loads((case / "case.json").read_text())
    if (truth := verify_label(meta)) is None:
        return None
    started = time.monotonic()
    prompt = VERIFY_PROMPT.format(goal=meta["goal"], claim="The goal is done.")
    try:
        answer = endpoint.ask(prompt, (case / "rgb.png",), attempt=out / case.name, tick=lambda: None,
                              cancelled=lambda: False)
    except ProviderError as error:
        return save(out, case, {"error": str(error)[:300], "truth": truth}, started)
    return save(out, case, {"answer": answer[:300], "achieved": verdict(answer)[0], "truth": truth}, started)


def run(args, kind: str, ask):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"kind": kind, "model": args.label, "cases": str(Path(args.cases))}))
    cases = cases_of(args)
    with ThreadPoolExecutor(args.jobs) as pool:
        for n, record in enumerate(pool.map(lambda c: ask(c, out), cases), 1):
            if record is not None:
                shown = record.get("skill") or record.get("point") or record.get("achieved", "")
                print(f"[{n}/{len(cases)}] {record['case']}: {shown} {record.get('error', '')[:80]}", flush=True)


def run_turns(args):
    settings = Agent(model=args.model, local_layout=args.layout, local_image_labels=not args.no_labels,
                     local_temperature=args.temperature,
                     **{k: v for k, v in (("local_url", args.local_url), ("second_url", args.second_url)) if v})
    args.label = args.label or f"{args.model} {args.layout}{' unlabelled' if args.no_labels else ''}"
    run(args, "turns", lambda case, out: ask_turn(settings, case, out))


def run_endpoint(args, kind, ask):
    endpoint = Endpoint(args.served, args.format if kind == "point" else "yesno", args.url, 300.0)
    args.label = args.label or f"{args.served} ({endpoint.format})"
    run(args, kind, lambda case, out: ask(endpoint, case, out))


# --- report ---
def score(args):
    rows = [score_run(Path(out)) for out in args.outs]
    keys = sorted({k for row in rows for k in row} - {"model", "kind"})
    print("| model | kind | " + " | ".join(keys) + " |\n|" + " --- |" * (len(keys) + 2))
    for row in rows:
        print(f"| {row['model']} | {row['kind']} | " + " | ".join(str(row.get(k, "")) for k in keys) + " |")


def template(args):
    """The text the vLLM server renders for the first case's request (its chat template, vision placeholders)."""
    provider = make(Agent(model=args.model, local_url=args.local_url))
    case = sorted(p.parent for p in Path(args.cases).glob("*/case.json"))[0]
    meta = json.loads((case / "case.json").read_text())
    if hasattr(provider, "turn"):
        provider.turn(tuple(meta["allowed"]))
    body = provider.request((case / "prompt.txt").read_text(), tuple(case / n for n in IMAGES), tuple(meta["offered"]))
    base = args.local_url.rstrip("/").removesuffix("/v1")
    headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}", "content-type": "application/json"}
    extra = {k: body[k] for k in ("chat_template_kwargs",) if k in body}
    tokens = post(f"{base}/tokenize", {"model": body["model"], "messages": body["messages"], **extra}, headers)
    text = post(f"{base}/detokenize", {"model": body["model"], "tokens": tokens["tokens"]}, headers)["prompt"]
    print(f"{case.name}: {tokens['count']} tokens\n--- head ---\n{text[:1500]}\n--- tail ---\n{text[-900:]}")


def post(url, body, headers):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("sessions", nargs="+", help="folders holding <split>/<scenario>/r<k>/session/<session>/")
    for name in ("turns", "point", "verify"):
        r = sub.add_parser(name)
        r.add_argument("cases")
        r.add_argument("out")
        r.add_argument("--limit", type=int)
        r.add_argument("--jobs", type=int, default=8)
        r.add_argument("--label", help="the run's name in the report")
        if name == "turns":
            r.add_argument("--model", required=True, help="a session/config.py MODELS name")
            r.add_argument("--local-url")
            r.add_argument("--second-url")
            r.add_argument("--layout", default="text-first", choices=("text-first", "images-first"))
            r.add_argument("--no-labels", action="store_true")
            r.add_argument("--temperature", type=float, default=0.0)
        else:
            r.add_argument("--served", required=True, help="the server's --served-model-name")
            r.add_argument("--url", required=True)
            if name == "point":
                r.add_argument("--format", required=True, choices=sorted(POINTER_PROMPTS))
    s = sub.add_parser("score")
    s.add_argument("outs", nargs="+")
    t = sub.add_parser("template")
    t.add_argument("cases")
    t.add_argument("--model", required=True)
    t.add_argument("--local-url", required=True)
    args = parser.parse_args(argv)
    commands = {"collect": collect, "score": score, "template": template, "turns": run_turns,
                "point": lambda a: run_endpoint(a, "point", ask_point),
                "verify": lambda a: run_endpoint(a, "verify", ask_verify)}
    commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
