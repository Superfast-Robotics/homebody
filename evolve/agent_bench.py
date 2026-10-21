"""Offline turn bench for the robot agent's models: real decision turns replayed without the simulator, so plumbing,
chat templates, reply formats and pointing are checked in minutes.

    python3 -m evolve.agent_bench collect OUT SESSION_DIRS...   turns of recorded sessions (e.g. Sol's b002) as cases
    python3 -m evolve.agent_bench run CASES OUT --model M [--local-url U] [--second-url U] [--limit N] [--jobs J]
    python3 -m evolve.agent_bench score OUT...                  one row per run
    python3 -m evolve.agent_bench template CASES --model M --local-url U   the text the server renders for one case

A case is one turn a session decided: its prompt, the four images it sent, the label image of that frame, the
decisions the turn allowed (the plan phase, read from the prompt's plan note) and the reference model's decision and
result. `run` asks the model through the production provider (vlm/catalog.make), as a session would; `score` reports
replies that parse, allowed skills, agreement with the reference, and clicks against the frame's labels: a pick hits
when its pixel carries the object the reference picked successfully."""
import argparse
import json
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
from homebody.vlm.provider import ProviderError

IMAGES = ("rgb.png", "depth_preview.png", "map.png", "topdown.png")
PLAN_DECISIONS = (PLAN, GOAL_DONE, GOAL_FAILED)


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
    """(step, decision event, result event) of each decision the session executed."""
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines() if line.strip()]
    decisions = {e["step"]: e for e in events if e["kind"] == "decision"}
    results = {e["step"]: e for e in events if e["kind"] == "result"}
    for step, decision in sorted(decisions.items()):
        if step in results:
            yield step, decision, results[step]


def collect_session(session: Path, scenario: str, out: Path) -> int:
    count = 0
    for step, decision, result in turns(session):
        folder = session / "tasks" / "task-000" / "decisions" / f"step-{step:03d}"
        frame = session / "frames" / f"{decision['epoch']:03d}-{decision['sequence']:08d}"
        if not (folder / "prompt.txt").is_file() or not all((frame / name).is_file() for name in IMAGES):
            continue
        prompt = (folder / "prompt.txt").read_text()
        request = json.loads((folder / "request.json").read_text())
        offered = tuple(find_enum(request))
        case = out / f"{scenario}-{session.name[-8:]}-{step:03d}"
        case.mkdir(parents=True, exist_ok=True)
        for name in IMAGES:
            shutil.copy(frame / name, case / name)
        np.savez_compressed(case / "labels.npz", labels=np.load(frame / "sensors.npz")["labels"])
        (case / "prompt.txt").write_text(prompt)
        reference = decision["decision"]
        (case / "case.json").write_text(json.dumps({
            "scenario": scenario, "step": step, "offered": offered, "allowed": allowed(prompt, offered),
            "reference": reference, "result": result["result"].get("code"),
            "target": target(reference, result["result"], np.load(frame / "sensors.npz")["labels"])}, indent=1))
        count += 1
    return count


def find_enum(value):
    """The skill enum of a saved request (any provider's schema shape)."""
    if isinstance(value, dict):
        if value.get("type") == "string" and "enum" in value and "done" in value["enum"]:
            return value["enum"]
        for item in value.values():
            if (found := find_enum(item)) is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            if (found := find_enum(item)) is not None:
                return found
    return None


def pixel(point, shape):
    """observations.normalized_pixel: (u, v) of point_normalized_1000 on an image of SHAPE, or None."""
    if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(
            type(value) in (int, float) and 0 <= value <= 1000 for value in point):
        return None
    height, width = shape
    return tuple(np.floor(np.asarray(point) * [width - 1, height - 1] / 1000).astype(int).tolist())


def target(decision, result, labels):
    """The label the reference clicked when its pick or place succeeded (the object to hit), else None."""
    if decision["skill"] not in ("pick", "place") or result.get("code") != "OK":
        return None
    uv = pixel(decision["arguments"].get("point_normalized_1000"), labels.shape)
    return int(labels[uv[1], uv[0]]) if uv else None


def collect(args):
    out, total = Path(args.out), 0
    for root in args.sessions:
        for events in sorted(Path(root).glob("**/session/*/events.jsonl")):
            session = events.parent
            scenario = session.parents[2].name  # <split>/<scenario>/r<k>/session/<session>
            total += collect_session(session, scenario, out)
    print(f"{total} cases in {out}")


# --- run ---
def ask(settings: Agent, case: Path, out: Path):
    meta = json.loads((case / "case.json").read_text())
    provider = make(settings)
    if hasattr(provider, "turn"):
        provider.turn(tuple(meta["allowed"]))
    attempt = out / case.name
    started, record = time.monotonic(), dict[str, object](case=case.name)
    try:
        decision = provider.generate((case / "prompt.txt").read_text(), images=tuple(case / n for n in IMAGES),
                                     names=tuple(meta["offered"]), attempt=attempt, tick=lambda: None,
                                     cancelled=lambda: False)
        record |= {"skill": decision.skill, "arguments": decision.arguments, "text": decision.text}
    except ProviderError as error:
        record |= {"error": str(error)[:500]}
    record["seconds"] = round(time.monotonic() - started, 2)
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "bench.json").write_text(json.dumps(record, indent=1))
    return record


def run(args):
    settings = Agent(model=args.model, **{k: v for k, v in (("local_url", args.local_url),
                                                            ("second_url", args.second_url)) if v})
    cases = sorted(p.parent for p in Path(args.cases).glob("*/case.json"))[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"model": args.model, "cases": str(args.cases)}))
    with ThreadPoolExecutor(args.jobs) as pool:
        for n, record in enumerate(pool.map(lambda c: ask(settings, c, out), cases), 1):
            print(f"[{n}/{len(cases)}] {record['case']}: {record.get('skill') or record.get('error', '')[:80]}",
                  flush=True)


# --- score ---
def click(arguments, labels):
    """'hit target label' as the label under a point_normalized_1000, 'invalid' or 'background'."""
    uv = pixel(arguments.get("point_normalized_1000"), labels.shape)
    if uv is None:
        return "invalid"
    label = int(labels[uv[1], uv[0]])
    return label if label > 0 else "background"


def score_run(out: Path) -> dict:
    model = json.loads((out / "run.json").read_text())
    cases_dir = Path(model["cases"])
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*/bench.json"))]
    stats = {"model": model["model"], "turns": len(rows), "parsed": 0, "allowed": 0, "same_skill": 0,
             "picks_scored": 0, "pick_hits": 0, "pick_background": 0, "pick_invalid": 0, "errors": {}}
    for row in rows:
        meta = json.loads((cases_dir / row["case"] / "case.json").read_text())
        if "error" in row:
            kind = row["error"].split(":")[0][:60]
            stats["errors"][kind] = stats["errors"].get(kind, 0) + 1
            continue
        stats["parsed"] += 1
        stats["allowed"] += row["skill"] in meta["allowed"]
        stats["same_skill"] += row["skill"] == meta["reference"]["skill"]
        if meta["target"] is not None and meta["reference"]["skill"] == "pick" and row["skill"] == "pick":
            got = click(row["arguments"], np.load(cases_dir / row["case"] / "labels.npz")["labels"])
            stats["picks_scored"] += 1
            stats["pick_hits"] += got == meta["target"]
            stats["pick_background"] += got == "background"
            stats["pick_invalid"] += got == "invalid"
    stats["seconds_mean"] = round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1)
    return stats


def score(args):
    keys = ("model", "turns", "parsed", "allowed", "same_skill", "picks_scored", "pick_hits", "pick_background",
            "pick_invalid", "seconds_mean")
    print("| " + " | ".join(keys) + " | errors |\n|" + " --- |" * (len(keys) + 1))
    for out in args.outs:
        s = score_run(Path(out))
        print("| " + " | ".join(str(s[k]) for k in keys) + f" | {s['errors'] or ''} |")


# --- template ---
def template(args):
    """The text the vLLM server renders for the first case's request (its chat template, vision placeholders)."""
    settings = Agent(model=args.model, local_url=args.local_url)
    provider = make(settings)
    case = sorted(p.parent for p in Path(args.cases).glob("*/case.json"))[0]
    meta = json.loads((case / "case.json").read_text())
    if hasattr(provider, "turn"):
        provider.turn(tuple(meta["allowed"]))
    body = provider.request((case / "prompt.txt").read_text(), tuple(case / n for n in IMAGES), tuple(meta["offered"]))
    base = args.local_url.rstrip("/").removesuffix("/v1")
    headers = {"Authorization": f"Bearer {Path('~/vlm/api_key').expanduser().read_text().strip()}"
               if Path("~/vlm/api_key").expanduser().is_file() else "", "content-type": "application/json"}
    tokens = post(f"{base}/tokenize", {"model": body["model"], "messages": body["messages"]}, headers)
    text = post(f"{base}/detokenize", {"model": body["model"], "tokens": tokens["tokens"]}, headers)["prompt"]
    print(f"{case.name}: {tokens['count']} tokens\n--- head ---\n{text[:1500]}\n--- tail ---\n{text[-800:]}")


def post(url, body, headers):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("sessions", nargs="+", help="folders holding <split>/<scenario>/r<k>/session/<session>/")
    r = sub.add_parser("run")
    r.add_argument("cases")
    r.add_argument("out")
    r.add_argument("--model", required=True, help="a session/config.py MODELS name")
    r.add_argument("--local-url")
    r.add_argument("--second-url")
    r.add_argument("--limit", type=int)
    r.add_argument("--jobs", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("outs", nargs="+")
    t = sub.add_parser("template")
    t.add_argument("cases")
    t.add_argument("--model", required=True)
    t.add_argument("--local-url", required=True)
    args = parser.parse_args(argv)
    {"collect": collect, "run": run, "score": score, "template": template}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
