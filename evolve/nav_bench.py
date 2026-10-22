"""Navigation bench: which model should choose where the open setup walks (2026-10-22). On the turns where the
reference (Sol) navigated, each model is asked twice: as the turn allowed it (does it choose navigate at all?) and
with navigate the only decision (how good are its points?). The navigate skill snaps the robot square to the mapped
edge on the goal side of the facing point (skills/navigate.py), so the points are scored by what decides the
stance: the facing point within 0.5 m of the reference's (the same spot to work on), the goal on the same side of it
as the reference's goal, and on free floor of the map; a stance is usable when all three hold.

    python3 -m evolve.nav_bench ask CASES OUT --list NAMES.txt --model M --mode turn|forced [--local-url U] [--jobs N]
    python3 -m evolve.nav_bench score CASES OUT...       one row per run

CASES are gepa_eval cases (case.json with reference, observation, map.png); M a session/config.py MODELS name, given
the seed's open-model guidance (agent.local_notes reach,claims,recover) as the open setup has it."""
import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.session.config import Agent
from homebody.vlm.catalog import make
from homebody.vlm.provider import ProviderError

from .agent_bench import IMAGES
from .agent_capabilities import map_click

NOTES = "reach,claims,recover"
NEAR_M, SIDE_DEG, FREE_LEVEL = 0.5, 60.0, 150


def ask(case: Path, out: Path, settings: Agent, mode: str) -> dict:
    done = out / case.name / "nav.json"
    if done.is_file():
        return json.loads(done.read_text())
    meta = json.loads((case / "case.json").read_text())
    allowed = ("navigate",) if mode == "forced" else tuple(meta["allowed"])
    names = tuple(meta["offered"] or meta["allowed"])
    names = names if "navigate" in names else (*names, "navigate")
    attempt = out / case.name / "request"
    shutil.rmtree(attempt, ignore_errors=True)
    provider, started = make(settings), time.monotonic()
    if hasattr(provider, "turn"):
        provider.turn(allowed)
    try:
        decision = provider.generate((case / "prompt.txt").read_text(), images=tuple(case / n for n in IMAGES),
                                     names=names, attempt=attempt, tick=lambda: None, cancelled=lambda: False)
        row = {"skill": decision.skill, "arguments": decision.arguments, "text": decision.text}
    except ProviderError as error:
        row = {"error": str(error)[:300]}
    row = {"case": case.name, "mode": mode, **row, "seconds": round(time.monotonic() - started, 1)}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(row, indent=1))
    return row


def free_at(click, map_png: Path) -> bool:
    """The map pixel under a 0-1000 click is light (free floor), not dark (blocked)."""
    with Image.open(map_png) as image:
        grey = np.asarray(image.convert("L"))
    h, w = grey.shape
    u, v = int(click[0] * (w - 1) / 1000), int(click[1] * (h - 1) / 1000)
    return 0 <= u < w and 0 <= v < h and int(grey[v, u]) >= FREE_LEVEL


def metres(a, b, scale) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) / scale


def side_angle(goal, theirs, facing) -> float:
    """Degrees between the goal's and the reference goal's directions from the facing point."""
    a = math.atan2(goal[1] - facing[1], goal[0] - facing[0])
    b = math.atan2(theirs[1] - facing[1], theirs[0] - facing[0])
    return abs((math.degrees(a - b) + 180) % 360 - 180)


def score_case(case: Path, row: dict) -> dict:
    meta = json.loads((case / "case.json").read_text())
    observation, reference = meta["observation"], meta["reference"]["arguments"]
    scale = abs(float(np.asarray(observation["map_click_from_world"])[0][0]))
    theirs = {p: map_click(reference, p, observation) for p in ("goal", "facing")}
    result = {"navigated": row.get("skill") == "navigate"}
    if not result["navigated"] or theirs["goal"] is None or theirs["facing"] is None:
        return result
    mine = {p: map_click(row["arguments"], p, observation) for p in ("goal", "facing")}
    if mine["goal"] is None or mine["facing"] is None:
        return result | {"parsed": False}
    facing_m = metres(mine["facing"], theirs["facing"], scale)
    angle = side_angle(mine["goal"], theirs["goal"], theirs["facing"])
    free = free_at(mine["goal"], case / "map.png")
    return result | {"parsed": True, "facing_m": round(facing_m, 2), "side_deg": round(angle, 1), "goal_free": free,
                     "usable": facing_m <= NEAR_M and angle <= SIDE_DEG and free}


def run_ask(args):
    settings = Agent(model=args.model, local_url=args.local_url, local_notes=NOTES)
    cases, out = Path(args.cases), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"model": args.model, "mode": args.mode, "cases": str(cases)}))
    names = [n.strip() for n in Path(args.list).read_text().splitlines() if n.strip()]
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda n: ask(cases / n, out, settings, args.mode), names))
    print(f"{args.model} {args.mode}: {sum('error' not in r for r in rows)} of {len(rows)} answered")


def share(values) -> str:
    values = list(values)
    return f"{100 * sum(values) / len(values):.0f}%" if values else "-"


def run_score(args):
    print("| model | mode | cases | navigated | parsed | facing <0.5 m | same side | goal free | usable | "
          "median facing m |\n|" + " --- |" * 10)
    for out in map(Path, args.outs):
        run = json.loads((out / "run.json").read_text())
        rows = [json.loads(p.read_text()) for p in sorted(out.glob("*/nav.json"))]
        scored = [score_case(Path(run["cases"]) / r["case"], r) for r in rows]
        pointed = [s for s in scored if s.get("parsed")]
        facing = sorted(s["facing_m"] for s in pointed)
        print(f"| {run['model']} | {run['mode']} | {len(scored)} | {share(s['navigated'] for s in scored)} | "
              f"{share(s.get('parsed', False) for s in scored if s['navigated'])} | "
              f"{share(s['facing_m'] <= NEAR_M for s in pointed)} | {share(s['side_deg'] <= SIDE_DEG for s in pointed)} | "
              f"{share(s['goal_free'] for s in pointed)} | {share(s.get('usable', False) for s in scored)} | "
              f"{facing[len(facing) // 2] if facing else '-'} |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask")
    a.add_argument("cases")
    a.add_argument("out")
    a.add_argument("--list", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--mode", choices=("turn", "forced"), required=True)
    a.add_argument("--local-url", default="http://127.0.0.1:8000/v1")
    a.add_argument("--jobs", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("outs", nargs="+")
    args = parser.parse_args(argv)
    {"ask": run_ask, "score": run_score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
