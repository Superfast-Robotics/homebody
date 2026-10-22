"""The GEPA data from the open setup's own rollouts (astra-evolve/e2e/gepa_open_models_design.md): its decision turns
as bench cases, the hard ones chosen, and Sol's decision at each as the reference (evolve/gepa_eval.py scores a
candidate against it), so the prompts are optimised on the states the open models reach, not on Sol's.

    python3 -m evolve.gepa_data collect OUT ROOT...      every turn of ROOT's train and val sessions as a case
    python3 -m evolve.gepa_data select CASES LIST --train N --val N [--seed S]   the hard turns to label
    python3 -m evolve.gepa_data label CASES LIST OUT [--jobs N]   Sol's decision at each listed turn as the reference

ROOT is an e2e run's folder on the volume (<profile>-<id>/b000, sessions under <split>/<scenario>/r<k>/session/).
A turn of a multi-model agent keeps its prompt in the planner's sub-attempt (01-base/01-planner/prompt.txt, with
the pointer's target note after it, which is dropped) and the enum of the turn's allowed decisions in its
request.json. A case keeps the open model's own decision and result as "own" (and, as agent_bench's cases, the
turn before as "previous"); `label` replaces its "reference" with Sol's decision (result OK; the clicked label as
the target of a pick or place)."""
import argparse
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from homebody.session.config import Agent
from homebody.session.planned import GOAL_DONE

from .agent_bench import ACTIVE_GOAL, IMAGES, OBSERVATION, allowed, ask_turn, find_enum, turns
from .agent_capabilities import label_at

NOTE_START = "\n\nWhen your skill takes point_normalized_1000, also end your text with `TARGET:"
SOL = "sol61-litellm"
HARD = ("pick", "place", GOAL_DONE, "done")


# --- collect ---
def planner_folder(step: Path) -> Path | None:
    """The sub-attempt that holds the turn's prompt: the step itself, or its planner's (first attempt)."""
    for folder in (step, *sorted(step.glob("*-base/*-planner")), *sorted(step.glob("*-planner")),
                   *sorted(step.glob("*-base"))):
        if (folder / "prompt.txt").is_file() and (folder / "request.json").is_file():
            return folder
    return None


def skill_enum(value):
    """The skill enum of a saved request: the "skill" property's enum (a turn's allowed decisions, which need not
    include done), else agent_bench's search."""
    if isinstance(value, dict):
        skill = value.get("skill")
        if isinstance(skill, dict) and isinstance(skill.get("enum"), list):
            return skill["enum"]
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            if (found := skill_enum(item)) is not None:
                return found
    return None


def session_prompt(text: str) -> str:
    """The session's prompt without the pointer composite's target note."""
    return text.split(NOTE_START)[0]


def collect_session(session: Path, scenario: str, split: str, out: Path) -> int:
    count, previous = 0, None
    for step, decision, result in turns(session):
        here = {"decision": decision and decision["decision"], "code": result["result"].get("code"),
                "message": result["result"].get("message", "")[:300]}
        folder = planner_folder(session / "tasks" / "task-000" / "decisions" / f"step-{step:03d}")
        frame = decision and session / "frames" / f"{decision['epoch']:03d}-{decision['sequence']:08d}"
        if folder is None or frame is None or not all((frame / name).is_file() for name in IMAGES):
            previous = here
            continue
        prompt = session_prompt((folder / "prompt.txt").read_text())
        request = json.loads((folder / "request.json").read_text())
        offered = tuple(skill_enum(request) or find_enum(request) or ())
        sensors = np.load(frame / "sensors.npz")
        case = out / f"{scenario}-{session.name[-8:]}-{step:03d}"
        case.mkdir(parents=True, exist_ok=True)
        for name in (*IMAGES, "rgb_masked.png"):
            if (frame / name).is_file():
                shutil.copy(frame / name, case / name)
        np.savez_compressed(case / "labels.npz", labels=sensors["labels"])
        if {"depth", "intrinsics", "world_camera", "world_torso"} <= set(sensors.files):
            np.savez_compressed(case / "geometry.npz", depth=sensors["depth"].astype(np.float16),
                                intrinsics=sensors["intrinsics"], world_camera=sensors["world_camera"],
                                world_torso=sensors["world_torso"])
        (case / "prompt.txt").write_text(prompt)
        goal, observation = ACTIVE_GOAL.search(prompt), OBSERVATION.search(prompt)
        (case / "case.json").write_text(json.dumps({
            "scenario": scenario, "split": split, "step": step, "offered": offered,
            "allowed": allowed(prompt, offered), "goal": goal[1].strip() if goal else None,
            "observation": json.loads(observation[1]) if observation else None, "previous": previous,
            "own": {"decision": decision["decision"], "code": here["code"], "message": here["message"]},
            "reference": None, "result": None, "message": "", "target": None}, indent=1))
        previous, count = here, count + 1
    return count


def collect(args):
    out, total = Path(args.out), 0
    for root in map(Path, args.roots):
        for events in sorted(root.glob("*/*/r*/session/*/events.jsonl")):
            session = events.parent
            split, scenario = session.parents[3].name, session.parents[2].name
            if split in ("train", "val"):
                total += collect_session(session, scenario, split, out)
    print(f"{total} cases in {out}")


# --- select ---
def hard(meta) -> bool:
    """A turn after a failed or refused result, or one that clicks or claims."""
    previous = meta.get("previous") or {}
    return previous.get("code") not in (None, "OK") or meta["own"]["decision"]["skill"] in HARD


def select(args):
    """Per split, N turns: the hard ones first, spread over the sessions (a few per session at a time), then others."""
    rng = random.Random(args.seed)
    metas = {p.parent.name: json.loads(p.read_text()) for p in sorted(Path(args.cases).glob("*/case.json"))}
    chosen = []
    for split, wanted in (("train", args.train), ("val", args.val)):
        names = [n for n, m in metas.items() if m["split"] == split and "plan" not in m["allowed"]]
        rng.shuffle(names)
        ordered = sorted(names, key=lambda n: not hard(metas[n]))  # stable: the hard ones first, shuffled
        sessions: dict[str, int] = {}
        picked = []
        for cap in (2, 4, 8, 10**6):  # round-robin over sessions so no single session dominates
            for name in ordered:
                key = name.rsplit("-", 1)[0]
                if name not in picked and sessions.get(key, 0) < cap and len(picked) < wanted:
                    picked.append(name)
                    sessions[key] = sessions.get(key, 0) + 1
        chosen += picked
        print(f"{split}: {len(picked)} of {len(names)} turns ({sum(hard(metas[n]) for n in picked)} hard)")
    Path(args.list).write_text("\n".join(chosen) + "\n")


# --- label ---
def labelled(case: Path, row: dict, out: Path) -> Path | None:
    """CASE copied to OUT with Sol's decision as its reference, or None when Sol's reply did not parse."""
    if "error" in row:
        return None
    target = out / case.name
    if not target.exists():
        shutil.copytree(case, target, ignore=shutil.ignore_patterns("request", "bench.json"))
    meta = json.loads((case / "case.json").read_text())
    reference = {k: row[k] for k in ("text", "skill", "arguments")}
    clicked = None
    if reference["skill"] in ("pick", "place"):
        clicked = label_at(reference["arguments"].get("point_normalized_1000"),
                           np.load(case / "labels.npz")["labels"]) or None
    (target / "case.json").write_text(json.dumps(meta | {"reference": reference, "result": "OK", "target": clicked,
                                                         "labeller": SOL}, indent=1))
    return target


def label(args):
    cases, out = Path(args.cases), Path(args.out)
    asked = out.parent / f"{out.name}-sol"  # Sol's replies (agent_bench's records), resumable
    asked.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in Path(args.list).read_text().splitlines() if n.strip()]
    settings = Agent(model=SOL, reasoning_effort="medium")

    def one(name):
        done = asked / name / "bench.json"
        row = json.loads(done.read_text()) if done.is_file() else ask_turn(settings, cases / name, asked)
        return labelled(cases / name, row, out)

    with ThreadPoolExecutor(args.jobs) as pool:
        made = list(pool.map(one, names))
    print(f"{sum(m is not None for m in made)} of {len(names)} turns labelled in {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("roots", nargs="+")
    s = sub.add_parser("select")
    s.add_argument("cases")
    s.add_argument("list")
    s.add_argument("--train", type=int, required=True)
    s.add_argument("--val", type=int, required=True)
    s.add_argument("--seed", type=int, default=7)
    lab = sub.add_parser("label")
    lab.add_argument("cases")
    lab.add_argument("list")
    lab.add_argument("out")
    lab.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    {"collect": collect, "select": select, "label": label}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
