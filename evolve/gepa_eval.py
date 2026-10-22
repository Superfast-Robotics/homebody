"""The capability side of the GEPA prompt optimisation (evolve/gepa_open.py): one candidate's open-model texts asked
on bench cases (evolve/agent_bench.py) through the open setup's own stack, the Qwen3.8 planner with the RoboBrain
pointer (c-q38-robobrain) and the session's open-model help (the pointer's view masked beyond reach, a place click
beyond reach moved within it), each case scored in [0, 1] with feedback a reflection model can act on.

    python3 -m evolve.gepa_eval ask CANDIDATE.json CASES OUT --list NAMES.txt [--jobs N] [--local-url U] [--second-url U]
    python3 -m evolve.gepa_eval index CASES          one JSON line per case: its name, scenario and capabilities
    python3 -m evolve.gepa_eval episodes WORKDIR     an end-to-end baseline's train sessions as reflection material

CANDIDATE.json holds the texts ({"guidance", "target_note", "pick_note", "place_note"}); NAMES.txt the case
folders to ask (one per line). Each case's record goes to OUT/<case>/gepa.json (a rerun keeps finished ones) and
every asked case's record to OUT/results.jsonl. A case is scored on each capability it tests, as
agent_capabilities.py measures them, and its score is their mean:

choose    acting inside a goal: the skill the reference chose and that worked
judge     claiming goal_done exactly when the reference's claim passed the plan's check
recover   after a failed or refused result, not repeating that decision unchanged
pick      the click on the object the reference picked
place     the click (after the move within reach) on the reference's support and within reach of the torso
navigate  the goal and facing points within 0.5 m of the reference's (half a point each)
A reply that does not parse scores 0 on every capability of its case."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from homebody.session.config import Agent
from homebody.session.planned import GOAL_DONE
from homebody.vlm.composite import POINTER_PROMPTS, Endpoint, section, to_1000
from homebody.vlm.provider import ProviderError

from .agent_bench import ask_turn
from .agent_capabilities import NEAR_M, judged, label_at, map_click, metres
from homebody.helpers.reach import reach_of, world_of

from .agent_place import SCORE_REACH_M, clamped, geometry, masked

AGENT = "c-q38-robobrain"
ACTING = ("navigate", "pick", "place")
TEXTS = ("guidance", "target_note", "pick_note", "place_note")


def settings_for(candidate: dict, local_url: str, second_url: str) -> Agent:
    return Agent(model=AGENT, local_url=local_url, second_url=second_url, local_notes="",
                 local_guidance=candidate.get("guidance", ""), local_target_note=candidate.get("target_note", ""),
                 local_pick_note=candidate.get("pick_note", ""), local_place_note=candidate.get("place_note", ""))


def families(meta) -> tuple[str, ...]:
    """The capabilities a case tests (none for a plan turn: the open setup's first plans are Sol's)."""
    reference, found = meta["reference"], []
    if reference["skill"] == "plan":
        return ()
    if meta["goal"] is not None and reference["skill"] in ACTING and meta["result"] == "OK":
        found.append("choose")
    if judged(meta) is not None:
        found.append("judge")
    previous = meta.get("previous") or {}
    if previous.get("decision") and previous.get("code") not in (None, "OK"):
        found.append("recover")
    if meta["target"] is not None and reference["skill"] == "pick":
        found.append("pick")
    if meta["target"] is not None and reference["skill"] == "place" and "point_normalized_1000" in reference[
            "arguments"]:
        found.append("place")
    if reference["skill"] == "navigate" and meta["result"] == "OK" and meta["observation"]:
        found.append("navigate")
    return tuple(found)


def pointer_call(case_out: Path) -> dict:
    """The composite's pointer call of the turn (its target words and answer), {} when it asked none."""
    routing = case_out / "request" / "routing.json"
    if not routing.is_file():
        return {}
    calls = json.loads(routing.read_text())["calls"]
    return next((c for c in calls if c["role"] == "pointer"), {})


def described(label, target) -> str:
    if label is None:
        return "no valid point"
    if label == 0:
        return "the background (no object)"
    return "the target object" if label == target else "another object"


def score_choose(meta, row):
    want = meta["reference"]["skill"]
    if row["skill"] == want:
        return 1.0, None
    return 0.0, f"chose {row['skill']} while the goal still needed {want} (the reference's {want} worked)."


def score_judge(meta, row):
    truth, claimed = judged(meta), row["skill"] == GOAL_DONE
    if claimed == truth:
        return 1.0, None
    if claimed:
        return 0.0, (f"claimed goal_done although the goal was not achieved yet (the reference went on with "
                     f"{meta['reference']['skill']}).")
    return 0.0, f"did not claim goal_done although the goal was achieved (the last result: {last_result(meta)})."


def score_recover(meta, row):
    previous = meta["previous"]["decision"]
    if row["skill"] == previous["skill"] and row["arguments"] == previous["arguments"]:
        return 0.0, f"repeated the decision that just failed, unchanged ({last_result(meta)})."
    return 1.0, None


def score_pick(meta, row, case):
    if row["skill"] != "pick":
        return 0.0, f"chose {row['skill']} where the step was a pick of the target object."
    labels = np.load(case / "labels.npz")["labels"]
    got = label_at(row["arguments"].get("point_normalized_1000"), labels)
    if got == meta["target"]:
        return 1.0, None
    return 0.0, f"the pick click landed on {described(got, meta['target'])}, not the target object."


def score_place(meta, row, case):
    if row["skill"] != "place":
        return 0.0, f"chose {row['skill']} where the step was a placement."
    point = row["arguments"].get("point_normalized_1000")
    if not isinstance(point, list) or len(point) != 2:
        return 0.0, "the place decision had no point_normalized_1000 click."
    point = clamped(point, case)
    labels = np.load(case / "labels.npz")["labels"]
    depth, intrinsics, world_camera, world_torso = geometry(case)
    on = label_at(point, labels) == meta["target"]
    world = world_of(point, depth, intrinsics, world_camera)
    reach = reach_of(world, world_torso) if world is not None else None
    near = reach is not None and reach <= SCORE_REACH_M
    if on and near:
        return 1.0, None
    faults = [] if on else ["the click was not on the destination surface the reference placed on"]
    if not near:
        faults.append(f"the spot was {reach:.2f} m from the torso, beyond the arm's {SCORE_REACH_M:.2f} m reach"
                      if reach is not None else "the spot had no depth")
    return 0.5 * (on + near), "; ".join(faults).capitalize() + "."


def score_navigate(meta, row):
    if row["skill"] != "navigate":
        return 0.0, f"chose {row['skill']} where the step was a navigation."
    parts, faults = [], []
    for prefix in ("goal", "facing"):
        mine = map_click(row["arguments"], prefix, meta["observation"])
        theirs = map_click(meta["reference"]["arguments"], prefix, meta["observation"])
        if theirs is None:
            continue
        distance = metres(mine, theirs, meta["observation"]) if mine else float("inf")
        parts.append(distance <= NEAR_M)
        if distance > NEAR_M:
            faults.append(f"the {prefix} point was {distance:.2f} m from the reference's working one"
                          if mine else f"the {prefix} point was missing")
    if not parts:
        return None
    return sum(parts) / len(parts), ("; ".join(faults).capitalize() + ".") if faults else None


def last_result(meta) -> str:
    previous = meta.get("previous") or {}
    if not previous:
        return "none yet"
    skill = (previous.get("decision") or {}).get("skill")
    return f"{skill} -> {previous.get('code')}: {(previous.get('message') or '')[:160]}"


def score_case(case: Path, row: dict, out: Path) -> dict:
    meta = json.loads((case / "case.json").read_text())
    wanted = families(meta)
    record = {"case": case.name, "scenario": meta["scenario"], "families": list(wanted),
              "task": section((case / "prompt.txt").read_text(), "Current task:"), "goal": meta["goal"],
              "previous": last_result(meta), "allowed": meta["allowed"],
              "reference": {k: meta["reference"][k] for k in ("skill", "text", "arguments")},
              "decision": {k: row.get(k) for k in ("skill", "text", "arguments", "error") if k in row}}
    pointer = pointer_call(out / case.name)
    if pointer:
        record["pointer"] = {k: pointer.get(k) for k in ("target", "answer", "error", "used") if k in pointer}
    scores, feedback = {}, []
    for family in wanted:
        if "error" in row:
            scores[family], text = 0.0, f"the reply did not parse ({row['error'][:200]})."
        else:
            scorer = {"choose": score_choose, "judge": score_judge, "recover": score_recover,
                      "navigate": score_navigate}.get(family)
            got = scorer(meta, row) if scorer else {"pick": score_pick, "place": score_place}[family](meta, row, case)
            if got is None:
                continue
            scores[family], text = got
        if text:
            feedback.append(f"{family}: {text}")
    record |= {"scores": scores, "score": round(sum(scores.values()) / len(scores), 4) if scores else None,
               "feedback": " ".join(feedback) or "correct on every capability this turn tests."}
    return record


def evaluate(case: Path, out: Path, settings: Agent) -> dict:
    done = out / case.name / "gepa.json"
    if done.is_file():
        return json.loads(done.read_text())
    if (case / "geometry.npz").is_file() and not (case / "rgb_masked.png").is_file():
        masked(case, case)  # agent.local_reach_mask: the pointer takes rgb_masked.png beside rgb.png
    (out / case.name / "bench.json").unlink(missing_ok=True)
    try:
        row = ask_turn(settings, case, out)
    except FileExistsError:  # a request folder left by an interrupted ask of the same case: ask it afresh
        row = ask_turn(settings, case, out)
    record = score_case(case, row, out) | {"seconds": row["seconds"]}
    done.write_text(json.dumps(record, indent=1))
    return record


# --- the pointer group: RoboBrain asked again on frozen planner turns ---
POINTER = ("robobrain2.5-8b", "robobrain")


def point_case(case: Path, frozen: dict, notes: dict, endpoint, out: Path) -> dict:
    """FROZEN (a seed record: the planner's pick or place and its target words) with the pointer asked again under
    NOTES; scored on that skill only."""
    done = out / case.name / "gepa.json"
    if done.is_file():
        return json.loads(done.read_text())
    decision, target = frozen["decision"], frozen["pointer"]["target"]
    skill = decision["skill"]
    note = notes.get(f"{skill}_note", "").strip()
    question = POINTER_PROMPTS[endpoint.format].format(target=target) + (f" {note}" if note else "")
    image = case / "rgb_masked.png" if (case / "rgb_masked.png").is_file() else case / "rgb.png"
    try:
        answer = endpoint.ask(question, (image,), attempt=out / case.name / "request", tick=lambda: None,
                              cancelled=lambda: False)
        point = to_1000(answer, endpoint.format, case / "rgb.png")
        row = {"skill": skill, "text": decision.get("text"),
               "arguments": {**decision["arguments"], "point_normalized_1000": point}}
    except ProviderError as error:
        answer, row = str(error), {"error": str(error)[:300]}
    meta = json.loads((case / "case.json").read_text())
    if "error" in row:
        score, text = 0.0, f"the pointer's answer gave no click ({row['error'][:160]})."
    else:
        score, text = {"pick": score_pick, "place": score_place}[skill](meta, row, case)
    record = score_case(case, row, out) | {"families": [skill], "scores": {skill: score}, "score": score,
                                           "feedback": f"{skill}: {text or 'correct.'}",
                                           "pointer": {"target": target, "answer": answer[:300]}}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(record, indent=1))
    return record


def point(args):
    notes = json.loads(Path(args.candidate).read_text())
    frozen = {r["case"]: r for r in (json.loads(line) for line in Path(args.frozen).read_text().splitlines()
                                     if line.strip())}
    endpoint = Endpoint(*POINTER, args.second_url, 120.0)
    cases, out = Path(args.cases), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in Path(args.list).read_text().splitlines() if n.strip()]
    with ThreadPoolExecutor(args.jobs) as pool:
        records = list(pool.map(lambda n: point_case(cases / n, frozen[n], notes, endpoint, out), names))
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"{len(records)} pointer cases, mean {sum(r['score'] or 0 for r in records) / max(len(records), 1):.3f}")


def index(cases: Path):
    for case in sorted(p.parent for p in cases.glob("*/case.json")):
        meta = json.loads((case / "case.json").read_text())
        print(json.dumps({"case": case.name, "scenario": meta["scenario"], "families": families(meta),
                          "split": meta.get("split"), "source": meta.get("labeller", "recorded")}))


def episodes(workdir: Path):
    """For each train session of WORKDIR's baseline (results/b000): its scenario, objects placed, the steps that
    failed or were refused (decision text, skill, code, message) and the final done summary, one JSON line each."""
    archive = [json.loads(line) for line in (workdir / "archive.jsonl").read_text().splitlines() if line.strip()]
    placed = archive[-1]["train"]["placed_by_layout"]
    for events in sorted((workdir / "results" / "b000" / "train").glob("*/r*/session/*/events.jsonl")):
        scenario = events.parents[3].name
        rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
        decisions = {e["step"]: e["decision"] for e in rows if e["kind"] == "decision"}
        steps, summary = [], ""
        for step, result in ((e["step"], e["result"]) for e in rows if e["kind"] == "result"):
            decision = decisions.get(step, {})
            if decision.get("skill") == "done":
                summary = str(decision.get("arguments", {}).get("summary", decision.get("text", "")))[:300]
            steps.append(f"{result.get('skill')} -> {result.get('code')}"
                         + (f": {result.get('message', '')[:140]} (decision: {decision.get('text', '')[:160]})"
                            if result.get("code") != "OK" else ""))
        print(json.dumps({"scenario": scenario, "placed": placed.get(scenario, "?"), "steps": steps,
                          "done_summary": summary}))


def ask(args):
    settings = settings_for(json.loads(Path(args.candidate).read_text()), args.local_url, args.second_url)
    cases, out = Path(args.cases), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in Path(args.list).read_text().splitlines() if n.strip()]
    with ThreadPoolExecutor(args.jobs) as pool:
        records = list(pool.map(lambda name: evaluate(cases / name, out, settings), names))
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    scored = [r["score"] for r in records if r["score"] is not None]
    print(f"{len(records)} cases, mean {sum(scored) / max(len(scored), 1):.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask")
    a.add_argument("candidate")
    a.add_argument("cases")
    a.add_argument("out")
    a.add_argument("--list", required=True, help="the case folder names to ask, one per line")
    a.add_argument("--jobs", type=int, default=16)
    a.add_argument("--local-url", default="http://127.0.0.1:8000/v1")
    a.add_argument("--second-url", default="http://127.0.0.1:8001/v1")
    pt = sub.add_parser("point")
    pt.add_argument("candidate")
    pt.add_argument("cases")
    pt.add_argument("out")
    pt.add_argument("--list", required=True)
    pt.add_argument("--frozen", required=True, help="the seed's records (JSON lines): planner decisions and targets")
    pt.add_argument("--jobs", type=int, default=16)
    pt.add_argument("--second-url", default="http://127.0.0.1:8001/v1")
    sub.add_parser("index").add_argument("cases")
    sub.add_parser("episodes").add_argument("workdir")
    args = parser.parse_args(argv)
    if args.command in ("ask", "point"):
        {"ask": ask, "point": point}[args.command](args)
    else:
        {"index": index, "episodes": episodes}[args.command](Path(args.cases if args.command == "index"
                                                                  else args.workdir))


if __name__ == "__main__":
    sys.exit(main())
