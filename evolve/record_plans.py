"""Record the first plans a run's sessions accepted as a plan file the "plans" composites read
(configs/plans/*.json): {"source", "plans": [{"scenario", "task", "text", "tree"}]}, one per scenario (its first
repeat with an accepted plan), leaving out tasks a file of SKIP already records.

    python3 -m evolve.record_plans RESULTS OUT.json --source TEXT [--skip configs/plans/a.json ...]

RESULTS is an e2e baseline's results folder (results/b000, its train/val/test/<scenario>/r<k>/session/<id>/), e.g.
an SPR run whose composite had Sol write the first plan of every task no file recorded (c-solfirst-q38-robobrain):
recorded once, every later run of those tasks starts from the same plan. Standard library only."""
import argparse
import json
import re
import sys
from pathlib import Path

TASK = re.compile(r"^Current task:\s*\n(.+)$", re.MULTILINE)


def first_plan(session: Path) -> dict | None:
    """The session's first plan decision the plan check accepted, with its task, or None."""
    rows = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines() if line.strip()]
    decision = next((e for e in rows if e["kind"] == "decision" and e["decision"].get("skill") == "plan"), None)
    if decision is None:
        return None
    result = next((e for e in rows if e["kind"] == "result" and e["step"] == decision["step"]), None)
    step = session / "tasks" / "task-000" / "decisions" / f"step-{decision['step']:03d}"
    prompts = sorted(step.rglob("prompt.txt"), key=lambda p: len(p.parts))  # a composite's: in its sub-attempt
    task = TASK.search(prompts[0].read_text()) if prompts else None
    if result is None or result["result"].get("code") != "OK" or task is None:
        return None
    return {"task": task[1].strip(), "text": decision["decision"]["text"], "tree": decision["decision"]["arguments"]["tree"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("results")
    parser.add_argument("out")
    parser.add_argument("--source", required=True)
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args(argv)
    known = {entry["task"] for name in args.skip for entry in json.loads(Path(name).read_text())["plans"]}
    plans, missing = {}, []
    for scenario in sorted({p.name for p in Path(args.results).glob("*/*") if p.is_dir()}):
        for session in sorted(Path(args.results).glob(f"*/{scenario}/r*/session/*")):
            if (plan := first_plan(session)) is not None:
                if plan["task"] not in known:
                    plans[scenario] = {"scenario": scenario, **plan}
                break
        else:
            missing.append(scenario)
    Path(args.out).write_text(json.dumps({"source": args.source, "plans": list(plans.values())}, indent=1) + "\n")
    print(f"{len(plans)} plans recorded; no accepted plan: {', '.join(missing) or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
