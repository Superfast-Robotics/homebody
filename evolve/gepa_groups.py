"""The GEPA groups (evolve/gepa_open.py): the open setup's texts optimised in three independent runs.

planner   guidance + target note: Qwen3.8's decision turns asked through the whole stack (gepa_eval ask), scored
          against Sol's labels; checked end to end (Sol planning) every few iterations
pointer   pick note + place note: the starting texts' planner turns frozen (their decision and target words), only
          RoboBrain asked again (gepa_eval point); no end-to-end check of its own
planning  plan guidance: Qwen3.8's first plan for a task (plan_bench ask), judged blind in both orders against Sol's
          plan by Claude Opus 5.5 (plan_bench judge); its score is the plan's mean rubric score over 10

A group's state names its texts; a candidate always carries all of them (the others stay the seed's)."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GROUPS = {
    "planner": {"texts": ("guidance", "target_note"), "e2e": True},
    "pointer": {"texts": ("pick_note", "place_note"), "e2e": False},
    "planning": {"texts": ("plan_guidance",), "e2e": False},
}
POINTER_HOST = "loki-gh200"  # RoboBrain on its 127.0.0.1:8001
PLAN_HOST, PLAN_CASES = "loki-gh200", "~/bench/plan-cases"
PLAN_REPO, PLAN_PY = "~/hb-bench", "~/vllm-venv/bin/python"


def plan_score(judged: dict) -> float | None:
    """The open plan's mean rubric score over 10, both orders; None when the judge gave no verdict."""
    scores = [s for v in judged.get("verdicts", []) for s in v["open"].values()]
    return round(sum(scores) / len(scores) / 10, 4) if scores else None


def plan_feedback(judged: dict) -> str:
    if not judged.get("verdicts"):
        return "the plan check refused the tree, so it would never run."
    low = sorted({c for v in judged["verdicts"] for c, s in v["open"].items() if s <= 5})
    reasons = " / ".join(v["reason"] for v in judged["verdicts"])
    return (f"judge preferred {judged['preferred']}; weak on {', '.join(low) or 'nothing'}. " + reasons)[:900]


def plan_record(case: Path, plan: dict, judged: dict | None) -> dict:
    from .plan_bench import tree_text  # noqa: PLC0415 -- plan_bench imports the session stack

    meta = json.loads((case / "case.json").read_text())
    judged = judged or {}
    return {"case": case.name, "scenario": meta["scenario"], "families": ["plan"], "task": meta["task"],
            "plan": {"open": tree_text(plan["tree"]) if plan.get("tree") else plan.get("error", ""),
                     "sol": tree_text(meta["sol"]["tree"])},
            "score": plan_score(judged) if plan.get("accepted") else 0.0, "scores": {},
            "feedback": plan_feedback(judged) if plan.get("accepted") else "the plan check refused the tree."}


def evaluate_planning(run: Path, cid: str, texts: dict, names: list[str], ssh) -> str:
    """CID's plans for NAMES (asked on the GH200 with its plan guidance), judged here; JSON lines."""
    from .plan_bench import judge_case  # noqa: PLC0415

    remote = f"~/bench/gepa/{run.name}"
    ssh(PLAN_HOST, f"mkdir -p {remote}/cands {remote}/lists && cat > {remote}/cands/{cid}.txt",
        texts.get("plan_guidance", ""))
    ssh(PLAN_HOST, f"cat > {remote}/lists/{cid}.txt", "\n".join(names))
    ssh(PLAN_HOST, f"cd {PLAN_REPO} && export VLLM_API_KEY=$(cat ~/vlm/api_key) && PYTHONPATH=src:. {PLAN_PY} -m "
                   f"evolve.plan_bench ask {PLAN_CASES} {remote}/plans/{cid} --model qwen38-27b-vllm --jobs 16 "
                   f"--plan-guidance {remote}/cands/{cid}.txt --list {remote}/lists/{cid}.txt")
    local = run / "plans" / cid
    for name in names:
        (local / name).mkdir(parents=True, exist_ok=True)
        (local / name / "plan.json").write_text(ssh(PLAN_HOST, f"cat {remote}/plans/{cid}/{name}/plan.json"))
    cases = run.parent.parent / "gepa-plan-cases"  # local copy of the plan cases (images, Sol's plan)

    def one(name):
        plan = json.loads((local / name / "plan.json").read_text())
        judged = judge_case(cases / name, local) if plan.get("accepted") else None
        return plan_record(cases / name, plan, judged)

    with ThreadPoolExecutor(4) as pool:
        return "".join(json.dumps(r) + "\n" for r in pool.map(one, names))


def evaluate_pointer(run: Path, cid: str, texts: dict, names: list[str], ssh) -> str:
    """CID's pointer clicks on NAMES (the frozen planner turns of state["frozen"] on the GH200); JSON lines."""
    remote = f"~/bench/gepa/{run.name}"
    ssh(POINTER_HOST, f"mkdir -p {remote}/cands {remote}/lists && cat > {remote}/cands/{cid}.json", json.dumps(texts))
    ssh(POINTER_HOST, f"cat > {remote}/lists/{cid}.txt", "\n".join(names))
    ssh(POINTER_HOST, f"cd ~/hb-bench && export VLLM_API_KEY=$(cat ~/vlm/api_key) && PYTHONPATH=src:. "
                      f"~/vllm-venv/bin/python -m evolve.gepa_eval point {remote}/cands/{cid}.json "
                      f"~/bench/gepa-cases-v2 {remote}/out/{cid} --list {remote}/lists/{cid}.txt "
                      f"--frozen {remote}/frozen.jsonl --jobs 16")
    return ssh(POINTER_HOST, f"cat {remote}/out/{cid}/results.jsonl")
