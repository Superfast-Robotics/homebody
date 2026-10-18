"""What the coding sub-agent is told and shown: its standing instructions (coder_agents.md, given as
the workspace's AGENTS.md), each call's REQUEST.md, and how check and execute report back."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODER_HISTORY_STEPS = 6  # the agent's recent steps copied into REQUEST.md
CODER_START = "Carry out REQUEST.md, following AGENTS.md."
FIXES = HERE / "fixes"  # validated coder fixes from train runs (evolve/harvest_fixes.py); one JSON file each
FIXES_SHOWN = 3


def fixes_for(skill, code, limit=FIXES_SHOWN):
    """The library fixes for SKILL's refusal CODE, most often validated first, as text for a request."""
    entries = [json.loads(p.read_text()) for p in sorted(FIXES.glob("*.json"))] if FIXES.is_dir() else []
    matching = [e for e in entries if e.get("skill") == skill and e.get("code") == code]
    matching.sort(key=lambda e: -int(e.get("validated", 1)))
    return [f"{e['code']} when {e.get('when', '')}: {e.get('repair', '')}\n  sketch:\n" +
            "\n".join("    " + line for line in str(e.get("sketch", "")).splitlines()[:40]) for e in matching[:limit]]


def coder_instructions():
    return (HERE / "coder_agents.md").read_text()


def coder_request(*, target, origin, request, task, recent, source_root, python, limits, library, trace=None,
                  fixes=()):
    """REQUEST.md for one call: the agent's request and the context it was made in, the failed call's trace
    and the library fixes for its refusal."""
    lines = [
        f"# Request: skill `{target}`",
        f"`skills/{target}.py` starts as {origin}.",
        "",
        f"Goal: {request.get('goal', '')}",
        f"Success looks like: {request.get('success', '')}",
        f"Tried so far: {request.get('tried', '') or 'nothing reported'}",
        f"Constraints: {request.get('constraints', '') or 'none'}",
        "",
        f"The operator's task: {task}",
        "",
        "The agent's recent steps, oldest first (decision, result, and what_changed: what the step did to the robot):",
        *(f"- {line}" for line in recent or ["none yet"]),
        "",
        (f"Limits for this call: {limits['seconds']:g} s of wall time including motion, "
         f"{limits['checks']} checks, {limits['executes']} executes."),
        *(["", "Trace of the failed call (keyframes in trace/failed/; the last ones are attached):", trace, ""]
          if trace else []),
        *(["Fixes that worked before for this refusal (check that they apply before reusing):",
           *(f"- {fix}" for fix in fixes), ""] if fixes else []),
        f"Library files (read-only): {', '.join(library) or 'none yet'}.",
        f"Source tree (read-only reference): {source_root}/homebody",
        f"Python: {python}",
    ]
    return "\n".join(lines) + "\n"


def check_report(report):
    """What check returns to the coder (a JSON-able dict)."""
    return json.loads(json.dumps(report, default=str))


def execute_report(report):
    """What execute returns to the coder (a JSON-able dict)."""
    return json.loads(json.dumps(report, default=str))
