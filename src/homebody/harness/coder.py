"""What the coding sub-agent is told and shown: its standing instructions (coder_agents.md, given as
the workspace's AGENTS.md), each call's REQUEST.md, and how check and execute report back."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODER_HISTORY_STEPS = 6  # the agent's recent steps copied into REQUEST.md
CODER_START = "Carry out REQUEST.md, following AGENTS.md."


def coder_instructions():
    return (HERE / "coder_agents.md").read_text()


def coder_request(*, target, origin, request, task, recent, source_root, python, limits, library):
    """REQUEST.md for one call: the agent's request and the context it was made in."""
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
