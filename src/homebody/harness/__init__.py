"""The harness a model writes robot programs against: the env API (env.py), its reference
(API.md), the prompt that asks for a program (prompt.md) and how an attempt's outcome is reported
back (feedback.py). The outer evolve loop improves this package; the inner loop only uses it.
"""
from pathlib import Path

from .env import Env
from .feedback import report

HERE = Path(__file__).resolve().parent


def build_env(ctx, memory, labels):
    return Env(ctx, memory, labels)


def api_doc():
    return (HERE / "API.md").read_text()


def render_prompt(task, memory, attempts, max_attempts):
    """The inner prompt: TASK text, scene MEMORY, the API reference and earlier ATTEMPTS' reports."""
    lines = []
    for item in memory:
        anchor = ", ".join(f"{v:.2f}" for v in item["anchor_xyz"])
        lines.append(f"- {item['id']}: {item['label']}; {item['location']} (anchor [{anchor}]"
                     f"{', portable' if item['portable'] else ''})")
    history = "\n\n".join(f"### Attempt {a['number']} (dry run)\n{a['report']}" for a in attempts) or "None yet."
    return (HERE / "prompt.md").read_text().format(task=task, memory="\n".join(lines), api=api_doc(), history=history,
                                                 attempt=len(attempts) + 1, max_attempts=max_attempts)


__all__ = ["Env", "api_doc", "build_env", "render_prompt", "report"]
