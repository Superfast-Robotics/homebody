"""The harness the agent writes customized skills against: the agent instructions
(agent_prompt.md), the primitives (env.py) and their reference (API.md), how the task's customized
skills are shown each turn, and how a customized skill's run is reported back (feedback.py).

The session (frozen) calls exactly these functions; the outer evolve loop improves this package.
"""
from pathlib import Path

from .env import Env, Target
from .feedback import result as custom_result

HERE = Path(__file__).resolve().parent


def build_env(ctx, frame):
    return Env(ctx, frame)


def api_doc():
    return (HERE / "API.md").read_text()


def agent_instructions(settings):
    """The instructions that open every turn's prompt; SETTINGS fills its {group.value} fields."""
    instructions = (HERE / "agent_prompt.md").read_text().format_map(settings).strip()
    return instructions + "\n\n" + api_doc().strip()


def custom_section(customs):
    """This task's customized skills as a prompt section, or "" when there are none.
    CUSTOMS maps a name to {description, code, version}."""
    if not customs:
        return ""
    parts = ["Customized skills this task (current versions):"]
    for name, item in customs.items():
        parts.append(f"- {name} (version {item['version']}): {item['description']}\n```python\n{item['code'].strip()}\n```")
    return "\n".join(parts)


def source_section(name, source, customized):
    """The source read_skill asked for, shown for one turn."""
    kind = "your customized version" if customized else "the built-in skill (reference only: edits apply to customized skills)"
    return f"Source of {name}, {kind}:\n```python\n{source.rstrip()}\n```"


__all__ = ["Env", "Target", "agent_instructions", "api_doc", "build_env", "custom_result", "custom_section",
           "source_section"]
