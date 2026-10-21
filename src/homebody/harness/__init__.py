"""The harness the agent and its coding sub-agent work with: the agent's turn instructions
(agent_prompt.md), the sub-agent's instructions, request and reports (coder.py, coder_agents.md),
and the higher-level primitives a skill may build on (env.py, documented in API.md).

The session (frozen) calls exactly these functions; the outer evolve loop improves this package.
"""
import re
from pathlib import Path

from .change_report import change_report
from .coder import (
    CODER_HISTORY_STEPS,
    CODER_START,
    check_report,
    coder_instructions,
    coder_request,
    execute_report,
    fixes_for,
)
from .env import Env, Target

HERE = Path(__file__).resolve().parent


def api_doc():
    return (HERE / "API.md").read_text()


def coder_text(text, settings):
    """TEXT with its <coder>...</coder> parts kept (markers removed) when the session has coder calls, else removed."""
    if coder_on(settings):
        return text.replace("<coder>", "").replace("</coder>", "")
    return re.sub(r"<coder>.*?</coder>", "", text, flags=re.S)


def coder_on(settings):
    """Whether the coding assistant is part of this session: [coder] calls above 0."""
    return getattr(settings.get("coder"), "calls", 1) > 0


def plan_instructions(settings):
    """How the agent writes and works through a plan (plan mode); SETTINGS fills its {group.value} fields."""
    return coder_text((HERE / "plan_prompt.md").read_text(), settings).format_map(settings).strip()


def agent_instructions(settings):
    """The instructions that open every turn's prompt; SETTINGS fills its {group.value} fields."""
    return coder_text((HERE / "agent_prompt.md").read_text(), settings).format_map(settings).strip()


__all__ = [
    "CODER_HISTORY_STEPS",
    "CODER_START",
    "Env",
    "Target",
    "agent_instructions",
    "api_doc",
    "change_report",
    "check_report",
    "coder_instructions",
    "coder_request",
    "execute_report",
    "fixes_for",
    "plan_instructions",
]
