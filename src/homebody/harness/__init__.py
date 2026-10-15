"""The harness the agent and its coding sub-agent work with: the agent's turn instructions
(agent_prompt.md), the sub-agent's instructions, request, source map and reports (coder.py, coder_agents.md,
source_map.py),
and the higher-level primitives a skill may build on (env.py, documented in API.md).

The session (frozen) calls exactly these functions; the outer evolve loop improves this package.
"""
from pathlib import Path

from .coder import (
    CODER_HISTORY_STEPS,
    CODER_START,
    check_report,
    coder_instructions,
    coder_request,
    execute_report,
)
from .env import Env, Target
from .source_map import source_map

HERE = Path(__file__).resolve().parent


def api_doc():
    return (HERE / "API.md").read_text()


def agent_instructions(settings):
    """The instructions that open every turn's prompt; SETTINGS fills its {group.value} fields."""
    return (HERE / "agent_prompt.md").read_text().format_map(settings).strip()


__all__ = ["CODER_HISTORY_STEPS", "CODER_START", "Env", "Target", "agent_instructions", "api_doc", "check_report",
           "coder_instructions", "coder_request", "execute_report", "source_map"]
