"""The registered skills by name. Naming a module `skills/<name>.py` in SKILLS registers it."""
import os
from collections.abc import Mapping
from importlib import import_module

from .contract import declared

# HOMEBODY_SKILLS (comma-separated module names) swaps the set for one run, e.g. "navigate,vla".
SKILLS = tuple(name.strip() for name in os.environ.get("HOMEBODY_SKILLS", "navigate,pick,place").split(",")
               if name.strip())


class Prompts(Mapping):
    """Each registered skill's PROMPT, read on access."""

    def __getitem__(self, name):
        return REGISTRY[name].PROMPT

    def __iter__(self):
        return iter(REGISTRY)

    def __len__(self):
        return len(REGISTRY)


REGISTRY = {name: declared(name, import_module(f".{name}", __package__)) for name in SKILLS}
PROMPTS = Prompts()
