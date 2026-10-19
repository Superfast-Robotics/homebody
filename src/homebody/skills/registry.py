"""The registered skills by name. Naming a module `skills/<name>.py` in SKILLS registers it."""
import os
from collections.abc import Mapping
from importlib import import_module

from .contract import declared

# HOMEBODY_SKILLS (comma-separated module names) swaps the set for one run, e.g. "navigate,vla".
SKILLS = tuple(name.strip() for name in os.environ.get("HOMEBODY_SKILLS", "navigate,pick,place").split(",")
               if name.strip())


# A hand's own versions of skills (skills/<hands>/<name>.py), used in place of the built-in modules
HAND_SKILLS = {"dex3": (), "dex1": ("pick", "place")}


class Prompts(Mapping):
    """Each registered skill's PROMPT, read on access."""

    def __init__(self, registry):
        self.registry = registry

    def __getitem__(self, name):
        return self.registry[name].PROMPT

    def __iter__(self):
        return iter(self.registry)

    def __len__(self):
        return len(self.registry)


def _load(hands):
    return {name: declared(name, import_module(f".{hands}.{name}" if name in HAND_SKILLS[hands] else f".{name}",
                                               __package__)) for name in SKILLS}


REGISTRY = _load("dex3")
PROMPTS = Prompts(REGISTRY)
_REGISTRIES = {"dex3": REGISTRY}


def registry(hands="dex3"):
    """The registered skills for the robot's HANDS (Robot.hands); REGISTRY itself for Dex3."""
    if hands not in HAND_SKILLS:
        raise ValueError(f"No skills for hands {hands!r}")
    if hands not in _REGISTRIES:
        _REGISTRIES[hands] = _load(hands)
    return _REGISTRIES[hands]
