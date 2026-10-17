"""The skills exploration runs, by name. Naming a module `skills/<name>.py` in LIVE_SKILLS registers it.

A skill that declares PARAMETERS, the JSON schema of its arguments, is offered to the model; the
others are motions the explorer runs itself. A model-callable skill takes four places: its module,
its name here, its branch in vlm_pick.parse_decision and its branch in decision_interpretation."""
from importlib import import_module

from homebody.skills.contract import declared

LIVE_SKILLS = ("pick_point", "look_at", "update_region", "ask_operator", "request_help", "explored",
               "walk_route", "survey", "departure_recovery")
REGISTRY = {name: declared(name, import_module(f"homebody.skills.{name}")) for name in LIVE_SKILLS}
ACTIONS = tuple(name for name, skill in REGISTRY.items() if hasattr(skill, "PARAMETERS"))
