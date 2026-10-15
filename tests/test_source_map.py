"""The coder's SOURCE_MAP.md: an outline of the skill it edits and of the source it imports, with line numbers."""
import inspect
from pathlib import Path

from homebody.harness.source_map import source_map
from homebody.skills.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1] / "src"

SKILL = '''"""A probe skill."""
from homebody.helpers.geometry import wrap
from .contract import Result, SkillFailure

FAILURES = ("TOO_FAR", "BLOCKED")


def prepare(arguments, selected, current, planning):
    """Plan the probe without moving."""
    if arguments["r"] > 1:
        raise SkillFailure("TOO_FAR", "out of reach")
    return wrap(arguments["r"])


class Mover:
    """Moves the probe."""

    def step(self, ctx, speed=0.1):
        return Result("probe", "BLOCKED", "a wall")
'''


def test_the_skill_outline_lists_definitions_with_lines_and_docstrings():
    text = source_map("probe", SKILL, ROOT)
    assert "L8 `prepare(arguments, selected, current, planning)`: Plan the probe without moving." in text
    assert "L15 class `Mover`: Moves the probe." in text
    assert "L18 `Mover.step(self, ctx, speed=0.1)`" in text


def test_each_failure_code_points_at_the_lines_that_return_it():
    text = source_map("probe", SKILL, ROOT)
    assert "`TOO_FAR`: L11" in text
    assert "`BLOCKED`: L19" in text


def test_imported_names_resolve_to_their_file_line_and_signature():
    text = source_map("probe", SKILL, ROOT)
    assert "helpers/geometry.py" in text
    assert "`wrap(" in text
    assert "skills/contract.py" in text and "`SkillFailure`" in text


def test_the_built_in_pick_map_covers_its_imports_and_stays_short():
    module = REGISTRY["pick"]
    text = source_map("pick", inspect.getsource(module), ROOT)
    for path in ("motion/arm.py", "motion/carry.py", "helpers/grasp.py", "skills/navigate.py"):
        assert path in text
    assert "`move_path(" in text
    assert len(text.splitlines()) < 400


def test_a_skill_that_does_not_parse_still_gets_a_note():
    assert "could not be parsed" in source_map("probe", "def broken(:\n", ROOT)
