"""The fix library: entries for a skill's refusal, most often validated first."""
import json

from homebody.harness import coder


def test_fixes_match_skill_and_refusal_and_rank_by_validations(tmp_path, monkeypatch):
    monkeypatch.setattr(coder, "FIXES", tmp_path)
    for name, skill, code, validated in (("a", "pick", "NO_GRASP", 1), ("b", "pick", "NO_GRASP", 3),
                                         ("c", "pick", "COLLISION", 5), ("d", "place", "NO_GRASP", 9)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"skill": skill, "code": code, "validated": validated,
                                                           "when": f"w{name}", "repair": f"r{name}", "sketch": "+x"}))
    shown = coder.fixes_for("pick", "NO_GRASP")
    assert len(shown) == 2 and shown[0].startswith("NO_GRASP when wb: rb") and "    +x" in shown[0]
    assert coder.fixes_for("navigate", "STALL") == []
