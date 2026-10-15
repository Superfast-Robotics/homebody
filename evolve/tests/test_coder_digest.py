"""coder_digest: what the coder changed across episodes, and which fixes recur among its successes."""
import json

from evolve.skill_loop import coder_digest

BASE = "def plan(x):\n    return x\n\n\ndef lift(h):\n    return h\n"


def call(split, episode, final, code="FAILED", held=False):
    """One episode with one coder call that turned BASE into FINAL."""
    (split / episode).mkdir(parents=True)
    (split / episode / "result.json").write_text(json.dumps({"episode": episode, "success": held}))
    folder = split / episode / "session" / "s" / "tasks" / "t" / "coder" / "call-1"
    (folder / "workspace" / "skills").mkdir(parents=True)
    (folder / "start_pick.py").write_text(BASE)
    (folder / "workspace" / "skills" / "pick.py").write_text(final)
    (folder / "result.json").write_text(json.dumps({"code": code, "message": "m"}))


def test_a_function_fixed_in_several_successful_episodes_is_listed_as_a_helper_candidate(tmp_path):
    lift_fix = BASE.replace("return h", "return h + 0.05")
    call(tmp_path, "bin_a", lift_fix, code="OK", held=True)
    call(tmp_path, "counter_b", lift_fix, code="FAILED", held=True)  # ended held although the call failed
    call(tmp_path, "shelf_c", BASE.replace("return x", "return -x"), code="OK", held=True)
    call(tmp_path, "bin_d", lift_fix, code="FAILED", held=False)
    text = coder_digest(tmp_path)
    recurring = text.split("## Recurring fixes")[1].split("##")[0]
    assert "`lift`" in recurring and "2 successful episodes" in recurring
    assert "bin_a" in recurring and "counter_b" in recurring
    assert "`plan`" not in recurring  # fixed in one successful episode only
    assert "| lift | 3 | 1 |" in text


def test_no_recurring_fixes_says_so(tmp_path):
    call(tmp_path, "bin_a", BASE.replace("return h", "return 2 * h"), code="OK", held=True)
    assert "none yet" in coder_digest(tmp_path).split("## Recurring fixes")[1]
