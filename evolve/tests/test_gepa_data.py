import json

from evolve import gepa_data


def test_a_composite_turn_keeps_its_prompt_in_the_planners_sub_attempt(tmp_path):
    planner = tmp_path / "step-001" / "01-base" / "01-planner"
    planner.mkdir(parents=True)
    (planner / "prompt.txt").write_text("prompt" + gepa_data.NOTE_START + " <spot>`.")
    (planner / "request.json").write_text("{}")
    assert gepa_data.planner_folder(tmp_path / "step-001") == planner
    assert gepa_data.session_prompt((planner / "prompt.txt").read_text()) == "prompt"
    assert gepa_data.planner_folder(tmp_path / "step-002") is None


def test_selection_takes_hard_turns_first_spread_over_sessions(tmp_path):
    def case(name, split, skill, previous_code=None):
        (tmp_path / name).mkdir()
        meta = {"split": split, "allowed": ["pick", "done"], "own": {"decision": {"skill": skill}},
                "previous": {"code": previous_code} if previous_code else None}
        (tmp_path / name / "case.json").write_text(json.dumps(meta))

    for i in range(6):
        case(f"s-aaaaaaaa-{i:03d}", "train", "pick" if i < 3 else "navigate")
    case("s-bbbbbbbb-000", "train", "navigate", previous_code="NO_ROUTE")
    case("s-cccccccc-000", "val", "navigate")
    gepa_data.select(type("A", (), {"cases": str(tmp_path), "list": str(tmp_path / "l.txt"), "train": 3, "val": 1,
                                    "seed": 0})())
    chosen = (tmp_path / "l.txt").read_text().split()
    assert "s-bbbbbbbb-000" in chosen and "s-cccccccc-000" in chosen and len(chosen) == 4
    assert sum(n.startswith("s-aaaaaaaa") for n in chosen) == 2  # two per session in the first pass
