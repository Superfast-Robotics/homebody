"""The end-to-end loop's pure parts: session rows, missing sessions, summaries, layout clusters, judging by placed
objects, the editable screen and the prompts."""
import json
import string
import tomllib
from pathlib import Path

import pytest

from evolve import skill_score
from evolve.e2e_loop import (
    RESCORABLE,
    E2EMixin,
    host_of,
    layout_cluster,
    ledger_total,
    read_split,
    session_row,
    summarise,
    unsafe,
)
from evolve.skill_loop import SkillLoop

HERE = Path(__file__).resolve().parents[1]
PREFIXES = {"wh_shelf": "wh_shelf", "wh_picking": "wh_picking"}


def write_result(folder, stages, status="model_done", credits=2.5, wall=600.0, exit_code="0", disturbed=()):
    folder.mkdir(parents=True)
    objects = {f"o{i}": {"stage": s, "reason": None if s == 3 else "outside_target", "dropped": False}
               for i, s in enumerate(stages)}
    (folder / "result.json").write_text(json.dumps({
        "success": all(s == 3 for s in stages), "stage": sum(stages) / len(stages), "objects": objects,
        "status": status, "steps": 12, "customizations": 1, "wall_seconds": wall, "cost": {"credits": credits},
        "disturbed": list(disturbed), "invalid": []}))
    (folder / "usage.json").write_text(json.dumps({"credits": credits}))
    (folder / "exit_code").write_text(exit_code + "\n")


def test_a_session_scores_its_placed_share_and_a_missing_one_is_infrastructure(tmp_path):
    write_result(tmp_path / "train" / "wh_proc_00" / "r0", [3, 1, 0, 3])
    row = session_row("wh_proc_00", "r0", tmp_path / "train" / "wh_proc_00" / "r0", ["o0", "o1", "o2", "o3"])
    assert row["placed"] == 0.5 and abs(row["stage"] - 1.75) < 1e-9 and not row["crashed"] and row["credits"] == 2.5
    rows = read_split(tmp_path / "train", [("wh_proc_00", "r0"), ("wh_proc_00", "r1"), ("wh_shelf_02", "r0")])
    assert sorted(rows) == ["wh_proc_00__r0", "wh_proc_00__r1", "wh_shelf_02__r0"]
    assert rows["wh_proc_00__r1"]["infra"] and not rows["wh_proc_00__r1"]["crashed"]


def test_a_crashed_session_counts_as_a_crash_but_keeps_its_credits(tmp_path):
    folder = tmp_path / "t" / "r0"
    write_result(folder, [3, 0], status="error", credits=4.0)
    row = session_row("t", "r0", folder)
    assert row["crashed"] and not row["infra"] and row["credits"] == 4.0 and row["placed"] == 0.5
    (folder / "result.json").unlink()
    (folder / "exit_code").write_text("124\n")
    row = session_row("t", "r0", folder)
    assert row["crashed"] and row["status"] == "timed out" and row["credits"] == 4.0 and row["seconds"] == 7200


def test_results_that_cannot_be_judged_are_infrastructure(tmp_path):
    folder = tmp_path / "t" / "r0"
    write_result(folder, [3, 3])
    assert session_row("t", "r0", folder, ["o0", "other"])["infra"]  # scores objects the task does not name
    write_result(tmp_path / "p" / "r0", [3], status="provider_error")
    assert session_row("p", "r0", tmp_path / "p" / "r0")["infra"]
    (tmp_path / "s" / "r0").mkdir(parents=True)
    (tmp_path / "s" / "r0" / "skipped").write_text("stopped: budget")
    assert session_row("s", "r0", tmp_path / "s" / "r0")["infra"]
    (tmp_path / "u" / "r0").mkdir(parents=True)
    (tmp_path / "u" / "r0" / "exit_code").write_text("0\n")  # ran, but the scorer failed
    (tmp_path / "u" / "r0" / "usage.json").write_text(json.dumps({"credits": 2.0}))
    row = session_row("u", "r0", tmp_path / "u" / "r0")
    assert row["infra"] and row["status"] == "unscored" and row["status"] in RESCORABLE


def test_a_session_whose_usage_cannot_be_read_is_charged_the_cap_and_never_judged(tmp_path):
    for name, usage in (("missing", None), ("failed", {"credits": None, "unmetered": True, "failed": True}),
                        ("list", [1]), ("no_credits", {"main_input": 5}), ("negative", {"credits": -1.0}),
                        ("text", {"credits": "2.0"}), ("boolean", {"credits": True})):
        folder = tmp_path / name / "r0"
        write_result(folder, [3, 3])
        if usage is None:
            (folder / "usage.json").unlink()
        else:
            (folder / "usage.json").write_text(json.dumps(usage))
        row = session_row(name, "r0", folder, ["o0", "o1"], cap=8.0)
        assert row["infra"] and row["status"] == "unmetered" and row["credits"] == 8.0 and row["placed"] == 0.0
    folder = tmp_path / "nan" / "r0"  # json writes NaN as a bare literal, which Python reads back as nan
    write_result(folder, [3, 3])
    (folder / "usage.json").write_text('{"credits": NaN}')
    assert session_row("nan", "r0", folder, ["o0", "o1"], cap=8.0)["status"] == "unmetered"


def test_a_session_with_unmetered_parts_is_judged_and_charged_its_worst_case(tmp_path):
    folder = tmp_path / "t" / "r0"
    write_result(folder, [3, 0])
    (folder / "usage.json").write_text(json.dumps({"credits": 1.2, "unmetered": True, "main_unmetered": 2}))
    row = session_row("t", "r0", folder, ["o0", "o1"], cap=102.0)
    assert not row["infra"] and row["placed"] == 0.5 and row["credits"] == 102.0 and not row["metered"]


def test_a_result_that_is_not_a_record_or_has_no_status_never_gets_credit(tmp_path):
    folder = tmp_path / "l" / "r0"
    write_result(folder, [3])
    (folder / "result.json").write_text("[1, 2]")
    row = session_row("l", "r0", folder, ["o0"])
    assert row["infra"] and row["status"] == "invalid result" and row["placed"] == 0.0
    for name, exit_code in (("ok", "0"), ("failed", "1")):
        folder = tmp_path / name / "r0"
        write_result(folder, [3], exit_code=exit_code)
        record = json.loads((folder / "result.json").read_text())
        del record["status"]
        (folder / "result.json").write_text(json.dumps(record))
        row = session_row(name, "r0", folder, ["o0"])
        assert row["placed"] == 0.0 and (row["infra"] if exit_code == "0" else row["crashed"] and not row["infra"])


def test_the_ledger_counts_the_last_amount_of_every_key():
    text = ("session:a@1 8.0\nsession:b@2 8.0\nastra 30.5\nsession:a@1 2.75\n"
            "session:a@3 8.0\nastra 41.0\nbroken line\n")  # a@3: a re-run attempt, still in flight
    assert ledger_total(text) == 2.75 + 8.0 + 8.0 + 41.0


def test_disturbing_other_products_costs_a_share_of_an_object(tmp_path):
    write_result(tmp_path / "t" / "r0", [3, 3, 0, 0], disturbed=["can_2", "jar_5"])
    assert session_row("t", "r0", tmp_path / "t" / "r0", penalty=0.25)["placed"] == (2 - 0.5) / 4
    write_result(tmp_path / "z" / "r0", [0, 0], disturbed=["a", "b", "c"])
    assert session_row("z", "r0", tmp_path / "z" / "r0", penalty=0.25)["placed"] == 0.0


def test_every_version_runs_a_task_repeat_on_the_same_host_and_repeats_split_hosts():
    hosts = [host_of(task, repeat, 2) for task in range(16) for repeat in range(2)]
    assert hosts.count(0) == hosts.count(1) == 16
    assert all(host_of(t, 0, 2) != host_of(t, 1, 2) for t in range(16))


def test_the_safety_screen_flags_system_access_and_settings_writes():
    found = unsafe(["import subprocess", "    open(path, 'w').write(x)", "settings.grasp.squeeze = 2",
                    "    x = settings.grasp.squeeze == 2  # compare", "    pose = env.observe()"])
    assert set(found) == {"imports a system module", "opens a file for writing", "writes the settings"}
    assert not unsafe(["    reply = env.plan_path(pres, goals)  # import os is only a comment"])


def test_variants_of_one_scanned_layout_are_one_cluster():
    assert layout_cluster("train/wh_shelf_03__r1", PREFIXES) == "wh_shelf"
    assert layout_cluster("wh_picking_original__r0", PREFIXES) == "wh_picking"
    assert layout_cluster("val/wh_proc_07__r1", PREFIXES) == "wh_proc_07"


def test_summary_has_what_the_judge_and_the_search_read(tmp_path):
    write_result(tmp_path / "wh_proc_00" / "r0", [3, 3])
    write_result(tmp_path / "wh_shelf_01" / "r0", [0, 3])
    write_result(tmp_path / "wh_shelf_01" / "r1", [0, 0], status="error")
    rows = read_split(tmp_path, [("wh_proc_00", "r0"), ("wh_shelf_01", "r0"), ("wh_shelf_01", "r1")])
    s = summarise(rows, lambda e: layout_cluster(e, PREFIXES))
    assert s["episodes"] == 3 and s["errors"] == 1 and s["placed"] == round(1.5 / 3, 3) and not s["infra"]
    assert s["placed_values"]["wh_proc_00__r0"] == 1.0 and s["held_values"]["wh_proc_00__r0"] == 1.0
    assert s["placed_by_layout"] == {"wh_proc_00": "2/2", "wh_shelf": "1/4"}
    assert s["untimed"] == ["wh_shelf_01__r1"] and set(s["by_task"]) == {"wh_proc_00", "wh_shelf_01"}


def test_placed_objective_counts_shared_layout_variants_as_one_sample():
    def side(placed):
        return {"placed_values": placed, "values": dict(placed), "held_values": {}, "seconds": {}, "credit_values": {},
                "episodes": len(placed), "errors": 0}
    episodes = [f"wh_shelf_{i:02d}__r0" for i in range(4)] + [f"wh_proc_{i:02d}__r0" for i in range(2)]
    parent = {"train": side(dict.fromkeys(episodes, 0.0)), "val": side({"wh_proc_09__r0": 0.0, "wh_proc_10__r0": 0.0})}
    child = {"train": side({**dict.fromkeys(episodes, 0.0), **{e: 1.0 for e in episodes[:4]}}),
             "val": side({"wh_proc_09__r0": 0.0, "wh_proc_10__r0": 0.0})}
    by_task = skill_score.judge(parent, child, "placed_values", speed=False)
    by_layout = skill_score.judge(parent, child, "placed_values", speed=False,
                                  clusters=lambda e: layout_cluster(e, PREFIXES))
    assert by_task["metric"] == "placed"
    assert by_layout["train"][1] > by_task["train"][1]  # four shelf variants are one sample, not four


def test_lost_sessions_count_as_failures_for_the_placed_objective():
    parent = {"placed_values": {"a": 1.0, "b": 1.0}}
    child = skill_score.lost_as_failures(parent, {"placed_values": {"a": 1.0}})
    assert child["placed_values"] == {"a": 1.0, "b": 0.0} and child["lost"] == ["b"]


class Screen(E2EMixin, SkillLoop):
    def __init__(self, editable):
        self.editable, self.skill = tuple(editable), "x"


def test_editable_screen_also_refuses_frozen_files():
    screen = Screen(["src/homebody/skills/dex1/*", "src/homebody/harness/*", "tests/*"])
    assert screen.allowed(["src/homebody/harness/env.py", "src/homebody/skills/dex1/pick.py"]) == []
    assert screen.allowed(["tests/test_session.py", "src/homebody/session/coder.py"]) == \
        ["src/homebody/session/coder.py", "tests/test_session.py"]


def test_the_profile_and_its_prompts_fit_together():
    profile = tomllib.loads((HERE / "config.toml").read_text())["e2e"]["demo-wh"]
    assert set(profile["suite"]) == {"train", "val", "test", "legacy"} and profile["objective"] == "placed"
    assert profile["parallel"] <= 16 and profile["val_margin"] > 0 and profile["service_checks"]
    operators = {"workspace", "skill", "source", "diagnosis", "mechanism", "parent", "parent_results", "other",
                 "other_results", "other_diff", "ruff", "python", "editable"}
    supplied = {"brief": {"objective", "editable", "repeats"}, "diagnose": {"workspace", "skill", "source"},
                "lessons": {"skill", "history", "lessons"}}
    for name in ("brief", "diagnose", "lessons", "propose", "diff", "crossover", "rewrite"):
        text = (HERE / "prompts" / f"{profile['prompts']}_{name}.md").read_text()
        used = {field for _, field, _, _ in string.Formatter().parse(text) if field}
        assert used <= supplied.get(name, operators), (name, used - supplied.get(name, operators))


def test_a_preregistered_val_margin_needs_val_shown_non_inferior():
    def side(values):
        return {"placed_values": values, "values": dict(values), "held_values": {}, "seconds": {},
                "credit_values": {}, "episodes": len(values), "errors": 0}
    train = [f"wh_proc_{i:02d}__r0" for i in range(12)]
    val = [f"wh_proc_v{i:02d}__r0" for i in range(6)]
    parent = {"train": side(dict.fromkeys(train, 0.0)), "val": side(dict.fromkeys(val, 0.5))}
    child = {"train": side(dict.fromkeys(train, 0.6)),
             "val": side({**dict.fromkeys(val, 0.5), val[0]: 0.0, val[1]: 0.25})}
    assert skill_score.judge(parent, child, "placed_values", speed=False)["ok"]  # train carries the pooled gain
    held = skill_score.judge(parent, child, "placed_values", speed=False, val_margin=0.05, val_z=1.0)
    assert not held["ok"] and held["basis"].startswith("val not shown non-inferior")


def round_loop(folder, archive, ran):
    from types import SimpleNamespace
    loop = SimpleNamespace(work=folder, log=lambda m: None, archive=lambda: archive,
                           archived_rounds=lambda: E2EMixin.archived_rounds(loop),
                           barren=lambda entries: E2EMixin.barren(loop, entries))
    loop.round_state = lambda: SkillLoop.round_state(loop)
    loop.save_round_state = lambda state: SkillLoop.save_round_state(loop, state)
    loop.completed_rounds = lambda: SkillLoop.completed_rounds(loop)

    def round_(n):
        assert json.loads((folder / "rounds.json").read_text())["started"] == n  # recorded before any proposal
        ran.append(n)
        archive.append({"id": f"e{n:03d}c1", "round": n, "status": "rejected"})
    loop.round = round_
    return loop


def test_an_unfinished_round_resumes_and_a_barren_first_round_stays_stopped(tmp_path):
    ran, archive = [], [{"id": "b000", "round": 0, "status": "baseline"}]
    loop = round_loop(tmp_path, archive, ran)
    SkillLoop.run(loop, 3)
    state = json.loads((tmp_path / "rounds.json").read_text())
    assert ran == [1] and state["finished"] == [1] and state["stop"]["round"] == 1  # written together
    SkillLoop.run(loop, 3)
    assert ran == [1]  # the stop decision persists
    (tmp_path / "STOP.json").unlink()
    (tmp_path / "rounds.json").write_text(json.dumps({"finished": [1], "started": 2, "stop": None}))
    archive.append({"id": "e002c1", "round": 2, "status": "rejected"})  # round 2 started, never finished
    SkillLoop.run(loop, 1)
    assert ran == [1, 2]  # round 2 is resumed, not skipped


def test_an_archived_rejection_never_marks_an_end_to_end_round_finished(tmp_path):
    ran = []
    archive = [{"id": "b000", "round": 0, "status": "baseline"},
               {"id": "e001c1", "round": 1, "status": "rejected"}]  # round 1 died before rounds.json was written
    SkillLoop.run(round_loop(tmp_path, archive, ran), 1)
    assert ran == [1]


def test_a_codex_call_killed_at_its_time_limit_still_records_its_charge(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from evolve import astra

    def killed(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 5, output=b'{"type": "turn.started"}\n', stderr=b"")
    monkeypatch.setattr(astra, "settings", lambda: {"model": "gpt-6-astra", "effort": "medium", "timeout": 5})
    monkeypatch.setattr(astra.subprocess, "run", killed)
    with pytest.raises(RuntimeError, match="timed out"):
        astra.codex("prompt", tmp_path, "read-only", tmp_path / "note.md")
    usage = json.loads((tmp_path / "note.usage.json").read_text())
    assert usage["credits"] == astra.TIMED_OUT_CREDITS and not usage["metered"]
    assert astra.credits_in(tmp_path) == astra.TIMED_OUT_CREDITS


def test_malformed_scoring_records_are_infrastructure_never_exceptions_or_credit(tmp_path):
    malformed = {"objects_number": {"objects": 7}, "stage_text": {"objects": {"o0": {"stage": "3"}}},
                 "stage_bool": {"objects": {"o0": {"stage": True}}}, "wall_text": {"wall_seconds": "bad"},
                 "wall_nan": {"wall_seconds": float("nan")}, "disturbed_number": {"disturbed": 1},
                 "disturbed_items": {"disturbed": [1]}, "invalid_none": {"invalid": None},
                 "success_text": {"success": "yes"}, "stage_high": {"stage": 9}, "steps_negative": {"steps": -1},
                 "status_unknown": {"status": "all_good"}, "status_number": {"status": 3}}
    for name, change in malformed.items():
        folder = tmp_path / name / "r0"
        write_result(folder, [3])
        record = {**json.loads((folder / "result.json").read_text()), **change}
        (folder / "result.json").write_text(json.dumps(record))
        row = session_row(name, "r0", folder, ["o0"])
        assert row["infra"] and row["status"] == "invalid result" and row["placed"] == 0.0, name
        assert row["status"] in RESCORABLE  # scored again, then run again
    folder = tmp_path / "unreadable" / "r0"
    write_result(folder, [3])
    (folder / "exit_code").write_bytes(b"\xff\xfe")  # nothing a session leaves can raise out of the row
    assert session_row("unreadable", "r0", folder, ["o0"])["placed"] == 0.0


def test_finished_statuses_are_judged_and_every_other_known_one_is_a_crash_or_infrastructure(tmp_path):
    from evolve.e2e_loop import CRASHED, FINISHED, INFRA_STATUS
    for status in FINISHED + CRASHED + INFRA_STATUS:
        folder = tmp_path / status / "r0"
        write_result(folder, [3], status=status)
        row = session_row(status, "r0", folder, ["o0"])
        assert (not row["infra"] and not row["crashed"]) if status in FINISHED else (
            row["crashed"] if status in CRASHED else row["infra"])


def test_an_old_quota_failure_kept_as_an_attempt_is_not_the_sessions_failure(tmp_path):
    from evolve import score
    write_result(tmp_path / "train" / "t" / "r0", [3])
    old = tmp_path / "train" / "t" / "r0.attempt-1"
    old.mkdir(parents=True)
    (old / "run.log").write_text("error: You've hit your usage limit\n")
    assert not score.quota_hit(tmp_path)
    (tmp_path / "train" / "t" / "r0" / "run.log").write_text("usage limit reached\n")
    assert score.quota_hit(tmp_path)


def test_raising_a_credit_cap_changes_no_identity():
    from evolve.skill_loop import SPENDING, measured
    profile = tomllib.loads((HERE / "config.toml").read_text())["e2e"]["demo-wh"]
    raised = {**profile, **{key: 99999.0 for key in SPENDING}}
    assert measured(raised) == measured(profile)
    assert measured({**profile, "repeats": 3}) != measured(profile)  # what is measured still counts


def worst_case_loop(cfg):
    from types import SimpleNamespace
    loop = SimpleNamespace(cfg=cfg)
    loop.session_worst_case = lambda: E2EMixin.session_worst_case(loop)
    loop.cap = E2EMixin.cap.fget(loop)
    return loop


def test_a_session_reserves_its_worst_case_from_its_settings():
    profile = tomllib.loads((HERE / "config.toml").read_text())["e2e"]["demo-wh"]
    settings = tomllib.loads((HERE / profile["settings"]).read_text())
    loop = worst_case_loop(profile)
    decisions = settings["agent"]["max_steps"] + 1 + settings["plan"]["replans"]
    expected = decisions * profile["decision_credit_ceiling"] + settings["coder"]["calls"] * profile["coder_credit_ceiling"]
    assert E2EMixin.session_worst_case(loop) == expected == 102.0
    assert E2EMixin.cap.fget(loop) == expected  # no fixed session_credit_cap: the worst case
    # a baseline's own allowance, not the round's cap: the profile's evaluation_credit_budget, else its sessions' worst case
    assert E2EMixin.budget(loop, 48) == float(profile.get("evaluation_credit_budget") or 48 * expected)
    loop.round_dir = Path("e001")
    assert E2EMixin.budget(loop, 48) == profile["round_credit_budget"]


def run_ledger(tmp_path, budget, reserve, keys, ledger_text=""):
    """run_jobs.sh's reserve() for KEYS in turn (each settled at once unless it is the last), returning the codes."""
    import subprocess
    ledger = tmp_path / "ledger.txt"
    ledger.write_text(ledger_text)
    script = (HERE / "run_jobs.sh").read_text()
    functions = script[script.index("LEDGER_PY="):script.index("jailed()")]
    settle = script[script.index("settle()"):script.index("\n", script.index("settle()"))]
    body = "\n".join(f'reserve {k}; echo "rc=$?"' for k in keys)
    program = (f"{functions}\n{settle}\nLEDGER={ledger}; STOP={tmp_path}/STOP; RESERVE={reserve}; BUDGET={budget}\n"
               f"sleep() {{ echo waited; exit 0; }}\n{body}")
    result = subprocess.run(["bash", "-c", program], capture_output=True, text=True, timeout=60, check=False)
    return result.stdout, ledger.read_text()


def test_a_session_reservation_waits_for_sessions_in_flight_and_is_refused_when_none_are(tmp_path):
    out, ledger = run_ledger(tmp_path, budget=250, reserve=102, keys=["a", "b"])
    assert out.count("rc=0") == 2 and ledger_total(ledger) == 204  # both fit
    out, ledger = run_ledger(tmp_path, budget=250, reserve=102, keys=["c"], ledger_text="reserve:a 102\nreserve:b 102\n")
    assert "waited" in out and "rc=" not in out  # two in flight: it waits (sleep stubbed to end the test)
    out, ledger = run_ledger(tmp_path, budget=250, reserve=102, keys=["c"], ledger_text="a 160\n")
    assert "rc=1" in out and "exceeded" in (tmp_path / "STOP").read_text()  # nothing in flight: refused
    assert ledger_total("reserve:a 102\nreserve:a 0\na 140.5\n") == 140.5  # a settlement that passes its reservation


def test_every_astra_call_of_a_round_reserves_and_settles_in_the_round_ledger(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from evolve import astra
    from evolve.e2e_loop import BudgetExceeded
    remote = tmp_path / "remote"

    def ssh(command, timeout=None, check=True):  # the remote's commands, run here on a local folder
        return subprocess.run(["bash", "-c", command.replace("/r/", f"{remote}/")], capture_output=True, text=True,
                              check=check).stdout
    from types import SimpleNamespace
    loop = SimpleNamespace(cfg={"round_credit_budget": 300, "astra_call_credit_ceiling": 120.0, "workdir": "w"},
                           remote=SimpleNamespace(work="/r", ssh=ssh), skill="s",
                           round_dir=tmp_path / "e001", jail=lambda: None)
    for name in ("ledger", "budget", "reserve", "settle_charge", "astra_call"):
        setattr(loop, name, getattr(E2EMixin, name).__get__(loop))

    def fake_codex(prompt, cd, sandbox, out, jail=None, **model):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        (Path(out).parent / (Path(out).stem + ".usage.json")).write_text(json.dumps({"credits": 70.0}))
        if "fail" in prompt:
            raise RuntimeError("codex exited 1")
        Path(out).write_text("note")
        return "note"
    monkeypatch.setattr(astra, "codex", fake_codex)
    loop.astra_call("diagnose", tmp_path, "read-only", tmp_path / "e001" / "diagnosis.md")
    with pytest.raises(RuntimeError):
        loop.astra_call("fail", tmp_path, "read-only", tmp_path / "e001" / "c1_note.md")
    ledger = remote / "e2e" / "w" / "ledger-e001.txt"
    assert ledger_total(ledger.read_text()) == 140.0  # both settled at their recorded credits, the failed one too
    loop.astra_call("lessons", tmp_path, "read-only", tmp_path / "e001" / "lessons.md")  # 140 + 120 fits under 300
    with pytest.raises(BudgetExceeded):  # 210 + 120 would not: the call is never started
        loop.astra_call("more", tmp_path, "read-only", tmp_path / "e001" / "more.md")
    assert ledger_total(ledger.read_text()) == 210.0 and not (tmp_path / "e001" / "more.md").exists()


def test_a_failed_codex_call_that_reported_no_turn_is_charged(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from evolve import astra
    monkeypatch.setattr(astra, "settings", lambda: {"model": "gpt-6-astra", "effort": "medium", "timeout": 5})
    monkeypatch.setattr(astra.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"auth error"))
    with pytest.raises(RuntimeError, match="exited 1"):
        astra.codex("prompt", tmp_path, "read-only", tmp_path / "note.md")
    assert astra.call_credits(tmp_path / "note.md", 0) == astra.TIMED_OUT_CREDITS


def test_a_decision_without_usage_marks_the_session_unmetered(tmp_path):
    import runpy
    episode = runpy.run_path(str(HERE / "score_episode.py"), run_name="score_episode")
    task = tmp_path / "task-000"
    for i, body in enumerate(('{"usage": {"input_tokens": 1000, "output_tokens": 100}}', '{"output": []}')):
        (task / "decisions" / f"{i:03d}").mkdir(parents=True)
        (task / "decisions" / f"{i:03d}" / "reply.json").write_text(body)
    cost = episode["cost"](task)
    assert cost["main_unmetered"] == 1 and cost["unmetered"] and cost["credits"] > 0


def test_candidates_are_judged_against_a_freshly_measured_setup():
    import inspect

    from evolve.evo_loop import EvoLoop
    source = inspect.getsource(EvoLoop.judge_candidates)
    assert "refresh=index == 0" in source


def test_coder_usage_with_more_cached_than_input_tokens_is_unmetered(tmp_path):
    from evolve import score_episode
    call = tmp_path / "task" / "coder" / "call-1"
    call.mkdir(parents=True)
    usage = {"input_tokens": 100, "cached_input_tokens": 200, "output_tokens": 100}
    (call / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": usage}) + "\n")
    record = score_episode.cost(tmp_path / "task")
    assert record["coder_unmetered"] == 1 and record["unmetered"]


def test_an_astra_call_with_incomplete_usage_is_unmetered_and_charged_its_reservation(tmp_path):
    from evolve import astra
    path = tmp_path / "call.usage.json"
    stdout = (json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000}}) + "\n").encode()
    astra.record_usage(stdout, "gpt-6-astra", path, unreported=50.0)
    record = json.loads(path.read_text())
    assert record["metered"] is False and record["credits"] >= 50.0
    assert astra.call_credits(tmp_path / "call.md", 120.0) == 120.0


def test_unjudged_sessions_are_retried_in_the_same_call_but_quota_and_other_stops_are_not():
    from types import SimpleNamespace

    from evolve import e2e_loop
    calls = []

    def once(failures, error=None):
        def run(items, splits, videos):
            calls.append(1)
            if len(calls) <= failures:
                raise error or e2e_loop.InfrastructureFailure(f"c1: {e2e_loop.UNJUDGED}: {{'train': ['t/r1']}}")
            return {"c1": "summaries"}
        return run

    loop = SimpleNamespace(log=lambda message: None)
    loop.evaluate_once = once(e2e_loop.UNJUDGED_RETRIES)
    assert e2e_loop.E2EMixin.evaluate_many(loop, []) == {"c1": "summaries"}
    assert len(calls) == e2e_loop.UNJUDGED_RETRIES + 1
    calls.clear()
    loop.evaluate_once = once(e2e_loop.UNJUDGED_RETRIES + 1)
    with pytest.raises(e2e_loop.InfrastructureFailure):
        e2e_loop.E2EMixin.evaluate_many(loop, [])
    calls.clear()
    loop.evaluate_once = once(1, e2e_loop.InfrastructureFailure("planner servers not ready"))
    with pytest.raises(e2e_loop.InfrastructureFailure):
        e2e_loop.E2EMixin.evaluate_many(loop, [])
    assert len(calls) == 1
