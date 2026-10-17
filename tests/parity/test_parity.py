"""Synthetic controller and exploration scenarios: the behaviour the code must keep.

Each runner patches module clocks, so it runs in a fresh interpreter. A fresh report must equal
expected/<scenario>.json under compare.py's rules: timings, wall clocks and the prompt-text hash
are ignored;
offered action names, chosen actions, arguments, result codes, poses, routes and view counts
must match exactly. A difference is a behaviour change to fix in the code, not a fixture to
update. Regenerate the expected reports only deliberately, with

    PYTHONPATH=src:vendor python tests/parity/run_controller.py --source-root . \
        --output tests/parity/expected/controller.json
    PYTHONPATH=src:vendor python tests/parity/run_explore.py --source-root . \
        --workdir "$(mktemp -d)" --output tests/parity/expected/explore.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import compare

TICK_P95_MS = 5.
DECISION_P95_MS = 3000.


def report(tmp_path_factory, script, *workdir):
    output = tmp_path_factory.mktemp("parity")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HOMEBODY_")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "vendor")])
    completed = subprocess.run(
        [sys.executable, str(HERE / script), "--source-root", str(ROOT),
         "--output", str(output / "report.json"),
         *(["--workdir", str(output / "work")] if workdir else [])],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((output / "report.json").read_text())


@pytest.fixture(scope="module")
def controller(tmp_path_factory):
    return report(tmp_path_factory, "run_controller.py")["controller"]


@pytest.fixture(scope="module")
def explore(tmp_path_factory):
    return report(tmp_path_factory, "run_explore.py", "work")["explore"]


def test_clear_route_arrives_without_refusal(controller):
    clear = controller["clear"]
    assert clear["terminal"]["status"] == "returned" and clear["arrived"]
    assert clear["refusal_codes"] == []
    assert clear["goal_error_m"] <= .40
    assert max(abs(p[1]) for p in clear["waypoints"]) > .3  # the route bends around the box
    assert clear["timings"]["tick"]["p95_ms"] < TICK_P95_MS


def test_blocked_goal_is_refused_without_motion(controller):
    blocked = controller["blocked"]
    assert blocked["terminal"]["code"] == "NO_COLLISION_FREE_PATH"
    assert blocked["commands"]["moving"] == 0
    assert blocked["final_pose"] == [0., 0., 0.]


def test_scripted_decisions_complete_the_exploration(explore):
    kinds = [d["kind"] for d in explore["decisions"]]
    assert kinds == ["update_region", "point", "look", "point", "explored"]
    assert [d["outcome"] for d in explore["decisions"]] == [
        "updated", "arrived", "arrived", "arrived", "accepted"]
    assert explore["phase"] == "done" and explore["scope_ready"]
    assert explore["region"]["revision"] == 1
    assert [c["call"] for c in explore["calls"][:7]] == ["arm"] + ["turn"] * 6
    assert explore["calls"][-1]["call"] == "hold"
    point = explore["decisions"][3]
    assert abs(point["pose_after"][0] - .2) < .05 and abs(point["pose_after"][1] - 1.2) < .05
    assert explore["timings"]["decision"]["p95_ms"] < DECISION_P95_MS


def expected(name):
    return json.loads((HERE / "expected" / f"{name}.json").read_text())


def differences(name, fresh):
    return [{"path": path, "expected": a, "fresh": b} for path, a, b in compare.diffs(
        compare.trimmed(expected(name)), compare.trimmed({name: fresh}))]


def test_controller_matches_the_expected_report(controller):
    assert differences("controller", controller) == []


def test_exploration_matches_the_expected_report(explore):
    assert differences("explore", explore) == []
