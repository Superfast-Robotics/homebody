"""The monitor app with fake harnesses and simulation servers over a real dashboard: the Servers rows, a scenario
run step by step (and each step's refusal), and frames from the simulation server's stream reaching the dashboard's
views and the monitor."""
import types

import numpy as np
from fakes import fake_dashboard
from test_monitor_config import CONFIG

from homebody.server.monitor_app import MonitorApp, MonitorControls
from homebody.server.watch import Status
from homebody.ui.servers import server_rows

NAME = "Warehouse: one jam jar into the cart"


class Watcher:
    def __init__(self, statuses):
        self.statuses = statuses

    def status(self, kind, name):
        return self.statuses.get((kind, name))

    def use(self, config):
        self.config = config


def up(controller=True):
    return Status("warehouse_cart", "sim", True, 1.0,
                  {"profile": "homebody-warehouse_cart", "controller": {"running": controller, "error": None if controller else "boom"},
                   "stream": "tcp://127.0.0.1:5721"})


def app(tmp_path, states, sim=None, monitor=None):
    path = tmp_path / "servers.toml"
    path.write_text(CONFIG)
    sent = []
    client = types.SimpleNamespace(send=lambda kind, **f: sent.append((kind, f)), close=lambda: None,
                                   closed=types.SimpleNamespace(is_set=lambda: False))
    state_iter = iter(states)
    video = types.SimpleNamespace(watched=None)
    video.watch = lambda endpoint: setattr(video, "watched", endpoint)
    controls = MonitorControls(None)
    dashboard = fake_dashboard(controls=controls)
    made = MonitorApp(path, console=dashboard, monitor=monitor, make_client=lambda harness: client,
                      harness_state=lambda harness: next(state_iter),
                      watcher=Watcher({("sim", "warehouse_cart"): sim or up()}), video=video)
    controls.app = made
    made.attach(dashboard)
    return made, sent, video


def test_a_scenario_attaches_its_harness_to_its_server_then_sends_its_task(tmp_path):
    states = [{"status": "Ready", "sim": None},
              {"status": "Loading", "sim": {"endpoint": "tcp://127.0.0.1:5720", "connected": True}},
              {"status": "Ready", "sim": {"endpoint": "tcp://127.0.0.1:5720", "connected": True}}]
    made, sent, video = app(tmp_path, states)
    made.wait_attached = lambda harness, sim: MonitorApp.wait_attached(made, harness, sim, timeout=5, poll=0)
    assert made.run_scenario(NAME, "Pick one jam jar, edited.")
    assert sent[0] == ("attach_sim", {"target": made.config.sims["warehouse_cart"].target()})
    assert sent[1] == ("submit", {"text": "Pick one jam jar, edited."})
    assert video.watched == "tcp://127.0.0.1:15721" and made.harness == "demo_with_monitor"
    assert "task sent" in made.message


def test_an_attached_harness_is_not_reattached_and_a_busy_one_gets_no_task(tmp_path):
    made, sent, _ = app(tmp_path, [{"status": "Ready", "sim": {"endpoint": "tcp://127.0.0.1:5720", "connected": True}}])
    assert made.run_scenario(NAME) and [kind for kind, _ in sent] == ["submit"]
    made, sent, _ = app(tmp_path, [{"status": "Executing", "sim": {"endpoint": "tcp://127.0.0.1:5720", "connected": True}}])
    assert not made.run_scenario(NAME) and sent == []
    assert "running a task" in made.message


def test_a_scenario_stops_with_its_reason_when_the_server_is_down_or_has_no_controller(tmp_path):
    down = Status("warehouse_cart", "sim", False, error="TimeoutError: describe timed out")
    made, sent, _ = app(tmp_path, [], sim=down)
    assert not made.run_scenario(NAME) and sent == []
    assert "does not answer" in made.message and "timed out" in made.message
    made, sent, _ = app(tmp_path, [], sim=up(controller=False))
    assert not made.run_scenario(NAME) and "boom" in made.message
    assert not made.run_scenario("nonsense") and "no scenario" in made.message


def test_stream_frames_feed_the_watched_view_and_the_monitors_frames(tmp_path):
    captured = []
    monitor = types.SimpleNamespace(capture_due=lambda t: True, capture=lambda rgb, t: captured.append(t))
    made, _, _ = app(tmp_path, [], monitor=monitor)
    hub = made.console.hub
    made.on_frame({"camera": "ego", "stamp": {"sim_time_s": 2.5}, "rgb": np.zeros((4, 4, 3), np.uint8)})
    assert captured == [2.5] and [f[0] for f in hub.frames] == ["robot"]
    assert hub.telemetries[-1]["sim"] == 2.5 and "pose" not in hub.telemetries[-1]  # frames arrive; no pose known
    hub.views = {"observer"}  # the page watches the observer: an ego frame only feeds the monitor and the history
    made.on_frame({"camera": "ego", "stamp": {"sim_time_s": 2.6}, "rgb": np.ones((4, 4, 3), np.uint8)})
    made.on_frame({"camera": "observer", "stamp": {"sim_time_s": 2.7}, "rgb": np.full((4, 4, 3), 7, np.uint8)})
    assert [f[0] for f in hub.frames] == ["robot", "observer"] and captured == [2.5, 2.6]
    assert hub.frames[-1][1]["label"] == "Simulation: observer"  # no scenario ran yet: no server named


def test_a_refreshed_config_offers_its_scenarios_and_a_bad_one_changes_nothing(tmp_path):
    made, _, _ = app(tmp_path, [])
    made.config_path.write_text(CONFIG.replace("one jam jar into the cart", "two jam jars into the cart"))
    assert "reloaded" in made.refresh_config()
    assert [p["name"] for p in made.console._snapshot()["scenarios"]] == ["Warehouse: two jam jars into the cart"]
    made.config_path.write_text("[[harness]]\nname = 1\n")
    assert "not reloaded" in made.refresh_config() and "Warehouse: two jam jars into the cart" in made.config.scenarios


def test_the_page_runs_a_scenario_and_refuses_one_the_config_lacks(tmp_path):
    made, _, _ = app(tmp_path, [])
    started = []
    made.run_scenario_soon = lambda name, text: started.append((name, text))
    assert made.console.command({"type": "run_scenario", "name": NAME, "text": "edited"})["ok"]
    assert started == [(NAME, "edited")]
    assert not made.console.command({"type": "run_scenario", "name": "nonsense"})["ok"]
    reply = made.console.command({"type": "submit", "text": "pick"})  # no harness connected yet
    assert not reply["ok"]


def test_the_servers_rows_say_who_answers_what_runs_where_and_the_harness_link():
    import tempfile
    from pathlib import Path

    from homebody.server.config import load
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "servers.toml"
        path.write_text(CONFIG)
        config = load(path)
    statuses = {("harness", "demo_with_monitor"): Status("demo_with_monitor", "harness", True, 3.0,
                                                         {"status": "Ready", "sim": {"connected": False, "endpoint": "tcp://127.0.0.1:5720", "error": "does not answer"}},
                                                         checked=100.0),
                ("sim", "warehouse_cart"): Status("warehouse_cart", "sim", True, 1.0,
                                                  {"profile": "homebody-warehouse_cart", "controller": {"running": True},
                                                   "stream": "tcp://127.0.0.1:5721", "overshoot": {"left_ankle_roll_joint": 0.04}},
                                                  checked=100.0)}
    harness, cart, kitchen = server_rows(config, statuses, "demo_with_monitor", "warehouse_cart", now=101.0,
                                         last_ok={("sim", "warehouse_cart"): 100.0})
    assert harness["health"] == "degraded" and "lost its simulation server tcp://127.0.0.1:5720" in harness["detail"]
    assert harness["error"] == "does not answer" and harness["active"] and harness["impact"] == "critical"
    assert cart["health"] == "healthy" and "controller running" in cart["detail"] and "video on" in cart["detail"]
    assert "left_ankle_roll_joint 0.040" in cart["detail"] and cart["last_ok"] == 100.0
    assert cart["impact"] == "critical" and "active harness runs on it" in cart["dependency"]  # the harness says so
    assert kitchen["name"] == "kitchen_baskets" and kitchen["health"] == "unknown"
    assert kitchen["impact"] == "noncritical"  # not the active harness's server
    stalled = server_rows(config, statuses, now=200.0)[0]  # a stalled watcher shows as unavailable
    assert stalled["health"] == "unavailable" and "no status for" in stalled["error"]
    assert server_rows(config, {}, None, "warehouse_cart", now=101.0)[1]["impact"] == "unknown"  # link not confirmed
    slow = {("harness", "demo_with_monitor"): Status("demo_with_monitor", "harness", error="TimeoutError", checked=110.0)}
    once = server_rows(config, slow, "demo_with_monitor", now=111.0, last_ok={("harness", "demo_with_monitor"): 105.0})[0]
    assert once["health"] == "degraded" and "slow to answer" in once["detail"] and once["connected"] is False
    gone = server_rows(config, slow, "demo_with_monitor", now=111.0, last_ok={("harness", "demo_with_monitor"): 90.0})[0]
    assert gone["health"] == "unavailable"  # silent for longer than the grace: a lost server


def test_the_robot_row_says_which_harness_runs_where_and_when_its_simulation_server_is_lost(tmp_path):
    made, _, _ = app(tmp_path, [])
    assert made.robot_link()["state"] == "unknown"
    made.client = types.SimpleNamespace(closed=types.SimpleNamespace(is_set=lambda: False))
    made.harness = "demo_with_monitor"
    made.statuses = {("harness", "demo_with_monitor"): Status("demo_with_monitor", "harness", True, 3.0, {
        "sim": {"connected": False, "endpoint": "tcp://127.0.0.1:5720", "error": "does not answer"}})}
    link = made.robot_link()
    assert link == {"state": "lost", "detail": "Harness demo_with_monitor on tcp://127.0.0.1:5720: does not answer"}
    assert made.console._snapshot()["robot_link"] == link


def test_the_controller_state_is_the_simulation_servers_own_report(tmp_path):
    made, _, _ = app(tmp_path, [])
    assert made.controller_state()["state"] == "unknown"
    made.sim = made.config.sims["warehouse_cart"]
    made.statuses = {("sim", "warehouse_cart"): up(controller=False)}
    state = made.controller_state()
    assert state["state"] == "error" and "boom" in state["detail"]
    made.statuses = {("sim", "warehouse_cart"): up()}
    assert made.controller_state()["state"] == "running"


def test_apply_binds_the_pair_and_the_session_is_what_the_harness_confirms(tmp_path):
    attached = {"status": "Ready", "sim": {"endpoint": "tcp://127.0.0.1:5720", "connected": True}}
    made, sent, video = app(tmp_path, [{"status": "Ready", "sim": None}, attached])
    made.wait_attached = lambda harness, sim: MonitorApp.wait_attached(made, harness, sim, timeout=5, poll=0)
    assert made.session()["harness"] is None  # nothing active before an Apply
    assert made.apply_session("demo_with_monitor", "warehouse_cart")
    assert [kind for kind, _ in sent] == ["attach_sim"] and video.watched == "tcp://127.0.0.1:15721"
    made.statuses[("harness", "demo_with_monitor")] = Status("demo_with_monitor", "harness", True, 1.0, attached)
    session = made.session()
    assert (session["harness"], session["sim"], session["sim_confirmed"], session["lifecycle"]) == \
        ("demo_with_monitor", "warehouse_cart", True, "Ready")
    assert session["applying"] is None and session["error"] == ""


def test_apply_is_refused_for_a_server_that_does_not_answer_or_under_a_running_task(tmp_path):
    down = Status("warehouse_cart", "sim", False, error="TimeoutError: describe timed out")
    made, sent, _ = app(tmp_path, [], sim=down)
    assert "does not answer" in made.apply_refusal("demo_with_monitor", "warehouse_cart")
    reply = made.console.command({"type": "apply_session", "harness": "demo_with_monitor", "sim": "warehouse_cart"})
    assert not reply["ok"] and "does not answer" in reply["message"] and sent == []
    assert "not a configured" in made.apply_refusal("nonsense", "warehouse_cart")
    made, sent, _ = app(tmp_path, [])
    made.harness, made.client = "demo_with_monitor", types.SimpleNamespace(closed=types.SimpleNamespace(is_set=lambda: False))
    made.statuses[("harness", "demo_with_monitor")] = Status("demo_with_monitor", "harness", True, 1.0, {"status": "Executing"})
    assert "running a task" in made.apply_refusal("demo_with_monitor", "warehouse_cart")
    assert not made.apply_session("demo_with_monitor", "warehouse_cart") and "running a task" in made.session()["error"]


def test_a_refresh_never_drops_the_active_session_but_warns_when_the_config_no_longer_names_it(tmp_path):
    made, _, _ = app(tmp_path, [])
    made.harness = "demo_with_monitor"
    made.client = types.SimpleNamespace(closed=types.SimpleNamespace(is_set=lambda: False), close=lambda: None)
    made.sim = made.config.sims["warehouse_cart"]
    made.config_path.write_text(CONFIG.replace("demo_with_monitor", "renamed_harness"))  # the harness and its scenario
    made.watcher.refresh = lambda: made.statuses
    assert made.console.command({"type": "refresh_config"})["ok"]
    session = made.session()
    assert session["harness"] == "demo_with_monitor" and made.client is not None  # still active
    assert any("no longer in the configuration" in w for w in session["warnings"]) and session["refreshed"]


SECOND = '\n[[harness]]\nname = "demo_openmodels"\nurl = "ws://127.0.0.1:8766"\ntoken_file = "~/.homebody/server_token"\n'


def with_second_harness(made):
    made.config_path.write_text(CONFIG + SECOND)
    made.watcher.refresh = lambda: made.statuses
    assert made.console.command({"type": "refresh_config"})["ok"] and "demo_openmodels" in made.config.harnesses


def test_a_failed_apply_returns_to_the_harness_it_was_on_and_a_busy_target_or_second_click_is_refused(tmp_path):
    made, _, video = app(tmp_path, [{"status": "Ready", "sim": None}])
    with_second_harness(made)
    old = types.SimpleNamespace(closed=types.SimpleNamespace(is_set=lambda: False), close=lambda: None, send=lambda kind, **f: None)
    made.harness, made.client = "demo_with_monitor", old
    connected = []
    made.make_client = lambda harness: connected.append(harness.name) or old
    made.wait_attached = lambda harness, sim: (_ for _ in ()).throw(TimeoutError("never attached"))
    assert not made.apply_session("demo_openmodels", "warehouse_cart")
    assert made.harness == "demo_with_monitor" and connected == ["demo_openmodels", "demo_with_monitor"]
    assert "never attached" in made.session()["error"] and video.watched is None  # the video never moved
    made.watcher.statuses[("harness", "demo_openmodels")] = Status(
        "demo_openmodels", "harness", True, 1.0, {"status": "Executing", "sim": {"endpoint": "tcp://elsewhere:1"}})
    assert "on another server" in made.apply_refusal("demo_openmodels", "warehouse_cart")
    made.running.acquire()  # an Apply already in progress: the second click is refused at once
    reply = made.console.command({"type": "apply_session", "harness": "demo_with_monitor", "sim": "warehouse_cart"})
    assert not reply["ok"] and "Another session change" in reply["message"]
    made.running.release()


def test_an_active_harness_dropped_from_the_config_keeps_its_running_task_guard(tmp_path):
    made, _, _ = app(tmp_path, [])
    with_second_harness(made)
    made.harness = "demo_with_monitor"
    made.client = types.SimpleNamespace(closed=types.SimpleNamespace(is_set=lambda: False), close=lambda: None)
    made.statuses = {("harness", "demo_with_monitor"): Status("demo_with_monitor", "harness", True, 1.0, {"status": "Executing"})}
    assert made.lifecycle() == "Executing"
    made.statuses = {}  # a refresh pruned it
    assert "running a task" in made.apply_refusal("demo_openmodels", "warehouse_cart")


def test_an_attach_the_harness_reports_failed_ends_the_wait_at_once(tmp_path):
    failed = {"status": "Error", "text": "The simulation server tcp://127.0.0.1:5720 failed: serves another model", "sim": None}
    made, _, _ = app(tmp_path, [failed, {"status": "Loading", "sim": None}, failed])
    harness, sim = made.config.harnesses["demo_with_monitor"], made.config.sims["warehouse_cart"]
    try:
        made.wait_attached(harness, sim, timeout=5, poll=0)  # the first Error predates the attach: it waits on
        raise AssertionError("expected the failed attach to raise")
    except RuntimeError as error:
        assert "serves another model" in str(error)


def test_a_harness_whose_session_faulted_on_its_server_is_attached_again(tmp_path):
    endpoint = {"endpoint": "tcp://127.0.0.1:5720", "connected": True}
    states = [{"status": "Reset required", "sim": endpoint}, {"status": "Reset required", "sim": endpoint},
              {"status": "Loading", "sim": None}, {"status": "Ready", "sim": endpoint}]
    made, sent, video = app(tmp_path, states)
    made.wait_attached = lambda harness, sim: MonitorApp.wait_attached(made, harness, sim, timeout=5, poll=0)
    assert made.apply_session("demo_with_monitor", "warehouse_cart")
    assert [kind for kind, _ in sent] == ["attach_sim"] and video.watched == "tcp://127.0.0.1:15721"
