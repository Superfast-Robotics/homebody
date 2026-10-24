"""The monitor app with fake harnesses and simulation servers: the Servers rows, a scenario run step by step (and
each step's refusal), and frames from the simulation server's stream reaching the live view and the monitor."""
import types

import numpy as np
from test_monitor_config import CONFIG

from homebody.server.monitor_app import MonitorApp
from homebody.server.watch import Status
from homebody.ui.servers import servers_html

NAME = "Warehouse: one jam jar into the cart"


class Console:
    def __init__(self):
        self.updates, self.camera, self.replaying = [], "robot", False
        self.ego = types.SimpleNamespace(image=None, label="")
        self.presets = None

    def update(self, status, text=""):
        self.updates.append((status, text))

    def set_presets(self, presets, scene=""):
        self.presets = presets


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
    made = MonitorApp(path, console=Console(), monitor=monitor, make_client=lambda harness: client,
                      harness_state=lambda harness: next(state_iter),
                      watcher=Watcher({("sim", "warehouse_cart"): sim or up()}), video=video)
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
    assert "task sent" in made.console.updates[-1][1]


def test_an_attached_harness_is_not_reattached_and_a_busy_one_gets_no_task(tmp_path):
    made, sent, _ = app(tmp_path, [{"status": "Ready", "sim": {"endpoint": "tcp://127.0.0.1:5720", "connected": True}}])
    assert made.run_scenario(NAME) and [kind for kind, _ in sent] == ["submit"]
    made, sent, _ = app(tmp_path, [{"status": "Executing", "sim": {"endpoint": "tcp://127.0.0.1:5720", "connected": True}}])
    assert not made.run_scenario(NAME) and sent == []
    assert "running a task" in made.console.updates[-1][1]


def test_a_scenario_stops_with_its_reason_when_the_server_is_down_or_has_no_controller(tmp_path):
    down = Status("warehouse_cart", "sim", False, error="TimeoutError: describe timed out")
    made, sent, _ = app(tmp_path, [], sim=down)
    assert not made.run_scenario(NAME) and sent == []
    assert "does not answer" in made.console.updates[-1][1] and "timed out" in made.console.updates[-1][1]
    made, sent, _ = app(tmp_path, [], sim=up(controller=False))
    assert not made.run_scenario(NAME) and "boom" in made.console.updates[-1][1]
    assert not made.run_scenario("nonsense") and "no scenario" in made.console.updates[-1][1]


def test_stream_frames_feed_the_chosen_live_view_and_the_monitors_frames(tmp_path):
    captured = []
    monitor = types.SimpleNamespace(capture_due=lambda t: True, capture=lambda rgb, t: captured.append(t))
    made, _, _ = app(tmp_path, [], monitor=monitor)
    rgb = np.zeros((4, 4, 3), np.uint8)
    made.on_frame({"camera": "ego", "stamp": {"sim_time_s": 2.5}, "rgb": rgb})
    assert captured == [2.5] and made.console.ego.image is rgb
    made.console.camera = "observer"
    made._last_view = 0.0
    made.on_frame({"camera": "ego", "stamp": {"sim_time_s": 2.6}, "rgb": np.ones((4, 4, 3), np.uint8)})
    assert made.console.ego.image is rgb  # the live view shows the observer, so an ego frame only feeds the monitor
    made.on_frame({"camera": "observer", "stamp": {"sim_time_s": 2.7}, "rgb": np.full((4, 4, 3), 7, np.uint8)})
    assert made.console.ego.image[0, 0, 0] == 7 and captured == [2.5, 2.6]


def test_a_refreshed_config_offers_its_scenarios_and_a_bad_one_changes_nothing(tmp_path):
    made, _, _ = app(tmp_path, [])
    made.config_path.write_text(CONFIG.replace("one jam jar into the cart", "two jam jars into the cart"))
    assert "reloaded" in made.refresh_config()
    assert [p["name"] for p in made.console.presets] == ["Warehouse: two jam jars into the cart"]
    made.config_path.write_text("[[harness]]\nname = 1\n")
    assert "not reloaded" in made.refresh_config() and "Warehouse: two jam jars into the cart" in made.config.scenarios


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
    html = servers_html(config, statuses, "demo_with_monitor", "warehouse_cart", now=101.0)
    assert "lost its simulation server tcp://127.0.0.1:5720" in html and "does not answer" in html
    assert "controller running" in html and "video on" in html and "left_ankle_roll_joint 0.040" in html
    assert "kitchen_baskets" in html and "not checked yet" in html
    assert "no status for" in servers_html(config, statuses, now=200.0)  # a stalled watcher shows as down
