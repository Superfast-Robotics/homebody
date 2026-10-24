"""The monitor app's config: harnesses, simulation servers and scenarios, checked whole before any is used."""
import pytest

from homebody.server.config import load

CONFIG = '''
[[harness]]
name = "demo_with_monitor"
url = "ws://127.0.0.1:8765"
token_file = "~/.homebody/server_token"

[[sim]]
name = "warehouse_cart"
endpoint = "tcp://127.0.0.1:5720"
stream = "tcp://127.0.0.1:5721"
local_endpoint = "tcp://127.0.0.1:15720"
local_stream = "tcp://127.0.0.1:15721"
profile = "/lambda/exports/homebody_warehouse_cart/profile.toml"
scene = "/lambda/scenes/warehouse_cart"
roots = { HOMEBODY_EFFECTIVE_ROOT = "/lambda/exports/homebody_warehouse_cart" }

[[sim]]
name = "kitchen_baskets"
endpoint = "tcp://127.0.0.1:5730"
profile = "/lambda/exports/homebody_kitchen_baskets/profile.toml"
scene = "/lambda/scenes/kitchen_baskets"

[[scenario]]
name = "Warehouse: one jam jar into the cart"
harness = "demo_with_monitor"
sim = "warehouse_cart"
task = "Pick one jam jar and place it in the blue picking cart."
'''


def write(tmp_path, text):
    path = tmp_path / "servers.toml"
    path.write_text(text)
    return path


def test_a_config_names_harnesses_sims_and_scenarios_with_both_views_of_each_sim(tmp_path):
    config = load(write(tmp_path, CONFIG))
    sim = config.sims["warehouse_cart"]
    assert sim.local_endpoint == "tcp://127.0.0.1:15720" and sim.local_stream == "tcp://127.0.0.1:15721"
    assert sim.target() == {"endpoint": "tcp://127.0.0.1:5720", "profile": "/lambda/exports/homebody_warehouse_cart/profile.toml",
                            "scene": "/lambda/scenes/warehouse_cart",
                            "roots": {"HOMEBODY_EFFECTIVE_ROOT": "/lambda/exports/homebody_warehouse_cart"},
                            "engine": "warehouse"}
    kitchen = config.sims["kitchen_baskets"]
    assert kitchen.local_endpoint == kitchen.endpoint and kitchen.stream is None  # no tunnel: the same addresses
    scenario = config.scenarios["Warehouse: one jam jar into the cart"]
    assert scenario.harness == "demo_with_monitor" and config.harnesses[scenario.harness].url == "ws://127.0.0.1:8765"


@pytest.mark.parametrize("change, error", [
    (lambda t: t.replace('sim = "warehouse_cart"', 'sim = "garage"'), "unknown sim"),
    (lambda t: t.replace('harness = "demo_with_monitor"', 'harness = "nope"'), "unknown harness"),
    (lambda t: t.replace('name = "kitchen_baskets"', 'name = "warehouse_cart"'), "repeated"),
    (lambda t: t.replace('stream = "tcp://127.0.0.1:5721"', 'stream = "tcp://10.0.0.5:5721"'), "loopback"),
    (lambda t: t.replace('url = "ws://127.0.0.1:8765"', 'url = "http://127.0.0.1:8765"'), "ws://"),
    (lambda t: t + '\n[[sim]]\nname = "x"\nendpoint = "tcp://127.0.0.1:1"\nprofile = "/p"\nscene = "/s"\ncolour = "red"\n',
     "unknown"),
    (lambda t: t.replace('task = "Pick one jam jar and place it in the blue picking cart."', 'task = ""'), "task"),
])
def test_a_bad_config_is_refused_whole_with_the_reason(tmp_path, change, error):
    with pytest.raises(ValueError, match=error):
        load(write(tmp_path, change(CONFIG)))


def test_an_unreadable_config_names_the_file(tmp_path):
    with pytest.raises(ValueError, match="servers.toml"):
        load(write(tmp_path, "[[harness]\nname = 1"))


def test_one_watch_round_checks_every_server_at_once(tmp_path):
    import time

    from homebody.server.config import load
    from homebody.server.watch import Status, Watcher
    path = tmp_path / "servers.toml"
    path.write_text(CONFIG)
    config = load(path)

    def slow(kind, server):
        time.sleep(0.3)  # a server that does not answer costs its timeout
        return Status(server.name, kind, True, checked=time.time())
    watcher = Watcher(config, check=slow)
    start = time.monotonic()
    statuses = watcher.refresh()
    assert len(statuses) == len(config.harnesses) + len(config.sims) >= 3
    assert time.monotonic() - start < 0.3 * 2


HOMEBODY_SIM = '''
[[sim]]
name = "kitchen_hb"
engine = "homebody"
endpoint = "tcp://127.0.0.1:5740"
stream = "tcp://127.0.0.1:5741"
scene = "/scenes/kitchen_baskets"
'''


def test_a_homebody_sim_server_needs_no_profile_and_its_target_names_its_engine(tmp_path):
    from homebody.backends.warehouse import SimTarget
    path = tmp_path / "servers.toml"
    path.write_text(CONFIG + HOMEBODY_SIM)
    sim = load(path).sims["kitchen_hb"]
    assert sim.engine == "homebody" and sim.target()["engine"] == "homebody" and sim.target()["profile"] is None
    target = SimTarget.from_json(sim.target())
    assert target.engine == "homebody" and target.profile is None
    assert load(path).sims["warehouse_cart"].target()["engine"] == "warehouse"
    path.write_text(CONFIG + HOMEBODY_SIM.replace('engine = "homebody"', 'engine = "warehouse"'))
    with pytest.raises(ValueError, match="needs the profile"):
        load(path)
    path.write_text(CONFIG + HOMEBODY_SIM.replace('"homebody"', '"isaac"'))
    with pytest.raises(ValueError, match="engine"):
        load(path)


def test_the_status_check_asks_a_homebody_sim_server_its_own_way(tmp_path, monkeypatch):
    from homebody.server import watch
    from homebody.simserver import client
    path = tmp_path / "servers.toml"
    path.write_text(CONFIG + HOMEBODY_SIM)
    asked = []
    monkeypatch.setattr(client, "describe", lambda endpoint, timeout: asked.append(endpoint) or {
        "profile": {"id": "homebody-kitchen_baskets"}, "mode": "continuous", "controller": {"running": True},
        "camera_stream": "tcp://127.0.0.1:5741"})
    status = watch.sim_status(load(path).sims["kitchen_hb"])
    assert asked == ["tcp://127.0.0.1:5740"] and status.connected
    assert status.detail["controller"]["running"] and status.detail["profile"] == "homebody-kitchen_baskets"
