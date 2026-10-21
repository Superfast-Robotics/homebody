import json
import tomllib

import pytest

from evolve import agent_experiments as ax
from evolve import loop, score_episode
from homebody.session.config import COMPOSITES, MODELS


def test_every_experiment_names_a_models_row_and_served_models():
    table = ax.experiments()
    assert list(table) == [f"E{n}" for n in range(1, 21)]
    served = {name for name, model in ax.MODELS.items() if model.agent}
    for eid, exp in table.items():
        assert exp.agent in MODELS or exp.agent.endswith("-vllm"), eid
        assert exp.local.model in ax.MODELS and (exp.second is None or exp.second.model in ax.MODELS), eid
    for spec in COMPOSITES.values():  # the endpoints a composite asks are served models of the table
        for member in spec.values():
            if isinstance(member, dict) and "served" in member:
                assert member["served"] in ax.MODELS
    assert served >= set(ax.SMALL)


def test_blocked_and_runnable_experiments():
    table = ax.experiments()
    blocked = {eid for eid, exp in table.items() if exp.blocked()}
    assert blocked == {"E3", "E6", "E9", "E11", "E12", "E13", "E14", "E15", "E16", "E18"}
    assert "quota" in table["E11"].blocked() and "4-bit" in table["E3"].blocked()


def test_runnable_pairs_fit_one_a100():
    for eid, exp in ax.experiments().items():
        if not exp.blocked():
            weights, gpu = exp.memory()
            assert gpu <= 0.92 * ax.GPU_GB and weights < gpu - 4, eid  # room for activations and KV cache


def test_a_96_gb_gpu_runs_the_32b_and_hy_embodied_experiments():
    table = ax.experiments(gpu_gb=96)
    blocked = {eid for eid, exp in table.items() if exp.blocked(96)}
    assert blocked == {"E11", "E12", "E18"}  # only the Sol ones wait
    for eid, exp in table.items():
        weights, gpu = exp.memory(96)
        assert gpu <= 0.92 * 96 and weights < gpu - 4, eid
    assert table["E3"].local == ax.single("cosmos-reason2-32b")  # alone: BF16
    assert table["E13"].second.online_fp8 and not table["E13"].local.online_fp8  # beside an 8B: the 32B in FP8


def test_an_ssh_server_is_driven_through_ssh_and_reached_through_its_tunnel(evolve_dir, monkeypatch):
    gh = {"name": "gh", "ip": "10.0.0.9", "ssh": "gh-host", "ports": [9000, 9001], "gpu_gb": 96}
    assert ax.on_host(gh, ["pgrep", "-af", "vllm serve .*--port 8000( |$)"]) == [
        "ssh", "gh-host", "pgrep -af 'vllm serve .*--port 8000( |$)'"]
    assert ax.on_host({"ip": "1"}, ["pgrep"]) == ["pgrep"] and ax.bind(gh) == "127.0.0.1"
    sent = []
    monkeypatch.setattr(ax.subprocess, "run", lambda argv, **kw: sent.append(argv))
    config = cfg() | {"agents": cfg()["agents"] | {"venv": "~/vllm-venv", "logs": "~/vlm"}}
    ax.start(config, gh, ax.Serve("cosmos-reason2-32b", 0.9), 8000)
    host, line = sent[0][1:]
    assert host == "gh-host" and line.startswith("HF_HOME=~/hf VLLM_API_KEY=$(cat ~/vlm/api_key) setsid nohup "
                                                 "~/vllm-venv/bin/vllm serve nvidia/Cosmos-Reason2-32B ")
    assert "--host 127.0.0.1 --port 8000" in line and line.endswith("> ~/vlm/cosmos-reason2-32b-8000.log 2>&1 < /dev/null &")
    ax.write_settings(cfg(), "E13", ax.experiments(gpu_gb=96)["E13"], gh, ("a", "b"))
    agent = tomllib.loads((evolve_dir / "episode_settings_pp_E13.toml").read_text())["agent"]
    assert (agent["local_url"], agent["second_url"]) == ("http://10.0.0.9:9000/v1", "http://10.0.0.9:9001/v1")


def test_the_picks_change_the_served_models():
    table = ax.experiments(best_small="qwen3-vl-8b", best_base="cosmos-reason2-8b")
    assert table["E10"].local.model == "qwen3-vl-8b" and table["E17"].second.model == "qwen3-vl-8b"
    assert table["E19"].second.model == "robobrain2.5-8b" and table["E20"].local.model == "cosmos-reason2-8b"
    assert ax.experiments()["E19"].second is None  # the base already serves the verifier's model


def test_vllm_command_carries_the_share_and_quantisation():
    args = ax.command(ax.Serve("robobrain2.5-8b", 0.38, 16384, online_fp8=True), "10.0.0.1", 8001)
    assert args[0] == "BAAI/RoboBrain2.5-8B-NV" and args[args.index("--port") + 1] == "8001"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.38" and args[-2:] == ["--quantization", "fp8"]
    assert "--trust-remote-code" in ax.command(ax.Serve("molmo2-er", 0.33), "10.0.0.1", 8001)


@pytest.fixture
def evolve_dir(tmp_path, monkeypatch):
    (tmp_path / "base.toml").write_text('[agent]\nmax_steps = 60\n\n[coder]\ncalls = 0\n')
    monkeypatch.setattr(ax, "HERE", tmp_path)
    return tmp_path


def cfg():
    return {"agents": {"ports": [8000, 8001], "work": "~/w", "parallel": 6, "base_profile": "demo-pp"},
            "e2e": {"demo-pp": {"name": "dex1-pickplace", "settings": "base.toml", "parallel": 12, "clusters": {},
                                "services": ["start"], "suite": {"train": ["a/*"], "val": ["b/*"]}}}}


def test_settings_add_the_agent_and_its_servers(evolve_dir):
    table = ax.experiments()
    ax.write_settings(cfg(), "E7", table["E7"], {"ip": "10.0.0.9"}, ("robobrain25-8b-vllm", "robobrain25-8b-vllm"))
    settings = tomllib.loads((evolve_dir / "episode_settings_pp_E7.toml").read_text())
    assert settings["agent"] == {"max_steps": 60, "model": "e7-qwen30b-molmo",
                                 "local_url": "http://10.0.0.9:8000/v1", "second_url": "http://10.0.0.9:8001/v1",
                                 "best_small": "robobrain25-8b-vllm", "best_base": "robobrain25-8b-vllm"}
    assert settings["coder"] == {"calls": 0}
    ax.write_settings(cfg(), "E2", table["E2"], {"ip": "10.0.0.9"}, ("a", "b"))
    one = tomllib.loads((evolve_dir / "episode_settings_pp_E2.toml").read_text())["agent"]
    assert one["second_url"] == one["local_url"]


def test_overlay_profiles_are_demo_pp_with_their_own_agent(evolve_dir):
    ax.write_overlay(cfg(), ax.experiments(), ["E7", "E10"])
    overlay = tomllib.loads((evolve_dir / loop.LOCAL_CONFIG).read_text())
    profile = overlay["e2e"]["demo-pp-E7"]
    assert profile["agent_model"] == "e7-qwen30b-molmo" and profile["workdir"] == "demo-pp-E7"
    assert profile["settings"] == "episode_settings_pp_E7.toml" and profile["parallel"] == 6
    assert profile["services"] == ["start"] and profile["clusters"] == {} and profile["suite"]["val"] == ["b/*"]
    assert overlay["local"]["proxy"] == "" and overlay["local"]["work"].endswith("/w")


def test_loop_config_merges_the_overlay_table_by_table():
    merged = loop.merged({"local": {"work": "a", "proxy": "p"}, "e2e": {"demo-pp": {"x": 1}}},
                         {"local": {"work": "b"}, "e2e": {"demo-pp-E7": {"x": 2}}})
    assert merged == {"local": {"work": "b", "proxy": "p"}, "e2e": {"demo-pp": {"x": 1}, "demo-pp-E7": {"x": 2}}}


def test_the_queue_prefers_an_experiment_on_the_loaded_models(tmp_path, monkeypatch):
    table = ax.experiments()
    loaded = {8000: ("qwen3-vl-30b-a3b", ""), 8001: ("embodied-r1-3b", "")}
    monkeypatch.setattr(ax, "served", lambda server, port: loaded.get(port))
    config, here = cfg(), {"ip": "10.0.0.9"}
    assert ax.next_experiment(config, here, ["E7", "E8", "E10"], table, tmp_path) == "E8"
    assert ax.next_experiment(config, here, ["E7", "E8", "E10"], table, tmp_path) == "E7"  # E8 is claimed
    (tmp_path / "rows").mkdir()
    (tmp_path / "rows" / "E10.json").write_text("{}")
    assert ax.next_experiment(config, here, ["E7", "E8", "E10"], table, tmp_path) is None


def test_summary_row_from_the_baseline_entry(tmp_path):
    split = {"placed": 0.5, "success_rate": 0.25, "minutes": 9.0, "credits": 0.0, "disturbed": 1,
             "outcomes": {"model_done, placed 50%": 3, "provider_error": 1}, "placed_by_layout": {"a": "1/2", "b": "2/2"}}
    folder = tmp_path / "e2e" / "demo-pp-E2"
    folder.mkdir(parents=True)
    (folder / "archive.jsonl").write_text(json.dumps({"id": "b000", "status": "baseline", "commit": "abc",
                                                      "train": split, "val": split}) + "\n")
    row = ax.summary_row("E2", ax.experiments()["E2"], tmp_path)
    assert row["all_placed"] == 0.75 and row["train_provider_errors"] == 1 and row["agent"] == "robobrain25-8b-vllm"


def usage(folder, prompt, cached, output):
    folder.mkdir(parents=True)
    (folder / "reply.json").write_text(json.dumps({"usage": {"prompt_tokens": prompt, "completion_tokens": output,
                                                             "prompt_tokens_details": {"cached_tokens": cached}}}))


def test_a_multi_model_decision_is_priced_per_call(tmp_path):
    step = tmp_path / "decisions" / "step-000"
    usage(step / "01-routine", 1000, 0, 100)
    usage(step / "02-fallback", 1_000_000, 0, 0)
    (step / "03-pointer").mkdir(parents=True)  # a local call that failed before replying
    (step / "routing.json").write_text(json.dumps({"calls": [
        {"role": "routine", "model": "robobrain25-8b-vllm", "local": True, "dir": "01-routine"},
        {"role": "fallback", "model": "sol61-litellm", "local": False, "dir": "02-fallback"},
        {"role": "pointer", "model": "molmo2-er", "local": True, "dir": "03-pointer"}]}))
    record = score_episode.cost(tmp_path, "e11-robobrain-sol")
    assert record["credits"] == 50.0 and not record["unmetered"] and record["main_input"] == 1_001_000
    assert record["models"]["robobrain25-8b-vllm"] == {"calls": 1, "input": 1000, "cached": 0, "output": 100,
                                                       "credits": 0.0}
    assert record["models"]["sol61-litellm"]["credits"] == 50.0 and record["models"]["molmo2-er"]["calls"] == 1


def test_a_remote_call_without_usage_is_unmetered(tmp_path):
    step = tmp_path / "decisions" / "step-000"
    (step / "01-fallback").mkdir(parents=True)
    (step / "routing.json").write_text(json.dumps({"calls": [
        {"role": "fallback", "model": "sol61-litellm", "local": False, "dir": "01-fallback"}]}))
    assert score_episode.cost(tmp_path, "e11-robobrain-sol")["unmetered"]


def test_a_single_model_decision_is_priced_as_before(tmp_path):
    usage(tmp_path / "decisions" / "step-000", 1_000_000, 0, 0)
    assert score_episode.cost(tmp_path, "sol61-litellm")["credits"] == 50.0
