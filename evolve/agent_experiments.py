"""Open-weight robot agents (astra-evolve/e2e/agent_experiments.md): serve an experiment's models with vLLM on a GPU
host, run the demo-pp baseline (17 sessions) with the experiment's agent, write its summary row.

    python3 -m evolve.agent_experiments plan                     the experiments, their models and GPU memory
    python3 -m evolve.agent_experiments worker [IDS] [--commit C] [--best-small M] [--best-base M] [--keep]
    python3 -m evolve.agent_experiments summary                  every row written so far

A worker runs on each GPU host ([agents] servers, found by private IP, or named with --server), from a checkout of a commit that has the
agent code (the vllm provider and vlm/composite.py). Workers share [agents] shared: each claims the next unclaimed
experiment of IDS (default: every runnable one, in order), preferring one whose models it already serves, so two
GPU hosts run two experiments at once. It writes evolve/config.local.toml (the machine's work folder and one
demo-pp-<id> profile per experiment) and evolve/episode_settings_pp_<id>.toml (the agent and its server URLs), then
runs `evolve.e2e_loop --profile demo-pp-<id> baseline COMMIT`.

A server with `ssh` is a GPU host without the shared volume (the GH200): the worker runs on a machine that has it,
starts vLLM there over ssh on 127.0.0.1 [agents] ports, and holds a tunnel from its own `ip` and the server's `ports`
to them, which sessions use. `gpu_gb` (default 40) sets which experiments fit and their shares."""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .loop import HERE, LOCAL_CONFIG, REPO, config, git

GPU_GB = 40.0
BIG_GB = 90.0  # a GPU this large (GH200 96 GB) serves the 32B and Hy-Embodied models in BF16
SMALL = ("robobrain2.5-8b", "cosmos-reason2-8b", "qwen3-vl-8b")  # the Phase-1 8B models best-small may name


@dataclass(frozen=True)
class Model:
    """One model as we serve it: its MODELS name (None for a pointer or verifier asked in its own words), the
    checkpoint vLLM loads (a quantised build where one is needed), weights in GB as loaded, its point format."""
    agent: str | None
    repo: str
    weights_gb: float
    precision: str
    point: str = "qwen1000"  # molmo: <point x y> percentages; qwen1000: [x, y] in 0-1000; pixels: Qwen2.5-VL pixels
    args: tuple[str, ...] = ()
    blocked: str = ""  # why it cannot be served on one A100 40 GB
    big: bool = False  # served only on a BIG_GB GPU (blocked says why)
    whitespace: bool = False  # its JSON needs free whitespace: Cosmos3 garbles its text fields and the Qwen3.5-family
    #                           models (Qwen3.6/3.8, RynnBrain1.1) close their arguments early without it


MODELS = {
    "robobrain2.5-8b": Model("robobrain25-8b-vllm", "BAAI/RoboBrain2.5-8B-NV", 17.5, "BF16"),
    "cosmos-reason2-8b": Model("cosmos-r2-8b-vllm", "nvidia/Cosmos-Reason2-8B", 17.5, "BF16"),
    "qwen3-vl-8b": Model("qwen3vl-8b-vllm", "Qwen/Qwen3-VL-8B-Instruct", 17.5, "BF16"),
    "qwen3-vl-30b-a3b": Model("qwen3vl-30b-vllm", "cyankiwi/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit", 17.0, "AWQ 4-bit"),
    "cosmos-reason2-32b": Model("cosmos-r2-32b-vllm", "nvidia/Cosmos-Reason2-32B", 66.8, "BF16",
                                blocked="no standard 4-bit build (only GGUF and embedl's FlashHead W4A16 with a "
                                        "custom architecture); BF16 needs 67 GB", big=True),
    "hy-embodied": Model("hy-embodied-vllm", "tencent/Hy-Embodied-VLM-1.0", 61.0, "BF16",
                         args=("--trust-remote-code",),
                         blocked="no quantised build; BF16 needs 61 GB", big=True),
    "molmo2-er": Model(None, "allenai/Molmo2-ER", 9.7, "BF16", point="molmo",
                       args=("--trust-remote-code", "--dtype", "bfloat16")),
    "embodied-r1-3b": Model(None, "IffYuan/Embodied-R1-3B-v1", 8.0, "BF16", point="embodied_r1"),
    # 2026 candidates (vLLM 0.32 on loki-gh200; the 8B Qwen3-VL fine-tunes also on loki-a100's 0.11.2)
    "embodied-r1.5": Model("embodied-r15-vllm", "IffYuan/Embodied-R1.5", 17.5, "BF16", point="embodied_r15"),
    "rynnbrain-8b": Model("rynnbrain-8b-vllm", "Alibaba-DAMO-Academy/RynnBrain-8B", 17.5, "BF16", point="rynnbrain"),
    "rynnbrain1.1-9b": Model("rynnbrain11-9b-vllm", "Alibaba-DAMO-Academy/RynnBrain1.1-9B", 19.0, "BF16",
                             point="rynnbrain", big=True, whitespace=True, args=("--reasoning-parser", "qwen3"), blocked="needs vLLM >= 0.17 (loki-gh200)"),
    "mimo-embodied-7b": Model("mimo-embodied-7b-vllm", "XiaomiMiMo/MiMo-Embodied-7B", 16.6, "BF16", point="mimo"),
    "qwen3.8-27b": Model("qwen38-27b-vllm", "Qwen/Qwen3.8-27B-FP8", 29.0, "FP8", point="qwen_json", big=True, whitespace=True, args=("--reasoning-parser", "qwen3"),
                         blocked="needs vLLM >= 0.17 (loki-gh200)"),
    "qwen3.6-35b-a3b": Model("qwen36-35b-vllm", "Qwen/Qwen3.6-35B-A3B-FP8", 36.0, "FP8", point="qwen_json", big=True, whitespace=True, args=("--reasoning-parser", "qwen3"),
                             blocked="needs vLLM >= 0.17 (loki-gh200)"),
    "cosmos3-nano": Model("cosmos3-nano-vllm", "nvidia/Cosmos3-Nano", 17.5, "BF16", point="cosmos3", whitespace=True,
                          args=("--mm-encoder-tp-mode", "data"), big=True, blocked="needs vLLM >= 0.23 (loki-gh200)"),
    "cosmos3-super": Model("cosmos3-super-vllm", "nvidia/Cosmos3-Super", 66.7, "BF16", point="cosmos3", whitespace=True,
                           args=("--mm-encoder-tp-mode", "data"), big=True, blocked="66.7 GB; vLLM >= 0.23"),
    "gemma-4-31b": Model("gemma4-31b-vllm", "google/gemma-4-31B-it", 62.0, "BF16", point="gemma", big=True,
                         blocked="62 GB; vLLM >= 0.19"),
}
# built and not chosen: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 (31 GB weight-only on Ampere: alone only),
# QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ, cyankiwi/...-AWQ-8bit (31 GB), Qwen/Qwen3-VL-8B-Instruct-FP8


@dataclass(frozen=True)
class Serve:
    """One vLLM server of an experiment: the model, on the local (first) or second port, its share of the GPU."""
    model: str
    share: float
    max_len: int = 32768
    online_fp8: bool = False  # weight-only FP8 (Marlin on Ampere) to fit beside a bigger model


@dataclass(frozen=True)
class Experiment:
    agent: str  # session/config.py MODELS name: a vllm model or a COMPOSITES one
    local: Serve
    second: Serve | None = None
    quota: bool = False  # uses Sol: waits for the ChatGPT quota reset
    best: tuple[str, ...] = field(default=())  # the picks it takes: best-small, best-base

    def blocked(self, gpu_gb=GPU_GB):
        reasons = [MODELS[s.model].blocked for s in (self.local, self.second)
                   if s and MODELS[s.model].blocked and not (MODELS[s.model].big and gpu_gb >= BIG_GB)]
        if self.quota:
            reasons.append("uses Sol: after the Oct 27 quota reset")
        return "; ".join(reasons)

    def memory(self, gpu_gb=GPU_GB):
        """(weights, the vLLM shares as GB) on one GPU."""
        serves = [s for s in (self.local, self.second) if s]
        weights = sum(MODELS[s.model].weights_gb / (2 if s.online_fp8 else 1) for s in serves)
        return weights, sum(s.share for s in serves) * gpu_gb


def single(model):
    return Serve(model, 0.90)


def experiments(best_small="robobrain2.5-8b", best_base="robobrain2.5-8b", gpu_gb=GPU_GB):
    """E1-E20 with the Phase-1 8B pick (best_small) and the verifier experiments' base (best_base, one 8B: a
    composite base would need its own servers beside the verifier's). On a BIG_GB GPU a 32B beside an 8B is
    weight-only FP8 (native on Hopper, 33 GB) so both keep KV room; alone it is BF16."""
    qwen30 = Serve("qwen3-vl-30b-a3b", 0.55, 16384)
    big = gpu_gb >= BIG_GB
    c32 = Serve("cosmos-reason2-32b", 0.55, 16384, online_fp8=True) if big else single("cosmos-reason2-32b")

    def beside_32b(model):
        return Serve(model, 0.35, 16384) if big else Serve(model, 0.45)

    return {
        "E1": Experiment("cosmos-r2-8b-vllm", single("cosmos-reason2-8b")),
        "E2": Experiment("robobrain25-8b-vllm", single("robobrain2.5-8b")),
        "E3": Experiment("cosmos-r2-32b-vllm", single("cosmos-reason2-32b")),
        "E4": Experiment("qwen3vl-8b-vllm", single("qwen3-vl-8b")),
        "E5": Experiment("qwen3vl-30b-vllm", single("qwen3-vl-30b-a3b")),
        "E6": Experiment("hy-embodied-vllm", single("hy-embodied")),
        "E7": Experiment("e7-qwen30b-molmo", qwen30, Serve("molmo2-er", 0.33, 8192)),
        "E8": Experiment("e8-qwen30b-er1", Serve("qwen3-vl-30b-a3b", 0.58, 16384), Serve("embodied-r1-3b", 0.30, 8192)),
        "E9": Experiment("e9-hy-molmo", Serve("hy-embodied", 0.72, 16384) if big else single("hy-embodied"),
                         Serve("molmo2-er", 0.20 if big else 0.33, 8192)),
        "E10": Experiment("e10-small-molmo", Serve(best_small, 0.55, 16384), Serve("molmo2-er", 0.33, 8192), best=("best-small",)),
        "E11": Experiment("e11-robobrain-sol", single("robobrain2.5-8b"), quota=True),
        "E12": Experiment("e12-cosmos8b-sol", single("cosmos-reason2-8b"), quota=True),
        "E13": Experiment("e13-robobrain-cosmos32b", beside_32b("robobrain2.5-8b"), c32),
        "E14": Experiment("e14-cosmos8b-cosmos32b", beside_32b("cosmos-reason2-8b"), c32),
        "E15": Experiment("e15-cosmos32b-robobrain", c32, beside_32b("robobrain2.5-8b")),
        "E16": Experiment("e16-cosmos32b-cosmos8b", c32, beside_32b("cosmos-reason2-8b")),
        "E17": Experiment("e17-qwen30b-small", Serve("qwen3-vl-30b-a3b", 0.52, 16384),
                          Serve(best_small, 0.38, 16384, online_fp8=True), best=("best-small",)),
        "E18": Experiment("e18-sol-small", single(best_small), quota=True, best=("best-small",)),
        # the verifier's model is served apart unless the base already serves it (then second = local)
        "E19": Experiment("e19-base-robobrain", single(best_base) if best_base == "robobrain2.5-8b"
                          else Serve(best_base, 0.45, 16384),
                          None if best_base == "robobrain2.5-8b" else Serve("robobrain2.5-8b", 0.45, 16384),
                          best=("best-base",)),
        "E20": Experiment("e20-base-molmo", Serve(best_base, 0.55, 16384), Serve("molmo2-er", 0.33, 8192),
                          best=("best-base",)),
        # bench-selected replacements (evolve/agent_bench.py): Qwen3.8-27B plans, alone or with a pointer that may
        # also check done claims; on loki-gh200 (its FP8 build at 0.55 beside an 8B pointer in BF16)
        "Q38": Experiment("qwen38-27b-vllm", Serve("qwen3.8-27b", 0.90, 16384)),
        "Q38R": Experiment("c-q38-robobrain-v", Serve("qwen3.8-27b", 0.55, 16384), Serve("robobrain2.5-8b", 0.33, 8192)),
        "Q38M": Experiment("c-q38-mimo", Serve("qwen3.8-27b", 0.55, 16384), Serve("mimo-embodied-7b", 0.33, 8192)),
        "Q38L": Experiment("c-q38-molmo-v", Serve("qwen3.8-27b", 0.55, 16384), Serve("molmo2-er", 0.33, 8192)),
    }


# --- the GPU host ---
def expand(path):
    return os.path.expanduser(path)


def this_server(cfg, name=None):
    """The [agents] servers entry NAME, or this machine's by its private IP (one without ssh)."""
    if name:
        found = [s for s in cfg["agents"]["servers"] if s["name"] == name]
    else:
        ips = set(subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split())
        found = [s for s in cfg["agents"]["servers"] if s["ip"] in ips and "ssh" not in s]
    if not found:
        raise SystemExit(f"{name or socket.gethostname()} is none of [agents] servers")
    return found[0]


def on_host(server, argv):
    """ARGV as run on the GPU host: here, or through ssh."""
    return ["ssh", server["ssh"], shlex.join(argv)] if "ssh" in server else argv


def bind(server):
    """The address vLLM listens on: the private IP here, 127.0.0.1 behind a tunnel."""
    return "127.0.0.1" if "ssh" in server else server["ip"]


def client_ports(cfg, server):
    """The ports sessions reach the experiment's two model servers on."""
    return server.get("ports", cfg["agents"]["ports"])


def tunnel(cfg, server):
    """For an ssh server, forward its ip:ports here to the host's vLLM ports, for the worker's lifetime."""
    if "ssh" not in server:
        return None
    forwards = [arg for c, r in zip(client_ports(cfg, server), cfg["agents"]["ports"])
                for arg in ("-L", f"{server['ip']}:{c}:127.0.0.1:{r}")]
    process = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                                *forwards, server["ssh"]], stdin=subprocess.DEVNULL)
    time.sleep(5)
    if process.poll() is not None:
        raise SystemExit(f"the tunnel to {server['ssh']} did not open")
    return process


def api_key():
    return Path(expand("~/vlm/api_key")).read_text().strip()


def served(server, port):
    """(model, vLLM arguments) of the vLLM server on PORT of the GPU host, or None."""
    lines = subprocess.run(on_host(server, ["pgrep", "-af", f"vllm serve .*--port {port}( |$)"]),
                           capture_output=True, text=True).stdout.splitlines()
    for line in lines:
        words = line.split()
        if "--served-model-name" in words:
            return words[words.index("--served-model-name") + 1], " ".join(words[words.index("serve"):])
    return None


def command(serve: Serve, ip, port):
    model = MODELS[serve.model]
    args = [model.repo, "--served-model-name", serve.model, "--host", ip, "--port", str(port),
            "--max-model-len", str(serve.max_len), "--enable-prefix-caching", "--limit-mm-per-prompt",
            '{"image": 4}', "--gpu-memory-utilization", f"{serve.share:.2f}", "--max-num-seqs", "16",
            # no free whitespace in a JSON grammar: Cosmos-Reason2-32B looped on it until max_tokens
            "--structured-outputs-config",
            json.dumps({"backend": "xgrammar", "disable_any_whitespace": not model.whitespace}), *model.args,
            *(("--quantization", "fp8") if serve.online_fp8 else ())]
    return args


def ready(ip, port, name):
    request = urllib.request.Request(f"http://{ip}:{port}/v1/models", headers={"Authorization": f"Bearer {api_key()}"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return name in response.read().decode(errors="replace")
    except OSError:
        return False


def stop(server, port):
    subprocess.run(on_host(server, ["pkill", "-f", f"vllm serve .*--port {port}( |$)"]), check=False)
    for _ in range(60):
        if served(server, port) is None:
            return
        time.sleep(2)
    raise SystemExit(f"the vLLM server on port {port} did not stop")


def start(cfg, server, item, port):
    """Start ITEM's vLLM server on PORT of the GPU host, its output in [agents] logs there."""
    agents = cfg["agents"]
    vllm = f"{server.get('venv', agents['venv'])}/bin/vllm"  # a server may run its own vLLM
    args = ["serve", *command(item, bind(server), port)]
    logfile = f"{agents['logs']}/{item.model}-{port}.log"
    if "ssh" in server:  # the host's own key and HF_HOME (and env, e.g. a CUDA compat library); ~ unquoted
        assigned = ["HF_HOME=~/hf", "HF_HUB_OFFLINE=1", "VLLM_API_KEY=$(cat ~/vlm/api_key)",
                    *(f"{k}={v}" for k, v in server.get("env", {}).items())]
        line = (f"{' '.join(assigned)} setsid nohup {vllm} {shlex.join(args)} > {logfile} 2>&1 < /dev/null &")
        subprocess.run(["ssh", server["ssh"], line], check=True)
        return logfile
    env = {**os.environ, "HF_HOME": expand("~/hf"), "VLLM_API_KEY": api_key(),
           **{k: expand(v) for k, v in server.get("env", {}).items()}}
    with Path(expand(logfile)).open("w") as out:
        subprocess.Popen([expand(vllm), *args], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, start_new_session=True)
    return logfile


def serve(cfg, server, wanted):
    """Make the GPU serve exactly WANTED ({port: Serve}): servers already serving the same are kept, the rest stopped
    and started, the bigger first, each waited for until it answers (through the tunnel for an ssh server)."""
    agents, ports = cfg["agents"], cfg["agents"]["ports"]
    outside = dict(zip(ports, client_ports(cfg, server)))
    for port in ports:
        now, want = served(server, port), wanted.get(port)
        if now and (want is None or " ".join(command(want, bind(server), port)) not in now[1]):
            log(f"stopping {now[0]} on {port}")
            stop(server, port)
    for port, item in sorted(wanted.items(), key=lambda kv: -kv[1].share):
        if served(server, port):
            log(f"keeping {item.model} on {port}")
            continue
        log(f"starting {item.model} on {port} ({item.share:.2f} of the GPU)")
        logfile = start(cfg, server, item, port)
        until = time.monotonic() + agents["ready_seconds"]
        while not ready(server["ip"], outside[port], item.model):
            if served(server, port) is None:
                raise RuntimeError(f"{item.model} exited while starting: see {logfile}")
            if time.monotonic() > until:
                raise RuntimeError(f"{item.model} not ready after {agents['ready_seconds']} s: see {logfile}")
            time.sleep(10)
        log(f"ready {item.model} on {port}")


# --- profiles and runs ---
def profile_name(eid):
    return f"demo-pp-{eid}"


def settings_name(eid):
    return f"episode_settings_pp_{eid}.toml"


def write_settings(cfg, eid, exp, server, picks):
    """The experiment's episode settings: the demo-pp ones with its agent and model servers."""
    base = (HERE / cfg["e2e"][cfg["agents"]["base_profile"]]["settings"]).read_text()
    ports = client_ports(cfg, server)
    local = f"http://{server['ip']}:{ports[0]}/v1"
    second = f"http://{server['ip']}:{ports[1]}/v1" if exp.second else local
    agent = (f'\n# {eid} (evolve/agent_experiments.py)\n[agent]\nmodel = "{exp.agent}"\nlocal_url = "{local}"\n'
             f'second_url = "{second}"\nbest_small = "{picks[0]}"\nbest_base = "{picks[1]}"\n')
    lines, out, done = base.splitlines(), [], False
    for line in lines:  # the agent keys join the base's [agent] table
        out.append(line)
        if line.strip() == "[agent]" and not done:
            out += agent.split("[agent]\n", 1)[1].splitlines()
            done = True
    (HERE / settings_name(eid)).write_text("\n".join(out if done else lines + agent.splitlines()) + "\n")


def toml(value):
    """VALUE as a TOML value (JSON strings, numbers and booleans are TOML's)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {toml(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(toml(v) for v in value) + "]"
    return json.dumps(value)


def write_overlay(cfg, table, ids):
    """evolve/config.local.toml: this machine's work folder and a demo-pp-<id> profile per experiment (demo-pp's
    keys with its own workdir, settings, agent and sessions per host)."""
    agents, base = cfg["agents"], cfg["e2e"][cfg["agents"]["base_profile"]]
    lines = ["# written by evolve/agent_experiments.py: this machine's loop work folder and experiment profiles",
             "[local]", f"work = {toml(expand(agents['work']))}", 'proxy = ""', ""]
    for eid in ids:
        own = {"workdir": profile_name(eid), "settings": settings_name(eid), "agent_model": table[eid].agent,
               "parallel": agents["parallel"]}
        keys = {k: v for k, v in base.items() if k != "suite"} | own
        lines += [f"[e2e.{profile_name(eid)}]", *(f"{k} = {toml(v)}" for k, v in keys.items()), "",
                  f"[e2e.{profile_name(eid)}.suite]", *(f"{k} = {toml(v)}" for k, v in base["suite"].items()), ""]
    (HERE / LOCAL_CONFIG).write_text("\n".join(lines))


def summary_row(eid, exp, work):
    """The experiment's row from its baseline's archive entry."""
    entry = next(json.loads(line) for line in (work / "e2e" / profile_name(eid) / "archive.jsonl").read_text()
                 .splitlines() if line.strip() and json.loads(line)["status"] == "baseline")
    row = {"id": eid, "agent": exp.agent, "commit": entry["commit"]}
    placed = total = 0
    for split in ("train", "val"):
        s = entry[split]
        row |= {f"{split}_placed": s["placed"], f"{split}_success": s["success_rate"],
                f"{split}_minutes": s["minutes"], f"{split}_credits": s["credits"], f"{split}_disturbed": s["disturbed"],
                f"{split}_provider_errors": sum(n for k, n in s["outcomes"].items() if "provider_error" in k)}
        for layout in s["placed_by_layout"].values():
            got, out_of = (int(n) for n in layout.split("/"))
            placed, total = placed + got, total + out_of
    row["all_placed"] = round(placed / total, 3) if total else 0.0
    return row


def run(cfg, server, eid, exp, picks, commit):
    wanted = {cfg["agents"]["ports"][0]: exp.local, **({cfg["agents"]["ports"][1]: exp.second} if exp.second else {})}
    serve(cfg, server, wanted)
    write_settings(cfg, eid, exp, server, picks)
    log(f"{eid}: baseline of {exp.agent} at {commit[:10]}")
    result = subprocess.run([sys.executable, "-m", "evolve.e2e_loop", "--profile", profile_name(eid), "baseline",
                             commit, "--id", "b000"], cwd=REPO, check=False)
    if result.returncode:
        raise RuntimeError(f"{eid}: the baseline exited {result.returncode}")
    return summary_row(eid, exp, Path(expand(cfg["agents"]["work"])))


def claim(shared, eid):
    try:
        (shared / "claims" / eid).mkdir(parents=True)  # atomic on the shared volume: one worker per experiment
        return True
    except FileExistsError:
        return False


def next_experiment(cfg, server, ids, table, shared):
    """The next unclaimed experiment, one that keeps this GPU's servers first."""
    now = [served(server, port) for port in cfg["agents"]["ports"]]
    keeps = [eid for eid in ids if [s.model if s else None for s in (table[eid].local, table[eid].second)]
             == [n[0] if n else None for n in now]]
    for eid in keeps + [e for e in ids if e not in keeps]:
        if not (shared / "rows" / f"{eid}.json").exists() and claim(shared, eid):
            return eid
    return None


def worker(args):
    cfg = config()
    server, shared = this_server(cfg, args.server), Path(cfg["agents"]["shared"])
    gpu_gb = server.get("gpu_gb", GPU_GB)
    table = experiments(args.best_small, args.best_base, gpu_gb)
    picks = (MODELS[args.best_small].agent, MODELS[args.best_base].agent)
    ids = args.ids or [eid for eid, exp in table.items() if not exp.blocked(gpu_gb)]
    for eid in ids:
        if eid not in table:
            raise SystemExit(f"unknown experiment {eid}")
        if table[eid].blocked(gpu_gb):
            raise SystemExit(f"{eid} is blocked: {table[eid].blocked(gpu_gb)}")
    commit = git("rev-parse", args.commit or "HEAD")
    write_overlay(cfg, table, ids)
    (shared / "rows").mkdir(parents=True, exist_ok=True)
    (shared / "failed").mkdir(parents=True, exist_ok=True)
    forward = tunnel(cfg, server)
    while (eid := next_experiment(cfg, server, ids, table, shared)) is not None:
        log(f"{server['name']} takes {eid}")
        try:
            row = run(cfg, server, eid, table[eid], picks, commit) | {"server": server["name"]}
            (shared / "rows" / f"{eid}.json").write_text(json.dumps(row, indent=2))
            log(f"{eid}: placed {row['all_placed']} (train {row['train_placed']}, val {row['val_placed']})")
        except (RuntimeError, OSError, StopIteration, KeyError) as error:
            (shared / "failed" / f"{eid}.txt").write_text(f"{server['name']}: {error}\n")
            log(f"{eid} failed: {error}")
    if not args.keep:
        for port in cfg["agents"]["ports"]:
            if served(server, port):
                stop(server, port)
    if forward is not None:
        forward.terminate()
    log(f"{server['name']}: no experiment left")


def cell(item: Serve | None) -> str:
    if item is None:
        return "-"
    return (f"{item.model} ({MODELS[item.model].precision}{', FP8 online' if item.online_fp8 else ''}, "
            f"{item.share:.2f})")


def plan(args):
    gpu_gb = args.gpu_gb
    table = experiments(args.best_small, args.best_base, gpu_gb)
    print(f"GPU {gpu_gb:g} GB")
    print("| ID | Agent | Local server | Second server | Weights GB | GPU GB | Status |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for eid, exp in table.items():
        weights, gpu = exp.memory(gpu_gb)
        print(f"| {eid} | {exp.agent} | {cell(exp.local)} | {cell(exp.second)} | {weights:.1f} | {gpu:.0f} | "
              f"{exp.blocked(gpu_gb) or 'runnable'} |")


def summary(args):
    shared = Path(config()["agents"]["shared"])
    rows = [json.loads(p.read_text()) for p in sorted((shared / "rows").glob("*.json"))]
    keys = ("id", "agent", "all_placed", "train_placed", "val_placed", "train_success", "val_success",
            "train_disturbed", "val_disturbed", "train_minutes", "val_minutes", "train_credits", "val_credits",
            "train_provider_errors", "val_provider_errors", "server")
    print("| " + " | ".join(keys) + " |\n|" + " --- |" * len(keys))
    for row in rows:
        print("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    for failed in sorted((shared / "failed").glob("*.txt")):
        print(f"failed {failed.stem}: {failed.read_text().strip()}")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--best-small", default="robobrain2.5-8b", choices=SMALL, help="the Phase-1 8B pick")
    parser.add_argument("--best-base", default="robobrain2.5-8b", choices=SMALL, help="the verifier experiments' base")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").add_argument("--gpu-gb", type=float, default=GPU_GB, help="the GPU's memory")
    sub.add_parser("summary")
    work = sub.add_parser("worker")
    work.add_argument("ids", nargs="*")
    work.add_argument("--commit", help="the version sessions run (default: this checkout's HEAD)")
    work.add_argument("--keep", action="store_true", help="leave the model servers running at the end")
    work.add_argument("--server", help="the [agents] servers entry to drive (default: this machine's)")
    args = parser.parse_args(argv)
    {"plan": plan, "summary": summary, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
