<h1 align="center">HomeBody: A Humanoid That Explores, Remembers, and Acts on Its Own</h1>

<p align="center">
  <a href="https://www.linkedin.com/in/gio-huh-789188374/">Gio Huh</a><sup>1,2</sup> &nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/caydengu/">Cayden Gu</a><sup>2</sup> &nbsp;&nbsp;
  <a href="https://www.takaratruong.com/">Takara E. Truong</a><sup>2</sup> &nbsp;&nbsp;
  <a href="https://profiles.stanford.edu/c-karen-liu">C. Karen Liu</a><sup>2,†</sup> &nbsp;&nbsp;
  <a href="https://guytevet.github.io/">Guy Tevet</a><sup>2,†</sup>
</p>

<p align="center">
  <sup>1</sup> Caltech &nbsp; <sup>2</sup> Stanford University &nbsp; <sup>†</sup> Equal advising
</p>

<p align="center">
  <a href="https://tml.stanford.edu/homebody/">
    <img src="https://img.shields.io/badge/Project_Page-8C1515?style=for-the-badge" alt="Project Page">
  </a>
</p>

<p align="center">
  <img src="docs/figures/overview-full.webp" alt="HomeBody overview: explore and collect scene data, reconstruct the kitchen, and deploy humanoid skills in the real world." width="100%">
</p>

### Code release schedule

| Target date | Release | Scope | Status |
|---|---|---|---|
| October 5, 2026 | **SIM** | Simulation environment with navigation and pick-and-place. | Released |
| October 16, 2026 | **REAL2SIM** | Exploration and scene-reconstruction pipeline. | Released |
| October 23, 2026 | **REAL** | Real-world deployment code and setup instructions. | Planned |

## Installation

**1.** Get the code and create a Python 3.11+ environment:
```bash
git clone https://github.com/Stanford-TML/homebody.git && cd homebody
conda create -n homebody python=3.11 -y && conda activate homebody
```

**2.** Install the package (it brings MuJoCo, PyTorch and the dashboard). Keep `-e`: it runs
from this folder's assets.
```bash
pip install -e .
```

**3.** Give it a model. Install and log in to one of the CLIs, or export an API key:
```bash
npm install -g @openai/codex && codex login               # for --model astra
npm install -g @anthropic-ai/claude-code && claude        # for --model opus, then /login
export OPENAI_API_KEY=...         # or ANTHROPIC_API_KEY=..., for the -api models
```

**4.** Check that everything is in place. `check` starts nothing, and `provider-check` sends the
model one short request:
```bash
homebody check
homebody provider-check --model astra
```

<details>
<summary>No GPU, or on a Mac? (SIM)</summary>

For SIM a GPU is optional: the walking policy runs on the CPU when CUDA is missing. Rendering is
chosen for you (`egl` on Linux with a GPU, `osmesa` on Linux without one, `cgl` on macOS).
Set `MUJOCO_GL` to override it. Linux without a GPU needs `sudo apt install libosmesa6`, and
`pip install torch --index-url https://download.pytorch.org/whl/cpu` before step 2 skips the
large CUDA download. SIM runs on macOS (Apple silicon) and Linux; REAL2SIM and REAL need the Linux/amd64
workstation with an NVIDIA GPU described next.
</details>

### 🔧 Robot hardware, for REAL2SIM and REAL

A Unitree G1 with its MID-360 LiDAR, and a Linux/amd64 workstation or laptop with an NVIDIA GPU (an
RTX 4080 or 4090 is preferred) and Docker, plugged into the robot's Ethernet. 3D-print the chest mount
(`hardware/g1_iphone_mount.stl`), fit your iPhone in it and connect the phone to the G1's Jetson with a
USB-C to USB-C cable. Put the robot's SSH user, IP and password and the phone stream in `configs/exploration.toml`
(`[robot]`, `[iphone]`), then install:

```bash
bash hardware/g1/install.sh          # or: --robot unitree@192.168.123.164 --password 123 --iface enp3s0
```

The [exploration guide](guides/exploration.md) and the [phone guide](guides/phone-camera.md)
have the details.

## Usage

### 💻 SIM

The HomeBody agent drives a Unitree G1 humanoid through a real2sim kitchen in MuJoCo, one
skill at a time, deciding from the robot's camera, its map and its memory of the scene.
The walking policy, arms, Dex3 hands and every object are simulated with full physics.
The simulation reproduces the long-horizon runs in the SRC Kitchen.

**Try it out in interactive mode.** Open http://127.0.0.1:8080, tell the agent what to do and press
Enter. The dashboard shows the camera, the agent's behavior tree and a 60-second monitoring
timeline; the 3D scene is a camera choice, served on the next port. It has no login, so keep
`--host` on localhost unless the network is trusted:
```bash
homebody --env src_kitchen --model astra
```

**Run one task without the dashboard:**
```bash
homebody --headless --env src_kitchen --model opus --task "Put the coffee bags on the island and throw away the used cartons."
```

**Run headless with scripted moves** (no model, four objects moved in one run):
```bash
python tools/check_manipulation.py --sequence configs/continuous_four.json --no-visuals
```

**After a run.** Every run is saved under `runs/session-<time>-<id>/`.

```bash
run=$(ls -d runs/session-* | tail -1)   # the newest run
python tools/timeline.py $run            # step-by-step decisions and results
python tools/score_session.py $run       # did every object end up in place
python tools/compose_cockpit.py $run     # a replay video, cockpit.mp4
```

<details>
<summary>Things to try</summary>

Type any of these into the dashboard, or pass one with `--task`. Each worked for both models
in every run (two runs per model, scored by `configs/waves/fun.json`).

- *The juice is for whoever sits at the round table nearest the recycling bin. Take it to them.*
- *Show the red and white carton to someone at the island, then put it back where it was.*
- *If the orange juice is next to the toaster, move it to the island; otherwise throw it away.*
- *Swap the two coffee bags between their counters.*
- *Throw away the carton that isn't orange juice.*

**Or give it a wrong memory.** Move an object before the run. The scan still describes and
places it where it was, so the model has to notice and go looking. This takes longer, often several
minutes of searching, and is the hardest test here. In recent runs Opus found the carton both
times, once after 29 minutes.
```bash
homebody --model opus --place-object juice_carton=3.19,3.42,-1.86 --task "Throw the orange juice carton in the recycling bin."
```
`--place-object` takes the object's map x and y in metres and its yaw in radians.
</details>

<details>
<summary>Models, environments and settings</summary>

**Models.** Pick one with `--model` (default `astra`). Each runs through its CLI with your
login, or with `-api` through the API with a key (e.g. `opus-api`). Only `astra` and `opus`
were validated, and smaller models fail more often.

| Codex (OpenAI) | Claude (Anthropic) |
|---|---|
| `astra`: GPT-6 Astra | `opus`: Claude Opus 5.5 |
| `sol`: GPT-5.6 Sol | `fable`: Claude Fable 5.1 |
| `luna`: GPT-5.6 Luna | `sonnet`: Claude Sonnet 5.5 |
| `terra`: GPT-5.6 Terra | `haiku`: Claude Haiku 4.5 |

**Environments.** Pick one with `--env <name>`. Each is a real2sim room, reconstructed from the real world,
in a folder under `assets/real2sim/`. For now there is `src_kitchen` (the default), a kitchen
with two coffee bags and two cartons, and more are coming soon.

**Settings.** Defaults are in `src/homebody/session/config.py`. `configs/simulation.toml` and
the scene's `settings.toml` override them, and `--settings my.toml` overrides those with
only the values it names.
</details>

<details>
<summary>Live monitor (watching a rollout)</summary>

`--settings configs/monitor.toml` turns on plan mode and the live monitor (`src/homebody/monitor/`). When a goal
of the agent's plan opens, Qwen3.8-27B (`qwen3.8-27b` on the LiteLLM proxy, key in `LITELLM_API_KEY`) writes the
actions the robot may be seen doing for it. While the robot works, the OpenAI Decisions API (`gpt-6-luna`, key in
`OPENAI_DECISIONS_API_KEY`, which nothing else reads) reads the newest head-camera frames and returns a probability
for each action, for `unsafe`, for `human_needed`, for whether each done goal still holds, and for progress. Calls
never overlap; the next one waits for the p95 latency. A failure the readings agree on for several calls in a row
stops the running skill (`MONITOR_STOPPED`), marks the goal in the plan, or reopens a done goal, and the agent
decides what to do next. `human_needed` pauses the run until **Resume**. On the dashboard each reading joins the
Automatic Monitoring timeline (risk, threshold, what the model saw), an abort or reopened goal raises an alert, and
resolving it dismisses the monitor's mark on the plan. Each task writes `monitor/readings.jsonl`.

```bash
python tools/monitor_probe.py decisions                     # latency and cost per call
python tools/monitor_record.py --drop-after 2 -- --sequence configs/continuous_four.json --no-visuals
python tools/monitor_bench.py runs/checks/manipulation_*/   # choose tau and ticks (configs/monitor_bench.toml)
```

</details>

<details>
<summary>Rollouts: export, import, evaluate, replay</summary>

A recorded session can travel as one `.tar.gz` bundle: the session's files plus `rollout.json`, which says what it
holds (sim or real, the scene, the tasks, whether it carries monitor logs or physics) and lists every file's
sha256. An import checks every file against the manifest and unpacks under `runs/imported/`. A rollout without
monitor logs (or one you want to judge with other settings) can be evaluated: the monitor runs offline over its
recorded frames and writes, beside the live logs, what it *would* have stopped, flagged or reopened. It prints the
number of calls and the cost first, and stops at `rollouts.offline_max_cost_usd`.

```bash
homebody rollout export runs/session-*             # light by default; --full keeps sensors.npz and depth
homebody rollout import session-*.tar.gz
homebody rollout --settings configs/monitor.toml evaluate runs/imported/session-* --dry-run
homebody rollout view [runs/session-*]             # a dashboard on recorded and imported rollouts
```

On every dashboard, **More → Rollouts** exports a session (to `runs/exports/`), imports a bundle by its path, and
evaluates with the monitor (the cost is shown before it runs). `homebody rollout view` also opens them: the task
appears as it ran, and clicking the timeline inspects any moment (camera frame, plan, alerts). Rollouts recorded
before plans were saved replay without the tree.

</details>

<details>
<summary>A harness server (start a rollout from another machine)</summary>

`homebody serve` runs the harness without a dashboard and listens on `server.host`:`server.port` (this machine
only by default; reach it through an SSH tunnel). `homebody connect` runs the dashboard and the monitor on your
machine: the harness sends them the camera (the view the page watches: robot, observer or hand), the plan, the
agent's decisions and skills, and the frames to watch, and they send back the operator's commands and the
monitor's stops and verdicts. The harness and its skills cannot tell the monitor is remote. A client needs the
server's token (`server.token_file`, made owner-only on first start). One client commands; others watch. When the
commanding dashboard goes away mid-task the run pauses (`server.on_disconnect`).

```bash
homebody serve --settings configs/monitor.toml                    # on the simulator host
ssh -N -L 8765:127.0.0.1:8765 sim-host &                          # here
scp sim-host:.homebody/server_token ~/.homebody/server_token      # keep it owner-only
homebody connect --url ws://127.0.0.1:8765 --settings configs/monitor.toml
```

`--backend real` is refused until the real-G1 backend arrives with the REAL release.

</details>

<details>
<summary>Three servers: simulation, harness and the monitor app</summary>

The physics can also run in its own server, [warehouse-sim](https://github.com/Superfast-Robotics/warehouse-sim)
(branch `homebody-servers`): it steps the scene in real time, runs the robot's balance and arm controller as a
plugin, renders the cameras and publishes them as a stream. A harness attaches to it with `--sim`, and the
monitor app watches every harness and simulation server named in one config and starts scenarios on them.

| Server | Command | Port |
| --- | --- | --- |
| Simulation | `python -m warehouse_sim.cli serve --profile P --cameras --evaluation --endpoint tcp://127.0.0.1:5720 --stream tcp://127.0.0.1:5721 --controller warehouse_homebody.plugin:make_plugin --controller-config C` | 5720 (RPC), 5721 (video) |
| Harness | `homebody serve --sim tcp://127.0.0.1:5720 --sim-profile P --sim-root HOMEBODY_EFFECTIVE_ROOT=E --scene S --settings configs/warehouse_sim.toml` | 8765 (`[server] port`) |
| Monitor app | `homebody monitor --config configs/servers_lambda.toml --settings configs/monitor_warehouse_sim.toml` | 8080 (the dashboard) |

A scene is exported for warehouse-sim with `warehouse_homebody.export` (its `profile.toml` and `effective.mjb`);
the harness and the simulation server read the same files, so they share a volume. Every port is loopback-only:
from another machine, tunnel the harness ports and the simulation ports to the `local_endpoint` / `local_stream`
the config names. The header's session chip (`harness → simulation server`) opens **Session configuration**: a
harness and a simulation server chosen from the config, each with its health, controller and video readiness;
**Apply selection** attaches them (refused while a task runs, and the active pair changes only when the harness
confirms the attachment), and **Refresh config** reads the config (`[[harness]]`, `[[sim]]`, `[[scenario]]`) again. Choosing a scenario above the instruction fills
it; sending runs it: its harness attaches to its simulation server (only when idle), the camera views switch to that
server's stream and the task is submitted. The monitor watches the frames from the stream.

The `demo_openmodels-server` and `demo-homebody-baseline-server` branches carry `homebody serve` (with `--sim`)
without the live monitor, so the monitor app can run scenarios on those harnesses too.

</details>

### 📷 REAL2SIM

Check, explore (open http://127.0.0.1:8099 and press **Start exploration**; **STOP** ends the run,
recorded under `runs/explore/`), then reconstruct the recording into a HomeBody scene with a coding agent:

```bash
homebody real2sim explore check
homebody real2sim explore --task "Explore this room." --iphone-input "$PHONE_STREAM_URL"
homebody real2sim reconstruct --exploration-data runs/explore/<run> --output assets/real2sim/my_room --task "Reconstruct the observed room"
homebody --env my_room                 # the new scene runs in SIM like src_kitchen
```

The coding agent reconstructs the scene from the capture on its own, following the
[HomeBody REAL2SIM skill](.agents/skills/homebody-real2sim/SKILL.md); the [REAL2SIM guide](guides/real2sim.md)
describes the scene package it delivers. Settings are the `[group]` tables in `configs/exploration.toml`,
overridden by `--settings my.toml`.

\*This release is an initial version of REAL2SIM, built on the vanilla open-source SuperOdometry; the real-world
rollouts use the full version with Amazon FAR's SuperOdometry.

### 🚀 REAL

Coming soon!

## Contributing

The skills, the grasp planner and the prompt are a working baseline with plenty of room to
grow, and every pull request and bug report helps.

<details>
<summary>How do I add a skill?</summary>

Write one module in `src/homebody/skills/`, for example `push.py`, and add its name to
`SKILLS` in `skills/registry.py`. The module declares

- `PROMPT`, one line starting with `push: ` that tells the model what the skill does and
  what arguments it takes
- `NEEDS`, the `Robot` commands it uses (`OPTIONAL` lists commands it uses when the backend has
  them), and `FAILURES`, the result codes it can return
- `prepare(arguments, selected, current, planning)`, which plans from one camera frame
- `execute(plan, ctx)`, which runs the plan and returns a `Result` (and `abort`, if it must
  undo something)

`navigate.py`, `pick.py` and `place.py` are full examples, and `tests/test_custom_skill.py`
builds a new skill, `wave`, from scratch and runs it. A skill with tunable numbers also adds a
settings group in `session/config.py` and `session/settings.py`. A SIM skill talks to the robot
only through `Robot`, so it runs unchanged on a real backend, and its prompt line describes the
robot, not one kitchen or task. The skills the exploration model can call are listed in
`real2sim/exploration/skills.py` and declare their commands against the live actions (`SURFACE`)
instead of `Robot`. Adding one takes four places: its module, its name in `LIVE_SKILLS`, its branch in
`vlm_pick.parse_decision` and its branch in `decision_interpretation.py`; the
[architecture guide](guides/architecture.md) shows where each piece goes.
</details>

<details>
<summary>How do I add a model?</summary>

A model from a service that is already supported (Codex, Claude Code, the OpenAI or Claude
API) is one line in `MODELS` in `src/homebody/session/config.py`.

A new service needs a provider, one class in `src/homebody/vlm/` with a `LABEL`, the
`EXECUTABLE` or API `KEY` it needs, and a `generate` method that sends the prompt and images
and returns a `Decision`. Add it to `PROVIDERS` in `vlm/catalog.py`. `vlm/claude.py` is a
short example.
</details>

<details>
<summary>How do I add a scene?</summary>

A scene is one folder under `assets/real2sim/`, run with `--env <folder>`. It needs
`scene.xml` (MuJoCo), `task_scene.json` (the robot's start and the movable objects),
`map.npz` (the navigation map), `physics.json` and `semantics.json` (the labels and
descriptions the model is given), and optionally a `settings.toml`. `src_kitchen/` is the
example. Follow the [scene build and validation steps](guides/real2sim.md)
for a new room. `tools/verify_assets.py` checks the repository's shipped manifest;
it does not certify a new scene's geometry or physics.
</details>

<details>
<summary>How do I add a robot backend?</summary>

Implement the `Robot` protocol in `src/homebody/primitives/robot.py`, nine members covering
time, camera frames, measurements and commands to the base, arms and hands.
`backends/mujoco.py` is the full simulator version; `backends/real.py` is an empty
placeholder. The G1 exploration driver lives in `backends/g1/`. A hardware backend also needs
capture timestamps, a deadman that stops the base, an operator stop and calibration
([CLAUDE.md](CLAUDE.md) has the list).
</details>

<details>
<summary>How do I report a bug?</summary>

Open an issue with the command you ran, what you expected and what happened. If it happened
in a run, attach the output of `python tools/timeline.py runs/session-<time>-<id>` and, if
you can, the replay video from `tools/compose_cockpit.py`.
</details>

<details>
<summary>What is in this repository</summary>

| Path | What it is |
|---|---|
| `src/homebody/session/` | The decision loop, settings and recording, for SIM and for exploration |
| `src/homebody/skills/` | The SIM skills (`navigate`, `pick`, `place`) and the skills exploration runs on the G1 |
| `src/homebody/primitives/` | The `Robot` interface, observation records and the live sensor and navigation primitives |
| `src/homebody/vlm/` | The model providers and the prompt |
| `src/homebody/backends/` | The MuJoCo robot and the G1 driver |
| `src/homebody/real2sim/` | Exploration (live map, decisions, capture), export and reconstruction |
| `src/homebody/motion/exploration/` | Gait, drive math and localization checks for the G1 |
| `hardware/g1/` | The G1 launcher and the SLAM, recorder and phone relay recipes |
| `configs/` | `simulation.toml` and `exploration.toml` settings, wave sets and the capture schema |
| `.agents/skills/homebody-real2sim/` | Capture audit and reconstruction workflow |
| `assets/` | The robot, its walking policy and the SRC Kitchen scene |
| `evaluation/` | The physics scorer |
| `tools/` | Checks, scoring, timelines, replays and scene building |

[CLAUDE.md](CLAUDE.md) explains how the code fits together, and the
[architecture guide](guides/architecture.md) shows the SIM and REAL2SIM layers.
</details>

## Bibtex

If you find this code useful in your research, please cite:

```bibtex
@misc{huh2026homebody,
  author = {Huh, Gio and Gu, Cayden and Truong, Takara E. and Liu, C. Karen and Tevet, Guy},
  title = {{HomeBody}: A Humanoid That Explores, Remembers, and Acts on Its Own},
  year = {2026},
  url = {https://tml.stanford.edu/homebody/}
}
```

## License

HomeBody is released under the [Apache License 2.0](LICENSE). Third-party components keep
their own licenses, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
