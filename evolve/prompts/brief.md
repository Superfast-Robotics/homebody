# Brief: improve the harness the robot agent and its coding assistant work with

HomeBody drives a simulated Unitree G1 humanoid (MuJoCo; Dex3 three-finger hands; a head RGB-D
camera). On every task, a model (astra) acts as the agent: each turn it sees the head image and
three derived images, the scene memory from a scan, the recent steps and the instructions, and
replies with one action: a skill call (navigate, pick, place), `call_coder`, or `done`.
`call_coder(skill, goal, tried, success, constraints)` hands one skill to a coding assistant (Sol,
running in Codex). The assistant gets a workspace with the skill's current source, the task's
helper library (`lib/`) and the current camera view; it edits the skill module, dry-runs it with the
`check` tool and runs it on the robot with the `execute` tool (a few minutes, a few checks, at most
two executes), then answers success, failed or infeasible with the reason, the state it left the
world in and how to call the skill. A version that executed with OK replaces that skill for the rest
of the task. Coded skills and `lib/` do not carry over between tasks.

You are improving the HARNESS that makes this work: what the agent and the assistant are told and
given, not the skills of any one task. The harness is `src/homebody/harness/`:
- `agent_prompt.md`: the agent's instructions opening every turn (when to call the coder, how to ask),
- `coder_agents.md`: the assistant's standing instructions (its AGENTS.md: workspace, protocol, how to work, when to stop),
- `coder.py`: the assistant's REQUEST.md and how check and execute report back to it,
- `env.py` and `API.md`: higher-level primitives a skill may build on (API.md must match env.py exactly),
- `__init__.py`: how these enter the prompts.
It builds on `src/homebody/toolkit/` (also yours, with its API.md) and on HomeBody's skills, motion
and helpers. Prompt tuning and expanding or tuning the primitives are the expected kinds of change.

## How a harness version is judged

Tasks come from three procedurally generated families, randomised in layout, sizes, heights and
colours: objects in an open bin to load into a carton; cups (and plates, trays) on a counter to
return to a shelf level; objects close to a backsplash to move into a side basket. Each targeted
object scores a stage: 0 untouched, 1 lifted, 2 ended in the target region, 3 physics-verified
placement. A version's score is the mean stage / 3 over tasks and repeats.

`results.md` and `runs/<task>/<repeat>/` show the train sessions with the current harness:
`transcript.md` is every decision and result of the agent, and each `coder-call-N.md` is one
assistant session: the request, what the assistant did, every check and execute with its answer,
the change it made to the skill and its final answer.
Acceptance also uses held-out tasks you never see: other draws of the same families and a real
scanned kitchen with a four-object task. Each run's value is half its stage (/3)
and half its task success, so clean completions count most. A version is accepted only if, compared
scene by scene with the current harness, neither split's value falls by more than two standard errors
and at least one rises by more than one and a half, confirmed by a second evaluation.
Accuracy comes first, then cost, then time: with accuracy unchanged within that band, a version is
also accepted when its credits per run (agent decisions plus coding-assistant calls, which dominate)
fall by at least a fifth, or, at no higher cost, when its minutes per run fall by at least a fifth.
So a change that makes the assistant's calls fewer, shorter or more often successful is valuable
even when the stage scores stay the same. The final harness is
used, unchanged, on other scenes. So improve the harness in general: never add anything about
particular scenes, objects, colours or coordinates, and never weaken a check to make a task look done.

## What you may change (evolve/tiers.toml)

- Tier 0, change freely: `src/homebody/harness/`, `src/homebody/toolkit/`, `tests/harness/`.
- Tier 1, with a stated reason: HomeBody's built-in skills, motion, helpers, the cuRobo server, other tests.
- Tier 2, only with a clear case: the robot interface, backend, walking, settings, anything else.
- Frozen, never: the agent loop and the `call_coder` mechanism with its tools and limits
  (`src/homebody/session/`, `src/homebody/vlm/`), `evolve/`, `evaluation/`, `assets/`, the scene generator. A patch touching
  them is rejected unseen.

The harness reaches the robot only through the skill `Context` (the Robot interface): camera
frames, measured robot state, commands. Never use simulator ground truth (object poses from the
physics engine, the scorer, scene files). Make the smallest change that addresses the problem; keep
API.md in exact agreement with env.py. You cannot run the simulator here; lint and the unit tests
run on every candidate before the suite, and a failure rejects it.
