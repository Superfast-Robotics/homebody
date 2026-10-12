# Brief: improve the harness the robot agent customizes its skills with

HomeBody drives a simulated Unitree G1 humanoid (MuJoCo; Dex3 three-finger hands; a head RGB-D
camera). On every task, a model (astra) acts as the agent: each turn it sees the head image and
three derived images, the scene memory from a scan, the recent steps and the instructions, and
replies with one action: a skill call (navigate, pick, place), `customize_skill`, or `done`.
`customize_skill(name, description, code)` writes Python `def run(env, **arguments)` against the
robot primitives (`env`), and that version replaces or adds the skill for the rest of the task; the
agent then calls it, reads its result and printed lines, and may revise it, with full code or with
find/replace `edits` to its current version. `read_skill(name)` shows any skill's current source (a
built-in's, or its own customized version) for one turn. Customized skills do not carry over between
tasks.

You are improving the HARNESS that makes those per-task customized skills work: what the agent is
told and given, not the skills of any one task. The harness is `src/homebody/harness/`:
- `agent_prompt.md`: the instructions opening every turn (when and how to customize, how to read results),
- `env.py`: the primitives a customized skill calls (perception by click, built-in skills, arm, hand, planning),
- `API.md`: the primitives' reference shown to the agent (it must match env.py exactly),
- `feedback.py`: how a customized skill's run is reported back,
- `__init__.py`: how instructions, the reference and the task's customized skills enter the prompt.
It builds on `src/homebody/toolkit/` (also yours) and on HomeBody's skills, motion and helpers.
Prompt tuning and expanding or tuning the primitives are the expected kinds of change.

## How a harness version is judged

Tasks come from three procedurally generated families, randomised in layout, sizes, heights and
colours: objects in an open bin to load into a carton; cups (and plates, trays) on a counter to
return to a shelf level; objects close to a backsplash to move into a side basket. Each targeted
object scores a stage: 0 untouched, 1 lifted, 2 ended in the target region, 3 physics-verified
placement. A version's score is the mean stage / 3 over tasks and repeats.

`results.md` and `runs/<task>/<repeat>/transcript.md` show the train sessions with the current
harness: every decision, the full code of every customization, every result and what it printed.
Acceptance also uses held-out tasks you never see: other draws of the same families and a real
scanned kitchen with a four-object task. A version is accepted only if neither the train nor the
held-out score drops and at least one rises, confirmed by a second evaluation. The final harness is
used, unchanged, on other scenes. So improve the harness in general: never add anything about
particular scenes, objects, colours or coordinates, and never weaken a check to make a task look done.

## What you may change (evolve/tiers.toml)

- Tier 0, change freely: `src/homebody/harness/`, `src/homebody/toolkit/`, `tests/harness/`.
- Tier 1, with a stated reason: HomeBody's built-in skills, motion, helpers, the cuRobo server, other tests.
- Tier 2, only with a clear case: the robot interface, backend, walking, settings, anything else.
- Frozen, never: the agent loop and the `customize_skill` mechanism (`src/homebody/session/`,
  `src/homebody/vlm/`), `evolve/`, `evaluation/`, `assets/`, the scene generator. A patch touching
  them is rejected unseen.

The harness reaches the robot only through the skill `Context` (the Robot interface): camera
frames, measured robot state, commands. Never use simulator ground truth (object poses from the
physics engine, the scorer, scene files). Make the smallest change that addresses the problem; keep
API.md in exact agreement with env.py. You cannot run the simulator here; lint and the unit tests
run on every candidate before the suite, and a failure rejects it.
