# Brief: improve the harness that robot programs are written against

HomeBody drives a simulated Unitree G1 humanoid (MuJoCo; Dex3 three-finger hands; a head RGB-D
camera). On every task, a model (astra) WRITES a Python program `run(env)` against a harness and
runs it: it gets the task, a scan's scene memory, the head image and the harness's API reference,
replies with a program, the program is dry-run on a saved simulation, the harness reports back,
and after up to 4 attempts one program is executed for real and scored from physics.

You are improving that HARNESS, not writing task programs. A better harness lets the program
author succeed more often on tasks it has never seen: clearer and more capable functions, better
feedback after a dry run, a better prompt. The harness is `src/homebody/harness/`:
- `env.py`: the `env` object programs call (perception, base, HomeBody's skills, arm, hand, planning),
- `API.md`: the reference the author reads (it must match env.py exactly),
- `prompt.md`: the author's instructions,
- `feedback.py`: what a dry run reports back (text and images),
- `__init__.py`: how the prompt is assembled.
It builds on `src/homebody/toolkit/` (also yours) and on HomeBody's skills, motion and helpers.

## How a harness version is judged

Tasks come from three procedurally generated families, randomised in layout, sizes, heights and
colours: objects in an open bin to load into a carton; cups (and plates, trays) on a counter to
return to a shelf level; objects close to a backsplash to move into a side basket. Each targeted
object scores a stage: 0 untouched, 1 lifted, 2 ended in the target region, 3 physics-verified
placement. A version's score is the mean stage / 3 over tasks and repeats.

`results.md` and `runs/` show the train tasks with the current harness: per run, each attempt's
reply (the program) and report, the result, and images. Acceptance also uses held-out tasks you
never see: other draws of the same families and a real scanned kitchen with a four-object task.
A version is accepted only if neither the train nor the held-out score drops and at least one
rises. The final harness will be used, unchanged, on other scenes. So improve the harness in
general: never add anything about particular scenes, entity names, colours or coordinates, and
never weaken a check to make a task look done.

## What you may change (evolve/tiers.toml)

- Tier 0, change freely: `src/homebody/harness/`, `src/homebody/toolkit/`, `tests/harness/`.
- Tier 1, with a stated reason: HomeBody's skills, motion, helpers, the cuRobo server, other tests.
- Tier 2, only with a clear case: the robot interface, backend, walking, settings, anything else.
- Frozen, never: `evolve/` (the inner runner, the attempt budget, the scorer), `evaluation/`,
  `assets/`, the scene generator. A patch touching them is rejected unseen.

The harness reaches the robot only through the skill `Context` (the Robot interface): camera
frames, measured robot state, commands. Never use simulator ground truth (object poses from the
physics engine, the scorer, scene files) in the harness. Make the smallest change that addresses
the problem; keep API.md in exact agreement with env.py. You cannot run the simulator here; lint
and the unit tests run on every candidate before the suite, and a failure rejects it.
