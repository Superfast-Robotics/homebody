# Brief: improve the HomeBody skills

HomeBody drives a simulated Unitree G1 humanoid (MuJoCo; Dex3 three-finger hands; a head RGB-D
camera) through skills: `navigate`, `pick`, `place` (`src/homebody/skills/`). Your job is to make
the robot succeed at tabletop transfer tasks by changing code, round after round.

## How a version is evaluated

Each task is a scripted transfer (`tools/check_manipulation.py`, read-only): navigate to a stance,
`pick` the named object, navigate to the destination, `place` it there. Stances and arguments come
from the task's sequence file; the skills decide everything else. The physics scorer then checks
that the object rests inside the destination and that nothing else was disturbed.

A run's stage is 0 (nothing), 1 (pick verified), 2 (place reported done) or 3 (physics-verified
success). A version's score is the mean stage / 3 over tasks and repeats.

Scenes are procedurally generated in three families, each randomised in layout, sizes, heights
and colours:
- objects standing in an open bin on a table, to load into an open carton on the same table;
- cups (and plates, trays) on a counter, to return to one level of an open shelf unit;
- objects close to a backsplash and to each other on a counter, to move into a side basket.

`results.md` and `traces/` show the train scenes. Acceptance also uses held-out scenes you never
see: other draws of the same families, and a real scanned kitchen where the current skills already
work. A version is accepted only if neither the train nor the held-out score drops and at least one
rises. So fix mechanisms, not scenes: never special-case object names, colours, coordinates or
scene ids, and do not break what works.

## What you may change (evolve/tiers.toml)

- Tier 0, change freely: `skills/navigate.py`, `skills/pick.py`, `skills/place.py`, a new
  `skills/lib/` package, tests under `tests/skills/`.
- Tier 1, change with a stated reason: other skill files, `toolkit/`, `motion/`, `helpers/`,
  `vlm/`, the cuRobo server, other tests.
- Tier 2, change only with a clear case: the robot interface, the backend, walking, settings,
  anything else. Its checks are the most expensive.
- Frozen, never: `evolve/`, `evaluation/`, `assets/`, the scripted runner and scorer, the scene
  generator. A patch touching them is rejected unseen.

Make the smallest change that fixes the mechanism. Keep each skill's name and `ARGUMENTS`
compatible with the scripted runner. Skills reach the robot only through `Context`/`Robot`, never
the simulator, and never use simulator ground truth. `src/homebody/toolkit/API.md` lists the
building blocks (IK diagnosis, collision gates with the hand's current shape, calibrated grasps,
cuRobo paths) and `trace(ctx, what, **data)`, which records a decision into the run's events:
trace the decisions you add, so the next round can see why a run went the way it did.

You cannot run the simulator here. Lint (ruff) and the unit tests run on every candidate before the
suite; a failure rejects it.
