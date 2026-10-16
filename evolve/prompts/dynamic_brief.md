# Brief: make the dynamic `{skill}` skill succeed more often

HomeBody drives a simulated Unitree G1 humanoid (MuJoCo; Dex3 three-finger hands; a head RGB-D
camera). The **dynamic `{skill}`** is the built-in `{skill}` skill plus a coding assistant: when the
skill refuses, the assistant ({coder_model}, in Codex) gets one call to change the skill's code and run it on the
robot itself (`src/homebody/harness/coder_agents.md` is its standing instructions; its request and the
reports of its `check`/`execute` tools come from `src/homebody/harness/coder.py`; it builds on
`src/homebody/toolkit/` (reference: `toolkit/API.md`), `motion/` and `helpers/`). You are improving
the dynamic skill as a whole, so that the skill plus its assistant succeed more often, at less cost
and time.

## The benchmark

Each episode is one generated scene (three families: an object in an open bin; a cup, plate or tray on
a counter; an object near a backsplash beside a basket). A scripted driver stands the robot at the
scene's stance (some episodes shifted by up to 8 cm), calls `{skill}` once on a click on the object; if
it refuses, the assistant gets one call ({coder_minutes} minutes, a few checks, two executes) limited to `{skill}`.
A held object is then carried along the task's carry path. Scored from physics: 0 not lifted, 1 lifted
but not held at the end, 2 held: success. The physics is deterministic, so every version runs exactly
the same episodes.

`results.md` lists every train episode: the skill's own result, the assistant's result and message,
the final stage and the seconds. `coder_edits.md` summarises what the assistant changed across
episodes and which changes worked, with the diffs of those that did, and lists the **recurring fixes**:
functions the assistant changed in several episodes that ended successfully. A fix it keeps rewriting by
hand is a strong candidate for a tested helper or an extension point. `episodes/<id>/` holds each
episode's result.json, the skill's trace.json and one `coder-call-1.md` per assistant session (its
request, actions, tool calls, code change and answer). `history.md` lists earlier versions and
their verdicts.

## How a version is judged

On held-out episodes you never see as well as the train episodes, compared episode by episode with the
current version: accuracy first (the value over train and val together must rise by more than 1.5
standard errors, and neither split may fall by more than two), then cost (the assistant's credits per
episode down by a fifth at unchanged accuracy), then time. A version that costs or takes over 30%
more needs a larger accuracy gain. A separate test set, never used here, decides at the end whether the
gains generalise.

## What you may change, and how

Make the dynamic skill better in GENERAL ways only. Good changes are:
- **helpers** the assistant (and the skill) can call: a tested function in `toolkit/` (documented in
  `toolkit/API.md`), `motion/` or `helpers/` that does, reliably, what the assistant keeps writing by
  hand or keeps getting wrong (see coder_edits.md);
- **the assistant's instructions** (`coder_agents.md`): how to approach a refused call, which helpers
  to try, what to check before spending an execute, when to stop;
- **what the assistant is shown** (`coder.py`): the request and the check/execute reports, e.g. the
  failed call's diagnostics in a form it can act on;
- **the skill's structure** (`skills/{skill}.py`): clear stages and extension points the assistant can
  override, and failure details that say what failed and by how much.

Do NOT tune the skill's own decisions (its thresholds, rankings, limits or verification rules): its
behaviour should change only through structure and the tools the assistant gets. Never add anything
about particular scenes, objects, colours, coordinates or episodes (a screen rejects source that names
the benchmark's scenes, objects or episodes), and every change must address a mechanism seen across
several episodes and families. The skill reaches the robot only through the skill `Context`; never use
simulator ground truth. The contact model and the robot are fixed. Skill code (`skills/{skill}.py` and
what it imports) is also loaded as a customised skill under an import allowlist (see
`session/skill_module.py`: numpy, scipy, math and other pure standard modules, and `homebody` skills, motion,
helpers, toolkit, harness, primitives); `time`, `os` and similar are refused, and `tests/test_coder.py`
checks that the built-in skills still load that way.

## Already in place, and a direction worth considering

From published work on coding agents for robot manipulation, two ideas are already built into the
harness: every execute answer has `what_changed` (base motion, each hand's closure and finger effort
before and after, and what each hand holds), and an execute is refused unless it repeats the checked
arguments on the current view (every motion starts a new view). Build on them rather than re-adding
them. One direction remains to weigh against the evidence: **promote recurring fixes**: turn a fix
from the recurring-fixes list into a tested helper or an extension point, so the assistant calls it
instead of rewriting it.
