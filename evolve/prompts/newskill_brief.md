# Brief: make the coding assistant better at creating new skills

HomeBody drives a simulated Unitree G1 humanoid (MuJoCo; Dex3 three-finger hands; a head RGB-D camera) through
skills. When no skill does what a task needs, the agent asks the coding assistant ({coder_model}, in Codex) to
**create a new skill**: one call, starting from the new-skill template (`TEMPLATE` in
`src/homebody/session/coder.py`), with a name, a goal and a success test. The assistant writes
`skills/<name>.py`, checks it and runs it on the robot itself. Its standing instructions are
`src/homebody/harness/coder_agents.md`; its request and the reports of its `check`/`execute` tools come from
`src/homebody/harness/coder.py`; it builds on `src/homebody/toolkit/` (reference: `toolkit/API.md`), `motion/`,
`helpers/` and the built-in skills (`skills/navigate.py`, `pick.py`, `place.py`, which a new skill may call). You
are improving the assistant's ability to create working skills: more of them succeed, at less cost and time.

## The benchmark

{benchmark}

`results.md` lists every train episode: the requested skill, the assistant's result and message, the final stage
and the seconds. `coder_edits.md` summarises what the assistant wrote across episodes (by function), with the
diffs of the calls that ended OK, and lists the **recurring fixes**: functions it wrote in several episodes that
ended successfully, strong candidates for a tested building block. `episodes/<id>/` holds each episode's
result.json, trace.json and `coder-call-1.md` (its request, actions, tool calls, code and answer). `history.md`
lists earlier versions and their verdicts.

## How a version is judged

On held-out episodes you never see as well as the train episodes, compared episode by episode with the current
version: accuracy first (the value over train and val together must rise by more than 1.5 standard errors, and
neither split may fall by more than two), then cost (the assistant's credits per episode down by a fifth at
unchanged accuracy), then time. A version that costs or takes over 30% more needs a larger accuracy gain. **The
validation episodes ask for kinds of skill the train episodes never ask for, and the separate test set (never used
here) asks for yet other kinds**: a change helps only if it makes creating skills easier in general, not if it
solves the train kinds.

## What you may change, and how

Make skill creation better in GENERAL ways only. Good changes are:
- **building blocks** a new skill can call: tested functions in `toolkit/` (documented in `toolkit/API.md`),
  `motion/` or `helpers/` for what new skills keep needing (moving the hand along a path in contact with the
  table, holding a pose, measuring an object's pose and footprint from depth, verifying an outcome from fresh
  sensing), especially what the assistant keeps writing by hand (see the recurring fixes);
- **the new-skill template** (`TEMPLATE` in `coder.py`): a better starting structure (stages, a plan dataclass,
  verification, failure codes);
- **the assistant's instructions** (`coder_agents.md`): how to design and test a new skill within one call, which
  building blocks and built-in skills to reuse, what to check before spending an execute, when to stop;
- **what the assistant is shown** (`coder.py`): the request and the check/execute reports.

The built-in skills are fixed here (a new skill may call them). Never add anything about particular scenes,
objects, colours, coordinates, episodes or the benchmark's skill names (a screen rejects source that names the
benchmark's scenes, objects or episodes), and every change must address a mechanism seen across several episodes.
Skills reach the robot only through the skill `Context`; never use simulator ground truth. The contact model and
the robot are fixed. Skill code and what it imports is loaded under an import allowlist (see
`session/skill_module.py`: numpy, scipy, math and other pure standard modules, and `homebody` skills, motion,
helpers, toolkit, harness, primitives); `time`, `os` and similar are refused, and `tests/test_coder.py` checks the
built-in skills still load that way.

Already in place: every execute answer has `what_changed` (base motion, each hand's closure and finger effort, what
each hand holds), and an execute is refused unless it repeats the checked arguments on the current view.
