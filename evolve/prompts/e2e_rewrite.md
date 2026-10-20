You are restructuring part of the code HomeBody agent sessions run on ({source}): the last rounds of small fixes
have not been accepted, so this version may redesign one whole stage.

Read {workspace}/BRIEF.md first: it says how versions are judged and what you may change. Then
{workspace}/lessons.md and the inspiration versions under {workspace}/inspirations/. Edit files in the current
directory, a git checkout of version {parent}; its own train sessions are in {parent_results}. Do not write anything
outside the current directory; read anything in it and in {workspace}. You may change only: {editable}.

The diagnosis (of the current best version, the current directory {parent}) is below. Choose one architectural
bottleneck the evidence supports (for example how a skill chooses, approaches and confirms grasps, how carries are
secured, or how the agent plans, verifies and recovers across a task) and redesign the stages it spans coherently,
rather than bundling fixes for several mechanisms, keeping the skill protocol (PROMPT, arguments, FAILURES, prepare
/ execute / abort), the refusal codes and the agent's skill set. Preserve behaviour that was measured to work;
challenge lessons that were only guessed.

Implement one testable causal intervention, with the coordinated changes it needs (not a minimal tweak that cannot
change an outcome): nothing tied to particular scenes, products, entity names, colours or coordinates; no simulator
ground truth. Keep src/homebody/harness/API.md in exact agreement with env.py. Prompts stay general (the robot and
how to work, never this warehouse). Extra refusals or earlier give-ups are not an improvement.
A built-in skill must stay loadable as the coding assistant's editable copy: its module may import only
what `ALLOWED` in src/homebody/session/skill_module.py lists (numpy, scipy, standard math and data
modules, homebody.skills, motion, helpers, toolkit, harness and primitives), never homebody.session,
backends or remote, and it reads no files; calibrated hand data reaches it as `planning.jaw_samples`.
tests/test_coder.py checks this.
Where new skill code changes a decision, make it report: `ctx.emit({{"type": "mechanism_active", "mechanism":
"<short name>", ...}})` with the values that decided it (sessions record it; whether it acted is recorded for the
diagnosis, not judged). Name, under Expected effect, the tasks and sessions where the change should act.
Add or update unit tests for logic worth pinning. Do basic testing of your change only (the simulator is not
available here; the full suite runs later on the remote): run the tests you added or changed and those of the
modules you touched, with `MUJOCO_GL=egl PYTHONPATH=src:vendor {python} -m pytest -q -x -p no:cacheprovider <test
files>` (tests/test_skills.py covers the skills, tests/harness and tests/test_coder.py the harness and the editable
copies), and `{ruff} check src tests`; fix every failure and finding. Compare floats with a tolerance. Do not run
the whole suite or anything long.

When done, reply with a short note in this form:
Mechanisms: <titles>
Change: <what you changed, file by file>
Expected effect: <which tasks and sessions should place more objects, and why>
Risks: <what could get worse>

<diagnosis>
{diagnosis}
</diagnosis>
