You are writing the first working version of the built-in `{skill}` skill for this robot.

Read {workspace}/BRIEF.md first: it says what the benchmark is, how versions are judged, what you may
change and how this robot differs from the one the current code was written for. `{source}` is, for
now, a copy of another robot's skill; {workspace}/results.md and {workspace}/episodes/ show what that
copy does on a sample of the train episodes here (its traces and results). Edit files in the current
directory (a git checkout); do not read or write anything outside it and {workspace}. You may change
only: {editable}.

Make `{skill}` work for this robot's hand: rewrite whatever in the copy assumes the other hand (grasp
poses and approach directions, closure, contact and hold checks, collision geometry, lift and
recovery), keeping the skill protocol (PROMPT, arguments, FAILURES, prepare / execute / abort) and
its refusal codes so the agent can call it the same way. Prefer reusing the copy's working structure
over starting from scratch. Nothing tied to particular scenes, objects, colours or coordinates; no
simulator ground truth.
A built-in skill must stay loadable as the coding assistant's editable copy: its modules may import only
what `ALLOWED` in src/homebody/session/skill_module.py lists (numpy, scipy, standard math and data
modules, homebody.skills, motion, helpers, toolkit, harness and primitives), never homebody.session,
backends or remote, and it reads no files; calibrated hand data reaches it as `planning.jaw_samples`
and the robot's hand state (`frame.hands[side]`).
Add unit tests for the new geometry and decisions. Do basic testing only (the simulator is not
available here; the full suite runs later on the remote): run the tests you added or changed and the
tests of the modules you touched, with
`MUJOCO_GL=egl PYTHONPATH=src:vendor {python} -m pytest -q -x -p no:cacheprovider <test files>`
(tests/test_skills.py covers the skills, tests/test_coder.py that they load as editable copies), and
`{ruff} check src tests`; fix every failure and finding. Compare floats with a tolerance.

When done, reply with a short note in this form:
Design: <how the skill now grasps with this hand>
Change: <what you changed, file by file>
Expected effect: <which episodes should now succeed, and why>
Risks: <what could still fail>
