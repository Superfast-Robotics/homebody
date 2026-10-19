You are restructuring the built-in `{skill}` skill ({source}): the last rounds of small fixes have
not been accepted, so this candidate may change more than one mechanism at once.

Read {workspace}/BRIEF.md first: it says how versions are judged and what you may change. Then
{workspace}/lessons.md (what earlier versions taught: do not repeat a change that already failed unless
you say what differs) and the inspiration programs under {workspace}/inspirations/ (other scored versions,
with their code against the current best: borrow what worked). Edit files in the current directory, a git
checkout of version {parent}; its own results on the train episodes are in {parent_results}. Do not read
or write anything outside the current directory and {workspace}. You may change only: {editable}.

The diagnosis (of the current best version, the current directory {parent}) is below. Address its top
mechanisms together with a coherent redesign of the stages they share (for example how grasps are chosen,
approached and confirmed), keeping the skill protocol (PROMPT, arguments, FAILURES, prepare / execute /
abort) and its refusal codes. Keep what lessons.md says works.

Make a general change: nothing tied to particular scenes,
objects, colours or coordinates; no simulator ground truth. Keep the other skills that use the shared
code working.
A built-in skill must stay loadable as the coding assistant's editable copy: its module may import only
what `ALLOWED` in src/homebody/session/skill_module.py lists (numpy, scipy, standard math and data
modules, homebody.skills, motion, helpers, toolkit, harness and primitives), never homebody.session,
backends or remote, and it reads no files; calibrated hand data reaches it as `planning.jaw_samples`.
tests/test_coder.py checks this.
Add or update unit tests for logic worth pinning. Do basic testing of your change only
(the simulator is not available here, and the full suite runs later on the remote): run the unit
tests you added or changed and the tests of the modules you touched, with
`MUJOCO_GL=egl PYTHONPATH=src:vendor {python} -m pytest -q -x -p no:cacheprovider <test files>`
(tests/test_skills.py covers the skills, tests/test_coder.py that they load as editable copies), and `{ruff} check src tests`; fix every failure and
finding. Compare floats with a tolerance. Do not run the whole suite or anything long.

When done, reply with a short note in this form:
Mechanisms: <titles>
Change: <what you changed, file by file>
Expected effect: <which episodes should now succeed, and why>
Risks: <what could get worse, including for other skills>

<diagnosis>
{diagnosis}
</diagnosis>
