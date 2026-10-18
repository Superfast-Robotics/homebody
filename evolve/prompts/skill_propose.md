You are improving the built-in `{skill}` skill to remove one failure mechanism.

Read {workspace}/BRIEF.md first: it says how versions are judged and what you may change. The
diagnosis is below; address **Mechanism {mechanism}** only. The episodes it cites are under
{workspace}/episodes/. Edit files in the current directory (a git checkout of the current version);
do not read or write anything outside it and {workspace}. You may change only: {editable}.

Make the smallest general change that removes the mechanism: nothing tied to particular scenes,
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
Mechanism: <title>
Change: <what you changed, file by file>
Expected effect: <which episodes should now succeed, and why>
Risks: <what could get worse, including for other skills>

<diagnosis>
{diagnosis}
</diagnosis>
