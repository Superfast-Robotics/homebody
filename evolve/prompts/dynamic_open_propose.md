You are improving the dynamic `{skill}` skill (the built-in skill plus a coding assistant that gets one
call when it refuses) to remove one failure mechanism.

Read {workspace}/BRIEF.md first: it says how versions are judged and what you may change. The
diagnosis is below; address **Mechanism {mechanism}** only. The episodes it cites are under
{workspace}/episodes/. Edit files in the current directory (a git checkout of the current version);
do not read or write anything outside it and {workspace}. You may change only: {editable}.

Make a GENERAL change: a tested helper the assistant can call (documented in toolkit/API.md), clearer
instructions in coder_agents.md, better requests or reports in coder.py, or structure in the skill
(stages, extension points, failure details), or a change to the skill's own decisions (how it chooses
stances, ranks grasps, approaches, retries and verifies) when the diagnosis shows a general mechanism:
prefer rules derived from what the robot senses over new constants. Never name particular scenes,
objects, colours, coordinates or episodes. Keep the other skills that use shared code working. Add or update unit tests
for logic worth pinning. Do basic testing of your change only (the simulator is not available here, and
the full suite runs later on the remote): run the unit tests you added or changed and the tests of the
modules you touched, with `MUJOCO_GL=egl PYTHONPATH=src:vendor {python} -m pytest -q -x -p no:cacheprovider
<test files>`, plus tests/harness, and `{ruff} check src tests`; fix every failure and finding.
Compare floats with a tolerance. Do not run the whole suite or anything long.

When done, reply with a short note in this form:
Mechanism: <title>
Change: <what you changed, file by file>
Why it generalises: <which recurring mechanism it addresses, across which families>
Expected effect: <how the dynamic skill should now succeed more often, or more cheaply>
Risks: <what could get worse, including for other skills>

<diagnosis>
{diagnosis}
</diagnosis>
