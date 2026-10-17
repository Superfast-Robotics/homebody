You are improving the coding assistant's ability to create new skills, to remove one failure mechanism.

Read {workspace}/BRIEF.md first: it says how versions are judged and what you may change. The diagnosis is below;
address **Mechanism {mechanism}** only. The episodes it cites are under {workspace}/episodes/. Edit files in the
current directory (a git checkout of the current version); do not read or write anything outside it and
{workspace}. You may change only: {editable}.

Make a GENERAL change that helps create kinds of skill the train episodes do not ask for: a tested building block
(documented in toolkit/API.md), a better new-skill template, clearer instructions in coder_agents.md, or better
requests or reports in coder.py. Never name particular scenes, objects, colours, coordinates, episodes or the
benchmark's skill names. Keep the built-in skills working. Add or update unit tests for logic worth pinning. Do basic
testing of your change only (the simulator is not available here): run the unit tests you added or changed and the
tests of the modules you touched, with `MUJOCO_GL=egl PYTHONPATH=src:vendor {python} -m pytest -q -x -p
no:cacheprovider <test files>`, plus tests/harness and tests/test_coder.py, and `{ruff} check src tests`; fix every
failure and finding. Compare floats with a tolerance. Do not run the whole suite or anything long.

When done, reply with a short note in this form:
Mechanism: <title>
Change: <what you changed, file by file>
Why it generalises: <which recurring mechanism it addresses, and why it helps unseen kinds of skill>
Expected effect: <how new skills should now succeed more often, or more cheaply>
Risks: <what could get worse>

<diagnosis>
{diagnosis}
</diagnosis>
