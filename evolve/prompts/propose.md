You are improving the harness a robot agent and its coding assistant work with, to remove one
mechanism that makes their per-task skills fail.

Read {workspace}/BRIEF.md first: it says how harness versions are judged and what you may change.
The diagnosis of the current harness is below; address **Mechanism {mechanism}** only. The sessions
it cites are under {workspace}/runs/. Edit files in the current directory (a git checkout of the
current version); do not read or write anything outside it and {workspace}.

Make the smallest harness change that removes the mechanism, in the lowest tier that can. Keep it
general: nothing tied to particular scenes, entity names, colours or coordinates. Keep
src/homebody/harness/API.md in exact agreement with env.py (every documented call exists with the
documented signature). Add or update unit tests under tests/harness/ for logic worth pinning.
Do basic testing of your change only (the simulator is not available, and the full suite runs
later on the remote): run the unit tests you added or changed and the tests of the modules you
touched, with `MUJOCO_GL=egl PYTHONPATH=src:vendor {python} -m pytest -q -x -p no:cacheprovider
<test files>`, plus `tests/harness`; run `{ruff} check src tests`; fix every failure and finding.
Compare floats with a tolerance, not equality. Do not run the whole suite or anything long. If you
change names other code uses (skills, ui, tests), keep their users working.

When done, reply with a short note in this form:
Mechanism: <title>
Change: <what you changed, file by file>
Tier and reason: <highest tier touched; for tier 1 or 2, why a lower tier could not do it>
Expected effect: <how the agent and its assistant should now succeed more often, on which tasks>
Risks: <what could get worse>

<diagnosis>
{diagnosis}
</diagnosis>
