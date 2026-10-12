You are improving the harness a robot agent customizes its skills with, to remove one mechanism
that makes its per-task skills fail.

Read {workspace}/BRIEF.md first: it says how harness versions are judged and what you may change.
The diagnosis of the current harness is below; address **Mechanism {mechanism}** only. The sessions
it cites are under {workspace}/runs/. Edit files in the current directory (a git checkout of the
current version); do not read or write anything outside it and {workspace}.

Make the smallest harness change that removes the mechanism, in the lowest tier that can. Keep it
general: nothing tied to particular scenes, entity names, colours or coordinates. Keep
src/homebody/harness/API.md in exact agreement with env.py (every documented call exists with the
documented signature). Add or update unit tests under tests/harness/ for logic worth pinning. You
cannot run the simulator or the test suite here (no MuJoCo), but before you finish, run
`{ruff} check src tests` and fix every finding, and make sure every changed file parses
(`python3 -m py_compile <file>`). If you change names other code uses (skills, ui, tests), keep
their users working.

When done, reply with a short note in this form:
Mechanism: <title>
Change: <what you changed, file by file>
Tier and reason: <highest tier touched; for tier 1 or 2, why a lower tier could not do it>
Expected effect: <how the agent's customized skills should now succeed more often, on which tasks>
Risks: <what could get worse>

<diagnosis>
{diagnosis}
</diagnosis>
