You are changing a humanoid robot's skill code to remove one failure mechanism.

Read {workspace}/BRIEF.md first: it says how versions are judged and what you may change. The
diagnosis of the current version is below; implement a fix for **Mechanism {mechanism}** only. The
traces it cites are under {workspace}/traces/. Edit files in the current directory (a git checkout
of the current version); do not read or write anything outside it and {workspace}.

Make the smallest change that removes the mechanism, in the lowest tier that can. Keep it general:
nothing tied to particular scenes, names, colours or coordinates. Record the decisions you add with
`trace(ctx, ...)` from homebody.toolkit. Add or update unit tests under tests/skills/ when the change
has logic worth pinning. You cannot run the simulator, but make sure the code at least parses
(`python3 -m py_compile <file>`).

When done, reply with a short note in this form:
Mechanism: <title>
Change: <what you changed, file by file>
Tier and reason: <highest tier touched; for tier 1 or 2, why a lower tier could not do it>
Expected effect: <which tasks should move, and how>
Risks: <what could get worse>

<diagnosis>
{diagnosis}
</diagnosis>
