You are diagnosing why the coding assistant fails to create working new skills on the benchmark episodes.

Read {workspace}/BRIEF.md first, then {workspace}/results.md, {workspace}/coder_edits.md and
{workspace}/history.md, then the episodes under {workspace}/episodes/<id>/ (result.json, trace.json,
coder-call-1.md). The code is in the current directory: the assistant's instructions
src/homebody/harness/coder_agents.md, its request, template and reports src/homebody/session/coder.py and
src/homebody/harness/coder.py, the building blocks under src/homebody/toolkit/ (toolkit/API.md), motion/ and
helpers/, and the built-in skills under src/homebody/skills/. Do not read anything outside these places.

Find the MECHANISMS that keep new skills from working, each across many episodes and several kinds of skill: how
the assistant designs a skill, what it writes by hand again and again, which building blocks it lacks or misuses,
where its skills fail on the robot (motion, contact, perception, verification), where it runs out of time, checks
or executes, and what information it lacked. Separate what a better building block, template, instruction or
report could fix from what the robot physically cannot do. Rank mechanisms by how many episodes fixing them would
turn into successes, and above all by how well the fix would carry over to kinds of skill the train episodes do
not ask for. Learn from history.md: do not re-propose what was rejected unless you say what differs.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", numbered in order of
expected impact (Mechanism 1 the highest): proposals are made for the first mechanisms only. Each section has:
**Evidence** (episode ids across kinds, counts, quotes from traces and assistant sessions), **Cause** (the missing
building block, the template, instruction or report at fault), **Fix direction** (a general building block,
template, instruction or report change) and **Expected gain**.
