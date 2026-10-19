You are diagnosing why the dynamic `{skill}` skill (the built-in skill plus a coding assistant that
gets one call when the skill refuses) fails on the benchmark episodes.

Read {workspace}/BRIEF.md first, then {workspace}/results.md, {workspace}/coder_edits.md and
{workspace}/history.md, then the episodes under {workspace}/episodes/<id>/ (result.json, trace.json,
coder-call-1.md). The code is in the current directory: {source}, the
assistant's instructions src/homebody/harness/coder_agents.md and its request and reports
src/homebody/harness/coder.py, and the helpers under src/homebody/toolkit/ (toolkit/API.md),
motion/ and helpers/. Do not read anything outside these places.

Find the MECHANISMS that keep the dynamic skill from succeeding, each across many episodes and
several families: where the skill refuses, what the assistant then tries, what it rewrites by hand
again and again, which of its changes work and which never do, where it runs out of time, checks or
executes, and what information it lacked. Separate what a better helper, instruction, report or skill
structure could fix from what the robot physically cannot do. Rank mechanisms by how many episodes
fixing them would turn into successes (or make much cheaper), and by how well the fix would carry over
to unseen scenes. Learn from history.md: do not re-propose what was rejected unless you say what
differs.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", numbered in
order of expected impact (Mechanism 1 the highest): proposals are made for the first mechanisms
only, so the order decides what gets fixed. Each section has:
**Evidence** (episode ids across families, counts, quotes from traces and assistant sessions), **Cause**
(file and function, or the instruction or report at fault), **Fix direction** (a general helper,
instruction, report, structural change, or a general change to the skill's own decisions, such as making
permanent a fix the assistant keeps making by hand) and **Expected gain**.
