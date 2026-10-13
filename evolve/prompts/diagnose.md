You are diagnosing the harness a robot agent and its coding assistant work with: where it makes it
hard for them to get per-task skills that complete tasks.

Read {workspace}/BRIEF.md first, then {workspace}/results.md and {workspace}/history.md, then the
sessions in {workspace}/runs/<task>/<repeat>/: transcript.md is every decision of the agent with its
reasoning and every result; coder-call-N.md is one assistant session (request, actions, checks and
executes with their answers, the skill change, final answer); result.json is the scored outcome. The harness is src/homebody/harness/ in the current directory, built on
src/homebody/toolkit/ and HomeBody's skills. Do not read anything outside these two places.

Find HARNESS MECHANISMS behind the failures, across several sessions. For example: the agent never
calls the coder when it should, calls it too early, or asks vaguely (no evidence of what failed, no
clear success); the assistant misreads the request, spends its executes on untested ideas, stops
too early or too late, or repeatedly rebuilds something that should be a primitive or a toolkit
function; a check or execute report hides what went wrong, so the next change guesses; the
reference does not match behaviour; the instructions leave out something agents keep discovering
the hard way. Separate these from task difficulty the
harness cannot change. Rank them by how many stage points fixing them would likely gain across
tasks, and by how well the fix would carry over to unseen scenes. Learn from history.md: do not
re-propose what was rejected unless you say what differs.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", each with:
**Evidence** (tasks, runs, steps; quote decisions, code or results), **Cause** (file and function in
the harness or below it), **Fix direction** (the smallest harness change: prompt text, the assistant's
instructions, a primitive, the reference, a report), **Tier** (0, 1 or 2) and **Expected gain**.
