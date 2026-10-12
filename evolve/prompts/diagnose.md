You are diagnosing the harness a robot agent customizes its skills with: where it makes it hard
for the agent to write and use per-task skills that complete tasks.

Read {workspace}/BRIEF.md first, then {workspace}/results.md and {workspace}/history.md, then the
sessions in {workspace}/runs/<task>/<repeat>/: transcript.md is every decision with its reasoning,
the full code of every customize_skill, and every result with what the skill printed; result.json
is the scored outcome. The harness is src/homebody/harness/ in the current directory, built on
src/homebody/toolkit/ and HomeBody's skills. Do not read anything outside these two places.

Find HARNESS MECHANISMS behind the failures, across several sessions. For example: the agent never
customizes when it should, or customizes too early; customized code repeatedly misuses a primitive
or rebuilds a capability that should be a primitive; a primitive fails without saying why; a result
hides what went wrong, so revisions guess; the reference does not match behaviour; the instructions
leave out something agents keep discovering the hard way. Separate these from task difficulty the
harness cannot change. Rank them by how many stage points fixing them would likely gain across
tasks, and by how well the fix would carry over to unseen scenes. Learn from history.md: do not
re-propose what was rejected unless you say what differs.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", each with:
**Evidence** (tasks, runs, steps; quote decisions, code or results), **Cause** (file and function in
the harness or below it), **Fix direction** (the smallest harness change: prompt text, a primitive,
the reference, the feedback), **Tier** (0, 1 or 2) and **Expected gain**.
