You are diagnosing a robot-programming harness: where it makes it hard for a model to write
programs that complete tasks.

Read {workspace}/BRIEF.md first, then {workspace}/results.md and {workspace}/history.md, then the
runs in {workspace}/runs/<task>/<repeat>/: NN_reply.md is the program the author wrote at attempt
NN, NN_report.md is what the harness reported back after its dry run, result.json is the executed
outcome, and the .jpg files are the images the author saw. The harness is
src/homebody/harness/ in the current directory, built on src/homebody/toolkit/ and HomeBody's
skills. Do not read anything outside these two places.

Find HARNESS MECHANISMS behind the failures: places where the API, its documentation, the prompt
or the dry-run report led authors astray or left them stuck, across several runs. For example: a
function that fails without saying why; a capability authors needed and had to improvise; a report
that hid what went wrong; documentation that does not match behaviour; a high-level skill that
fails on a recurring situation where a better primitive or composition would let authors succeed.
Separate these from task difficulty the harness cannot change. Rank them by how many stage points
fixing them would likely gain across tasks, and by how well the fix would carry over to unseen
scenes. Learn from history.md: do not re-propose what was rejected unless you say what differs.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", each with:
**Evidence** (tasks, runs, attempts; quote program or report lines), **Cause** (file and function
in the harness or below it), **Fix direction** (the smallest harness change), **Tier** (0, 1 or 2)
and **Expected gain**.
