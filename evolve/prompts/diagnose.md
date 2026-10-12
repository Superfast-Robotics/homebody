You are diagnosing why a humanoid robot's skills fail, to plan the next code changes.

Read {workspace}/BRIEF.md first, then {workspace}/results.md, {workspace}/history.md and the traces
in {workspace}/traces/ (run_summary.log has each skill's result line; skill_events.jsonl has the
skills' stages, plans, results and traces). The code is in the current directory; start from
src/homebody/skills/ and src/homebody/toolkit/API.md. Do not read anything outside these two
places.

Group the failed runs into failure MECHANISMS: a cause in the code that explains several runs, not
one run's symptom. For each, find where in the code it arises and what change would remove it.
Rank them by how many stage points fixing them would likely gain across the train tasks, and say
which you would expect to generalise to unseen scenes of the same families. Learn from history.md:
do not re-propose what was rejected unless you say what is different.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", each with:
**Evidence** (tasks, runs, quoted result or trace lines), **Cause** (file and function, and why),
**Fix direction** (the smallest change you would make), **Tier** (0, 1 or 2, from BRIEF.md) and
**Expected gain**.
