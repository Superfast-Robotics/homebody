You are diagnosing why the built-in `{skill}` skill fails on the benchmark episodes.

Read {workspace}/BRIEF.md first, then {workspace}/results.md, {workspace}/history.md and {workspace}/lessons.md
(what earlier versions taught), then the inspiration programs under {workspace}/inspirations/ (other scored
versions: their notes, results and code against this one), then the
episodes under {workspace}/episodes/<id>/ (trace.json, result.json; result.json's `hand_contacts` lists what the robot's hand touched and when, from the recorded physics: evidence for you only, which the skill itself cannot sense this way). The skill's code is
{source} in the current directory, with the shared code it uses under
src/homebody/motion/, src/homebody/helpers/ and src/homebody/toolkit/ (toolkit/API.md is the robot
interface reference). Do not read anything outside these places.

Find the MECHANISMS behind the failures, each across many episodes: group episodes by refusal code
and by what the traces show (which stage it reached, what it replanned, why it gave up, whether the
object slipped during the lift or the carry), then trace each group to the code that decides it
(a gate that rejects good grasps, a reach limit that is too strict, a closure too weak to hold during
walking, a verification that misreads the view...). Separate what the skill could do better from what
the robot physically cannot do (an object out of the arm's reach from any stance). Rank mechanisms by
how many episodes fixing them would likely turn into successes, and by how well the fix would carry
over to unseen scenes. Learn from history.md: do not re-propose what was rejected unless you say what
differs.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", numbered in
order of expected impact (Mechanism 1 the highest): proposals are made for the first mechanisms
only, so the order decides what gets fixed. Each section has:
**Evidence** (episode ids, counts, quoted trace events and messages), **Cause** (file and function),
**Fix direction** (the smallest change) and **Expected gain** (how many episodes, which families).
