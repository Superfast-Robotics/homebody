You are diagnosing why HomeBody agent sessions fail to place the warehouse tasks' objects, to decide what to change
in the robot's Dex1 skills or the agent harness ({source}).

Read {workspace}/BRIEF.md first, then {workspace}/results.md, {workspace}/history.md and, when present,
{workspace}/lessons.md (historical hypotheses and train measurements: separate what was measured from what was
guessed) and the inspiration versions under {workspace}/inspirations/ (other scored versions: their notes, results
and code against this one). Then read the sessions under {workspace}/runs/<task>/<repeat>/: transcript.md (the
agent's decisions and every result), skill_calls.md and keyframes/ (open the images of failed stages),
coder-call-N.md, result.json. The code is in the current directory: the skills in src/homebody/skills/dex1/, the
harness in src/homebody/harness/, the shared code they use in src/homebody/motion/, helpers/ and toolkit/. You may
read, without changing anything, the rest of the repository (the agent loop in src/homebody/session/, the robot
interface and backend, calibration in assets/robot/) to tell what the robot physically cannot do from what the code
assumes. Simulator evidence (the scorer's reasons, contacts) is for diagnosis only: a fix may not read it.

Find the MECHANISMS behind the failures, each across many sessions and several tasks: group sessions by how they
ended and where objects stopped (never lifted, dropped while walking, released outside the target, the agent gave
up or ran out of steps, the coder failed or wasted the session's time), then trace each group to the code that
decides it: a skill's gate, grasp choice, closure, verification or release; the agent's instructions that lead it
to a wrong skill call, a premature finish or a vague coder request; a primitive or report that hides what went
wrong. Separate what the code could do better from what the robot physically cannot do. Rank mechanisms by how
many placed objects fixing them would likely add across tasks, and by how well the fix would carry over to unseen
layouts. Learn from history.md: do not re-propose what was rejected unless you say what differs.

Answer in Markdown, with at most four sections titled exactly "## Mechanism N: <title>", numbered in order of
expected impact (Mechanism 1 the highest): proposals are made for the first mechanisms only. Each section has:
**Evidence** (tasks, sessions, steps; quote decisions, results and code), **Cause** (file and function),
**Competing explanations** (and the evidence that would separate them), **Fix direction** (one testable
intervention in the skills or the harness, the sessions it should change and how) and **Expected gain** (how many
objects, which tasks).
