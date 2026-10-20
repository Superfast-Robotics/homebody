# Brief: make agent sessions complete warehouse tasks

HomeBody drives a simulated Unitree G1 humanoid in MuJoCo. In every session a language model (the agent) reads
a task ("pick one product of each kind ... and place each on ..."), sees the head camera and decides one skill
call per turn: navigate, pick, place, look, and `call_coder` (a coding sub-agent that customizes a skill for the
task). It plans the task as a behavior tree first. You improve the code those sessions run on, so that more of
the tasks' objects end up placed, across warehouse layouts no round has seen.

## The suite

Train tasks (shown here) and held-out tasks (never shown) are warehouse scenes: shelving racks with bays and
boards, rows of packaged products (cans, jars, trays, bowls, boxes), aisles to walk. Some layouts are procedurally
generated, others are variants of two scanned warehouse scenes (different target products, destinations, robot
starts). Each task runs {repeats} sessions per version. A session is scored from the recorded physics: per target
object a stage (0 untouched, 1 lifted, 2 in the target region, 3 placed and settled); a dropped object stays at 1.

`results.md` lists every train session: the share placed, the stage value, how it ended, each object's stage and
the scorer's reason, coder calls, credits and minutes. `runs/<task>/<repeat>/` holds `transcript.md` (every
decision of the agent with its reasoning and every skill result), `coder-call-N.md` (each coding sub-agent
session), `skill_calls.md` (each skill call's stages, plan outline and result), `keyframes/` for sessions that did
not place everything (head and wrist images at each stage of each skill call; open them), `result.json` and
`runner_errors.log` when the runner failed. `coder_edits.md` summarizes what the coder changed across sessions.
`history.md` lists earlier versions with their train verdicts; `lessons.md` and `inspirations/`, when present,
summarize what earlier versions tried.

## How a version is judged

By {objective}. Train and held-out sessions are compared with the current best version task by task: rejected
when either split falls by more than two standard errors, or when more than 5% of its sessions crash; accepted
when train and held-out together rise by more than one and a half standard errors (two and a half when sessions
get more than 30% slower or costlier). The sessions of one task, and all tasks built from one scanned layout,
count as one sample in the standard error. The agent is a language model, so sessions vary from run to run:
change what recurs across many sessions and tasks, in general ways. Never add anything about particular scenes,
products, entity names, colours or coordinates.

## What you may change

Only these paths: {editable}. Anything else (the agent loop, its prompt assembly and the call_coder mechanism in
src/homebody/session and src/homebody/vlm, the scorer, the scenes, the physics) is rejected unseen; you may read
the whole repository. Two kinds of change are open:

- **The robot's skills** (`src/homebody/skills/dex1/`: pick, transport, place for the Dex1 grippers): how grasps are
  chosen, approached, closed, verified, carried and released. A skill reaches the robot only through its `Context`
  (camera frames, measured robot state, commands); never use simulator ground truth (object poses from the physics
  engine, scene files, the scorer, contacts).
- **The agent harness** (`src/homebody/harness/`): the agent's turn instructions (`agent_prompt.md`), the planning
  prompt (`plan_prompt.md`), the primitives customized skills are written against (`env.py`, documented in
  `API.md`, which must match it exactly), the coder's instructions (`coder_agents.md`, `coder.py`) and the reports
  a skill result or coder change gives back (`change_report.py`). Prompts stay general: they describe the robot and
  how to work, never this warehouse or these tasks.

Credits count: a version whose sessions cost clearly more needs a clearly larger gain.
