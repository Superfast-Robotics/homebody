# evolve: meta-optimising the harness robot programs are written against

Two loops, after Meta-Harness (arXiv 2603.28052) and Self-Harness (arXiv 2606.09498):

**Inner loop (`inner.py`, frozen; one task).** Astra gets the task, the scan's scene memory, the
head image and the harness's API reference (`src/homebody/harness/`), writes `run(env)`, which is
dry-run on a saved simulation; the harness reports back; after up to 4 attempts one program runs
for real and is scored from physics (0 untouched, 1 lifted, 2 in the target region, 3 placed).

**Outer loop (`loop.py`; the harness).** Each round:
1. **Workspace.** The best harness's train runs (programs, reports, images, results) become
   `rounds/rNNN/workspace/`, with a brief and the history. Held-out results never appear there.
2. **Diagnose.** Astra (high effort, read-only) finds harness mechanisms behind the failures.
3. **Propose.** For each of the top K mechanisms, astra patches the harness in its own worktree.
4. **Gate.** Tier = highest tier the patch touches (`tiers.toml`; the harness and toolkit are tier
   0); frozen files reject it; lint and unit tests (API.md must match env.py) run remotely.
5. **Evaluate.** The inner loop runs every train and held-out task with the candidate harness.
6. **Accept.** Neither split's score drops and one rises; the larger gain, then the smaller diff,
   wins. Every candidate is archived (`archive.jsonl`).

    python3 -m evolve.loop baseline
    python3 -m evolve.loop run --rounds 5
    python3 -m evolve.loop status

Train and held-out scenes come from `tools/scenegen.py`; the held-out set adds the scanned SRC
kitchen. The final harness is tested, unchanged, on scenes kept outside this repository.

## Skill loops (`skill_loop.py`, `evo_loop.py`)

One built-in skill (a profile's `name`, in its `source` module) is improved on a skill benchmark: scripted
episodes (`skill_bench.py`) that walk to a stance, call the skill once and score it from physics, with no model at
run time. A profile is a `[skills.<profile>]` table of `config.toml` (episodes, editable files, models, objective).

    python3 -m evolve.skill_loop --profile P init COMMIT          Astra writes the skill's first version (a new robot)
    python3 -m evolve.skill_loop --profile P baseline COMMIT [--id p001] [--videos]
    python3 -m evolve.skill_loop --profile P run --rounds N [--strategy greedy|evolve]
    python3 -m evolve.skill_loop --profile P remeasure ID ...     older versions' changes re-scored on the current best
    python3 -m evolve.skill_loop --profile P test ID | status

- **greedy** (`skill_loop.py`): diagnose the best version, one patch per mechanism, accept the largest clear gain.
- **evolve** (`evo_loop.py`, a profile's `strategy = "evolve"`): every scored version of the latest baseline's lineage
  is a possible parent (weighted by train score and few offspring, top versions and the Pareto front); each round
  writes focused diffs, a crossover of complementary versions and, after barren rounds, a rewrite; a UCB1 bandit picks
  each diff's model by gain per credit; lessons and inspiration programs go into every prompt; all candidates are
  evaluated in one pooled run (`evaluate_many`), each with its activation record (`mechanism_active` events).
- **Judging** (`skill_score.judge`): paired per episode against the best, cluster-robust standard errors (stance
  variants of a scene step are one cluster); no split may fall by 2 SE, the pooled score must rise by 1.5 SE (2.5
  when 30% slower). `objective = "held"` judges completed holds (placements) instead of stage value, `weights`
  weight episodes by id prefix, `speed_acceptance = false` turns off time- and cost-only wins. Lost, crashed or
  untimed episodes count as failures and block time wins.
- **Identity**: baselines record a manifest (candidate tree, profile, setup fingerprint of scenes, robot, episodes,
  scorer, planner and the seed's Python environment) before and after; rounds refuse to run when the setup changed
  (re-baseline); cached results are reused only for the same commit and setup.
- **Evidence** for the diagnosis: per-episode `hand_contacts`, `arm_track.json` (commanded, applied and measured
  arm joints) and the failed episodes' keyframes. Validation numbers never reach the proposer.
- **Services**: a profile's `services` (e.g. the Dex1 cuRobo planning servers) are started on every host first.

Benchmarks: `skill_episodes.py` (generated scenes), `warehouse_episodes.py` (front-row picks or places in the
warehouse scene packages, mixed with a sample of generated episodes). End to end: `warehouse_e2e.py` runs the
current checkout's agent sessions on the warehouse scenes. `review.py` runs one read-only Astra review of the loop;
past reviews are in `reviews/`.

## End-to-end loop (`e2e_loop.py`)

The same greedy and evolutionary search, judged by full agent sessions instead of skill episodes. A profile
(`[e2e.<P>]` in `config.toml`) names a task suite (train, val and test globs of scene packages with `sequence.json`
targets), sessions per task, the session settings and agent model, the hosts' services and the editable files.

    python3 -m evolve.e2e_loop [--profile P] baseline COMMIT [--id b000]
    python3 -m evolve.e2e_loop [--profile P] run --rounds N [--strategy greedy|evolve]
    python3 -m evolve.e2e_loop [--profile P] test CID       test tasks: only on this command
    python3 -m evolve.e2e_loop [--profile P] legacy CID     the original tasks, reported apart from test
    python3 -m evolve.e2e_loop [--profile P] status

An episode is one session of one task (`TASK__rN`), valued by the share of its target objects placed and settled in
a valid recording, less `disturbance_penalty` of an object per other product disturbed (`objective = "placed"`; stage
value reported). The loop scores with its own `evaluation/evaluate.py`, never the candidate's. A session its own code
ended (error status, nonzero exit, timeout) is a crash, charged its credits; one that could not be judged on its
merits (not run, provider failure, unscorable or invalid result) is an infrastructure failure: the evaluation stops
without a verdict and the same command resumes it, re-running only those sessions. Sessions of a task, and every task
whose name starts with one of the profile's `clusters` prefixes (the variants of one scanned scene), are one sample in
the standard error; val must also be shown non-inferior (`val_margin`, `val_z`, preregistered), and round winners are
exploratory until test. Each task-repeat runs on the same host for every version, and the planner services must
check ready, running this checkout's `tools/curobo_server.py`, before sessions start. One experiment id (suite and
its content, runner and scorer, settings and model, every host's environment and planner, scheduling and statistical
policy, profile; not the credit caps) guards baselines, parents and cached results; a baseline measures it again
after its sessions, a round before its first proposal and after its candidates' evaluation. A session whose usage
could not be read, or whose result is not a complete, type-checked scoring record with a known status, is never
judged: it is re-scored, then run again with its earlier attempt kept (`task/rN.attempt-K`, evidence and charges
kept, never read as the session's own failure); one with unmetered parts is judged and charged its worst case. A
persistent ledger per round (per command outside rounds) takes a worst-case reservation from every session (its
settings' decisions and coder calls at measured ceilings) and every Astra call before it starts, settles each at its
measured credits, and refuses what would pass the cap (`round_credit_budget`; outside a round the evaluation's own
allowance), across resumes. `rounds.json` records the round under
way before its first proposal, and its completion together with the stop decision (a first round with no accepted
candidate stops the run). The proposer works
in a source export (no git history, no suite generators, no loop code) and sees train sessions only; patches outside
the editable files, touching frozen files (`tiers.toml`) or adding system, file, network or settings access are
rejected unseen.

Isolation (`proposer_jail`, `session_sandbox`, both on): Astra's Codex calls run in a bubblewrap jail
(`astra.jailed`) that shows the system read-only, its source export and the round's folder (the train workspace and
its own notes), its tools, and a fresh Codex home holding only the credentials: no other round, result, archive,
checkout or Codex history. On the hosts each session and the candidate's tests run in bubblewrap (`remote.SANDBOX`,
`run_jobs.sh jailed`): the whole filesystem read-only except the session's own folder, the candidate's walking-policy
cache, /tmp (the planners' admission locks) and the coding sub-agent's home; the scorer runs outside from the loop's
copy. What remains: candidate code runs in the same process that records the session, so it can still alter its own
recording (physics.jsonl and outcome.json in the session folder) before the scorer reads it, and it can read files it
cannot write (the proxy key it is given anyway, other candidates). Closing that needs recording in a separate process
or user. The planner servers exit on a CUDA error (`tools/curobo_server.py`), and the loop watches them before and
every minute during a batch (`curobo_dex1_server.sh watch`), restarting any that is down or fails its CUDA check and
counting restarts in each version's `planner_restarts.txt`. Profile `demo-wh`: the Dex1 skills and the agent harness on the warehouse suite in
`dex1-demos/2026-10-20/warehouse_suite`. `curobo_load.py` is the planner load test that set its sessions per host.
