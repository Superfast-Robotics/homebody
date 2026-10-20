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
