**No-go for round 5 as written.** The changes improve the experiment, but do not yet make its measurements or planner trustworthy. A corrected baseline and controlled remeasurements are worthwhile.

References below are relative to `hb-demo`; `E/` means `/home/lohith/startup/homebody_claude/astra-evolve/skill/dex1-pick-wh/`. I used read-only inspection and arithmetic; no simulator, network, or file changes.

**1. Earlier findings, one by one**

| Earlier finding | Status and evidence |
|---|---|
| Partial credit rewarded lifts rather than completed carries | **Resolved for newly run ordinary pick episodes.** `evolve/skill_bench.py:103` requires outbound success before returning, records completion at line 107, and gates stage 2 at line 255. `evolve/config.toml:331` selects held outcomes. Legacy results remain contaminated: e004c4’s basil_03_v1 reports `STALL` at `E/results/e004c4/train/whtrain_warehouse_picking_basil_03_v1/result.json:26`, yet stage 2 at line 100. |
| Warehouse priority was rhetoric | **Resolved as weighting, not a warehouse-specific guarantee.** `evolve/config.toml:332` doubles warehouse weights; `evolve/skill_score.py:133` and `evolve/evo_loop.py:106` apply them to acceptance and parent fitness. Generated gains can still compensate warehouse regressions. |
| Speed acceptance did not establish unchanged accuracy | **Not resolved.** `evolve/skill_score.py:142` checks only significant split deterioration; lines 151–155 can then accept a faster, less successful skill. |
| Tiny population, confounded crossover donors, overcomplicated bandit | **Partly resolved.** Remeasurement exists (`evolve/evo_loop.py:250`); reward now accounts for proposal credits (line 183). However, crossover still considers *all* historical populations (line 230), including incompatible measurements. Negative gains remain clipped to zero (line 88); more bookkeeping has not created more evidence. |
| Identical probe outputs falsely established inactivity | **Partly resolved.** Named episodes and activation events are added (`evolve/evo_loop.py:375`, `:397`). Neither establishes causal activation; loopholes remain below. |
| Missing images, trajectories and controller evidence; overly narrow inspection | **Partly resolved.** Repository inspection is explicitly permitted (`evolve/prompts/evo_diagnose.md:9`). Workspace export adds arm tracks/images (`evolve/skill_loop.py:330`), but contains a concrete bug. Plans remain abbreviated (`evolve/skill_bench.py:436`); numerical depth clouds and full paths still are not exported. |
| Validation leaked through historical feedback | **Partly resolved, substantially incomplete.** `public_reason()` removes explicit validation deltas (`evolve/skill_loop.py:59`), but existing lessons, statuses, verdict basis and pooled timing still leak feedback. |
| Prompts encouraged timid patches, bundled rewrites and unsupported causal lessons | **Mostly resolved in wording.** Single interventions: `skill_propose.md:8`, `evo_diff.md:20`; competing explanations: both diagnosis prompts’ closing sections; component crossover: `evo_crossover.md:12`; architectural bottleneck: `evo_rewrite.md:10`; calibrated initialization: `skill_init.md:13`. Remaining contradictions below. |
| Wrong robot, scope, rescue and force descriptions | **Mostly resolved.** `skill_brief.md:3`, `:39`; `bench_pick.md:4`; `evolve/config.toml:353`. Benchmark interpolation is now wired (`evolve/skill_loop.py:306`). |
| Whole-span grasps, open-hand routing, nominal insertion, inadequate possession evidence, no transport recovery | **Not resolved by these commits.** They provide planning infrastructure, not a changed accepted skill. e004c4 still fits whole visible spans (`2400da5e:src/homebody/skills/dex1/pick.py:245`). cuRobo does not implement local contact proposals, loaded extraction or transport recovery. |
| Invisible collision clutter, incomplete contact evidence and uncertain physical causes | **Partly resolved.** Depth and segmentation render group 4 (`src/homebody/backends/mujoco.py:280`); contacts enter results (`evolve/skill_bench.py:542`). RGB still shows different geometry. Contacts remain eight aggregate pairs, not continuous bilateral target-contact evidence (`:482`). Torque, saturation and reach impossibility remain unproven. |
| Correlated scenes, inadequate generalization experiment; uncontrolled Dex3 comparison | **Not resolved.** Warehouse objects still receive two stance variants in the same scene packages (`evolve/warehouse_episodes.py:42`). No new independent layouts, clustered inference or controlled hardware comparison appears. |

**2. Bugs, statistical errors and loopholes**

**The new “applied reference” is not the applied reference.** `tools/check_manipulation.py:323` records `gate.q`; the controller subsequently effort-clamps that value before sending it (`src/homebody/backends/controller.py:71`, `:170`). This can falsely attribute stopping to downstream tracking. Record the actual sent reference, gains/limits, full torso pose and goal. Export is every approximately half-second (`evolve/skill_bench.py:543`), not per tick.

**Keyframes are selected using the wrong episode.** `evolve/skill_loop.py:333` uses stale `row` from the preceding table loop. For e004c4, the final sorted train row is successful tray_1_v1: **no episode gets images**. Use `train[episode]`. Also, retaining the first 40 filenames (`:365`) can discard late failure/carry frames.

**The weighted SE is not valid general inference for these priority weights.** `evolve/skill_score.py:115` multiplies a once-weighted residual variance by a Kish factor. An independent-episode HC1 estimate instead uses:

\[
SE^2=\frac{n}{n-1}\frac{\sum_i w_i^2(d_i-\bar d_w)^2}{(\sum_i w_i)^2}.
\]

Even that requires replacing episode independence with an appropriate object/layout sampling model. With 34 warehouse and 42 generated episodes, one isolated warehouse gain yields current SE **0.01631**, versus HC1 **0.01810**; one generated gain yields **0.01159**, versus **0.00914**. The error changes direction.

**Binary holds need a power plan.** Weighting reduces Kish effective size to about **68**, before correlations. For orientation, 76 independent, equally weighted pairs with 20% discordance give SE ≈0.051: thresholds require approximately **7.7 or 12.8 percentage points**, and approximate 80%-power effects are **12 or 17 points**. A few clean wins can pass—the unweighted formula permits three at 1.5 SE—but gains mixed with losses are difficult. Repeated five-candidate selection also invalidates a simple one-shot significance interpretation (`evolve/evo_loop.py:312`).

**Missing episodes can disappear from the denominator.** Collection marks splits complete without verifying the manifest (`evolve/skill_loop.py:218`); scoring reads existing result files and intersects IDs (`evolve/skill_score.py:31`, `:108`). Process death, timeout or failure before `result.json` can silently remove hard episodes. Require every expected result or fail the evaluation.

**Activation is self-certified.** Any `mechanism_active` event passes, including inherited events unrelated to the new intervention (`evolve/evo_loop.py:397`). Changing a message or duration also escapes “inert” (`:94`). Named probes match anywhere in notes, not just Expected effect (`:375`). Missing instrumentation does not prove inactivity. Require intervention-specific eligibility/decision evidence; keep unchanged probes informational.

**Validation remains visible.** Existing `E/lessons.md:2`–7 contains validation and pooled gains and is copied verbatim (`evolve/evo_loop.py:198`). `describe_train()` retains validation-dependent basis and pooled timing (`evolve/skill_score.py:177`); statuses and old notes persist. `evo_lessons.md:3` therefore falsely calls these train-only verdicts, and line 4 retroactively attributes activation instrumentation to old probes. Rewrite/sanitize historical evidence before round 5; reserve genuinely untouched final testing.

**Ground truth is intentionally exposed to the proposer, but not newly wired into runtime observations.** Contact identities and arm tracks are privileged training diagnostics. Runtime receives rendered depth/segmentation, including collision proxies—not deployable perception validation. Prompts prohibit misuse, but filename/string screening is no security boundary (`evolve/skill_loop.py:393`), and built-in skills run in-process. Audit generated code for backend access and scene-derived rules.

**cuRobo has material integration gaps:**

- **Global stale locks:** the client configures only when the server says unconfigured (`src/homebody/helpers/curobo.py:54`). Later episodes inherit the first configuration’s waist; the idle arm is assumed at rest, both hands open (`:44`). Concurrent initial pings can trigger redundant configurations. Make posture/configuration identity part of each atomic planning request.
- **Serialization and timing:** one REP socket processes requests sequentially (`tools/curobo_server.py:295`). At 28 episodes/host, 1.5–2.5-second service time implies roughly **42–70 seconds** for a burst, excluding initialization and extra calls. Timeout is 300 seconds (`src/homebody/session/config.py:156`), so this is not automatically a timeout—but pooled wall-time penalties become scheduling-dependent. The “1–3 s” prompt promise is false under load.
- **Frames/units:** the world→pelvis transform is coherent (`helpers/curobo.py:60`); poses/depth use metres, arm joints radians, Dex1 slide locks metres. But `env.plan_path` takes **torso-frame** poses and target objects, whereas direct `plan` takes **world-frame** poses and label IDs (`harness/env.py:227`). Document this distinction explicitly.
- **Collision claims exceed guarantees:** sphere grids leave box corners uncovered (`tools/build_dex1_hand.py:224`); depth occupancy treats unseen space as free and erases obstacles near starting spheres (`tools/curobo_server.py:181`, `:187`). There is no attached-object model. `env.follow` thins paths and disables live collision checking (`harness/env.py:242`, `:252`), so interpolation and tracking invalidate the original path’s clearance. Revalidate executed segments; do not advertise loaded extraction as supported.

Other prompt contradictions: `evo_rewrite.md:2` permits multiple mechanisms while line 12 forbids bundling; `skill_brief.md:32` claims unchanged score without equivalence; `evo_crossover.md:13` still explicitly tolerates stale donor measurements.

**3. My earlier guide, revised**

Keep completed carries as the selection objective, but **drop the idea that changing the objective alone repairs selection**. Add clustered uncertainty, explicit noninferiority for speed wins, complete manifests and an untouched final set.

Replace “build a collision-aware planner” with **validate this planner’s posture, geometry, execution and concurrency contract first**. Start with one bounded approach intervention; defer loaded extraction until attachment geometry exists.

Retain component ablations and richer diagnostics. Stop treating activation counts as evidence of benefit, and stop describing internal gate state as applied control. Treat old lessons as hypotheses requiring reconstruction.

**4. Go / no-go and required fixes**

- **New baseline: conditional go.** First create an explicit integration commit containing e004c4 (`2400da5e`) plus the infrastructure changes. **HEAD is not e004c4**: current `src/homebody/skills/dex1/pick.py:245` uses the copied candidate generator, lacking accepted overlap ranking. Conversely, baselining the old commit alone omits updated candidate-side helpers/tools. `baseline()` evaluates exactly its supplied commit (`evolve/skill_loop.py:233`). Use a fresh result ID.
- **Before baseline:** fix missing-result handling, actual-reference logging and keyframe selection. Freeze/hash manifests, settings, robot XML/URDF/meshes, scorer and server configuration; the current fingerprint covers only selected XML/JSON/calibration files (`:240`). Verify deployed server identity; its external launcher is not in this repository.
- **Remeasurements: conditional go afterward.** Port each listed candidate independently onto that integrated baseline. `remeasure` currently skips frozen checks, screening and tests (`evolve/evo_loop.py:250`); add them and handle conflicts explicitly. These measure component transfers, not original historical programs. Restrict donors to matching experiment identities.
- **Round 5: no-go** until statistics, speed noninferiority, validation sanitization, probe semantics and planner execution/concurrency are repaired. Then run controlled ablations before another five-way evolutionary selection.