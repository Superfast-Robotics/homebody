The premise needs updating: the archive contains **14 candidates, with `e004c4` accepted**, recorded at 18:16:36. It improved pooled stage value by **0.072 ± 0.028 SE**, with 14% less time. But warehouse holds remain **11/34 overall**: train falls 10→9, validation rises 1→2. The loop found a scoring improvement, not a warehouse breakthrough.

Sources: [archive.jsonl](/home/lohith/startup/homebody_claude/astra-evolve/skill/dex1-pick-wh/archive.jsonl), [loop.log](/home/lohith/startup/homebody_claude/astra-evolve/skill/dex1-pick-wh/loop.log). Everything below is read-only analysis; no files changed or simulator/network used.

**1. Why progress is poor**

**(a) Search and selection**

The acceptance threshold explains some rejection, but not most failure.

- `s001c2`, insertion recovery: **+0.026 ± 0.018**, just below 1.5 SE.
- `e003c2`, deeper seating: **+0.007 ± 0.036**.
- `e004c5`, route/posture search: **+0.033 ± 0.045**, costing 63% more time.
- `s002c2`: exactly zero score change.
- `e003c1/c5`: substantial regressions, **−0.171/−0.197**.

In [skill_score.py](/home/lohith/startup/homebody_claude/hb-demo/evolve/skill_score.py), acceptance uses paired stage differences across **76 episodes**, requiring 1.5 SE, or 2.5 SE when time/cost increases over 30%. For `e004c5`, its observed SE implies a required gain around **0.113**, versus 0.033 observed. That penalizes exploratory planning heavily.

This is not uniformly underpowered: three equal, isolated improvements without regressions can clear 1.5 SE. The difficult case is gains offset by regressions. Conversely, “accuracy unchanged” in the speed branch actually means **no statistically significant split regression**, not demonstrated equivalence; it could accept a faster, somewhat worse skill.

The objective is misaligned. `e004c4` changes train stages from **27/6/16** to **20/13/16** for stages 0/1/2: all net train value gain comes from partial credit. Warehouse is called the priority but supplies only **34/76 episodes**, with no extra weight.

Evolution has scarcely had a population to evolve. Both rounds follow rebaselines; lineage filtering leaves their initial parent pools effectively at `p002` and `p003`. Their plans confirm all focused candidates use those baselines. Crossovers borrow older programs measured under different camera/depth conditions. Their apparent complementarity is consequently confounded.

Four greedy candidates plus ten evolutionary candidates are a small search, mostly over insertion, clearance and verification. Rewrites bundle concurrent hypotheses before component results exist. UCB has too few observations, clips negative rewards to zero, and does not measure cost despite describing cheaper-model preference. Its sophistication exceeds its evidence.

Probes select eight episodes by outcome, warehouse first—not by mechanism activation. `fingerprint()` compares stage, code, message and rounded duration. Identical outputs **do not establish that code never ran**. Rejecting `e003c3/c4` as globally inert was unsupported; their benefit remains unknown.

**(b) Evidence supplied to models**

The workspace supplies train result tables, abbreviated plans/events, historical notes and later contact summaries. It does **not** supply replay images, depth clouds or full trajectories: `workspace()` copies only result/trace JSON, and collection excludes recordings. For example, basil’s plan contains `points: "array(8826, 3) float64"` and `approach_joints: "[32 items]"`. The model cannot inspect actual seating or reconstruct the route from that.

Diagnoser instructions also prohibit reading outside selected workspace/skill/helper locations, excluding backend, controller, calibration and benchmark implementation needed to distinguish physical limits from implementation assumptions.

`hand_contacts` is valuable but aggregates first/last contact, sample count and maximum force. It cannot alone establish continuous opposed contact, slip direction or causality. Earlier evidence was worse: `p003` exposed previously depth-invisible collision clutter, reducing train value **0.429→0.388** without changing holds.

Validation is not truly hidden: `history.md`, inspiration verdicts and lessons expose validation deltas. This is adaptive validation feedback, although episode details remain hidden. Lessons also turn “rejected” into “failed mechanism” too readily.

**(c) Skill architecture**

The evaluated baseline is **not** the copied top-down Dex3 planner still visible in this checkout. At commit `a54cb313ec`, Dex1 already searches forward/oblique approaches, pitches **0°, 30°, 60°**, four closure rolls, calibrated widths and insertion tiers.

Its limits are specific:

- Whole-visible-span fitting and centered placements restrict local grasp opportunities.
- It plans with open jaws and chooses the first routable candidate under heuristic ranking.
- Nominal insertion assumes commanded joints equal measured joints and a fixed torso.
- Recovery handles collision/target loss, but not insertion timeout.
- Possession verification relies on opposing effort/gap plus vision; it does not establish both jaws touch the target.
- Pick terminates before transport, while shared carry/navigation behavior is outside editable scope.

The accepted crossover isolates useful contact-overlap ranking from `s001c1`’s harmful settling requirement. That is evidence for **ablation**, not another indiscriminate verification gate.

**(d) Robot, controller and scene**

The [round-four diagnosis](/home/lohith/startup/homebody_claude/astra-evolve/skill/dex1-pick-wh/rounds/e004/diagnosis.md), checked against result/trace files, separates concrete failures:

- `whtrain_warehouse_picking_allulose_03_v0`: jaw 1 touches the target, jaw 2 `structure__observed_clutter_00385`; lift fails. Opposed resistance is not target enclosure.
- Seven insertion timeouts have **17–52 mm** position error, only **0.012–0.019 rad** joint-tracking error, and roughly **0.62–0.66 rad** remaining joint-limit margin. “Insufficient torque” or “joint saturation” is not established. Contact, servo conditioning and changing torso/target frames remain competing explanations.
- `pktrain_12_bin_to_carton_s0_v0` and warehouse bowl-1 variants have **36/24/26 reachable grasps but zero clear insertions**. Better ranking cannot manufacture a feasible candidate.
- Counter-clutter 05/08/20/23 fail the stance/range heuristic: **0.56–0.62 m** ahead versus estimated **0.48 m** reach. That does not prove impossibility from every stance.
- `pktrain_01_shelf_return_s0_v1` rises **0.168 m**, retains bilateral target contact, yet refuses verification.
- `whtrain_warehouse_picking_jam_07_v0` loses the object amid environmental contacts reaching **105.3 N**. Closure already reaches **1.12**; “squeeze harder” misses the problem.

The forearm-axis geometry and approximately 95 mm aperture constrain access. However, scene fidelity also needs auditing: `p003` renders simplified clutter boxes into depth while RGB shows scanned surfaces. Geometric mismatch is a plausible remaining issue, not demonstrated by these records. The prompt’s “about 16 N per jaw” also differs from the builder’s **20 N actuator limit**; command limit and realized contact force must be distinguished.

Dex3’s 0.37–0.53 on another scene distribution is not a controlled hardware comparison.

**2. Prompt review and concrete replacements**

Paths below are under [evolve/prompts](/home/lohith/startup/homebody_claude/hb-demo/evolve/prompts). Other prompt families in that directory are not selected by this profile.

| Prompt | Problem and exact replacement |
|---|---|
| `skill_brief.md` | Describes benchmark, scoring and scope, but hardcodes the wrong robot and scenes. Replace **“MuJoCo; Dex3 three-finger hands; a head RGB-D camera”** with **“MuJoCo; robot, sensors and calibration specified by the selected profile.”** Replace **“Each episode is one generated scene”** with **“The generated benchmark manifest defines scene families, counts, carry legs and rescue policy.”** |
| `skill_brief.md` | Shared-code permission contradicts the Dex1 profile. Replace **“Only `{source}` and the shared code under it”** with **“Only the profile’s explicit editable paths may change; all other code may be inspected.”** Replace the abbreviated judging paragraph with the actual rule, including the 30% slowdown/2.5-SE clause, and explicitly distinguish stage value from completed holds. |
| `skill_diagnose.md` | Requests causal grouping, but encourages gate-centric guesses and artificially narrow access. Replace **“Do not read anything outside these places”** with **“Inspect robot interface, controller, calibration and benchmark code read-only; simulator evidence is diagnostic only.”** Replace **“Fix direction (the smallest change)”** with **“Fix direction, competing explanations, missing evidence, activation conditions, and a falsifiable experiment.”** |
| `skill_propose.md` | Requests one mechanism and tests, but minimality encourages timid patches. Replace **“Make the smallest general change that removes the mechanism”** with **“Implement one testable causal intervention, including coordinated stage changes when required. State activation episodes, predicted trace changes, and regression controls; do not equate extra refusal with improvement.”** |
| `skill_init.md` | Requests hand adaptation but anchors architecture to Dex3. Replace **“Prefer reusing the copy’s working structure over starting from scratch”** with **“Reuse interfaces and validated primitives; derive grasp, insertion, extraction and recovery stages from this hand’s calibrated geometry.”** Add: **“Identify capabilities blocked by the interface rather than approximating them silently.”** |
| `evo_diagnose.md` | Adds lessons/inspirations but inherits diagnosis problems. Apply the diagnosis replacements above. Replace **“what earlier versions taught”** with **“historical hypotheses and measurements; separate lineage changes, uncertainty and causal evidence.”** Require parent-specific evidence where candidate parents differ. |
| `evo_diff.md` | Demands activation but threatens rejection using an invalid proxy. Replace **“a candidate that behaves exactly like its parent … is rejected”** with **“Instrument eligibility, branch execution and changed decisions. Select probes that exercise the intervention; unchanged final outcomes alone do not establish inactivity.”** Apply the single-intervention wording above. |
| `evo_crossover.md` | Assumes complementary outcomes identify complementary mechanisms. Replace **“bring … winning behaviour … without losing … successes”** with **“Identify separable causal components; remeasure donors on the current baseline, then test each component and their combination. Treat complementary outcomes as hypotheses.”** |
| `evo_rewrite.md` | Permits redesign but dictates combining top diagnoses. Replace **“Address its top mechanisms together”** with **“Choose one architectural bottleneck supported by evidence. Specify interfaces, staged experiments and component ablations before combining mechanisms.”** Replace **“Keep what lessons.md says works”** with **“Preserve independently verified behavior; challenge unsupported lessons.”** |
| `evo_lessons.md` | Compresses notes into causal conclusions and leaks validation. Replace **“‘inert’ means its change never ran”** with **“‘probe unchanged’ means sampled output fingerprints matched; activation is unknown unless instrumented.”** Replace **“changes that raised train or val value”** with **“train measurements, uncertainty, activation evidence and unresolved hypotheses; exclude validation feedback.”** |
| `bench_pick.md` | Says every refusal receives coder rescue, false here. Replace **“if it refuses, the assistant gets one call”** with **“Rescue occurs only when the profile enables it; report its budget explicitly.”** It is loaded but **not interpolated by `skill_brief.md`**, so currently changing it will not fix this profile’s brief. Wire it in or remove redundant loading. |
| Profile `brief_note` | Useful geometry, but stale provenance and unimplemented priority. Replace **“they began as copies”** with **“The evaluated skill is a Dex1-specific forward/oblique grasp search; inspect the evaluated commit.”** Replace **“Warehouse picking is the priority”** with **“Report warehouse and generated results separately; warehouse priority must be encoded in selection weights.”** Replace **“about 16 N per jaw”** with **“Read actuator limits from calibration/model; realized contact force depends on closure error and contact geometry.”** |

Keep the prohibitions on runtime simulator ground truth and scene-specific hardcoding.

**3. Missing capabilities, ranked by expected impact**

These rankings are engineering estimates, not measured gains.

1. **Joint grasp–stance–insertion–extraction planning.** Existing directional sampling is real but narrow; zero-clearance cases and extraction collisions demonstrate its limits. Add local contact-patch proposals, aperture-aware approach, alternative stances/postures and loaded extraction paths. Requires a bounded collision-aware planner over observed geometry, calibrated uncertainty and several weeks of integration.

2. **Target-aware enclosure and transport recovery.** Allulose’s split target/clutter contact and actual carry losses expose the gap. Add deployable tactile/force evidence where available, visual co-motion/slip estimation, bounded reseating/regrasp, and carry-path recovery. Requires interface/backend work plus hardware sensing validation—not exposing simulator object identities to skills.

3. **Measured controller diagnostics and recovery.** Ideal rollout does not predict live insertion. Log per-tick goal, torso, commanded/measured joints, progress and relevant controller limits; distinguish contact obstruction from conditioning before changing control. Instrumentation takes days; robust controller/planner integration longer.

4. **Active perception and predicate-level verification.** Shelf-01 is retained but unverified; allulose-02 has unexplained missing surfaces despite fresh observations. Add synchronized wrist/head evidence, individual verification predicates and bounded viewpoint/reacquisition actions. Requires sensor plumbing and tracking work, not looser thresholds.

5. **A valid benchmark and learning experiment.** Add independent object/layout samples, perturbations and object/scene-clustered uncertainty. Warehouse variants are correlated, not 34 independent objects. Score completed outbound/return navigation explicitly: currently `score_pick()` checks that `carry` exists, not that every leg succeeded. Freeze scene assets; reserve untouched final testing; use ablations and mechanism-targeted probes. This is days of harness work plus evaluation budget.

**4. Next three steps**

1. **Repair measurement before further optimization.** Freeze `p003` scene/sensor configuration; audit contact geometry, force calibration and carry completion scoring. Export synchronized visual/control evidence and intervention activation counters. Success: every representative failure can be assigned a supported cause or explicit unresolved hypothesis.

2. **Run controlled component experiments.** Start from the accepted overlap ranking. Independently test aperture-aware local grasps, alternative insertion/extraction routes, and possession recovery; rebase promising older components first. Success: activation is demonstrated, targeted held completions improve, and preserved controls remain successful. Unit tests alone do not qualify.

3. **Restart search against the actual objective.** Prioritize warehouse completed holds, retain stage value as diagnostic, expand independent scenes, isolate final testing, and separate development budgets from deployment latency limits. Success: repeatable warehouse hold gains across new objects/layouts, with explicit carry completion and acceptable runtime—not another increase in objects briefly lifted.