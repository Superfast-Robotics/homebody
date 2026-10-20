**Conditional go for p004 and controlled remeasurements; no-go for unrestricted round 5.** The fixes materially improve measurement, but several claimed repairs are incomplete.

References are relative to `hb-demo`; `E/` denotes the supplied evidence directory. I used read-only inspection and arithmetic only. I assume deployed servers match the commit, as requested.

**1. Required fixes from the update review**

| Required fix | Assessment |
|---|---|
| Integrate accepted e004c4 with robot infrastructure | **Fixed.** `demo-pick-p004` is `8691be0`, following `397cb74` and `2400da5`. Its diff from `2400da5` changes eight infrastructure/configuration files and leaves `src/homebody/skills/dex1/pick.py` unchanged, including accepted overlap ranking. |
| Missing-result handling | **Partly fixed.** `evolve/skill_loop.py:168` synthesizes crashed failures for missing job results before marking completion (`:237`). However, existing malformed/unscored results and setup failures can still disappear through `evolve/skill_score.py:35`, and pairing still intersects IDs (`:117`). Require exact manifest/result reconciliation, including explicit setup-failure policy. |
| Actual applied-reference logging | **Fixed for the erroneous reference.** `src/homebody/backends/controller.py:176` saves effort-clamped targets; `tools/check_manipulation.py:326` exports them and full floating-base pose. **Richer diagnostics remain partial:** no waist/torso articulation, Cartesian goal, gains or saturation history; export remains approximately half-second (`evolve/skill_bench.py:543`). |
| Keyframe selection | **Fixed.** `evolve/skill_loop.py:356` uses the current episode; `:388` retains the last 40 images. Early causal evidence can still be discarded, but the concrete stale-row bug is gone. |
| Freeze/hash experiment identity | **Partly fixed.** `evolve/skill_loop.py:261` hashes robot files, real2sim assets, manifests and external server scripts. It **omits generated `assets/skillbench` scenes**, referenced settings, scorer/harness code, dependencies and server runtime configuration. Candidate assets actually come from the remote seed (`evolve/remote.py:62`), not the baseline commit. |
| Weighted uncertainty and speed noninferiority | **Partly fixed overall.** CR1’s weighted cluster-residual formula is correct (`evolve/skill_score.py:120`); the pooled lower-bound ≥−0.02 requirement exists (`:164`). But stripping `_vN` (`:111`) only clusters stance variants—not shared layouts or related objects. No power plan or repeated-selection correction was added. |
| Sanitize validation feedback | **Partly fixed.** `describe_train()` removes verdict basis and pooled timing (`evolve/skill_score.py:190`); `E/lessons.md` is absent and the old file is archived. Acceptance statuses still enter history, inspirations and lessons (`evolve/skill_loop.py:361`; `evolve/evo_loop.py:209`, `:427`). This remains adaptive development feedback, not hidden validation. |
| Honest probe semantics | **Fixed for rejection semantics.** `evolve/evo_loop.py:380` restricts named probes to Expected effect; `:401` rejects crashes, while `:407` records new mechanism names without rejecting unchanged outcomes. Self-reported activation still does not establish causal execution or benefit. |
| Remeasure screening, tests and conflicts | **Partly fixed.** Frozen checks and remote lint/tests now run (`evolve/evo_loop.py:255`, `:273`). `allowed()` and `specific()` still do not. Failed three-way applications merely log and continue (`:267`), leaving conflict state for subsequent retries. |
| Compatible donors | **Fixed for these planned runs, incomplete generally.** Parents and donors now use lineage (`evolve/evo_loop.py:229`). Git ancestry (`:129`) is not experiment identity if the same commit is rebaselined under changed conditions. |
| Planner posture, execution and concurrency | **Partly fixed.** Three endpoints and first-wins configuration avoid redundant rebuilds (`src/homebody/helpers/curobo.py:90`; `tools/curobo_server.py:149`). Stale posture remains executable; path thinning and unchecked execution remain (`src/homebody/harness/env.py:242`, `:252`). Documentation now acknowledges these limits. |
| Prompt contradictions and collision geometry | **Mostly fixed.** Rewrite scope is coherent (`evolve/prompts/evo_rewrite.md:10`); frames, queueing and unsupported loaded extraction are explicit (`evolve/config.toml:371`). Cell-covering spheres replace the undercoverage construction (`tools/build_dex1_hand.py:225`), subject to rounding and increased conservatism. |

**2. New problems and remaining loopholes**

- **Missing results can manufacture speed wins.** Synthetic failures contain no timing (`evolve/skill_loop.py:180`); `seconds()` defaults to zero (`evolve/skill_score.py:80`). Up to three crashes among 76 episodes survive the 5% gate (`:154`). I reproduced acceptance with three previously unsuccessful, 1,000-second episodes replaced by missing results: identical holds, **−97.6% time, accepted**. Record timeout/runtime penalties or prohibit timing acceptance with missing timing/results.

- **The posture warning is discarded by the advertised wrapper.** Server replies include `posture_mismatch` while still returning success (`tools/curobo_server.py:267`, `:285`); `env.plan_path()` drops that field entirely (`src/homebody/harness/env.py:243`). Moreover, client locks substitute nominal idle-arm/open-hand values for measurements (`src/homebody/helpers/curobo.py:45`). An empty warning cannot certify actual posture. Reject incompatible plans or configure atomically from measured nonmoving joints.

- **Three servers add three persistent posture histories.** PID-modulo routing (`src/homebody/helpers/curobo.py:94`) neither balances load nor preserves episode/server pairing across evaluations. First-arriving clients determine each server’s geometry. Queueing and posture can therefore change with process scheduling, contaminating both outcomes and latency.

- **“Slight inflation” is not uniformly 2.5 mm.** Palm cells use radius 21.2 mm and approximately 10.35 mm half-height (`assets/robot/dex1/curobo.json:4`; generator at `tools/build_dex1_hand.py:231`): roughly **10.9 mm outward bulging** at face centers. This can remove feasible narrow routes. Measure clearance effects before interpreting planner refusals.

- **Probe evidence never reaches the lessons model.** `evolve/evo_loop.py:425` serializes status, reason and notes, omitting `probe`, although `evolve/prompts/evo_lessons.md:4` describes those records. Historical “its change never ran” reasons also survive `public_reason()` fallback (`evolve/skill_loop.py:62`).

**3. What I would defer**

I retract making a fully validated cuRobo planner a prerequisite for **these five measurements**: p004 and all four historical candidate implementations do not call it. Loaded extraction, tactile sensing, per-tick diagnostics, full depth/path export, broader grasp architecture and bandit redesign can wait.

Independent layouts, untouched final testing and stronger inference remain necessary for generalization claims. They need not block exploratory component measurements if the existing validation set is explicitly treated as development data. Preserve the prohibition on runtime ground truth and manually inspect transferred code.

**4. Decisions and conditions**

- **(a) p004: conditional GO.** Pin `8691be0`, run the harness from `0458c51`, use fresh ID `p004`, and record a complete experiment manifest covering the omitted assets/settings/scorer. Reconcile all **49 train + 27 validation** episodes; investigate infrastructure failures before accepting the baseline.

- **(b) Four remeasurements: conditional GO afterward.** Independently transfer each patch onto p004, inspect conflicts and semantic preservation of overlap ranking, apply normal screens and tests, and verify recorded commits match evaluated code. Treat results as component transfers. Disable timing-based acceptance until missing-runtime handling is repaired.

- **(c) Evolutionary round 5: NO-GO as written.** First close result/timing and identity gaps, complete remeasurements, repair evidence reporting, and explicitly define exploratory selection versus final testing. Either exclude cuRobo interventions for this round or enforce measured-posture compatibility and checked execution before allowing them.