**(a) NO-GO as currently documented; (b) NO-GO until p004 is established and transfers are inspected; (c) still NO-GO.** The repairs are substantial, but “complete identity” and “measured planner posture” overstate what the code guarantees.

References below are relative to `hb-demo`; `P/` means the p004 worktree, and `E/` the supplied evidence directory. Inspection was read-only; no simulator or network was used.

| Recheck condition | Status and evidence |
|---|---|
| Pin the integrated baseline and current harness | **Met as code preparation.** Replace the previous pins with candidate **055da62** and harness **835730c**. The branch contains `397cb74 → 8691be0 → 055da62`; its diff from `2400da5` leaves the accepted skill unchanged, including overlap ranking (`P/src/homebody/skills/dex1/pick.py:408`). |
| Fresh ID `p004` | **Met: available, not yet evaluated.** The supplied archive ends with e004c4; no p004 results exist (`E/archive.jsonl:18`). |
| Complete experiment manifest, including assets/settings/scorer | **Not met.** Generated scenes and three local driver/scorer files are now hashed, but settings, profile/objective/weights, dependencies and runtime configuration remain omitted (`evolve/skill_loop.py:264`). Assets still come from the remote seed (`evolve/remote.py:62`), not the pinned candidate. |
| Reconcile 49 train + 27 validation episodes; investigate infrastructure failures | **Not met yet.** No p004 evidence exists. Missing files receive synthetic failures, but existing files bypass reconciliation; setup failures disappear and comparisons intersect IDs (`evolve/skill_loop.py:182`; `evolve/skill_score.py:35`, `:118`). |
| Independently transfer all four patches onto p004; inspect conflicts and preserve overlap ranking | **Not met yet.** The implementation uses one fixed baseline for the batch and cleans failed applications (`evolve/evo_loop.py:256`, `:267`), but no transferred runs/commits are recorded (`E/archive.jsonl:18`). |
| Normal screens and tests | **Met in implementation; execution unproven.** Scope, generality and remote checks now run (`evolve/evo_loop.py:275`). |
| Recorded commit must match evaluated code | **Not met as an enforced guarantee.** Existing worktrees are reused without checking cleanliness or requested HEAD; completed results are reused by ID alone (`evolve/loop.py:85`; `evolve/skill_loop.py:206`). |
| Treat remeasurements as component transfers | **Met in implementation.** Parent-relative patches are applied separately and recorded as `remeasured` (`evolve/evo_loop.py:264`, `:288`). |
| Disable timing acceptance until missing-runtime handling is repaired | **Not fully met.** The specific missing-file exploit is repaired, but another missing-timing path still accepts; see below. |

The five **new problems from the recheck** now stand as follows:

| Problem | Status |
|---|---|
| Missing results manufacture speed wins | **Met for missing files:** charged 5,400 seconds; any child crash blocks time/cost acceptance (`evolve/skill_loop.py:188`; `evolve/skill_score.py:165`). **Not met for all missing timing.** |
| Wrapper discards posture warnings; client substitutes nominal posture | **Met for the reported idle-arm case**, accepting your deployment verification: measured locks, atomic rebuild and `reconfigured_for` propagation exist (`src/homebody/helpers/curobo.py:40`; `tools/curobo_server.py:268`; `src/homebody/harness/env.py:246`). Broader compatibility remains broken below. |
| Three persistent posture histories contaminate runs | **Not met.** PID routing remains; differing postures now cause expensive rebuilds, so scheduling still affects latency (`src/homebody/helpers/curobo.py:99`; `tools/curobo_server.py:268`). |
| Approximately 11 mm palm-sphere bulge | **Met geometrically, not experimentally.** Palm radius falls to 9.7 mm with approximately 5.2 mm half-height: about 4.5 mm bulge. Clearance effects remain unmeasured (`assets/robot/dex1/curobo.json:4`; `tools/build_dex1_hand.py:225`). |
| Probe records omitted from lessons; historical inactivity claims survive | **Met.** Lessons receive `probe`; old inert reasons become “activation unknown” (`evolve/evo_loop.py:436`; `evolve/skill_loop.py:67`). |

Additional findings:

- **Timing remains exploitable.** A scored, noncrashed result without timing still becomes zero seconds (`evolve/skill_score.py:81`). In-memory reproduction with 49+27 unchanged failures, parent timing 1,000 seconds and child timing absent, returns **accepted, time −100%**. Require finite valid timings on both sides or disable timing verdicts.
- **Planner hand compatibility is wrong.** Both planners share one lock dictionary, but compatibility skips the active hand (`tools/curobo_server.py:156`, `:255`). Switching sides can therefore plan an open-hand approach using previously closed geometry. Furthermore, the **0.05-m slide tolerance exceeds the entire 0.0445-m finger travel** (`:50`; `tools/build_dex1_hand.py:28`). Idle-hand opening changes need never trigger rebuilding.
- **Identity is recorded after evaluation**, without a before/after consistency check (`evolve/skill_loop.py:260`). A fingerprint cannot establish what actually ran.
- **Planner guidance is now stale:** it still describes first-configuration locks and `posture_mismatch` (`evolve/config.toml:380`).

To release **(a)**, record the complete frozen manifest before running 055da62 through 835730c, then audit all 76 outcomes before accepting p004. Release **(b)** afterward, with clean inspected transfers and timing acceptance disabled until validated. These five implementations do not call cuRobo; planner repairs need not block them.

Before **(c)**: finish those measurements; close reconciliation, timing, identity and cache gaps; explicitly treat existing validation as development data; reserve untouched final testing with appropriate layout clustering and selection-aware inference. Fix stale evidence guidance. Either exclude cuRobo interventions or repair hand compatibility, scheduling and checked execution first.