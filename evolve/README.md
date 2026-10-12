# evolve: astra-authored skills

A propose → gate → accept loop over HomeBody's own code, after Self-Harness (arXiv 2606.09498)
with tiered gates, an archive of versions (Darwin Gödel Machine) and the proposer reading full
traces from disk (Meta-Harness).

Each round:
1. **Workspace.** The best version's train runs become `rounds/rNNN/workspace/`: `BRIEF.md`,
   `results.md`, per-run traces (skill results, stages, plans, `skill_trace` events) and the
   history of earlier proposals. Held-out results are never written there.
2. **Diagnose.** Astra (Codex, read-only) groups failures into mechanisms (`diagnosis.md`).
3. **Propose.** For each of the top K mechanisms, astra edits its own git worktree
   (`evolve/<id>` branch) with a minimal patch and a note.
4. **Gate.** The patch's tier is the highest of the files it touches (`tiers.toml`); a frozen file
   rejects it. Lint and unit tests run with the candidate's code on the remote.
5. **Evaluate.** Train and held-out scripted suites (`config.toml`), repeats by tier, scored
   from physics (`score.py`): stage 0 nothing, 1 pick, 2 place, 3 physics-verified success.
6. **Accept.** Neither split's score drops and at least one rises; among several, the larger gain
   wins, then the smaller diff. Every candidate is archived (`archive.jsonl`).

Commands (from the repository root; the work directory is in `config.toml`):

    python3 -m evolve.loop baseline
    python3 -m evolve.loop run --rounds 5
    python3 -m evolve.loop status

The suite's scenes come from `tools/scenegen.py` (train and val draws). The evaluation scenes the
final version is tested on are kept outside this repository and outside the loop.
