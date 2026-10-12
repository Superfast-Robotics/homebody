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
