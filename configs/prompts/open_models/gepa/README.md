# GEPA-optimised open-model texts (2026-10-23)

The open setup's own texts after GEPA prompt optimisation (evolve/gepa_open.py), for the open models only (Sol's
prompt is unchanged): group A's best (c021: guidance, target note) with group B's (c005: pick and place notes).

| File | Setting | Read by |
| --- | --- | --- |
| guidance.md | `agent.local_notes = "gepa/guidance"` | Qwen3.8, every decision turn (replaces reach, claims, recover) |
| target_note.md | `agent.local_target_note` | Qwen3.8: how to name a click target for the pointer |
| pick_note.md | `agent.local_pick_note` | RoboBrain2.5, after its pick question |
| place_note.md | `agent.local_place_note` | RoboBrain2.5, after its place question |

Bench (val, Sol-labelled turns of the open setup's own rollouts): planner texts 0.719 -> 0.752, pointer notes 0.833 ->
0.870. End to end (Dex1: demo-pp and warehouse, Sol planning, one run): placed train 9-14% -> 23%, val 17-18% ->
26%. configs/open_models_gepa.toml holds them as settings; evolve/agent_experiments.py OPEN_GEPA as options.
