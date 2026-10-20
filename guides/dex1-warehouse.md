# The G1 with Dex1-1 grippers in the warehouse

The `demo` branch runs HomeBody's G1 with Unitree Dex1-1 two-jaw parallel grippers (from the simulation team's
warehouse handoff) on the scanned warehouse scenes. The default robot (Dex3 hands) is unchanged.

## Pieces

| Path | What it is |
| --- | --- |
| `configs/dex1.toml` | `[robot] hands = "dex1"` and the Dex1 planning servers; pass with `--settings` |
| `assets/robot/dex1/` | the Dex1 robot (model, URDF, calibration, cuRobo spheres), built by `tools/build_dex1_hand.py` |
| `src/homebody/skills/dex1/` | the Dex1 pick (`pick.py`, `transport.py`) and place (`place.py`, `place_geometry.py`), written by Astra and improved by the skill loops; `registry(hands)` selects them |
| `tools/package_warehouse_picking.py` | the handoff as a scene package (`assets/real2sim/warehouse_picking`), with its own lighting; its hidden clutter is seen by depth |
| `tools/curobo_server.py --hand`, `tools/curobo/curobo_dex1_server.sh` | the collision-aware arm planner with the Dex1 gripper, three servers per evaluation host |
| `evolve/` profiles `demo-pick-wh`, `demo-place-wh` | the skill loops on warehouse-weighted benchmarks (`evolve/README.md`) |

The basic versions (Astra's first Dex1 pick and place) on the original HomeBody code are on the branch
`demo-homebody-baseline`.

## Results (2026-10-20)

Completed holds and placements on the warehouse-weighted benchmarks (train / held-out; warehouse in brackets):

| Skill | Version | Train | Held-out |
| --- | --- | --- | --- |
| pick (holds after the whole carry) | first version p005 | 15/49 (8/24) | 5/27 (2/10) |
| pick | **e006c5** (round 6) | **19/49 (12/24)** | 5/27 (2/10) |
| place (placed, of scored) | first version p000 | 13/26 (6/12) | 7/14 (2/2) |
| place | **e001c5** (round 1) | **18/27 (9/13)** | 9/14 (2/2) |

End to end with the agent (8 sessions): warehouse_shelf placed 7 of 16 objects (can 4/4, jar 3/4, tray and bowl 0);
warehouse_picking placed none (basil lifted every time); score 0.354. The held-out warehouse sets are too small to
confirm warehouse gains (5 pick objects; 2 place episodes reach the place call), and some place episodes' setup
varies between runs.

## Running a warehouse simulation

The warehouse scenes are not in the repository. They are frozen on the group volume, one folder per date
(`/lambda/nfs/group-volume/dex1-demos/<date>/`, each with a README saying where its scenes came from):

| Scene | Folder |
| --- | --- |
| warehouse picking (the handoff) | `/lambda/nfs/group-volume/dex1-demos/2026-10-20/warehouse_picking` |
| warehouse shelf | `/lambda/nfs/group-volume/dex1-demos/2026-10-20/warehouse_shelf` |

On a GPU host that mounts the group volume, with this branch installed (`pip install -e .`, README step 2):

**1. Start the Dex1 planning servers** (the pick and place plan their arm paths with them; skip if
`status` already shows three running). The launcher is the deployed copy of `tools/curobo/curobo_dex1_server.sh`:

    /lambda/nfs/group-volume/workcell-sim/homebody-baseline/curobo_dex1/curobo_dex1_server.sh start
    /lambda/nfs/group-volume/workcell-sim/homebody-baseline/curobo_dex1/curobo_dex1_server.sh status

**2. Give it a model.** The evaluation hosts use GPT-6.1 Sol through the team's LiteLLM proxy
(`--model sol61-litellm`): set `LITELLM_BASE_URL` to the proxy and `LITELLM_API_KEY` to its key. Any other
`--model` from the README works too.

**3. Run one session.** Each scene carries its own task (`default_task` in `task_scene.json`):

    SCENE=/lambda/nfs/group-volume/dex1-demos/2026-10-20/warehouse_shelf
    MUJOCO_GL=egl python3 -m homebody --headless --root . --scene $SCENE --settings configs/dex1.toml \
        --model sol61-litellm --output runs/warehouse \
        --task "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['default_task'])" $SCENE/task_scene.json)"

Drop `--headless` (and `MUJOCO_GL`) to watch it in the live console. The session folder
(`runs/warehouse/session-*`) holds `room.mp4` (the observer camera), `observations.mp4` (what the agent
sees) and `events.jsonl`.

**4. Score it** from physics against the scene's targets (`configs/warehouse_shelf_four.json` or
`configs/warehouse_picking_three.json`):

    python3 tools/score_session.py runs/warehouse/session-* --scene $SCENE --sequence configs/warehouse_shelf_four.json

**Without a model**, the scripted pick-and-place sequence runs the same skills on fixed targets:

    MUJOCO_GL=egl python3 tools/check_manipulation.py --scene $SCENE --settings configs/dex1.toml \
        --sequence configs/warehouse_shelf_four.json

## Skill loops and repeated sessions

    python3 -m evolve.skill_loop --profile demo-pick-wh run --rounds 1   # evolve/README.md
    python3 -m evolve.warehouse_e2e --repeats 2                          # sessions on both scenes, every host
