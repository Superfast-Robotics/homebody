# The basic Dex1 pick and place in the warehouse

This branch is the original HomeBody code with the G1's Dex1-1 two-jaw grippers (`configs/dex1.toml`,
`assets/robot/dex1/`) and Astra's first Dex1 pick and place (`src/homebody/skills/dex1/`), with no skill-loop
improvements and no planning servers: the HomeBody baseline for the `demo` branch's skills.

## Running a warehouse simulation

The warehouse scenes are not in the repository. They are frozen on the group volume, one folder per date
(`/lambda/nfs/group-volume/dex1-demos/<date>/`, each with a README saying where its scenes came from):

| Scene | Folder | Scoring targets |
| --- | --- | --- |
| warehouse picking (the handoff) | `/lambda/nfs/group-volume/dex1-demos/2026-10-20/warehouse_picking` | `configs/warehouse_picking_three.json` |
| warehouse shelf | `/lambda/nfs/group-volume/dex1-demos/2026-10-20/warehouse_shelf` | `configs/warehouse_shelf_four.json` |

The original HomeBody renders the scenes with its own lighting, not the handoff's.

On a GPU host that mounts the group volume, with this branch installed (`pip install -e .`, README step 2):

**1. Give it a model.** The evaluation hosts use GPT-6.1 Sol through the team's LiteLLM proxy
(`--model sol61-litellm`): set `LITELLM_BASE_URL` to the proxy and `LITELLM_API_KEY` to its key. Any other
`--model` from the README works too.

**2. Run one session.** Each scene carries its own task (`default_task` in `task_scene.json`):

    SCENE=/lambda/nfs/group-volume/dex1-demos/2026-10-20/warehouse_shelf
    MUJOCO_GL=egl python3 -m homebody --headless --root . --scene $SCENE --settings configs/dex1.toml \
        --model sol61-litellm --output runs/warehouse \
        --task "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['default_task'])" $SCENE/task_scene.json)"

Drop `--headless` (and `MUJOCO_GL`) to watch it in the live console. The session folder
(`runs/warehouse/session-*`) holds `room.mp4` (the observer camera), `observations.mp4` (what the agent
sees) and `events.jsonl`.

**3. Score it** from physics against the scene's targets:

    python3 tools/score_session.py runs/warehouse/session-* --scene $SCENE --sequence configs/warehouse_shelf_four.json
