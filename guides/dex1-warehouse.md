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

## Running

    python3 -m homebody --headless --root . --scene assets/real2sim/warehouse_picking --settings configs/dex1.toml --task "..."
    python3 -m evolve.skill_loop --profile demo-pick-wh run --rounds 1
    python3 -m evolve.warehouse_e2e --repeats 2
