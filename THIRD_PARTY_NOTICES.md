# Third-party notices

This repository contains or depends on the following third-party components. Each entry
states only what the repository itself records; where no licence file ships here, the
component's own terms apply.

| Component | Where | Origin | Licence |
| --- | --- | --- | --- |
| G1 walking-controller kernel | `vendor/holosoma_inference/` | Derived from Amazon FAR Holosoma (revision `4ed2cebf9780f3efb59621657916006afadb87dc`) and the authors' AMO/Holosoma adapter; observation math from OpenTeleVision/AMO and Psi0 (module docstrings in `g1_control/config.py`, `g1_control/amo.py`, `g1_control/runtime.py`) | Apache-2.0, full text in `vendor/holosoma_inference/LICENSE`; Copyright Amazon.com, Inc. or its affiliates, all rights reserved (the original NOTICE, carried in `NOTICE`) |
| AMO walking-policy weights | `assets/models/amo_jit.pt`, `adapter_jit.pt`, `adapter_norm_stats.pt` | Byte-identical to OpenTeleVision/AMO revision `34caaf943660e6f9420e35f64e86dd56fb51dd0e` (`g1_control/config.py`) | Apache-2.0, Copyright 2025 Jialong Li, Xuxin Cheng, Tianshu Huang and Xiaolong Wang (original LICENSE); the original project releases the weights for research use only |
| Unitree G1 robot description and meshes (29-DoF with Dex3 hands) | `assets/robot/g1_29dof_with_hand.urdf`, `assets/robot/g1_grasp/`, `assets/robot/g1_viz/` | Unitree G1 model (`g1_29dof_with_hand_rev_1_0`); the URDF keeps joints only, the MuJoCo model and STL meshes are used for simulation | BSD-3-Clause, full text in `assets/robot/LICENSE` |
| G1 gravity dynamics model and adapter | `assets/robot/g1_body29_hand14.urdf`, `src/homebody/backends/g1/gravity.py` | Psi0 revision `a32e57a3fabb8590c80677f9cd3d1fc3db60eb06` (`physical-superintelligence-lab/Psi0`) | Apache-2.0, full text in `vendor/holosoma_inference/LICENSE`; the robot model itself is Unitree G1's |
| Collision-map, RRT/A* planning and waypoint-follower code | `src/homebody/motion/exploration/nav.py` | Derived from Isaac Lab Arena (`isaac-sim/IsaacLab-Arena`); Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (contributors: https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md) | Apache-2.0, full text in `LICENSE` |
| MuJoCo | Python dependency (`pyproject.toml`), not vendored | MuJoCo physics engine | Apache-2.0 (MuJoCo) |
| SuperOdometry | Built from source by `hardware/g1/superodom/` (revision in `source-lock.json`, one executor patch), not vendored | `superxslam/SuperOdom` | GPLv3 as declared by the project; the built image keeps its notices |
| Sophus, GTSAM, unitree_ros2 | Built from source by `hardware/g1/superodom/` and `hardware/g1/recorder/` at the revisions in `source-lock.json`, not vendored | `strasdat/Sophus`, `borglab/gtsam`, `unitreerobotics/unitree_ros2` | Each project's own licence at the pinned revision |
| MediaMTX | Downloaded onto the robot by `hardware/g1/install.sh` (v1.21.1, SHA-256 pinned), not vendored | `bluenviron/mediamtx` | MIT |
| Scanned kitchen | `assets/real2sim/src_kitchen/` | Captured by the authors | Not a third-party component |
| Project website assets (Three.js, HLS.js, DM Sans, provider icons) | `docs/` | As listed in `docs/LICENSES.txt` | MIT, Apache-2.0, SIL OFL 1.1 and MIT, full texts in `docs/LICENSES.txt`. Provider and institution logos are their owners' trademarks, shown for identification only. |

File hashes for every shipped asset are in `assets/manifest.json` and are checked by
`tools/verify_assets.py`. Integrity checks do not establish redistribution rights.
