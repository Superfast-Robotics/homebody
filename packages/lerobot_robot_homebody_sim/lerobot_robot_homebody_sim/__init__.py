"""HomeBody's simulated robots for LeRobot: importing this registers `--robot.type=homebody_g1` and the simulated
arms, whose actions and observations are their real followers': `so101_follower_sim`, `koch_follower_sim`,
`omx_follower_sim`, `openarm_follower_sim`, `bi_so101_follower_sim` and `lekiwi_sim`."""
from .config_homebody_g1 import HomebodyG1Config
from .homebody_g1 import HomebodyG1
from .sim_arm import (
    BiSo101FollowerSim,
    BiSo101FollowerSimConfig,
    KochFollowerSim,
    KochFollowerSimConfig,
    LekiwiSim,
    LekiwiSimConfig,
    OmxFollowerSim,
    OmxFollowerSimConfig,
    OpenarmFollowerSim,
    OpenarmFollowerSimConfig,
    SimArm,
    SimArmConfig,
    So101FollowerSim,
    So101FollowerSimConfig,
)

__all__ = ["BiSo101FollowerSim", "BiSo101FollowerSimConfig", "HomebodyG1", "HomebodyG1Config", "KochFollowerSim",
           "KochFollowerSimConfig", "LekiwiSim", "LekiwiSimConfig", "OmxFollowerSim", "OmxFollowerSimConfig", "OpenarmFollowerSim",
           "OpenarmFollowerSimConfig", "SimArm", "SimArmConfig", "So101FollowerSim", "So101FollowerSimConfig"]
