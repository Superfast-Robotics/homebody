"""HomeBody's simulated robots for LeRobot: importing this registers `--robot.type=homebody_g1` and the simulated
arms `so101_follower_sim`, `koch_follower_sim` and `omx_follower_sim` (actions and observations as their real
followers')."""
from .config_homebody_g1 import HomebodyG1Config
from .homebody_g1 import HomebodyG1
from .sim_arm import (
    KochFollowerSim,
    KochFollowerSimConfig,
    OmxFollowerSim,
    OmxFollowerSimConfig,
    SimArm,
    SimArmConfig,
    So101FollowerSim,
    So101FollowerSimConfig,
)

__all__ = ["HomebodyG1", "HomebodyG1Config", "KochFollowerSim", "KochFollowerSimConfig", "OmxFollowerSim",
           "OmxFollowerSimConfig", "SimArm", "SimArmConfig", "So101FollowerSim", "So101FollowerSimConfig"]
