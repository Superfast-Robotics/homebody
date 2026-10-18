"""HomeBody's simulated robots for LeRobot: importing this registers `--robot.type=homebody_g1`."""
from .config_homebody_g1 import HomebodyG1Config
from .homebody_g1 import HomebodyG1

__all__ = ["HomebodyG1", "HomebodyG1Config"]
