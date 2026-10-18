"""`--robot.type=homebody_g1`: HomeBody's simulated Unitree G1 (AMO walking, Dex3 hands) as a LeRobot robot."""
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig

HANDS = ("dex3",)
BACKENDS = ("mujoco",)


@RobotConfig.register_subclass("homebody_g1")
@dataclass(kw_only=True)
class HomebodyG1Config(RobotConfig):
    # The release root holding assets/ and configs/; the scene package, relative to it or absolute, or None
    # for the robot alone on a flat floor
    root: Path | None = None
    scene: Path | None = None
    hands: str = "dex3"
    backend: str = "mujoco"
    # {identity: [x, y, yaw]}: movable objects stood upright there at every reset
    spawns: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.hands not in HANDS:
            raise ValueError(f"robot.hands must be one of {', '.join(HANDS)}")
        if self.backend not in BACKENDS:
            raise ValueError(f"robot.backend must be one of {', '.join(BACKENDS)}")
