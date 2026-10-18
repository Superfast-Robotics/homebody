"""The robot a run drives, chosen as LeRobot chooses one: `--robot.type`. "mujoco" is the MuJoCo backend
itself (needs no LeRobot); "homebody_g1" is the LeRobot robot of that type, read through LeRobotSurface."""
ROBOT_TYPES = ("mujoco", "homebody_g1")


def make_robot(robot_type, root, settings, scene, spawns=None, hands="dex3", backend="mujoco"):
    if robot_type == "mujoco":
        from homebody.backends.mujoco import MujocoBackend
        return MujocoBackend(root, settings, scene=scene, spawns=spawns)
    if robot_type != "homebody_g1":
        raise ValueError(f"robot.type must be one of {', '.join(ROBOT_TYPES)}")
    try:
        from lerobot_robot_homebody_sim import HomebodyG1, HomebodyG1Config
    except ImportError as error:
        raise RuntimeError("robot.type=homebody_g1 needs LeRobot and packages/lerobot_robot_homebody_sim "
                           f"installed (Python 3.12): {error}") from error
    from homebody.robots.surface import LeRobotSurface
    robot = HomebodyG1(HomebodyG1Config(root=root, scene=scene, spawns=dict(spawns or {}), hands=hands,
                                        backend=backend), settings)
    robot.connect()
    return LeRobotSurface(robot)
