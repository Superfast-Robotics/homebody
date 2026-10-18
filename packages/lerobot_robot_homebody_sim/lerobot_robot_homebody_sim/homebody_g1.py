"""HomeBody's simulated G1 as a LeRobot `Robot`. Observations and actions are flat dicts named after LeRobot's
`unitree_g1` (`homebody.robots.g1`); a simulated robot also has a clock, so it adds `epoch`, `time`,
`advance` and `stop`, the camera-free `get_state`, the on-request `get_wrist`, the scene's `navigation` map
and `simulator` (hooks only a simulator has: the scorer's physics record, observer views, reset)."""
from lerobot.robots.robot import Robot

from homebody.robots import g1

from .config_homebody_g1 import HomebodyG1Config


class HomebodyG1(Robot):
    config_class = HomebodyG1Config
    name = "homebody_g1"

    def __init__(self, config: HomebodyG1Config, settings=None, backend=None):
        """SETTINGS are HomeBody's (loaded from ROOT's configs and the scene when None); BACKEND, a connected
        simulator to drive instead of building one (tests)."""
        super().__init__(config)
        self.config, self.settings, self._backend = config, settings, backend

    # --- LeRobot Robot ---
    @property
    def observation_features(self):
        camera = self._settings().camera
        return g1.observation_features(camera.height, camera.width)

    @property
    def action_features(self):
        return dict(g1.ACTION_FEATURES)

    @property
    def is_connected(self):
        return self._backend is not None

    def connect(self, calibrate=True):
        if self._backend is not None:
            return
        from homebody.backends.mujoco import MujocoBackend
        settings = self._settings()
        self._backend = MujocoBackend(self.config.root, settings, scene=self.config.scene,
                                      spawns=self.config.spawns)

    @property
    def is_calibrated(self):
        return True  # a simulator's joints and hands come calibrated with the model

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        return g1.encode_frame(self._connected().snapshot())

    def send_action(self, action):
        return g1.apply_action(self._connected(), action)

    def disconnect(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    # --- what a simulated robot adds ---
    @property
    def epoch(self):
        return self._connected().epoch

    @property
    def time(self):
        return self._connected().time

    def get_state(self):
        return g1.encode_state(self._connected().measure())

    def get_wrist(self, side):
        return g1.encode_wrist(self._connected().wrist_view(side))

    def advance(self, epoch, seconds):
        self._connected().advance(epoch, seconds)

    def stop(self, epoch):
        self._connected().stop(epoch)

    @property
    def navigation(self):
        return self._connected().navigation

    @property
    def simulator(self):
        return self._connected()

    def _connected(self):
        if self._backend is None:
            raise RuntimeError("homebody_g1 is not connected")
        return self._backend

    def _settings(self):
        if self.settings is None:
            from homebody.session.settings import Settings
            root = self.config.root
            if root is None:
                raise ValueError("robot.root must name the release root holding assets/ and configs/")
            scene = None if self.config.scene is None else root / self.config.scene
            self.settings = Settings.load(root / "configs/simulation.toml", scene=scene)
        return self.settings
