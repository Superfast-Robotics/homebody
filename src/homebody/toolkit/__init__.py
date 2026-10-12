"""Building blocks for writing and revising skills: reach diagnosis, collision gates with the
hand's current shape, calibrated grasps, planned arm paths and structured traces.

Everything here works on a captured Frame plus the Context's arms and settings, so a skill
composes them without touching the simulator. API.md in this folder is the reference.
"""
from .grasps import calibrated_grasps
from .reach import diagnose_ik, palm_in_torso, reach_report
from .scene import clearance, points_near
from .trace import trace

__all__ = ["calibrated_grasps", "clearance", "diagnose_ik", "palm_in_torso", "points_near", "reach_report", "trace"]
