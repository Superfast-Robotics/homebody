"""Fresh registered LiDAR points for exploration without a metric camera."""
from homebody.primitives.navigation.contracts import TermSpec

SPEC = TermSpec("lidar_obstacles", ("N", 3), "f32", "map", "m", staleness_s=0.5)


def measured(handle):
    """Measured registered LiDAR points in the exploration start frame, with sensor freshness."""
    return handle.lidar_obstacles()
