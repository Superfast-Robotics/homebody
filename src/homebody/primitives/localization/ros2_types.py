"""ROS 2 message types as CycloneDDS IDL structs, for reading SuperOdom's /ws topics from this
package's own interpreter (the host ROS is Humble on Python 3.10; this package runs on 3.12).
Type names follow the ROS 2 DDS mangling `pkg::msg::dds_::Name_`.
"""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float64, int32, sequence, uint8, uint32


@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class PointField(IdlStruct, typename="sensor_msgs::msg::dds_::PointField_"):
    name: str = ""
    offset: uint32 = 0
    datatype: uint8 = 0
    count: uint32 = 0


@dataclass
class PointCloud2(IdlStruct, typename="sensor_msgs::msg::dds_::PointCloud2_"):
    header: Header = field(default_factory=Header)
    height: uint32 = 0
    width: uint32 = 0
    fields: sequence[PointField] = field(default_factory=list)
    is_bigendian: bool = False
    point_step: uint32 = 0
    row_step: uint32 = 0
    data: sequence[uint8] = field(default_factory=list)
    is_dense: bool = False


@dataclass
class Point(IdlStruct, typename="geometry_msgs::msg::dds_::Point_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs::msg::dds_::Quaternion_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0
    w: float64 = 1.0


@dataclass
class Pose(IdlStruct, typename="geometry_msgs::msg::dds_::Pose_"):
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::PoseWithCovariance_"):
    pose: Pose = field(default_factory=Pose)
    covariance: array[float64, 36] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class Vector3(IdlStruct, typename="geometry_msgs::msg::dds_::Vector3_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Imu(IdlStruct, typename="sensor_msgs::msg::dds_::Imu_"):
    header: Header = field(default_factory=Header)
    orientation: Quaternion = field(default_factory=Quaternion)
    orientation_covariance: array[float64, 9] = field(default_factory=lambda: [0.0] * 9)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    angular_velocity_covariance: array[float64, 9] = field(default_factory=lambda: [0.0] * 9)
    linear_acceleration: Vector3 = field(default_factory=Vector3)
    linear_acceleration_covariance: array[float64, 9] = field(default_factory=lambda: [0.0] * 9)


@dataclass
class Bool(IdlStruct, typename="std_msgs::msg::dds_::Bool_"):
    data: bool = False


@dataclass
class Twist(IdlStruct, typename="geometry_msgs::msg::dds_::Twist_"):
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::TwistWithCovariance_"):
    twist: Twist = field(default_factory=Twist)
    covariance: array[float64, 36] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class Odometry(IdlStruct, typename="nav_msgs::msg::dds_::Odometry_"):
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)


@dataclass
class String(IdlStruct, typename="std_msgs::msg::dds_::String_"):
    data: str = ""
