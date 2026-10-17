# Headless Linux/amd64 build from pinned public sources.
FROM ros:humble-ros-base-jammy@sha256:6b4abb5340e430c632bcf4710fb34996df4f80e502993e5755f8fc59a39c7b2e

SHELL ["/bin/bash", "-c"]
ARG BUILD_JOBS=2
ARG SERIAL_IMU_EXECUTOR=1
ARG SOPHUS_REVISION=97e71617749a32b83cdd411591ebb2ede9d330f0
ARG GTSAM_REVISION=4abef9248edc4c49943d8fd8a84c028deb486f4c
ARG SUPERODOM_REVISION=57a6e233c348372485b7c64ae8238d53c1c7c2ad

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates tini libeigen3-dev libpcl-dev \
      libceres-dev libfmt-dev libtbb-dev libboost-all-dev \
      python3-colcon-common-extensions ros-humble-ament-cmake-auto \
      ros-humble-image-transport ros-humble-cv-bridge ros-humble-tf2-eigen \
      ros-humble-tf2-geometry-msgs ros-humble-pcl-conversions \
      ros-humble-visualization-msgs ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/sources/Sophus \
    && git -C /opt/sources/Sophus remote add origin https://github.com/strasdat/Sophus.git \
    && git -C /opt/sources/Sophus fetch --depth 1 origin "${SOPHUS_REVISION}" \
    && git -C /opt/sources/Sophus checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/sources/Sophus rev-parse HEAD)" = "${SOPHUS_REVISION}" \
    && cmake -S /opt/sources/Sophus -B /tmp/sophus-build -DBUILD_TESTS=OFF \
    && cmake --build /tmp/sophus-build --parallel "${BUILD_JOBS}" \
    && cmake --install /tmp/sophus-build \
    && rm -rf /tmp/sophus-build

RUN git init /opt/sources/gtsam \
    && git -C /opt/sources/gtsam remote add origin https://github.com/borglab/gtsam.git \
    && git -C /opt/sources/gtsam fetch --depth 1 origin "${GTSAM_REVISION}" \
    && git -C /opt/sources/gtsam checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/sources/gtsam rev-parse HEAD)" = "${GTSAM_REVISION}" \
    && cmake -S /opt/sources/gtsam -B /tmp/gtsam-build \
      -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_TESTS=OFF \
      -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=ON \
      -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_WITH_TBB=ON \
      -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
    && cmake --build /tmp/gtsam-build --parallel "${BUILD_JOBS}" \
    && cmake --install /tmp/gtsam-build \
    && rm -rf /tmp/gtsam-build && ldconfig

RUN git init /opt/superodom_ws/src/SuperOdom \
    && git -C /opt/superodom_ws/src/SuperOdom remote add origin https://github.com/superxslam/SuperOdom.git \
    && git -C /opt/superodom_ws/src/SuperOdom fetch --depth 1 origin "${SUPERODOM_REVISION}" \
    && git -C /opt/superodom_ws/src/SuperOdom checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/superodom_ws/src/SuperOdom rev-parse HEAD)" = "${SUPERODOM_REVISION}"

# The single-executor variant: one executor owns the shared IMU buffers.
RUN test "${SERIAL_IMU_EXECUTOR}" = 1 \
    && node=/opt/superodom_ws/src/SuperOdom/super_odometry/src/imuPreintegration_node.cpp \
    && test "$(grep -c 'rclcpp::executors::MultiThreadedExecutor executor;' "$node")" = 1 \
    && sed -i 's/rclcpp::executors::MultiThreadedExecutor executor;/rclcpp::executors::SingleThreadedExecutor executor;/' "$node"

RUN source /opt/ros/humble/setup.bash && cd /opt/superodom_ws \
    && CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" MAKEFLAGS="-j${BUILD_JOBS}" \
      colcon build --executor sequential \
      --packages-select super_odometry_msgs super_odometry --cmake-args -DBUILD_TESTING=OFF \
    && rm -rf build log && ldconfig \
    && dpkg-query -W > /opt/sources/debian-packages.tsv

COPY entrypoint.sh /usr/local/bin/superodom-entrypoint
LABEL org.opencontainers.image.source="https://github.com/superxslam/SuperOdom" \
      org.opencontainers.image.revision="${SUPERODOM_REVISION}" \
      org.homebody.sophus.revision="${SOPHUS_REVISION}" \
      org.homebody.gtsam.revision="${GTSAM_REVISION}" \
      org.homebody.superodom.serial-imu="${SERIAL_IMU_EXECUTOR}"
ENV ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/superodom-entrypoint"]
CMD ["bash"]
