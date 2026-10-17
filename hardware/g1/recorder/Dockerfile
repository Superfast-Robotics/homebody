# Build the declared public parent with ../superodom/build.sh odometry first.
ARG SUPERODOM_IMAGE=homebody-superodom:57a6e233-serial-imu
FROM ${SUPERODOM_IMAGE}
SHELL ["/bin/bash", "-c"]
ARG BUILD_JOBS=2
ARG UNITREE_REVISION=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ros-humble-rosbag2-storage-mcap ros-humble-rosidl-generator-dds-idl \
      python3-zmq python3-opencv \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/sources/unitree_ros2 \
    && git -C /opt/sources/unitree_ros2 remote add origin https://github.com/unitreerobotics/unitree_ros2.git \
    && git -C /opt/sources/unitree_ros2 fetch --depth 1 origin "${UNITREE_REVISION}" \
    && git -C /opt/sources/unitree_ros2 checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/sources/unitree_ros2 rev-parse HEAD)" = "${UNITREE_REVISION}" \
    && source /opt/ros/humble/setup.bash \
    && cd /opt/sources/unitree_ros2/cyclonedds_ws \
    && CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" MAKEFLAGS="-j${BUILD_JOBS}" \
      colcon build --executor sequential --packages-select unitree_api unitree_go unitree_hg \
      --install-base /opt/unitree_msgs --cmake-args -DBUILD_TESTING=OFF \
    && rm -rf build log \
    && dpkg-query -W > /opt/sources/debian-packages-recorder.tsv
COPY hardware/g1/recorder/entrypoint.sh /rec/entry.sh
COPY src/homebody/real2sim/exploration/record/color_bridge.py /rec/color_bridge.py
LABEL org.homebody.unitree_ros2.revision="${UNITREE_REVISION}"
ENTRYPOINT ["/bin/bash", "/rec/entry.sh"]
