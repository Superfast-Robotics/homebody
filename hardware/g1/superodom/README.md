# SuperOdometry build

This recipe builds the pinned SuperOdom revision and its dependencies from
public sources. It needs no prebuilt base image, submodule, existing checkout or
access to a robot. Linux/amd64 is the build target.

From the repository root:

```bash
hardware/g1/superodom/build.sh
```

The default builds `homebody-superodom:57a6e233-serial-imu` and
`homebody-recorder:humble-57a6e233`. `odometry` or `recorder` selects one build;
the recorder requires the first image. Compilation uses two jobs; `BUILD_JOBS`
can override this. Docker and Internet access are prerequisites. PCL's Ubuntu
development dependencies include Qt/VTK packages even though no desktop or GPU
is used by the build. Allow several gigabytes for images and intermediate layers.

The base digest and source revisions are recorded in `source-lock.json` and the
Dockerfiles. Ubuntu/ROS package repositories are not snapshot-pinned: the image
keeps `/opt/sources/debian-packages.tsv` (and the recorder's corresponding file)
to identify the resolved versions. This is a pinned-source build, not a claim of
bit-identical output from changing package mirrors.

The explicit `SERIAL_IMU_EXECUTOR=1` variant changes only the IMU node executor
from `MultiThreadedExecutor` to `SingleThreadedExecutor`, so one executor owns
the shared IMU buffers. The recipe checks the original
line before applying that patch. Zero is refused rather than building a different
variant under the same tag. Sophus and GTSAM revisions and build options are
pinned; GTSAM tests/examples are omitted from installation.

The three launch/configuration files hold the G1 parameters, separate
`/ws` output topics, and frame conventions. Inputs must already have the existing
G1 time/acceleration conversion applied. The recipe does not qualify a new
sensor's units, calibration, robot network, or localization quality.

Runtime containers use the launching user's UID/GID and writable home/log
directories created under `HOMEBODY_WORKSPACE/superodom_ws`. Both containers carry
the workspace's canonical path in the `homebody.exploration.workspace` label.
Launcher cleanup uses `run_superodom_ws.sh down-owned`: it removes only matching
containers, including a replacement started by a map reset. An adopted container
owned by another launch is left running. Explicit `down` still stops both fixed
container names; both commands report stop/removal failures. This lifecycle does
not stop the separately owned balancing body host.

## Offline image checks

These checks create only an isolated, device-free Docker namespace. They check the
four executables and shared libraries, the exact serial patch, and synthetic
scan/odometry transformation with preserved coordinates and timestamps. Run from
the repository root:

```bash
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp:rw,size=256m \
  -e HOME=/tmp -e ROS_DOMAIN_ID=71 -e ROS_LOCALHOST_ONLY=1 \
  -v "$PWD/hardware/g1/superodom:/checks:ro" \
  homebody-superodom:57a6e233-serial-imu python3 /checks/check_image.py

docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp:rw,size=256m \
  -e HOME=/tmp -e ROS_DOMAIN_ID=71 -e ROS_LOCALHOST_ONLY=1 \
  -v "$PWD/hardware/g1/superodom:/checks:ro" \
  --entrypoint /bin/bash homebody-recorder:humble-57a6e233 -lc \
  'source /opt/ros/humble/setup.bash; source /opt/superodom_ws/install/setup.bash; source /opt/unitree_msgs/setup.bash; python3 /checks/check_image.py --recorder'
```

The recorder check additionally loads the three Unitree message packages and
writes, closes, and reads a synthetic MCAP record. Neither command invokes the
recorder's normal entrypoint or connects to the robot. These checks establish
build/interface behavior; a supervised run still needs to validate deployment.

The pinned SuperOdom normalizer emits camera-configuration warnings without a
camera input. This release uses its LiDAR/odometry outputs only; it does not
provide calibrated phone colorization.

## Source and licensing

The image keeps the fetched source and its provenance under `/opt/sources` and
`/opt/superodom_ws/src/SuperOdom`. SuperOdom declares GPLv3 in its README and
package metadata; the recipe is not a license change. Keep the SuperOdom notices and
the explicit patch with any distributed image. Sophus, GTSAM and Unitree keep
their own licenses. No image is published by this build script.
