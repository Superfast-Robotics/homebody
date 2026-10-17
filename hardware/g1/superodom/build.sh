#!/usr/bin/env bash
# Build only; this never launches a sensor or robot process.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:-all}" in
  --help|-h)
    echo "Usage: hardware/g1/superodom/build.sh [all|odometry|recorder]"
    echo "Builds public pinned images with two compile jobs (BUILD_JOBS overrides)."
    exit 0 ;;
  all|odometry|recorder) target=${1:-all} ;;
  *) echo "Unknown build target: $1" >&2; exit 2 ;;
esac
jobs=${BUILD_JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "BUILD_JOBS must be positive" >&2; exit 2; }
if [[ "$target" != recorder ]]; then
  docker build --platform linux/amd64 --build-arg BUILD_JOBS="$jobs" \
    --build-arg SERIAL_IMU_EXECUTOR=1 \
    -t homebody-superodom:57a6e233-serial-imu "$here"
fi
if [[ "$target" != odometry ]]; then
  docker build --platform linux/amd64 --build-arg BUILD_JOBS="$jobs" \
    -f "$here/../recorder/Dockerfile" \
    -t homebody-recorder:humble-57a6e233 "$here/../../.."
fi
