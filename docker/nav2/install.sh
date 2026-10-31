#!/bin/sh
# Install the CPU-only ROS 2 Jazzy Navfn planner that the onboard football route
# runs inside the course container (lab/course_labs/nav2_local.py), on arm64 or amd64.
#
# Packages come from the dated official ROS snapshot that the course Nav2 lock
# (lab/nav2/image-lock.json) was taken from. apt checks the snapshot's signature
# against the pinned key below (snapshots.ros.org serves plain HTTP only). Each
# architecture installs that snapshot's own build: on arm64 these are the locked
# package identities; on amd64 the same upstream versions with amd64 build stamps.
# record_install.py fails the build if any direct package differs from the lock's
# upstream version and writes the installed inventory.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SNAPSHOT=${ROS_SNAPSHOT:-2026-09-11}
KEY=/usr/share/keyrings/ros-snapshots-jazzy.asc
KEY_SHA256=6d2ff4af9d56b304213de7664551f6986174a68bae76476b7ad21469b27a28c4
LOCK=${NAV2_LOCK:-$HERE/../../lab/nav2/image-lock.json}

echo "$KEY_SHA256  $HERE/ros-snapshots.asc" | sha256sum -c -
install -m 0644 "$HERE/ros-snapshots.asc" "$KEY"
echo "deb [signed-by=$KEY] http://snapshots.ros.org/jazzy/$SNAPSHOT/ubuntu noble main" \
    > /etc/apt/sources.list.d/ros2-jazzy-snapshot.list
apt-get update
# util-linux provides taskset, which pins the Nav2 service to one CPU.
# shellcheck disable=SC2046
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends util-linux $(cat "$HERE/packages.txt")
python3 "$HERE/record_install.py" --lock "$LOCK" --packages "$HERE/packages.txt" \
    --snapshot "$SNAPSHOT" --key "$KEY" --output /usr/local/share/microduck-nav2
rm -rf /var/lib/apt/lists/*
