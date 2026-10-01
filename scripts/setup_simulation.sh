#!/usr/bin/env bash

set -euo pipefail

if [ ! -f /opt/ros/humble/setup.bash ]
then
  echo "ROS 2 Humble is required on Ubuntu 22.04."
  exit 1
fi

sudo apt update

sudo apt install -y \
  build-essential \
  git \
  python3-pip \
  python3-rosdep \
  python3-venv \
  python3-vcstool \
  ros-humble-cv-bridge \
  ros-humble-moveit \
  ros-humble-mujoco-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-tf2-geometry-msgs \
  ros-humble-xacro

if apt-cache show ros-humble-mujoco-ros2-control-plugins \
  > /dev/null 2>&1
then
  sudo apt install -y ros-humble-mujoco-ros2-control-plugins
fi

source /opt/ros/humble/setup.bash

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]
then
  sudo rosdep init
fi

rosdep update

if [ ! -d src/franka_ros2/.git ]
then
  vcs import src < dependencies/franka_ros2.repos
fi

if [ -f src/franka_ros2/dependency.repos ]
then
  vcs import src < src/franka_ros2/dependency.repos
fi

rosdep install \
  --from-paths src \
  --ignore-src \
  --rosdistro humble \
  --recursive \
  --yes

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

colcon build \
  --symlink-install \
  --packages-up-to fr3_tomato_harvesting \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

echo
echo "Simulation setup complete."
echo "Run:"
echo "  source .venv/bin/activate"
echo "  source /opt/ros/humble/setup.bash"
echo "  source install/setup.bash"
echo "  ros2 launch fr3_tomato_harvesting autonomous_simulation.launch.py"
