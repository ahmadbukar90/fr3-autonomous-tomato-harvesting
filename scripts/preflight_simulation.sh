#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failed=0

require_file() {
  if [ -f "$1" ]; then
    printf 'PASS file: %s\n' "$1"
  else
    printf 'FAIL missing file: %s\n' "$1"
    failed=1
  fi
}

require_text() {
  if grep -q "$2" "$1"; then
    printf 'PASS check: %s\n' "$3"
  else
    printf 'FAIL check: %s\n' "$3"
    failed=1
  fi
}

require_file "scripts/setup_simulation.sh"
require_file "dependencies/franka_ros2.repos"
require_file "models/best_v2.pt"
require_file "src/fr3_tomato_harvesting/launch/autonomous_simulation.launch.py"
require_file "src/mujoco_fr3_bringup/simulation/franka_fr3/harvesting_scene.xml"
require_file "src/tomato_interfaces/msg/TomatoTarget.msg"

bash -n scripts/setup_simulation.sh

python3 -m py_compile \
  src/fr3_tomato_harvesting/fr3_tomato_harvesting/*.py \
  src/fr3_tomato_harvesting/launch/*.py

packages="$(colcon list --base-paths src | awk '{print $1}')"
for package in fr3_tomato_harvesting mujoco_fr3_bringup tomato_interfaces
do
  if printf '%s\n' "$packages" | grep -qx "$package"; then
    printf 'PASS package: %s\n' "$package"
  else
    printf 'FAIL package: %s\n' "$package"
    failed=1
  fi
done

require_text \
  "src/fr3_tomato_harvesting/launch/autonomous_simulation.launch.py" \
  "simulation_tomato_detector" \
  "simulation detector is launched"

require_text \
  "src/fr3_tomato_harvesting/launch/autonomous_simulation.launch.py" \
  "simulation_harvest_controller" \
  "simulation controller is launched"

if grep -RIn --exclude=preflight_simulation.sh --exclude=preflight_hardware.sh "/home/muhayy" \
  src models README.md scripts dependencies > /dev/null
then
  printf 'FAIL old absolute path found\n'
  failed=1
else
  printf 'PASS no old absolute paths\n'
fi

if [ "$failed" -ne 0 ]; then
  printf '\nPreflight failed. No software was installed or launched.\n'
  exit 1
fi

printf '\nPreflight passed. No software was installed, built, or launched.\n'
printf 'When ready, run: ./scripts/setup_simulation.sh\n'
