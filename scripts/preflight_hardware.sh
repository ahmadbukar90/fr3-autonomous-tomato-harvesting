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
  if grep -qF "$2" "$1"; then
    printf 'PASS check: %s\n' "$3"
  else
    printf 'FAIL check: %s\n' "$3"
    failed=1
  fi
}

CONTROLLER=src/fr3_tomato_harvesting/fr3_tomato_harvesting/hardware_harvest_controller.py
LAUNCH=src/fr3_tomato_harvesting/launch/hardware_harvesting.launch.py

require_file "$CONTROLLER"
require_file "$LAUNCH"
require_file models/best_v2.pt
require_file requirements-hardware.txt

bash -n "$0"
python3 -m py_compile "$CONTROLLER" "$LAUNCH"

require_text "$CONTROLLER" 'self.declare_parameter("armed", False)' \
  'hardware starts disarmed'
require_text "$CONTROLLER" 'goal blocked because hardware is disarmed.' \
  'future commands are blocked while disarmed'
require_text "$LAUNCH" 'DeclareLaunchArgument("armed", default_value="false")' \
  'launcher defaults to disarmed'
require_text "$LAUNCH" 'executable="hardware_tomato_detector"' \
  'hardware detector is launched'
require_text "$LAUNCH" 'executable="hardware_harvest_controller"' \
  'hardware controller is launched'

if grep -RInE '192\.168\.|10\.10\.|172\.16\.|/home/muhayy' \
  "$CONTROLLER" "$LAUNCH" > /dev/null
then
  printf 'FAIL private robot address or old path found\n'
  failed=1
else
  printf 'PASS no private robot address or old path\n'
fi

if [ "$failed" -ne 0 ]; then
  printf '\nHardware preflight failed. Nothing was installed, built, launched, or moved.\n'
  exit 1
fi

printf '\nHardware preflight passed. Nothing was installed, built, launched, connected, or moved.\n'
printf 'The launch file will still start DISARMED by default.\n'
