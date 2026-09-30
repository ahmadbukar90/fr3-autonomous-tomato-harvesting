# FR3 Autonomous Tomato Harvesting

ROS 2 Humble software for autonomous tomato harvesting with a Franka Research 3 (FR3).

## Scope

- MuJoCo simulation and physical-FR3 workflows
- RealSense D405 RGB-D perception with YOLO
- MoveIt harvesting control
- ArUco GridBoard eye-in-hand calibration support
- GelSight-compatible simulation logging

Only selected source code is included. Robot IPs, calibration results, model weights, datasets, recordings, build files, and vendor SDKs are excluded.

## ROS 2 executables

- `aruco_gridboard_detector`
- `hardware_tomato_detector`
- `hardware_harvest_controller`
- `simulation_tomato_detector`
- `simulation_harvest_controller`
- `harvest_dataset_logger`

## Requirements

Ubuntu 22.04, ROS 2 Humble, Franka ROS 2, MoveIt, and the custom `tomato_interfaces` package are required. Install Python packages with:

```bash
python3 -m pip install -r requirements.txt
```

## Local model and data

The default model path is `models/best_v2.pt`; the logger writes to `data/harvest_dataset/`. Both are ignored by Git.

```bash
export TOMATO_MODEL_PATH=/path/to/best_v2.pt
export TOMATO_DATASET_ROOT=/path/to/harvest_dataset
```

## Safety

Start in simulation. Before physical tests, validate calibration, TF, workspace limits, collision scene, controller state, speed limits, gripper behavior, and emergency-stop procedure. No hardware launch configuration is included.
