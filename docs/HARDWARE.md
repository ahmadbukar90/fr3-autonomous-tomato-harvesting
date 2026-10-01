# Guarded hardware quick start

This guide is for a separately configured physical FR3. Do not use the simulation launcher on a physical robot.

## Before a physical test

Keep robot addresses, calibration, TF, and workspace values in local untracked files. Validate the E-stop, collision scene, speed limits, D405 calibration, gripper behavior, and FR3 communication.

## Static repository check

```bash
./scripts/preflight_hardware.sh
```

This check does not install, build, launch, connect to, or move hardware.

## Safe startup order

1. FR3 driver and controllers
2. Franka gripper action server
3. MoveIt
4. Tomato detector and harvesting controller

Verify that `/franka_gripper/move` reports `Action servers: 1` before continuing.

## Guarded launcher

```bash
ros2 launch fr3_tomato_harvesting hardware_harvesting.launch.py
```

It launches with `armed:=false`: no target can start a harvest cycle.

Only after physical safety checks:

```bash
ros2 param set /tomato_harvest_bridge_full_cycle armed true
```

To block future MoveIt and gripper goals:

```bash
ros2 param set /tomato_harvest_bridge_full_cycle armed false
```

Disarming is not an emergency stop and does not cancel a trajectory already accepted by MoveIt. Use the normal Franka safe-stop or E-stop procedure for unexpected motion or a reflex abort.
