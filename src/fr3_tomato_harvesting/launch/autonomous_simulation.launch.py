#!/usr/bin/env python3

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package_name, relative_path):
    package_share = get_package_share_directory(package_name)
    path = os.path.join(package_share, relative_path)

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    bringup_share = get_package_share_directory("mujoco_fr3_bringup")
    franka_description_share = get_package_share_directory(
        "franka_description"
    )
    harvesting_share = get_package_share_directory(
        "fr3_tomato_harvesting"
    )

    headless = LaunchConfiguration("headless")
    camera_publish_rate = LaunchConfiguration("camera_publish_rate")
    model_path = LaunchConfiguration("model_path")

    robot_xacro = os.path.join(
        bringup_share, "urdf", "fr3_mujoco.urdf.xacro"
    )
    robot_description = {
        "robot_description": ParameterValue(
            Command([
                FindExecutable(name="xacro"), " ", robot_xacro,
                " headless:=", headless,
                " camera_publish_rate:=", camera_publish_rate,
            ]),
            value_type=str,
        )
    }

    semantic_xacro = os.path.join(
        franka_description_share, "robots", "fr3", "fr3.srdf.xacro"
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            Command([
                FindExecutable(name="xacro"), " ", semantic_xacro,
                " hand:=true", " arm_prefix:=",
            ]),
            value_type=str,
        )
    }

    ompl = {
        "planning_plugin": "ompl_interface/OMPLPlanner",
        "request_adapters": (
            "default_planner_request_adapters/"
            "AddTimeOptimalParameterization "
            "default_planner_request_adapters/"
            "ResolveConstraintFrames "
            "default_planner_request_adapters/"
            "FixWorkspaceBounds "
            "default_planner_request_adapters/"
            "FixStartStateBounds "
            "default_planner_request_adapters/"
            "FixStartStateCollision "
            "default_planner_request_adapters/"
            "FixStartStatePathConstraints"
        ),
        "start_state_max_bounds_error": 0.1,
    }
    ompl.update(
        load_yaml("franka_fr3_moveit_config", "config/ompl_planning.yaml")
    )

    moveit_controllers = {
        "moveit_simple_controller_manager": load_yaml(
            "franka_fr3_moveit_config", "config/fr3_controllers.yaml"
        ),
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/"
            "MoveItSimpleControllerManager"
        ),
    }

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_share, "launch", "fr3_mujoco.launch.py"
            )
        ),
        launch_arguments={
            "headless": headless,
            "camera_publish_rate": camera_publish_rate,
        }.items(),
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            load_yaml(
                "franka_fr3_moveit_config", "config/kinematics.yaml"
            ),
            {
                "planning_pipelines": ["ompl"],
                "default_planning_pipeline": "ompl",
                "ompl": ompl,
            },
            moveit_controllers,
            {
                "moveit_manage_controllers": False,
                "trajectory_execution.allowed_execution_duration_scaling": 1.2,
                "trajectory_execution.allowed_goal_duration_margin": 0.5,
                "trajectory_execution.allowed_start_tolerance": 0.05,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
                "use_sim_time": True,
            },
        ],
    )

    detector = Node(
        package="fr3_tomato_harvesting",
        executable="simulation_tomato_detector",
        output="screen",
        parameters=[{"use_sim_time": True}],
        additional_env={"TOMATO_MODEL_PATH": model_path},
    )

    controller = Node(
        package="fr3_tomato_harvesting",
        executable="simulation_harvest_controller",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument(
            "camera_publish_rate", default_value="10.0"
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value=os.path.join(
                harvesting_share, "models", "best_v2.pt"
            ),
        ),
        simulator,
        TimerAction(period=5.0, actions=[move_group]),
        TimerAction(period=10.0, actions=[detector]),
        TimerAction(period=12.0, actions=[controller]),
    ])
