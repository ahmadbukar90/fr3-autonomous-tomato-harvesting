#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


def launch_setup(context, *args, **kwargs):
    package_share = get_package_share_directory(
        "mujoco_fr3_bringup"
    )

    xacro_path = os.path.join(
        package_share,
        "urdf",
        "fr3_mujoco.urdf.xacro",
    )

    controllers_path = os.path.join(
        package_share,
        "config",
        "controllers.yaml",
    )

    mujoco_plugins_file = os.path.join(
        package_share,
        "config",
        "mujoco_plugins.yaml",
    )

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_path,
            " headless:=",
            LaunchConfiguration("headless"),
            " camera_publish_rate:=",
            LaunchConfiguration("camera_publish_rate"),
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content,
            value_type=str,
        )
    }

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[
                robot_description,
                {"use_sim_time": True},
            ],
        ),

        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            output="both",
            emulate_tty=True,
            parameters=[
                {"use_sim_time": True},
                ParameterFile(
                    controllers_path,
                    allow_substs=True,
                ),
                ParameterFile(
                    mujoco_plugins_file,
                    allow_substs=True,
                ),
            ],

            remappings=[
                (
                    "~/robot_description",
                    "/robot_description",
                )
            ],
            on_exit=Shutdown(),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "--param-file",
                controllers_path,
            ],
            output="both",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "fr3_arm_controller",
                "--param-file",
                controllers_path,
            ],
            output="both",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "gripper_controller",
                "--param-file",
                controllers_path,
            ],
            output="both",
        ),
    ]

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "camera_publish_rate",
                default_value="10.0",
            ),
            OpaqueFunction(
                function=launch_setup,
            ),
        ]
    )
