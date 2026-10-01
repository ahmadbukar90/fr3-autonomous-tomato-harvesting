#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    armed = LaunchConfiguration("armed")
    continuous_mode = LaunchConfiguration("continuous_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("fr3_tomato_harvesting"),
                        "models",
                        "best_v2.pt",
                    ]
                ),
            ),
            DeclareLaunchArgument("armed", default_value="false"),
            DeclareLaunchArgument(
                "continuous_mode",
                default_value="true",
            ),
            SetEnvironmentVariable("TOMATO_MODEL_PATH", model_path),
            Node(
                package="fr3_tomato_harvesting",
                executable="hardware_tomato_detector",
                output="screen",
            ),
            Node(
                package="fr3_tomato_harvesting",
                executable="hardware_harvest_controller",
                output="screen",
                parameters=[
                    {
                        "armed": ParameterValue(armed, value_type=bool),
                        "continuous_mode": ParameterValue(
                            continuous_mode,
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
