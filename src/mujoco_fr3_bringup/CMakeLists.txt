cmake_minimum_required(VERSION 3.8)
project(mujoco_fr3_bringup)

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mujoco_ros2_control REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)
find_package(std_msgs REQUIRED)

add_library(
  fr3_tomato_mujoco_system
  SHARED
  src/fr3_tomato_mujoco_system.cpp
)

ament_target_dependencies(
  fr3_tomato_mujoco_system
  hardware_interface
  pluginlib
  rclcpp
)

target_link_libraries(
  fr3_tomato_mujoco_system
  mujoco_ros2_control::mujoco_ros2_control
)

add_library(
  gelsight_contact_plugin
  SHARED
  src/gelsight_contact_plugin.cpp
)

ament_target_dependencies(
  gelsight_contact_plugin
  pluginlib
  rclcpp
  std_msgs
  mujoco_ros2_control_plugins
)

pluginlib_export_plugin_description_file(
  hardware_interface
  fr3_tomato_mujoco_system_plugin.xml
)

pluginlib_export_plugin_description_file(
  mujoco_ros2_control_plugins
  gelsight_contact_plugin.xml
)

install(
  TARGETS
    fr3_tomato_mujoco_system
    gelsight_contact_plugin
  LIBRARY DESTINATION lib
)

install(
  DIRECTORY
    launch
    config
    urdf
    simulation
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES
    fr3_tomato_mujoco_system_plugin.xml
    gelsight_contact_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
