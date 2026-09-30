#!/usr/bin/env python3

import math
from enum import Enum, auto
from typing import Callable, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    OrientationConstraint,
    PositionConstraint,
)
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Empty
from tomato_interfaces.msg import TomatoTarget

import tf2_ros
from tf2_geometry_msgs import do_transform_point

try:
    from franka_msgs.action import Move
except ImportError:
    Move = None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class HarvestState(Enum):
    SEARCHING = auto()
    MOVING_TO_TOMATO = auto()
    WAITING_BEFORE_GRASP = auto()
    CLOSING_GRIPPER = auto()
    WAITING_AFTER_GRASP = auto()
    RETRACTING_TOMATO = auto()
    MOVING_TO_BASKET = auto()
    WAITING_BEFORE_RELEASE = auto()
    OPENING_GRIPPER = auto()
    WAITING_AFTER_RELEASE = auto()
    RETURNING_HOME = auto()
    ERROR = auto()


class TomatoHarvestBridge(Node):
    """
    Autonomous harvesting cycle:

      SEARCHING
        -> lock first valid tomato
        -> save current TCP pose as home
        -> move to tomato
        -> wait 0.2 s
        -> close gripper once
        -> accept the final reachable width if fully closed is impossible
        -> wait 0.2 s
        -> retract backward by 0.10 m and downward by 0.10 m
        -> move to basket
        -> wait 0.2 s
        -> open gripper
        -> wait 0.2 s
        -> return to saved home pose
        -> automatically search for the next tomato

    The node does not repeatedly resend an unreachable zero-width close goal.
    """

    def __init__(self) -> None:
        super().__init__("tomato_harvest_bridge_full_cycle")

        # ============================================================
        # Frames and MoveIt configuration
        # ============================================================
        self.base_frame = "fr3_link0"
        self.camera_frame = "camera_link"
        self.tcp_frame = "fr3_hand_tcp"
        self.move_group_name = "fr3_arm"
        self.move_action_name = "/move_action"

        # ============================================================
        # Detection configuration
        # ============================================================
        self.minimum_confidence = 0.50

        # ============================================================
        # Workspace safety limits
        # ============================================================
        self.minimum_x = 0.05
        self.maximum_x = 0.85
        self.minimum_y = -0.50
        self.maximum_y = 0.50
        self.minimum_z = 0.02
        self.maximum_z = 0.75

        # ============================================================
        # Motion settings
        # ============================================================
        self.position_tolerance_m = 0.01
        self.velocity_scaling = 0.01
        self.acceleration_scaling = 0.01
        self.num_planning_attempts = 10
        self.allowed_planning_time_s = 10.0

        # Pause used between sequential actions.
        self.action_pause_s = 0.20

        # Combined retreat after grasping.
        self.retreat_distance_m = 0.10
        self.retreat_down_distance_m = 0.10

        # Faster speed used only for the retreat-and-down motion.
        self.retreat_velocity_scaling = 0.05
        self.retreat_acceleration_scaling = 0.03

        # ============================================================
        # Harvesting orientation
        # ============================================================
        # Measured using:
        # ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp
        # ROS quaternion order: x, y, z, w
        self.harvest_qx = 0.771
        self.harvest_qy = -0.063
        self.harvest_qz = 0.630
        self.harvest_qw = 0.069
        self._normalize_quaternion_fields("harvest")

        # Tight X/Z orientation constraints; more freedom around Y.
        self.harvest_x_tolerance_rad = math.radians(5.0)
        self.harvest_y_tolerance_rad = math.radians(20.0)
        self.harvest_z_tolerance_rad = math.radians(5.0)

        # Saved home pose and basket pose use tighter orientation tolerance.
        self.fixed_pose_orientation_tolerance_rad = math.radians(5.0)

        # ============================================================
        # Basket pose in fr3_link0
        # ============================================================
        self.basket_pose = Pose()
        self.basket_pose.position.x = 0.365
        self.basket_pose.position.y = 0.316
        self.basket_pose.position.z = 0.099
        self.basket_pose.orientation.x = 0.999
        self.basket_pose.orientation.y = 0.048
        self.basket_pose.orientation.z = 0.005
        self.basket_pose.orientation.w = -0.012
        self._normalize_pose_quaternion(self.basket_pose)

        # ============================================================
        # Franka gripper settings
        # ============================================================
        self.gripper_action_name = "/franka_gripper/move"
        self.gripper_joint_state_topic = "/franka_gripper/joint_states"

        self.gripper_min_width_m = 0.0
        self.gripper_max_width_m = 0.08
        self.gripper_close_width_m = 0.0
        self.gripper_open_width_m = 0.08
        self.gripper_speed_mps = 0.03

        self.measured_gripper_width_m: Optional[float] = None
        self.held_gripper_width_m: Optional[float] = None
        self.gripper_close_command_sent = False

        # ============================================================
        # State
        # ============================================================
        self.state = HarvestState.SEARCHING
        self.locked_track_id = None
        self.initial_tcp_pose: Optional[Pose] = None
        self.tomato_pose: Optional[Pose] = None
        self.retreat_pose: Optional[Pose] = None

        # Keep one-shot timers alive until they execute.
        self._one_shot_timers = []

        # ============================================================
        # TF
        # ============================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        # ============================================================
        # Action clients
        # ============================================================
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_action_name,
        )

        if Move is None:
            self.gripper_client = None
            self.get_logger().error(
                "Could not import franka_msgs.action.Move. "
                "Install and source franka_msgs before running this node."
            )
        else:
            self.gripper_client = ActionClient(
                self,
                Move,
                self.gripper_action_name,
            )

        # ============================================================
        # ROS interfaces
        # ============================================================
        self.target_sub = self.create_subscription(
            TomatoTarget,
            "/tomato_pick_target",
            self.target_callback,
            10,
        )

        # Manual reset remains available after an error.
        self.next_target_sub = self.create_subscription(
            Empty,
            "/next_tomato_target",
            self.next_target_callback,
            10,
        )

        self.gripper_joint_sub = self.create_subscription(
            JointState,
            self.gripper_joint_state_topic,
            self.gripper_joint_state_callback,
            10,
        )

        self.get_logger().info("Autonomous tomato harvest bridge started.")
        self.get_logger().info(
            "Cycle: tomato -> 0.2 s -> close -> 0.2 s "
            "-> retract back 10 cm and down 10 cm "
            "-> basket -> 0.2 s -> open -> 0.2 s -> home -> repeat."
        )

    # ================================================================
    # Quaternion and timer helpers
    # ================================================================

    def _normalize_quaternion_fields(self, prefix: str) -> None:
        qx = getattr(self, f"{prefix}_qx")
        qy = getattr(self, f"{prefix}_qy")
        qz = getattr(self, f"{prefix}_qz")
        qw = getattr(self, f"{prefix}_qw")

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            raise ValueError(f"Configured {prefix} quaternion is invalid.")

        setattr(self, f"{prefix}_qx", qx / norm)
        setattr(self, f"{prefix}_qy", qy / norm)
        setattr(self, f"{prefix}_qz", qz / norm)
        setattr(self, f"{prefix}_qw", qw / norm)

    @staticmethod
    def _normalize_pose_quaternion(pose: Pose) -> None:
        q = pose.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm <= 1e-9:
            raise ValueError("Pose quaternion is invalid.")
        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm

    @staticmethod
    def _copy_pose(source: Pose) -> Pose:
        target = Pose()
        target.position.x = source.position.x
        target.position.y = source.position.y
        target.position.z = source.position.z
        target.orientation.x = source.orientation.x
        target.orientation.y = source.orientation.y
        target.orientation.z = source.orientation.z
        target.orientation.w = source.orientation.w
        return target

    def schedule_once(
        self,
        delay_s: float,
        callback: Callable[[], None],
        description: str,
    ) -> None:
        """Run a callback once without blocking the ROS executor."""
        timer_holder = {}

        def timer_callback() -> None:
            timer = timer_holder.get("timer")
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)
                if timer in self._one_shot_timers:
                    self._one_shot_timers.remove(timer)

            self.get_logger().info(
                f"Pause complete ({delay_s:.2f} s): {description}."
            )
            callback()

        timer = self.create_timer(delay_s, timer_callback)
        timer_holder["timer"] = timer
        self._one_shot_timers.append(timer)

    # ================================================================
    # State helpers
    # ================================================================

    def set_state(self, new_state: HarvestState) -> None:
        self.state = new_state
        self.get_logger().info(f"Harvest state -> {new_state.name}")

    def reset_for_next_target(self) -> None:
        self.locked_track_id = None
        self.initial_tcp_pose = None
        self.tomato_pose = None
        self.retreat_pose = None
        self.gripper_close_command_sent = False
        self.held_gripper_width_m = None
        self.set_state(HarvestState.SEARCHING)
        self.get_logger().info("Ready to lock the next valid tomato.")

    # ================================================================
    # Subscriptions
    # ================================================================

    def gripper_joint_state_callback(self, msg: JointState) -> None:
        if len(msg.position) < 2:
            return

        width = float(msg.position[0] + msg.position[1])
        self.measured_gripper_width_m = clamp(
            width,
            self.gripper_min_width_m,
            self.gripper_max_width_m,
        )

    def next_target_callback(self, _msg: Empty) -> None:
        """Manual reset after an error or intentional operator reset."""
        if self.state != HarvestState.ERROR:
            self.get_logger().warn(
                "/next_tomato_target is only used to reset an ERROR state."
            )
            return

        self.get_logger().warn("Manual reset received.")
        self.reset_for_next_target()

    def target_callback(self, msg: TomatoTarget) -> None:
        if self.state != HarvestState.SEARCHING:
            return

        if msg.confidence < self.minimum_confidence:
            return

        # Detector convention:
        # x_cm = depth, y_cm = horizontal, z_cm = vertical.
        depth = msg.x_cm / 100.0
        horizontal = msg.y_cm / 100.0
        vertical = msg.z_cm / 100.0

        # Existing detector-to-camera_link conversion.
        x_cam = -vertical
        y_cam = horizontal
        z_cam = depth

        tomato_camera = PointStamped()
        tomato_camera.header.frame_id = self.camera_frame
        tomato_camera.header.stamp.sec = 0
        tomato_camera.header.stamp.nanosec = 0
        tomato_camera.point.x = x_cam
        tomato_camera.point.y = y_cam
        tomato_camera.point.z = z_cam

        try:
            camera_to_base = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
            )
            tomato_base = do_transform_point(
                tomato_camera,
                camera_to_base,
            )
        except Exception as error:
            self.get_logger().error(f"Tomato TF transform failed: {error}")
            return

        x_base = tomato_base.point.x
        y_base = tomato_base.point.y
        z_base = tomato_base.point.z

        if not self.target_is_safe(x_base, y_base, z_base):
            return

        current_tcp_pose = self.lookup_current_tcp_pose()
        if current_tcp_pose is None:
            return

        tomato_pose = Pose()
        tomato_pose.position.x = x_base
        tomato_pose.position.y = y_base
        tomato_pose.position.z = z_base
        tomato_pose.orientation.x = self.harvest_qx
        tomato_pose.orientation.y = self.harvest_qy
        tomato_pose.orientation.z = self.harvest_qz
        tomato_pose.orientation.w = self.harvest_qw

        # Build a combined backward-and-down retreat pose.
        # "Backward" is defined horizontally from the tomato toward the
        # TCP pose saved before the approach.
        retreat_pose = self._copy_pose(tomato_pose)
        dx = current_tcp_pose.position.x - tomato_pose.position.x
        dy = current_tcp_pose.position.y - tomato_pose.position.y
        horizontal_norm = math.hypot(dx, dy)

        if horizontal_norm <= 1e-6:
            self.get_logger().error(
                "Cannot calculate retreat direction because the horizontal "
                "distance between the tomato and saved home pose is too small."
            )
            return

        retreat_pose.position.x += (
            self.retreat_distance_m * dx / horizontal_norm
        )
        retreat_pose.position.y += (
            self.retreat_distance_m * dy / horizontal_norm
        )
        retreat_pose.position.z -= self.retreat_down_distance_m

        if not self.target_is_safe(
            retreat_pose.position.x,
            retreat_pose.position.y,
            retreat_pose.position.z,
        ):
            self.get_logger().error(
                "The backward-and-down retreat pose is outside the configured "
                "safe workspace."
            )
            return

        self.initial_tcp_pose = current_tcp_pose
        self.tomato_pose = tomato_pose
        self.retreat_pose = retreat_pose
        self.locked_track_id = msg.track_id
        self.gripper_close_command_sent = False
        self.held_gripper_width_m = None

        self.get_logger().info("==================================================")
        self.get_logger().info(
            f"LOCKED TOMATO id={msg.track_id}, class={msg.tomato_class}, "
            f"confidence={msg.confidence:.2f}"
        )
        self.get_logger().info(
            f"Target: x={x_base:.3f}, y={y_base:.3f}, z={z_base:.3f}"
        )
        self.get_logger().info(
            f"Retreat-and-down pose: x={retreat_pose.position.x:.3f}, "
            f"y={retreat_pose.position.y:.3f}, "
            f"z={retreat_pose.position.z:.3f}"
        )
        self.get_logger().info(
            f"Retreat distances: backward={self.retreat_distance_m:.3f} m, "
            f"down={self.retreat_down_distance_m:.3f} m"
        )
        self.get_logger().info("Saved current TCP pose as cycle home pose.")
        self.get_logger().info("==================================================")

        self.set_state(HarvestState.MOVING_TO_TOMATO)
        self.send_moveit_pose_goal(
            target_pose=tomato_pose,
            x_tolerance=self.harvest_x_tolerance_rad,
            y_tolerance=self.harvest_y_tolerance_rad,
            z_tolerance=self.harvest_z_tolerance_rad,
            completion_tag="tomato",
        )

    # ================================================================
    # TF and safety
    # ================================================================

    def lookup_current_tcp_pose(self) -> Optional[Pose]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                rclpy.time.Time(),
            )
        except Exception as error:
            self.get_logger().error(
                f"Could not read the current TCP pose: {error}"
            )
            return None

        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation.x = transform.transform.rotation.x
        pose.orientation.y = transform.transform.rotation.y
        pose.orientation.z = transform.transform.rotation.z
        pose.orientation.w = transform.transform.rotation.w
        self._normalize_pose_quaternion(pose)
        return pose

    def target_is_safe(self, x: float, y: float, z: float) -> bool:
        if not (self.minimum_x <= x <= self.maximum_x):
            self.get_logger().error(f"Unsafe x={x:.3f} m.")
            return False
        if not (self.minimum_y <= y <= self.maximum_y):
            self.get_logger().error(f"Unsafe y={y:.3f} m.")
            return False
        if not (self.minimum_z <= z <= self.maximum_z):
            self.get_logger().error(f"Unsafe z={z:.3f} m.")
            return False
        return True

    # ================================================================
    # MoveIt actions
    # ================================================================

    def send_moveit_pose_goal(
        self,
        target_pose: Pose,
        x_tolerance: float,
        y_tolerance: float,
        z_tolerance: float,
        completion_tag: str,
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None,
    ) -> None:
        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.fail_cycle(
                f"MoveIt action server {self.move_action_name} unavailable."
            )
            return

        goal = MoveGroup.Goal()
        goal.request.group_name = self.move_group_name
        goal.request.num_planning_attempts = self.num_planning_attempts
        goal.request.allowed_planning_time = self.allowed_planning_time_s
        if velocity_scaling is None:
            velocity_scaling = self.velocity_scaling
        if acceleration_scaling is None:
            acceleration_scaling = self.acceleration_scaling

        velocity_scaling = clamp(float(velocity_scaling), 0.001, 1.0)
        acceleration_scaling = clamp(float(acceleration_scaling), 0.001, 1.0)

        goal.request.max_velocity_scaling_factor = velocity_scaling
        goal.request.max_acceleration_scaling_factor = acceleration_scaling

        tolerance_sphere = SolidPrimitive()
        tolerance_sphere.type = SolidPrimitive.SPHERE
        tolerance_sphere.dimensions = [self.position_tolerance_m]

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.base_frame
        position_constraint.link_name = self.tcp_frame
        position_constraint.constraint_region.primitives.append(
            tolerance_sphere
        )
        position_constraint.constraint_region.primitive_poses.append(
            target_pose
        )
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.base_frame
        orientation_constraint.link_name = self.tcp_frame
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = x_tolerance
        orientation_constraint.absolute_y_axis_tolerance = y_tolerance
        orientation_constraint.absolute_z_axis_tolerance = z_tolerance
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        goal.request.goal_constraints.append(constraints)

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(
            f"Sending MoveIt {completion_tag} goal: "
            f"x={target_pose.position.x:.3f}, "
            f"y={target_pose.position.y:.3f}, "
            f"z={target_pose.position.z:.3f}, "
            f"velocity_scaling={velocity_scaling:.3f}, "
            f"acceleration_scaling={acceleration_scaling:.3f}"
        )

        send_future = self.move_group_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda future: self.moveit_goal_response_callback(
                future,
                completion_tag,
            )
        )

    def moveit_goal_response_callback(
        self,
        future,
        completion_tag: str,
    ) -> None:
        try:
            goal_handle = future.result()
        except Exception as error:
            self.fail_cycle(
                f"Failed to send MoveIt {completion_tag} goal: {error}"
            )
            return

        if not goal_handle.accepted:
            self.fail_cycle(f"MoveIt {completion_tag} goal was rejected.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self.moveit_result_callback(
                future,
                completion_tag,
            )
        )

    def moveit_result_callback(
        self,
        future,
        completion_tag: str,
    ) -> None:
        try:
            result_response = future.result()
            error_code = result_response.result.error_code.val
        except Exception as error:
            self.fail_cycle(
                f"Failed to receive MoveIt {completion_tag} result: {error}"
            )
            return

        if error_code != 1:
            self.fail_cycle(
                f"MoveIt {completion_tag} failed with error code {error_code}."
            )
            return

        self.get_logger().info(
            f"MoveIt {completion_tag} motion completed successfully."
        )

        if completion_tag == "tomato":
            self.set_state(HarvestState.WAITING_BEFORE_GRASP)
            self.schedule_once(
                self.action_pause_s,
                self.begin_close_gripper,
                "starting gripper closure",
            )
            return

        if completion_tag == "retreat":
            self.set_state(HarvestState.MOVING_TO_BASKET)
            self.send_moveit_pose_goal(
                target_pose=self.basket_pose,
                x_tolerance=self.fixed_pose_orientation_tolerance_rad,
                y_tolerance=self.fixed_pose_orientation_tolerance_rad,
                z_tolerance=self.fixed_pose_orientation_tolerance_rad,
                completion_tag="basket",
            )
            return

        if completion_tag == "basket":
            self.set_state(HarvestState.WAITING_BEFORE_RELEASE)
            self.schedule_once(
                self.action_pause_s,
                self.begin_open_gripper,
                "opening gripper at basket",
            )
            return

        if completion_tag == "home":
            self.get_logger().info(
                f"Harvest cycle completed for target {self.locked_track_id}."
            )
            self.reset_for_next_target()

    # ================================================================
    # Gripper actions
    # ================================================================

    def begin_close_gripper(self) -> None:
        self.set_state(HarvestState.CLOSING_GRIPPER)
        self.send_gripper_goal(
            width=self.gripper_close_width_m,
            completion_tag="close",
        )

    def begin_open_gripper(self) -> None:
        self.set_state(HarvestState.OPENING_GRIPPER)
        self.send_gripper_goal(
            width=self.gripper_open_width_m,
            completion_tag="open",
        )

    def send_gripper_goal(self, width: float, completion_tag: str) -> None:
        if Move is None or self.gripper_client is None:
            self.fail_cycle("Franka gripper Move action is unavailable.")
            return

        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.fail_cycle(
                f"Gripper action server {self.gripper_action_name} unavailable."
            )
            return

        if completion_tag == "close":
            if self.gripper_close_command_sent:
                self.get_logger().warn(
                    "Close command already sent; refusing to send it again."
                )
                return
            self.gripper_close_command_sent = True

        goal = Move.Goal()
        goal.width = float(
            clamp(width, self.gripper_min_width_m, self.gripper_max_width_m)
        )
        goal.speed = float(self.gripper_speed_mps)

        self.get_logger().info(
            f"Sending gripper {completion_tag} goal: "
            f"width={goal.width:.4f} m, speed={goal.speed:.3f} m/s."
        )

        send_future = self.gripper_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda future: self.gripper_goal_response_callback(
                future,
                completion_tag,
            )
        )

    def gripper_goal_response_callback(
        self,
        future,
        completion_tag: str,
    ) -> None:
        try:
            goal_handle = future.result()
        except Exception as error:
            if completion_tag == "close":
                self.get_logger().warn(
                    f"Close goal response failed: {error}. "
                    "Using the latest reachable width."
                )
                self.accept_reachable_close_and_continue()
            else:
                self.fail_cycle(f"Open goal response failed: {error}")
            return

        if not goal_handle.accepted:
            if completion_tag == "close":
                self.get_logger().warn(
                    "Close goal was rejected. Using the latest reachable width."
                )
                self.accept_reachable_close_and_continue()
            else:
                self.fail_cycle("Open gripper goal was rejected.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self.gripper_result_callback(
                future,
                completion_tag,
            )
        )

    def gripper_result_callback(
        self,
        future,
        completion_tag: str,
    ) -> None:
        try:
            result_wrapper = future.result()
            result = result_wrapper.result
            success = bool(getattr(result, "success", True))
            error_text = str(getattr(result, "error", ""))
        except Exception as error:
            success = False
            error_text = str(error)

        if completion_tag == "close":
            if self.measured_gripper_width_m is not None:
                self.held_gripper_width_m = self.measured_gripper_width_m
            else:
                self.held_gripper_width_m = self.gripper_close_width_m

            if success:
                self.get_logger().info(
                    "Gripper close action completed. "
                    f"Measured held width={self.held_gripper_width_m:.4f} m."
                )
            else:
                self.get_logger().warn(
                    "Requested fully closed width was not reached. "
                    f"Action message: '{error_text}'."
                )
                self.get_logger().warn(
                    f"Accepting final reachable width "
                    f"{self.held_gripper_width_m:.4f} m and not retrying."
                )

            self.set_state(HarvestState.WAITING_AFTER_GRASP)
            self.schedule_once(
                self.action_pause_s,
                self.begin_retreat_motion,
                "retracting backward and downward after grasping",
            )
            return

        if completion_tag == "open":
            if not success:
                self.fail_cycle(
                    f"Gripper failed to open at the basket: '{error_text}'."
                )
                return

            self.get_logger().info("Gripper opened successfully at basket.")
            self.set_state(HarvestState.WAITING_AFTER_RELEASE)
            self.schedule_once(
                self.action_pause_s,
                self.begin_return_home,
                "returning to the saved initial pose",
            )

    def accept_reachable_close_and_continue(self) -> None:
        if self.measured_gripper_width_m is not None:
            self.held_gripper_width_m = self.measured_gripper_width_m
        else:
            self.held_gripper_width_m = self.gripper_close_width_m

        self.get_logger().warn(
            f"Continuing with reachable/last measured gripper width "
            f"{self.held_gripper_width_m:.4f} m."
        )
        self.set_state(HarvestState.WAITING_AFTER_GRASP)
        self.schedule_once(
            self.action_pause_s,
            self.begin_retreat_motion,
            "retracting backward and downward after accepting the reachable gripper width",
        )

    # ================================================================
    # Sequential motion helpers
    # ================================================================

    def begin_retreat_motion(self) -> None:
        if self.retreat_pose is None:
            self.fail_cycle("Retreat-and-down pose is unavailable.")
            return

        self.set_state(HarvestState.RETRACTING_TOMATO)
        self.send_moveit_pose_goal(
            target_pose=self.retreat_pose,
            x_tolerance=self.harvest_x_tolerance_rad,
            y_tolerance=self.harvest_y_tolerance_rad,
            z_tolerance=self.harvest_z_tolerance_rad,
            completion_tag="retreat",
            velocity_scaling=self.retreat_velocity_scaling,
            acceleration_scaling=self.retreat_acceleration_scaling,
        )

    def begin_return_home(self) -> None:
        if self.initial_tcp_pose is None:
            self.fail_cycle("Saved initial TCP pose is unavailable.")
            return

        self.set_state(HarvestState.RETURNING_HOME)
        self.send_moveit_pose_goal(
            target_pose=self.initial_tcp_pose,
            x_tolerance=self.fixed_pose_orientation_tolerance_rad,
            y_tolerance=self.fixed_pose_orientation_tolerance_rad,
            z_tolerance=self.fixed_pose_orientation_tolerance_rad,
            completion_tag="home",
        )

    # ================================================================
    # Error handling
    # ================================================================

    def fail_cycle(self, message: str) -> None:
        self.set_state(HarvestState.ERROR)
        self.get_logger().error(message)
        self.get_logger().error(
            "Harvest cycle stopped. Inspect the robot, then reset with:"
        )
        self.get_logger().error(
            'ros2 topic pub --once /next_tomato_target '
            'std_msgs/msg/Empty "{}"'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TomatoHarvestBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
