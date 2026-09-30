#!/usr/bin/env python3

import math
from enum import Enum, auto
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Pose

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    OrientationConstraint,
    PositionConstraint,
)

from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory

from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from tomato_interfaces.msg import TomatoTarget

from std_msgs.msg import Empty, Float64MultiArray

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

    
class HarvestState(Enum):

    SEARCHING = auto()
    MOVING_TO_PREGRASP = auto()
    MOVING_TO_GRASP = auto()
    CLOSING_GRIPPER = auto()
    WAITING_AFTER_GRASP = auto()
    RETREATING = auto()
    MOVING_TO_BASKET = auto()
    MOVING_TO_BASKET_DROP = auto()
    OPENING_GRIPPER = auto()
    WAITING_AFTER_RELEASE = auto()
    RETURNING_HOME = auto()
    FINISHED = auto()
    ERROR = auto()

class TomatoHarvestBridge(Node):
    """
    Safe tomato approach-and-close test.

    Sequence:
      1. Wait for one valid /tomato_pick_target message.
      2. Transform the target from d405_color_optical_frame to fr3_link0.
      3. Save the current TCP pose.
      4. Move fr3_hand_tcp to the target while preserving the current
         end-effector orientation.
      5. Stop and hold at the reached pose.

    This test intentionally performs no gripper motion, retreat, basket
    motion, or automatic return-home motion.
    """

    def __init__(self) -> None:
        super().__init__("tomato_harvest_bridge_full_cycle")

        # Frames and MoveIt configuration.
        self.base_frame = "fr3_link0"
        self.camera_frame = "d405_color_optical_frame"
        self.tcp_frame = "fr3_grasp_center"
        self.move_group_name = "fr3_arm"
        self.move_action_name = "/move_action"
        
        # Track IDs rejected because they are inside the basket.
        self.ignored_basket_track_ids = set()

        # Detection filtering.
        self.minimum_confidence = 0.50
        
        # Permanent fine correction expressed in the TCP/gripper frame.
        # Tune these once from the physical GelSight midpoint alignment.
        self.grasp_tcp_x_offset_m = 0.0042
        self.grasp_tcp_y_offset_m = -0.015525
        self.grasp_tcp_z_offset_m = 0.00071

        # Workspace safety limits in fr3_link0.
        self.minimum_x = 0.05
        self.maximum_x = 0.85
        self.minimum_y = -0.50
        self.maximum_y = 0.50
        self.minimum_z = 0.02
        self.maximum_z = 1.05
        

        # MoveIt goal settings.
        self.pregrasp_position_tolerance_m = 0.01
        self.grasp_position_tolerance_m = 0.003
        # Accurate Cartesian return to scanning/HOME pose.
        self.home_position_tolerance_m = 0.003
        self.orientation_tolerance_rad = math.radians(5.0)
        self.velocity_scaling = 0.05
        self.acceleration_scaling = 0.05
        self.num_planning_attempts = 10
        self.allowed_planning_time_s = 10.0
        
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )

        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )


        
        
        self.final_tcp_error_threshold_m = 0.003
        self.max_grasp_correction_attempts = 4
        self.grasp_correction_attempts = 0
        
        self.grasp_correction_gain = 0.40
        self.max_grasp_correction_step_m = 0.003

        self.state = HarvestState.SEARCHING
        self.locked_track_id: Optional[str] = None
        self.initial_tcp_pose: Optional[Pose] = None
        self.pregrasp_pose: Optional[Pose] = None
        self.grasp_pose: Optional[Pose] = None
        
        self.retreat_pose: Optional[Pose] = None
        self.basket_hover_pose: Optional[Pose] = None
        self.basket_drop_pose: Optional[Pose] = None
        
        # ============================================================
        # Basket exclusion volume
        #
        # Tomatoes inside this box are still detected/tracked,
        # but are ignored by the harvest bridge.
        #
        # Coordinates are in fr3_link0.
        # ============================================================

        self.basket_exclusion_enabled = True

        self.basket_exclusion_center_x = 0.470182
        self.basket_exclusion_center_y = 0.36
        self.basket_exclusion_center_z = 0.20

        # Half-extents of the basket exclusion box.
        self.basket_exclusion_half_x = 0.15
        self.basket_exclusion_half_y = 0.15
        self.basket_exclusion_half_z = 0.20

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_action_name,
        )

        self.next_target_publisher = self.create_publisher(
            Empty,
            "/next_tomato_target",
            10,
        )
        
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            "/gripper_controller/commands",
            10,
        )
        
        self.gripper_command_publisher = self.create_publisher(
            Float64MultiArray,
            "/gripper_controller/commands",
            10,
        )

        self.gripper_timer = None
        
        
        # Startup observation pose

        self.startup_joint_positions = [
            0.0,
            -1.725,
            0.0,
            -2.25,
            0.0,
            2.125,
            0.0,
        ]
        
        self.target_sub = self.create_subscription(
            TomatoTarget,
            "/tomato_pick_target",
            self.target_callback,
            10,
        )

        self.next_target_sub = self.create_subscription(
            Empty,
            "/next_tomato_target",
            self.next_target_callback,
            10,
        )

        
        self.get_logger().info(
            "Autonomous tomato harvest bridge started."
        )

        self.get_logger().info(
            "Sequence: detect -> pre-grasp -> grasp -> close -> "
            "retreat -> basket -> open -> home -> repeat."
        )
        
        
        self.collision_object_publisher = self.create_publisher(
            CollisionObject,
            "/collision_object",
            10,
        )

        self.create_timer(
            2.0,
            self.publish_environment_collision_objects,
        )
        
        # Keep one-shot timers alive until they execute.
        self._one_shot_timers = []
        
        # Pause between sequential harvest actions.
        self.action_pause_s = 0.20

    def set_state(self, new_state: HarvestState) -> None:
        self.state = new_state
        self.get_logger().info(
            f"Harvest state -> {new_state.name}"
        )
        
    def schedule_once(
        self,
        delay_s: float,
        callback,
        description: str,
    ) -> None:
        timer_holder = {}

        def timer_callback() -> None:
            timer = timer_holder.get("timer")

            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)

                if timer in self._one_shot_timers:
                    self._one_shot_timers.remove(timer)

            self.get_logger().info(
                f"Pause complete ({delay_s:.2f} s): "
                f"{description}."
            )

            callback()

        timer = self.create_timer(
            delay_s,
            timer_callback,
        )

        timer_holder["timer"] = timer
        self._one_shot_timers.append(timer)

    def reset_for_next_target(self) -> None:
        self.grasp_correction_attempts = 0
        self.locked_track_id = None
        self.initial_tcp_pose = None
        self.target_tcp_pose = None
        self.set_state(HarvestState.SEARCHING)
        self.get_logger().info(
            "Ready to accept the next tomato target."
        )

    def next_target_callback(self, _msg: Empty) -> None:

        # While SEARCHING, /next_tomato_target may be published
        # automatically when a detected tomato lies inside the
        # basket exclusion volume.
        #
        # Do not reset or stop the bridge. Simply remain in
        # SEARCHING so the detector can provide another target.
        if self.state == HarvestState.SEARCHING:
            self.get_logger().info(
                "Next-target request received while searching. "
                "Waiting for another suitable tomato."
            )
            return

        # Allow manual recovery from an ERROR state.
        if self.state == HarvestState.ERROR:
            self.get_logger().warn(
                "Manual reset received after error. "
                "Accepting a new tomato target."
            )

            self.reset_for_next_target()
            return

        # Ignore next-target requests while the robot is
        # executing an active harvest cycle.
        self.get_logger().warn(
            "Ignoring /next_tomato_target because the robot "
            "is currently executing a harvest cycle."
        )

    def target_callback(self, msg: TomatoTarget) -> None:
        if self.state != HarvestState.SEARCHING:
            return
            
        if msg.track_id in self.ignored_basket_track_ids:
            return
            
        if msg.tomato_class.lower() != "red":
            self.get_logger().info(
                f"Ignoring non-red tomato id={msg.track_id}, "
                f"class={msg.tomato_class}, "
                f"confidence={msg.confidence:.2f}."
            )
            return

        if msg.confidence < self.minimum_confidence:
            self.get_logger().warn(
                f"Ignoring low-confidence target id={msg.track_id}, "
                f"confidence={msg.confidence:.2f}."
            )
            return

        
        # Detector publishes:
        # msg.x_cm = camera Z (forward/depth)
        # msg.y_cm = camera X (right)
        # msg.z_cm = camera Y (down)
        #
        # ROS optical frame expects:
        # point.x = rightgedit ~/tomato_mujoco/ros2_ws/src/tomato_detection/tomato_detection/tomato_harvest_bridge_full_cycle.py
        # point.y = down
        # point.z = forward

        
        tomato_camera = PointStamped()
        tomato_camera.header.frame_id = self.camera_frame
        tomato_camera.header.stamp.sec = 0
        tomato_camera.header.stamp.nanosec = 0

        # Detector now publishes TRUE camera coordinates.
        tomato_camera.point.x = msg.x_cm / 100.0
        tomato_camera.point.y = msg.y_cm / 100.0
        tomato_camera.point.z = msg.z_cm / 100.0

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
            self.get_logger().error(
                f"Tomato TF transform failed: {error}"
            )
            return

        
        tomato_x = float(tomato_base.point.x)
        tomato_y = float(tomato_base.point.y)
        tomato_z = float(tomato_base.point.z)


        # Do not harvest tomatoes that are already in the basket.
        # Do not harvest tomatoes that are already in the basket.
        if self.target_is_inside_basket(
            tomato_x,
            tomato_y,
            tomato_z,
        ):
            self.ignored_basket_track_ids.add(
                msg.track_id
            )

            self.get_logger().info(
                "Ignoring tomato inside basket exclusion volume: "
                f"id={msg.track_id}, "
                f"x={tomato_x:.3f}, "
                f"y={tomato_y:.3f}, "
                f"z={tomato_z:.3f}. "
                "Track ID added to basket ignore list."
            )

            return
        
        self.get_logger().info(
            f"Camera point: "
            f"x={tomato_camera.point.x:.3f}, "
            f"y={tomato_camera.point.y:.3f}, "
            f"z={tomato_camera.point.z:.3f}"
        )
        
        self.get_logger().info(
            f"Tomato center in {self.base_frame}: "
            f"x={tomato_x:.3f}, "
            f"y={tomato_y:.3f}, "
            f"z={tomato_z:.3f}"
        )
     
        if not self.target_is_safe(
            tomato_x,
            tomato_y,
            tomato_z,
        ):
            return

        current_tcp_pose = self.lookup_current_tcp_pose()
        if current_tcp_pose is None:
            return

        pregrasp_pose = Pose()
        grasp_pose = Pose()

        #
        # Preserve current TCP orientation.
        #

        pregrasp_pose.orientation.x = (
            current_tcp_pose.orientation.x
        )
        pregrasp_pose.orientation.y = (
            current_tcp_pose.orientation.y
        )
        pregrasp_pose.orientation.z = (
            current_tcp_pose.orientation.z
        )
        pregrasp_pose.orientation.w = (
            current_tcp_pose.orientation.w
        )

        grasp_pose.orientation.x = (
            current_tcp_pose.orientation.x
        )
        grasp_pose.orientation.y = (
            current_tcp_pose.orientation.y
        )
        grasp_pose.orientation.z = (
            current_tcp_pose.orientation.z
        )
        grasp_pose.orientation.w = (
            current_tcp_pose.orientation.w
        )

        #
        # Grasp pose.
        #

        # ---------------------------------------------------------
        # Apply permanent grasp correction in the TCP local frame.
        # ---------------------------------------------------------

        qx = current_tcp_pose.orientation.x
        qy = current_tcp_pose.orientation.y
        qz = current_tcp_pose.orientation.z
        qw = current_tcp_pose.orientation.w

        # Rotation matrix:
        # TCP local coordinates -> fr3_link0 coordinates.
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)

        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)

        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        local_dx = self.grasp_tcp_x_offset_m
        local_dy = self.grasp_tcp_y_offset_m
        local_dz = self.grasp_tcp_z_offset_m

        base_dx = (
            r00 * local_dx
            + r01 * local_dy
            + r02 * local_dz
        )

        base_dy = (
            r10 * local_dx
            + r11 * local_dy
            + r12 * local_dz
        )

        base_dz = (
            r20 * local_dx
            + r21 * local_dy
            + r22 * local_dz
        )

        grasp_pose.position.x = tomato_x + base_dx
        grasp_pose.position.y = tomato_y + base_dy
        grasp_pose.position.z = tomato_z + base_dz

        #
        # Pre-grasp 10 cm back in base X.
        #

        pregrasp_pose.position.x = (
            grasp_pose.position.x - 0.10
        )

        pregrasp_pose.position.y = (
            grasp_pose.position.y
        )

        pregrasp_pose.position.z = (
            grasp_pose.position.z
        )
        
        self.get_logger().info(
            "Grasp calibration:"
        )

        self.get_logger().info(
            f"  tomato = "
            f"[{tomato_x:.4f}, "
            f"{tomato_y:.4f}, "
            f"{tomato_z:.4f}]"
        )

        self.get_logger().info(
            f"  offsets = "
            f"[{self.grasp_tcp_x_offset_m:.4f}, "
            f"{self.grasp_tcp_y_offset_m:.4f}, "
            f"{self.grasp_tcp_z_offset_m:.4f}]"
        )

        self.get_logger().info(
            f"  grasp TCP = "
            f"[{grasp_pose.position.x:.4f}, "
            f"{grasp_pose.position.y:.4f}, "
            f"{grasp_pose.position.z:.4f}]"
        )

        self.initial_tcp_pose = current_tcp_pose
        self.pregrasp_pose = pregrasp_pose
        self.grasp_pose = grasp_pose
        
        self.retreat_pose = None
        self.basket_pose = None
        
        # ============================================================
        # Basket exclusion volume
        #
        # Any detected tomato whose CENTER lies inside this volume
        # will still be detected/tracked, but will not be harvested.
        #
        # Coordinates are expressed in fr3_link0.
        # ============================================================

        self.basket_exclusion_enabled = True

        self.basket_exclusion_center_x = 0.220182
        self.basket_exclusion_center_y = 0.36
        self.basket_exclusion_center_z = 0.20

        # Half-size of exclusion box around basket center.
        # Change these later when basket dimensions change.
        self.basket_exclusion_half_x = 0.15
        self.basket_exclusion_half_y = 0.15
        self.basket_exclusion_half_z = 0.20
        
        self.locked_track_id = str(msg.track_id)
        
        
        retreat_pose = Pose()

        retreat_pose.position.x = (
            grasp_pose.position.x - 0.10
        )

        retreat_pose.position.y = (
            grasp_pose.position.y
        )

        retreat_pose.position.z = (
            grasp_pose.position.z + 0.10
        )

        retreat_pose.orientation = grasp_pose.orientation

        self.retreat_pose = retreat_pose
        

        self.get_logger().info(
            "=================================================="
        )
        self.get_logger().info(
            f"LOCKED TOMATO id={msg.track_id}, "
            f"class={msg.tomato_class}, "
            f"confidence={msg.confidence:.2f}"
        )
        self.get_logger().info(
            f"Target in {self.base_frame}: "
            f"x={tomato_x:.3f}, "
            f"y={tomato_y:.3f}, "
            f"z={tomato_z:.3f}"
        )
        self.get_logger().info(
            "Approach-only mode: preserving current TCP orientation."
        )
        self.get_logger().info(
            "=================================================="
        )

        
        self.set_state(HarvestState.MOVING_TO_PREGRASP)
        self.send_moveit_pose_goal(
            self.pregrasp_pose,
            self.pregrasp_position_tolerance_m,
            orientation_mode="constrained",
        )
        
        
        
        
        basket_hover_pose = Pose()

        basket_hover_pose.position.x = 0.470182
        basket_hover_pose.position.y = 0.36
        basket_hover_pose.position.z = 0.32

        basket_hover_pose.orientation.x = grasp_pose.orientation.x
        basket_hover_pose.orientation.y = grasp_pose.orientation.y
        basket_hover_pose.orientation.z = grasp_pose.orientation.z
        basket_hover_pose.orientation.w = grasp_pose.orientation.w

        basket_drop_pose = Pose()

        basket_drop_pose.position.x = 0.470182
        basket_drop_pose.position.y = 0.36
        basket_drop_pose.position.z = 0.24

        basket_drop_pose.orientation.x = grasp_pose.orientation.x
        basket_drop_pose.orientation.y = grasp_pose.orientation.y
        basket_drop_pose.orientation.z = grasp_pose.orientation.z
        basket_drop_pose.orientation.w = grasp_pose.orientation.w

        self.basket_hover_pose = basket_hover_pose
        self.basket_drop_pose = basket_drop_pose

        self.basket_hover_pose = basket_hover_pose
        self.basket_drop_pose = basket_drop_pose


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

        if not self.normalize_pose_quaternion(pose):
            self.get_logger().error(
                "Current TCP orientation is invalid."
            )
            return None

        return pose

    @staticmethod
    def normalize_pose_quaternion(pose: Pose) -> bool:
        q = pose.orientation
        norm = math.sqrt(
            q.x * q.x
            + q.y * q.y
            + q.z * q.z
            + q.w * q.w
        )

        if norm <= 1.0e-9:
            return False

        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm
        return True
        
    def target_is_inside_basket(
        self,
        x: float,
        y: float,
        z: float,
    ) -> bool:

        if not self.basket_exclusion_enabled:
            return False

        inside_x = (
            abs(x - self.basket_exclusion_center_x)
            <= self.basket_exclusion_half_x
        )

        inside_y = (
            abs(y - self.basket_exclusion_center_y)
            <= self.basket_exclusion_half_y
        )

        inside_z = (
            abs(z - self.basket_exclusion_center_z)
            <= self.basket_exclusion_half_z
        )

        return (
            inside_x
            and inside_y
            and inside_z
        )

    def target_is_safe(
        self,
        x: float,
        y: float,
        z: float,
    ) -> bool:
        if not (self.minimum_x <= x <= self.maximum_x):
            self.get_logger().error(
                f"Unsafe x={x:.3f} m."
            )
            return False

        if not (self.minimum_y <= y <= self.maximum_y):
            self.get_logger().error(
                f"Unsafe y={y:.3f} m."
            )
            return False

        if not (self.minimum_z <= z <= self.maximum_z):
            self.get_logger().error(
                f"Unsafe z={z:.3f} m."
            )
            return False

        return True

    def send_moveit_pose_goal(
        self,
        target_pose: Pose,
        position_tolerance_m: float,
        velocity_multiplier: float = 1.0,
        acceleration_multiplier: float = 1.0,
        orientation_mode: str = "constrained",
    ) -> None:

        # ----------------------------------------------------
        # Validate orientation policy.
        #
        # constrained:
        #     TCP must maintain target_pose.orientation
        #     within orientation_tolerance_rad.
        #
        # free:
        #     No TCP orientation constraint is sent to MoveIt.
        #     MoveIt may use joints 6/7 freely, subject only
        #     to normal robot joint/collision limits.
        # ----------------------------------------------------

        if orientation_mode not in (
            "constrained",
            "free",
        ):
            self.fail_cycle(
                "Invalid MoveIt orientation mode: "
                f"{orientation_mode}"
            )
            return

        if not self.move_group_client.wait_for_server(
            timeout_sec=2.0
        ):
            self.fail_cycle(
                "MoveIt /move_action server "
                "is not available."
            )
            return

        # ----------------------------------------------------
        # Effective speed scaling.
        # ----------------------------------------------------

        velocity_scaling = clamp(
            self.velocity_scaling
            * velocity_multiplier,
            0.001,
            1.0,
        )

        acceleration_scaling = clamp(
            self.acceleration_scaling
            * acceleration_multiplier,
            0.001,
            1.0,
        )

        # ----------------------------------------------------
        # MoveGroup goal.
        # ----------------------------------------------------

        goal = MoveGroup.Goal()

        goal.request.group_name = (
            self.move_group_name
        )

        goal.request.num_planning_attempts = (
            self.num_planning_attempts
        )

        goal.request.allowed_planning_time = (
            self.allowed_planning_time_s
        )

        goal.request.max_velocity_scaling_factor = (
            velocity_scaling
        )

        goal.request.max_acceleration_scaling_factor = (
            acceleration_scaling
        )

        # ----------------------------------------------------
        # POSITION CONSTRAINT
        #
        # Position is ALWAYS constrained.
        # ----------------------------------------------------

        tolerance_sphere = SolidPrimitive()

        tolerance_sphere.type = (
            SolidPrimitive.SPHERE
        )

        tolerance_sphere.dimensions = [
            position_tolerance_m
        ]

        position_constraint = (
            PositionConstraint()
        )

        position_constraint.header.frame_id = (
            self.base_frame
        )

        position_constraint.link_name = (
            self.tcp_frame
        )

        position_constraint.constraint_region.primitives.append(
            tolerance_sphere
        )

        position_constraint.constraint_region.primitive_poses.append(
            target_pose
        )

        position_constraint.weight = 1.0

        # ----------------------------------------------------
        # Goal constraints.
        # ----------------------------------------------------

        constraints = Constraints()

        constraints.position_constraints.append(
            position_constraint
        )

        # ----------------------------------------------------
        # OPTIONAL ORIENTATION CONSTRAINT
        # ----------------------------------------------------

        if orientation_mode == "constrained":

            orientation_constraint = (
                OrientationConstraint()
            )

            orientation_constraint.header.frame_id = (
                self.base_frame
            )

            orientation_constraint.link_name = (
                self.tcp_frame
            )

            orientation_constraint.orientation = (
                target_pose.orientation
            )

            orientation_constraint.absolute_x_axis_tolerance = (
                self.orientation_tolerance_rad
            )

            orientation_constraint.absolute_y_axis_tolerance = (
                self.orientation_tolerance_rad
            )

            orientation_constraint.absolute_z_axis_tolerance = (
                self.orientation_tolerance_rad
            )

            orientation_constraint.weight = 1.0

            constraints.orientation_constraints.append(
                orientation_constraint
            )

        # orientation_mode == "free":
        #
        # Deliberately add NO OrientationConstraint.
        # MoveIt is free to choose wrist orientation.

        goal.request.goal_constraints.append(
            constraints
        )

        # ----------------------------------------------------
        # Planning options.
        # ----------------------------------------------------

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(
            "Sending MoveIt pose goal: "
            f"x={target_pose.position.x:.3f}, "
            f"y={target_pose.position.y:.3f}, "
            f"z={target_pose.position.z:.3f}, "
            f"orientation={orientation_mode}, "
            f"velocity_scaling="
            f"{velocity_scaling:.3f}, "
            f"acceleration_scaling="
            f"{acceleration_scaling:.3f}"
        )

        send_future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        send_future.add_done_callback(
            self.moveit_goal_response_callback
        )
        
    def send_home_motion(self) -> None:        # Pause between sequential harvest actions.
        self.action_pause_s = 0.20

        home_pose = Pose()
        home_pose.position.x = (
            self.initial_tcp_pose.position.x
        )

        home_pose.position.y = (
            self.initial_tcp_pose.position.y
        )

        home_pose.position.z = (
            self.initial_tcp_pose.position.z
        )

        home_pose.orientation = (
            self.initial_tcp_pose.orientation
        )

        self.send_moveit_pose_goal(
            home_pose,
            self.home_position_tolerance_m,
            orientation_mode="constrained",
        )

    def moveit_goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as error:
            self.fail_cycle(
                f"Failed to send MoveIt approach goal: {error}"
            )
            return

        if not goal_handle.accepted:
            self.fail_cycle(
                "MoveIt approach goal was rejected."
            )
            return

        self.get_logger().info(
            "MoveIt approach goal accepted."
        )

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self.moveit_result_callback
        )

    def moveit_result_callback(self, future) -> None:
        try:
            result_response = future.result()
            error_code = int(
                result_response.result.error_code.val
            )
        except Exception as error:
            self.fail_cycle(
                "Failed to receive MoveIt approach result: "
                f"{error}"
            )
            return

        if error_code != 1:
            self.fail_cycle(
                "MoveIt approach failed with error code "
                f"{error_code}."
            )
            return

        self.get_logger().info(
            "MoveIt motion completed successfully."
        )

        if self.state == HarvestState.MOVING_TO_PREGRASP:

            self.get_logger().info(
                "Moving from pre-grasp to grasp..."
            )

            self.set_state(
                HarvestState.MOVING_TO_GRASP
            )

            self.send_moveit_pose_goal(
                self.grasp_pose,
                self.grasp_position_tolerance_m,
                orientation_mode="constrained",
            )

            return
            

            
        if self.state == HarvestState.RETREATING:

            self.get_logger().info(
                "Post-grasp retreat completed."
            )

            # ------------------------------------------------
            # The gripper is now outside the vine region.
            #
            # Activate the temporary collision volume BEFORE
            # planning the basket motion.
            # ------------------------------------------------

            self.activate_vine_collision_region()

            # Give MoveIt a short moment to receive/update
            # the planning-scene collision object.
            self.schedule_once(
                0.20,
                self.start_basket_motion_after_retreat,
                "starting basket motion after vine "
                "collision region activation",
            )

            return
            
            
        if self.state == HarvestState.MOVING_TO_BASKET:

            self.set_state(
                HarvestState.MOVING_TO_BASKET_DROP
            )

            self.send_moveit_pose_goal(
                self.basket_drop_pose,
                self.pregrasp_position_tolerance_m,
                orientation_mode="free",
            )
            
            return
            
        if self.state == HarvestState.MOVING_TO_BASKET_DROP:

            self.get_logger().info(
                "Basket drop pose reached. Releasing tomato."
            )

            self.set_state(
                HarvestState.OPENING_GRIPPER
            )

            self.open_gripper()

            return
            

            
            
        if self.state == HarvestState.MOVING_TO_GRASP:

            self.get_logger().info(
                "Grasp motion finished. Waiting 0.20 s for TCP to settle."
            )

            self.schedule_once(
                0.20,
                self.verify_grasp_pose_and_close,
                "verifying final grasp TCP",
            )

            return
            
        if self.state == HarvestState.RETURNING_HOME:

            self.get_logger().info(
                "Harvest cycle completed."
            )

            # Next harvest needs to be able to enter the
            # vine region again.
            self.remove_vine_collision_region()

            self.reset_for_next_target()

            return
            
            
    def verify_grasp_pose_and_close(self) -> None:

        if self.grasp_pose is None:
            self.fail_cycle(
                "Cannot verify grasp pose: grasp pose is missing."
            )
            return

        actual_tcp_pose = self.lookup_current_tcp_pose()

        if actual_tcp_pose is None:
            self.fail_cycle(
                "Could not read TCP after grasp settling delay."
            )
            return

        error_x = (
            self.grasp_pose.position.x
            - actual_tcp_pose.position.x
        )

        error_y = (
            self.grasp_pose.position.y
            - actual_tcp_pose.position.y
        )

        error_z = (
            self.grasp_pose.position.z
            - actual_tcp_pose.position.z
        )

        error_norm = math.sqrt(
            error_x * error_x
            + error_y * error_y
            + error_z * error_z
        )

        self.get_logger().info(
            "ACTUAL TCP AFTER SETTLING: "
            f"x={actual_tcp_pose.position.x:.4f}, "
            f"y={actual_tcp_pose.position.y:.4f}, "
            f"z={actual_tcp_pose.position.z:.4f}"
        )

        self.get_logger().info(
            "COMMANDED TCP AT GRASP: "
            f"x={self.grasp_pose.position.x:.4f}, "
            f"y={self.grasp_pose.position.y:.4f}, "
            f"z={self.grasp_pose.position.z:.4f}"
        )

        self.get_logger().info(
            "FINAL TCP ERROR: "
            f"dx={error_x * 1000.0:.2f} mm, "
            f"dy={error_y * 1000.0:.2f} mm, "
            f"dz={error_z * 1000.0:.2f} mm, "
            f"norm={error_norm * 1000.0:.2f} mm"
        )

        self.get_logger().info(
            "Final grasp pose reached. Closing gripper."
        )

        self.set_state(
            HarvestState.CLOSING_GRIPPER
        )

        self.close_gripper()

        return
        
    def close_gripper(self) -> None:

        message = Float64MultiArray()
        message.data = [0.0, 0.0]

        self.get_logger().info(
            "Closing gripper: [0.0, 0.0]"
        )

        self.gripper_command_publisher.publish(
            message
        )

        self.set_state(
            HarvestState.WAITING_AFTER_GRASP
        )

        self.schedule_once(
            self.action_pause_s,
            self.gripper_close_finished,
            "retreating after grasping tomato",
        )

    def gripper_close_finished(self) -> None:

        self.get_logger().info(
            "Gripper close completed. "
            "Waiting 0.5 s before retreat."
        )

        self.schedule_once(
            0.5,
            self.start_retreat_after_grasp,
            "starting 20 cm post-grasp retreat",
        )
        

        
    def start_retreat_after_grasp(self) -> None:

        if self.grasp_pose is None:
            self.fail_cycle(
                "Cannot calculate retreat: "
                "grasp pose is missing."
            )
            return

        retreat_pose = Pose()

        # ----------------------------------------------------
        # Post-grasp collision-avoidance retreat
        #
        # Move exactly 20 cm in base-frame -X.
        #
        # Y and Z stay unchanged.
        # End-effector orientation stays unchanged.
        # ----------------------------------------------------

        retreat_pose.position.x = (
            self.grasp_pose.position.x
            - 0.20
        )

        retreat_pose.position.y = (
            self.grasp_pose.position.y
        )

        retreat_pose.position.z = (
            self.grasp_pose.position.z
        )

        retreat_pose.orientation.x = (
            self.grasp_pose.orientation.x
        )

        retreat_pose.orientation.y = (
            self.grasp_pose.orientation.y
        )

        retreat_pose.orientation.z = (
            self.grasp_pose.orientation.z
        )

        retreat_pose.orientation.w = (
            self.grasp_pose.orientation.w
        )

        if not self.target_is_safe(
            retreat_pose.position.x,
            retreat_pose.position.y,
            retreat_pose.position.z,
        ):
            self.fail_cycle(
                "Calculated 20 cm retreat pose "
                "is outside the safe workspace."
            )
            return

        self.retreat_pose = retreat_pose

        self.get_logger().info(
            "Starting post-grasp retreat: "
            "20 cm in base-frame -X."
        )

        self.set_state(
            HarvestState.RETREATING
        )

        self.send_moveit_pose_goal(
            retreat_pose,
            self.pregrasp_position_tolerance_m,
            orientation_mode="constrained",
        )
        
        
    def start_basket_motion_after_retreat(
        self,
    ) -> None:

        self.get_logger().info(
            "Starting basket motion with vine "
            "collision protection active."
        )

        self.set_state(
            HarvestState.MOVING_TO_BASKET
        )

        self.send_moveit_pose_goal(
            self.basket_hover_pose,
            self.pregrasp_position_tolerance_m,
            orientation_mode="free",
        )
        
        
    def open_gripper(self) -> None:
        message = Float64MultiArray()
        message.data = [0.06, 0.06]

        self.get_logger().info(
            "Opening gripper: [0.06, 0.06]"
        )

        self.gripper_command_publisher.publish(
            message
        )

        self.set_state(
            HarvestState.WAITING_AFTER_RELEASE
        )
        
        self.schedule_once(
            self.action_pause_s,
            self.gripper_open_finished,
            "returning home after releasing tomato",
        )
        
        
    def gripper_open_finished(self) -> None:
    
        self.get_logger().info(
            "Gripper opened."
        )

        self.set_state(
            HarvestState.RETURNING_HOME
        )

        self.send_home_motion()

    def fail_cycle(self, message: str) -> None:
        self.set_state(HarvestState.ERROR)
        self.get_logger().error(message)
        self.get_logger().error(
            "Approach test stopped. Inspect the robot, then reset with:"
        )
        self.get_logger().error(
            'ros2 topic pub --once /next_tomato_target '
            'std_msgs/msg/Empty "{}"'
        )
        
        
    def activate_vine_collision_region(self) -> None:
        """
        Add a temporary MoveIt collision box protecting the
        entire vine/frame region after the tomato has been
        extracted.

        The physical vine plane is approximately:
            x = 0.795182 m in fr3_link0

        The protected region extends 20 cm toward the robot
        in the -X direction.
        """

        collision = CollisionObject()

        collision.header.frame_id = (
            self.base_frame
        )

        collision.id = (
            "post_grasp_vine_protection"
        )

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX

        # ----------------------------------------------------
        # Protected region
        #
        # X:
        #   0.595182 -> 0.795182
        #   exactly 20 cm toward -X from vine plane
        #
        # Y:
        #   covers complete vine/support-frame width
        #
        # Z:
        #   tabletop -> top of vine/frame
        # ----------------------------------------------------

        primitive.dimensions = [
            0.20,
            0.92,
            1.00,
        ]

        pose = Pose()

        pose.position.x = (
            0.695182
        )

        pose.position.y = (
            0.0
        )

        pose.position.z = (
            0.50
        )

        pose.orientation.w = 1.0

        collision.primitives.append(
            primitive
        )

        collision.primitive_poses.append(
            pose
        )

        collision.operation = (
            CollisionObject.ADD
        )

        self.collision_object_publisher.publish(
            collision
        )

        self.get_logger().info(
            "POST-GRASP VINE COLLISION REGION ACTIVATED: "
            "X=[0.595, 0.795], "
            "Y=[-0.46, 0.46], "
            "Z=[0.00, 1.00] m."
        )


    def remove_vine_collision_region(self) -> None:
        """
        Remove the temporary vine protection region so the
        robot can enter the vine area for the next harvest.
        """

        collision = CollisionObject()

        collision.header.frame_id = (
            self.base_frame
        )

        collision.id = (
            "post_grasp_vine_protection"
        )

        collision.operation = (
            CollisionObject.REMOVE
        )

        self.collision_object_publisher.publish(
            collision
        )

        self.get_logger().info(
            "Post-grasp vine collision region removed."
        )
        
        
    def publish_environment_collision_objects(self) -> None:
        #
        # Publish table and basket collision geometry to MoveIt.
        #

        objects = []

        #
        # TABLE TOP
        #
        # MuJoCo:
        # body pos = [0.525, 0.0, 0.685] world
        #
        # fr3_link0 world origin = [0.22, 0.0, 0.70]
        #
        # Therefore in fr3_link0:
        # [0.305, 0.0, -0.015]
        #

        # ============================================================
        # TABLETOP COLLISION GEOMETRY
        #
        # Physical tabletop thickness = 0.03 m.
        # Add 0.02 m safety margin ABOVE the tabletop only.
        # ============================================================

        table_margin = 0.02

        table = CollisionObject()
        table.header.frame_id = "fr3_link0"
        table.id = "harvesting_table"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            1.00,
            1.22,
            0.03 + table_margin,
        ]

        pose = Pose()
        pose.position.x = 0.305
        pose.position.y = 0.0
        pose.position.z = (
            -0.015
            + table_margin / 2.0
        )
        pose.orientation.w = 1.0

        table.primitives.append(primitive)
        table.primitive_poses.append(pose)
        table.operation = CollisionObject.ADD

        objects.append(table)

        #
        # BASKET
        #
        # Basket body center in fr3_link0:
        # [0.48, 0.36, 0.0]
        #

        #
        # BASKET
        #
        # MuJoCo basket body:
        # world      = [0.45, 0.36, 0.70]
        # fr3_link0  = [0.23, 0.36, 0.00]
        #


        # ============================================================
        # BASKET COLLISION GEOMETRY
        #
        # Physical basket body center in fr3_link0:
        # [0.48, 0.36, 0.0]
        #
        # Safety margin:
        #   2 cm outward from each exterior wall
        #   2 cm above the physical top of each wall
        #
        # The basket interior is NOT reduced by this margin.
        # ============================================================

        basket_margin = 0.02

        basket_parts = [
            (
                "basket_bottom",
                [0.20, 0.25, 0.005],
                [0.48, 0.36, 0.0025],
            ),

            # Front wall:
            # extend 2 cm outward in -Y and 2 cm upward.
            (
                "basket_front",
                [0.20, 0.005 + basket_margin, 0.145 + basket_margin],
                [0.48, 0.2375 - basket_margin / 2.0, 0.0775 + basket_margin / 2.0],
            ),

            # Back wall:
            # extend 2 cm outward in +Y and 2 cm upward.
            (
                "basket_back",
                [0.20, 0.005 + basket_margin, 0.145 + basket_margin],
                [0.48, 0.4825 + basket_margin / 2.0, 0.0775 + basket_margin / 2.0],
            ),

            # Left wall:
            # extend 2 cm outward in -X and 2 cm upward.
            (
                "basket_left",
                [0.005 + basket_margin, 0.25, 0.145 + basket_margin],
                [0.3825 - basket_margin / 2.0, 0.36, 0.0775 + basket_margin / 2.0],
            ),

            # Right wall:
            # extend 2 cm outward in +X and 2 cm upward.
            (
                "basket_right",
                [0.005 + basket_margin, 0.25, 0.145 + basket_margin],
                [0.5775 + basket_margin / 2.0, 0.36, 0.0775 + basket_margin / 2.0],
            ),
        ]


        for name, size, position in basket_parts:
            obj = CollisionObject()
            obj.header.frame_id = "fr3_link0"
            obj.id = name

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = size

            pose = Pose()
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
            pose.orientation.w = 1.0

            obj.primitives.append(primitive)
            obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD

            objects.append(obj)

        for obj in objects:
            self.collision_object_publisher.publish(obj)


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
