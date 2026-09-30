#!/usr/bin/env python3

from __future__ import annotations
import os

import csv
import math
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import tf2_ros

from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from tomato_interfaces.msg import TomatoTarget


# ============================================================
# USER SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = Path(
    os.environ.get(
        "TOMATO_DATASET_ROOT",
        PROJECT_ROOT / "data" / "harvest_dataset",
    )
)

LOG_RATE_HZ = 20.0

HARVEST_BRIDGE_NODE_NAME = (
    "tomato_harvest_bridge_full_cycle"
)

MOVE_GROUP_NODE_NAME = (
    "move_group"
)

BASE_FRAME = "fr3_link0"
EE_FRAME = "fr3_grasp_center"

# Measured HOME position of fr3_grasp_center in fr3_link0.
HOME_POSITION = (
    0.036,
    0.000,
    0.832,
)

# Requested HOME tolerance.
HOME_POSITION_TOLERANCE_M = 0.01

# Robot must move clearly away from HOME before returning HOME
# can close the current tomato cycle.
HOME_DEPARTURE_DISTANCE_M = 0.05

# ============================================================
# TOPICS
# ============================================================

RGB_TOPIC = "/camera/color/image_raw"
LEFT_TACTILE_TOPIC = "/gelsight/left/tactile_image"
RIGHT_TACTILE_TOPIC = "/gelsight/right/tactile_image"

JOINT_STATES_TOPIC = "/joint_states"
GRIPPER_COMMAND_TOPIC = "/gripper_controller/commands"
CONTACT_TOPIC = "/gelsight/contact_data"

TARGET_TOPIC = "/tomato_pick_target"
TASK_PHASE_TOPIC = "/harvest/task_phase"

# Explicit marker used ONLY for manual recovery.
#
# Publish one Empty message immediately before running your manual
# FollowJointTrajectory HOME command. This is intentionally explicit:
# MoveIt and the manual command use the same arm controller, so they
# cannot be distinguished reliably from controller status alone.
MANUAL_RECOVERY_TOPIC = (
    "/harvest_logger/manual_recovery"
)


# ============================================================
# JOINTS
# ============================================================

ARM_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

FINGER_JOINTS = [
    "fr3_finger_joint1",
    "fr3_finger_joint2",
]


# ============================================================
# LOGGER
# ============================================================

class HarvestDatasetLogger(Node):

    def __init__(self) -> None:

        super().__init__(
            "harvest_dataset_logger"
        )

        DATASET_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Latest observations
        # ----------------------------------------------------

        self.latest_rgb: Optional[Image] = None
        self.latest_left_tactile: Optional[Image] = None
        self.latest_right_tactile: Optional[Image] = None
        self.latest_target: Optional[TomatoTarget] = None

        self.latest_joint_positions = {
            name: float("nan")
            for name in (
                ARM_JOINTS
                + FINGER_JOINTS
            )
        }

        self.latest_joint_velocities = {
            name: float("nan")
            for name in ARM_JOINTS
        }

        self.latest_gripper_command = [
            float("nan"),
            float("nan"),
        ]

        self.left_contact = 0
        self.right_contact = 0

        self.left_indentation_m = 0.0
        self.right_indentation_m = 0.0

        self.task_phase = "unknown"

        # ----------------------------------------------------
        # Tomato-cycle state
        #
        # tomato_N starts when:
        #   - the bridge is first detected, OR
        #   - after tomato_(N-1) closes, the robot next
        #     genuinely leaves HOME.
        #
        # tomato_N closes when:
        #   - it has genuinely left HOME, AND
        #   - it returns within 1 cm of HOME.
        # ----------------------------------------------------

        self.tomato_index = (
            self.find_next_tomato_index()
        )

        self.bridge_seen_once = False

        self.cycle_active = False
        self.departed_home = False

        self.frame_index = 0

        # After a completed cycle, do not record idle HOME time
        # into the next tomato. Wait for the next departure.
        self.waiting_for_next_departure = False

        # Explicit manual-recovery latch.
        #
        # This is set only when /harvest_logger/manual_recovery receives
        # an Empty message. Normal MoveIt/bridge motion never sets it.
        self.manual_recovery_latched = False

        # Kept in the CSV as a diagnostic.
        self.manual_motion_active = False

        # Prevent repeated handling while sitting at HOME.
        self.manual_home_handled = False

        # ----------------------------------------------------
        # Pause state
        #
        # ONLY ERROR/FATAL rosout messages from MoveIt or the
        # harvest bridge pause the dataset.
        #
        # Detector staleness, missing target updates, missing
        # image frames, and ordinary node presence do NOT pause.
        # ----------------------------------------------------

        self.paused = False
        self.pause_reason = ""

        # ----------------------------------------------------
        # TF
        # ----------------------------------------------------

        self.tf_buffer = (
            tf2_ros.Buffer()
        )

        self.tf_listener = (
            tf2_ros.TransformListener(
                self.tf_buffer,
                self,
            )
        )

        # ----------------------------------------------------
        # Non-blocking disk writer
        #
        # ROS callbacks never perform PNG compression or disk I/O.
        # This keeps the logger from starving the bridge/MoveIt.
        # ----------------------------------------------------

        self.write_queue: queue.Queue = (
            queue.Queue(
                maxsize=200
            )
        )

        self.writer_stop = (
            threading.Event()
        )

        self.writer_thread = (
            threading.Thread(
                target=self.writer_loop,
                daemon=True,
                name="harvest_dataset_writer",
            )
        )

        self.writer_thread.start()

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------

        self.create_subscription(
            Image,
            RGB_TOPIC,
            self.rgb_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            LEFT_TACTILE_TOPIC,
            self.left_tactile_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            RIGHT_TACTILE_TOPIC,
            self.right_tactile_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self.joint_state_callback,
            20,
        )

        self.create_subscription(
            Float64MultiArray,
            GRIPPER_COMMAND_TOPIC,
            self.gripper_command_callback,
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            CONTACT_TOPIC,
            self.contact_callback,
            20,
        )

        self.create_subscription(
            TomatoTarget,
            TARGET_TOPIC,
            self.target_callback,
            20,
        )

        self.create_subscription(
            String,
            TASK_PHASE_TOPIC,
            self.task_phase_callback,
            10,
        )

        self.create_subscription(
            Log,
            "/rosout",
            self.rosout_callback,
            50,
        )

        self.create_subscription(
            Empty,
            MANUAL_RECOVERY_TOPIC,
            self.manual_recovery_callback,
            10,
        )

        # ----------------------------------------------------
        # Main timer
        # ----------------------------------------------------

        self.create_timer(
            1.0 / LOG_RATE_HZ,
            self.timer_callback,
        )

        self.get_logger().info(
            "Simple harvest dataset logger started."
        )

        self.get_logger().info(
            f"Next tomato folder: "
            f"tomato_{self.tomato_index}"
        )

        self.get_logger().info(
            "Waiting for first harvest bridge start."
        )


    # ========================================================
    # TOMATO INDEX
    # ========================================================

    def find_next_tomato_index(
        self,
    ) -> int:

        highest = 0

        for path in DATASET_ROOT.glob(
            "tomato_*"
        ):

            if not path.is_dir():
                continue

            try:

                number = int(
                    path.name.split(
                        "_",
                        1,
                    )[1]
                )

                highest = max(
                    highest,
                    number,
                )

            except Exception:

                continue

        return highest + 1


    # ========================================================
    # ROS GRAPH
    # ========================================================

    def node_is_present(
        self,
        wanted_name: str,
    ) -> bool:

        try:

            names = (
                self.get_node_names()
            )

        except Exception:

            return False

        return any(
            (
                name
                == wanted_name
            )
            or
            name.endswith(
                "/" + wanted_name
            )
            for name in names
        )


    # ========================================================
    # TF / HOME
    # ========================================================

    def get_ee_pose(
        self,
    ):

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    BASE_FRAME,
                    EE_FRAME,
                    rclpy.time.Time(),
                )
            )

            p = (
                transform.transform.translation
            )

            q = (
                transform.transform.rotation
            )

            return (
                float(p.x),
                float(p.y),
                float(p.z),

                float(q.x),
                float(q.y),
                float(q.z),
                float(q.w),
            )

        except Exception:

            nan = float("nan")

            return (
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            )


    def home_distance(
        self,
        ee_pose,
    ) -> float:

        x = ee_pose[0]
        y = ee_pose[1]
        z = ee_pose[2]

        if not (
            math.isfinite(x)
            and
            math.isfinite(y)
            and
            math.isfinite(z)
        ):

            return float("inf")

        dx = (
            x
            - HOME_POSITION[0]
        )

        dy = (
            y
            - HOME_POSITION[1]
        )

        dz = (
            z
            - HOME_POSITION[2]
        )

        return math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )


    # ========================================================
    # CALLBACKS
    # ========================================================

    def rgb_callback(
        self,
        msg: Image,
    ) -> None:

        self.latest_rgb = msg


    def left_tactile_callback(
        self,
        msg: Image,
    ) -> None:

        self.latest_left_tactile = msg


    def right_tactile_callback(
        self,
        msg: Image,
    ) -> None:

        self.latest_right_tactile = msg


    def joint_state_callback(
        self,
        msg: JointState,
    ) -> None:

        for index, name in enumerate(
            msg.name
        ):

            if (
                name
                in self.latest_joint_positions
                and
                index < len(msg.position)
            ):

                self.latest_joint_positions[
                    name
                ] = float(
                    msg.position[index]
                )

            if (
                name
                in self.latest_joint_velocities
                and
                index < len(msg.velocity)
            ):

                self.latest_joint_velocities[
                    name
                ] = float(
                    msg.velocity[index]
                )


    def gripper_command_callback(
        self,
        msg: Float64MultiArray,
    ) -> None:

        if len(msg.data) >= 2:

            self.latest_gripper_command = [
                float(msg.data[0]),
                float(msg.data[1]),
            ]


    def contact_callback(
        self,
        msg: Float64MultiArray,
    ) -> None:

        if len(msg.data) < 10:
            return

        self.left_contact = int(
            msg.data[0] > 0.5
        )

        self.left_indentation_m = float(
            msg.data[4]
        )

        self.right_contact = int(
            msg.data[5] > 0.5
        )

        self.right_indentation_m = float(
            msg.data[9]
        )


    def target_callback(
        self,
        msg: TomatoTarget,
    ) -> None:

        # Detector target is metadata only.
        # Stale/frozen detector output never pauses the logger.
        self.latest_target = msg


    def task_phase_callback(
        self,
        msg: String,
    ) -> None:

        self.task_phase = str(
            msg.data
        )


    # ========================================================
    # MANUAL RECOVERY MARKER
    # ========================================================

    def manual_recovery_callback(
        self,
        msg: Empty,
    ) -> None:

        del msg

        if not self.cycle_active:

            self.get_logger().warning(
                "Manual recovery marker received while no tomato "
                "cycle is active; ignoring it."
            )

            return

        self.manual_recovery_latched = True
        self.manual_motion_active = True
        self.manual_home_handled = False

        self.get_logger().warning(
            f"Manual recovery MARKED for tomato_{self.tomato_index}. "
            "The next HOME return will NOT advance the tomato ID."
        )


    # ========================================================
    # ERROR PAUSE
    # ========================================================

    def rosout_callback(
        self,
        msg: Log,
    ) -> None:

        # ERROR=40, FATAL=50.
        if int(msg.level) < 40:
            return

        node_name = str(
            msg.name
        )

        if (
            HARVEST_BRIDGE_NODE_NAME
            in node_name
        ):

            self.paused = True

            self.pause_reason = (
                "harvest bridge ERROR"
            )

            self.get_logger().error(
                "DATASET PAUSED: "
                "harvest bridge ERROR."
            )

        elif (
            MOVE_GROUP_NODE_NAME
            in node_name
        ):

            self.paused = True

            self.pause_reason = (
                "MoveIt ERROR"
            )

            self.get_logger().error(
                "DATASET PAUSED: MoveIt ERROR."
            )


    # ========================================================
    # CYCLE CONTROL
    # ========================================================

    def begin_cycle(
        self,
        already_departed: bool,
    ) -> None:

        if self.cycle_active:
            return

        self.cycle_active = True

        self.departed_home = (
            already_departed
        )

        self.waiting_for_next_departure = False

        self.manual_home_handled = False
        self.manual_motion_active = False
        self.manual_recovery_latched = False

        self.frame_index = 0

        self.enqueue(
            {
                "type":
                    "open",

                "tomato_index":
                    self.tomato_index,
            }
        )

        self.get_logger().info(
            f"START LOGGING "
            f"tomato_{self.tomato_index}"
        )


    def finish_cycle(
        self,
    ) -> None:

        completed = (
            self.tomato_index
        )

        self.enqueue(
            {
                "type":
                    "close",

                "tomato_index":
                    completed,
            }
        )

        self.cycle_active = False

        self.departed_home = False

        self.frame_index = 0

        self.tomato_index += 1

        self.waiting_for_next_departure = True

        self.get_logger().info(
            "TOMATO ID UPDATE: "
            f"tomato_{completed} "
            f"-> tomato_{self.tomato_index}"
        )


    # ========================================================
    # MAIN TIMER
    # ========================================================

    def timer_callback(
        self,
    ) -> None:

        ee_pose = (
            self.get_ee_pose()
        )

        distance = (
            self.home_distance(
                ee_pose
            )
        )

        bridge_present = (
            self.node_is_present(
                HARVEST_BRIDGE_NODE_NAME
            )
        )

        moveit_present = (
            self.node_is_present(
                MOVE_GROUP_NODE_NAME
            )
        )

        # ----------------------------------------------------
        # First bridge start begins the first tomato log.
        # ----------------------------------------------------

        if (
            bridge_present
            and
            not self.bridge_seen_once
        ):

            self.bridge_seen_once = True

            self.paused = False
            self.pause_reason = ""

            self.begin_cycle(
                already_departed=(
                    math.isfinite(distance)
                    and
                    distance
                    >= HOME_DEPARTURE_DISTANCE_M
                )
            )

        if not self.bridge_seen_once:
            return

        # ----------------------------------------------------
        # Resume after an ERROR only when both relevant
        # terminals/nodes are present again.
        # ----------------------------------------------------

        if (
            self.paused
            and
            bridge_present
            and
            moveit_present
        ):

            self.paused = False
            self.pause_reason = ""

            self.get_logger().info(
                "DATASET RESUMED."
            )

        if self.paused:
            return

        # ----------------------------------------------------
        # After tomato_N completed at HOME, increment happened
        # immediately. Start tomato_N+1 when the robot next
        # genuinely leaves HOME.
        # ----------------------------------------------------

        if (
            not self.cycle_active
            and
            self.waiting_for_next_departure
            and
            math.isfinite(distance)
            and
            distance
            >= HOME_DEPARTURE_DISTANCE_M
        ):

            self.begin_cycle(
                already_departed=True
            )

        if not self.cycle_active:
            return

        # ----------------------------------------------------
        # Detect departure for the current cycle.
        # ----------------------------------------------------

        if (
            not self.departed_home
            and
            math.isfinite(distance)
            and
            distance
            >= HOME_DEPARTURE_DISTANCE_M
        ):

            self.departed_home = True

            self.get_logger().info(
                f"tomato_{self.tomato_index} "
                "departed HOME."
            )

        # ----------------------------------------------------
        # Save one observation.
        # ----------------------------------------------------

        self.snapshot_observation(
            ee_pose,
            distance,
        )

        # ----------------------------------------------------
        # Finish current tomato when it returns HOME.
        #
        # The HOME observation above is saved first.
        # ----------------------------------------------------

        if (
            self.departed_home
            and
            math.isfinite(distance)
            and
            distance
            <= HOME_POSITION_TOLERANCE_M
        ):

            if self.manual_recovery_latched:

                # Explicitly marked manual recovery:
                # keep the SAME tomato ID.
                if not self.manual_home_handled:

                    self.manual_home_handled = True
                    self.departed_home = False

                    self.get_logger().warning(
                        f"Manual recovery returned tomato_"
                        f"{self.tomato_index} to HOME. "
                        "Tomato ID NOT advanced."
                    )

                # The manual recovery has now been consumed.
                # Keep the current tomato active for a later retry.
                self.manual_recovery_latched = False
                self.manual_motion_active = False

            else:

                # No explicit manual marker -> normal autonomous bridge
                # completion. Advance the tomato ID.
                self.manual_motion_active = False

                self.get_logger().info(
                    f"tomato_{self.tomato_index} "
                    "returned HOME normally."
                )

                self.finish_cycle()


    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot_observation(
        self,
        ee_pose,
        home_distance,
    ) -> None:

        target = (
            self.latest_target
        )

        if target is None:

            target_data = {
                "track_id": "",
                "tomato_class": "",
                "confidence": float("nan"),
                "x_cm": float("nan"),
                "y_cm": float("nan"),
                "z_cm": float("nan"),
                "depth_cm": float("nan"),
            }

        else:

            target_data = {
                "track_id":
                    str(target.track_id),

                "tomato_class":
                    str(target.tomato_class),

                "confidence":
                    float(target.confidence),

                "x_cm":
                    float(target.x_cm),

                "y_cm":
                    float(target.y_cm),

                "z_cm":
                    float(target.z_cm),

                "depth_cm":
                    float(target.depth_cm),
            }

        item = {
            "type":
                "frame",

            "tomato_index":
                self.tomato_index,

            "frame_index":
                self.frame_index,

            "timestamp":
                time.time(),

            "rgb_msg":
                self.latest_rgb,

            "left_msg":
                self.latest_left_tactile,

            "right_msg":
                self.latest_right_tactile,

            "joint_positions":
                dict(
                    self.latest_joint_positions
                ),

            "joint_velocities":
                dict(
                    self.latest_joint_velocities
                ),

            "ee_pose":
                tuple(
                    ee_pose
                ),

            "home_distance":
                float(
                    home_distance
                ),

            "gripper_command":
                list(
                    self.latest_gripper_command
                ),

            "target":
                target_data,

            "left_contact":
                self.left_contact,

            "right_contact":
                self.right_contact,

            "left_indentation_m":
                self.left_indentation_m,

            "right_indentation_m":
                self.right_indentation_m,

            "task_phase":
                self.task_phase,

            "manual_motion_active":
                (
                    self.manual_motion_active
                    or
                    self.manual_recovery_latched
                ),

            "paused":
                self.paused,

            "pause_reason":
                self.pause_reason,
        }

        if self.enqueue(
            item
        ):

            self.frame_index += 1


    # ========================================================
    # BACKGROUND WRITER
    # ========================================================

    def enqueue(
        self,
        item,
    ) -> bool:

        try:

            self.write_queue.put_nowait(
                item
            )

            return True

        except queue.Full:

            self.get_logger().warning(
                "Dataset writer queue full. "
                "Dropping one frame rather than "
                "blocking harvest control."
            )

            return False


    @staticmethod
    def ros_image_to_numpy(
        msg: Optional[Image],
    ) -> Optional[np.ndarray]:

        if msg is None:
            return None

        try:

            if msg.encoding in (
                "rgb8",
                "bgr8",
            ):

                image = np.frombuffer(
                    msg.data,
                    dtype=np.uint8,
                ).reshape(
                    msg.height,
                    msg.width,
                    3,
                )

                if msg.encoding == "rgb8":

                    image = cv2.cvtColor(
                        image,
                        cv2.COLOR_RGB2BGR,
                    )

                return image.copy()

            if msg.encoding == "mono8":

                return np.frombuffer(
                    msg.data,
                    dtype=np.uint8,
                ).reshape(
                    msg.height,
                    msg.width,
                ).copy()

        except Exception:

            return None

        return None


    @staticmethod
    def csv_header():

        return [
            "frame_index",
            "timestamp",

            "track_id",
            "tomato_class",
            "confidence",

            "rgb_file",
            "tactile_left_file",
            "tactile_right_file",

            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",

            "dq1",
            "dq2",
            "dq3",
            "dq4",
            "dq5",
            "dq6",
            "dq7",

            "ee_x",
            "ee_y",
            "ee_z",

            "ee_qx",
            "ee_qy",
            "ee_qz",
            "ee_qw",

            "home_distance_m",
            "home_reached",

            "gripper_left_position",
            "gripper_right_position",

            "gripper_left_command",
            "gripper_right_command",

            "target_x_cm",
            "target_y_cm",
            "target_z_cm",
            "target_depth_cm",

            "left_contact",
            "right_contact",

            "left_indentation_m",
            "right_indentation_m",

            "task_phase",
            "manual_motion_active",

            "paused",
            "pause_reason",
        ]


    def writer_loop(
        self,
    ) -> None:

        csv_file = None
        csv_writer = None

        episode_dir = None

        while not self.writer_stop.is_set():

            try:

                item = self.write_queue.get(
                    timeout=0.25
                )

            except queue.Empty:

                continue

            try:

                kind = item.get(
                    "type"
                )

                if kind == "open":

                    if csv_file is not None:

                        csv_file.flush()
                        csv_file.close()

                    tomato_index = int(
                        item[
                            "tomato_index"
                        ]
                    )

                    episode_dir = (
                        DATASET_ROOT
                        / f"tomato_{tomato_index}"
                    )

                    rgb_dir = (
                        episode_dir
                        / "rgb"
                    )

                    left_dir = (
                        episode_dir
                        / "tactile_left"
                    )

                    right_dir = (
                        episode_dir
                        / "tactile_right"
                    )

                    for directory in (
                        episode_dir,
                        rgb_dir,
                        left_dir,
                        right_dir,
                    ):

                        directory.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                    csv_file = open(
                        episode_dir
                        / "observations.csv",
                        "w",
                        newline="",
                    )

                    csv_writer = (
                        csv.writer(
                            csv_file
                        )
                    )

                    csv_writer.writerow(
                        self.csv_header()
                    )

                    csv_file.flush()

                elif kind == "frame":

                    if (
                        csv_writer is None
                        or
                        csv_file is None
                        or
                        episode_dir is None
                    ):

                        continue

                    frame = int(
                        item[
                            "frame_index"
                        ]
                    )

                    frame_name = (
                        f"{frame:06d}.png"
                    )

                    rgb_file = ""
                    left_file = ""
                    right_file = ""

                    rgb = (
                        self.ros_image_to_numpy(
                            item[
                                "rgb_msg"
                            ]
                        )
                    )

                    if rgb is not None:

                        path = (
                            episode_dir
                            / "rgb"
                            / frame_name
                        )

                        if cv2.imwrite(
                            str(path),
                            rgb,
                        ):

                            rgb_file = (
                                f"rgb/{frame_name}"
                            )

                    left = (
                        self.ros_image_to_numpy(
                            item[
                                "left_msg"
                            ]
                        )
                    )

                    if left is not None:

                        path = (
                            episode_dir
                            / "tactile_left"
                            / frame_name
                        )

                        if cv2.imwrite(
                            str(path),
                            left,
                        ):

                            left_file = (
                                f"tactile_left/"
                                f"{frame_name}"
                            )

                    right = (
                        self.ros_image_to_numpy(
                            item[
                                "right_msg"
                            ]
                        )
                    )

                    if right is not None:

                        path = (
                            episode_dir
                            / "tactile_right"
                            / frame_name
                        )

                        if cv2.imwrite(
                            str(path),
                            right,
                        ):

                            right_file = (
                                f"tactile_right/"
                                f"{frame_name}"
                            )

                    target = item[
                        "target"
                    ]

                    joints = item[
                        "joint_positions"
                    ]

                    velocities = item[
                        "joint_velocities"
                    ]

                    ee_pose = item[
                        "ee_pose"
                    ]

                    home_distance = item[
                        "home_distance"
                    ]

                    gripper = item[
                        "gripper_command"
                    ]

                    row = [
                        frame,
                        item["timestamp"],

                        target["track_id"],
                        target["tomato_class"],
                        target["confidence"],

                        rgb_file,
                        left_file,
                        right_file,
                    ]

                    for joint in ARM_JOINTS:

                        row.append(
                            joints[
                                joint
                            ]
                        )

                    for joint in ARM_JOINTS:

                        row.append(
                            velocities[
                                joint
                            ]
                        )

                    row.extend(
                        [
                            ee_pose[0],
                            ee_pose[1],
                            ee_pose[2],

                            ee_pose[3],
                            ee_pose[4],
                            ee_pose[5],
                            ee_pose[6],

                            home_distance,

                            int(
                                home_distance
                                <=
                                HOME_POSITION_TOLERANCE_M
                            ),

                            joints[
                                "fr3_finger_joint1"
                            ],

                            joints[
                                "fr3_finger_joint2"
                            ],

                            gripper[0],
                            gripper[1],

                            target["x_cm"],
                            target["y_cm"],
                            target["z_cm"],
                            target["depth_cm"],

                            item[
                                "left_contact"
                            ],

                            item[
                                "right_contact"
                            ],

                            item[
                                "left_indentation_m"
                            ],

                            item[
                                "right_indentation_m"
                            ],

                            item[
                                "task_phase"
                            ],

                            int(
                                item[
                                    "manual_motion_active"
                                ]
                            ),

                            int(
                                item[
                                    "paused"
                                ]
                            ),

                            item[
                                "pause_reason"
                            ],
                        ]
                    )

                    csv_writer.writerow(
                        row
                    )

                    # Flush each row, but no fsync.
                    # This is crash-resistant enough without
                    # heavily blocking the system.
                    csv_file.flush()

                elif kind == "close":

                    if csv_file is not None:

                        csv_file.flush()
                        csv_file.close()

                    csv_file = None
                    csv_writer = None
                    episode_dir = None

                elif kind == "stop":

                    break

            except Exception as error:

                print(
                    "[harvest_dataset_writer] "
                    f"{error}",
                    flush=True,
                )

            finally:

                self.write_queue.task_done()

        if csv_file is not None:

            try:

                csv_file.flush()
                csv_file.close()

            except Exception:

                pass


    # ========================================================
    # CLEANUP
    # ========================================================

    def destroy_node(
        self,
    ):

        try:

            self.write_queue.join()

        except Exception:

            pass

        self.writer_stop.set()

        try:

            self.write_queue.put_nowait(
                {
                    "type":
                        "stop"
                }
            )

        except Exception:

            pass

        try:

            self.writer_thread.join(
                timeout=3.0
            )

        except Exception:

            pass

        super().destroy_node()


def main(
    args=None,
) -> None:

    rclpy.init(
        args=args
    )

    node = (
        HarvestDatasetLogger()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()


