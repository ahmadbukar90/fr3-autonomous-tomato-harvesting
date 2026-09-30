#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


class ArucoGridBoardDetector(Node):
    def __init__(self):
        super().__init__("aruco_gridboard_detector")

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        self.board_frame = "aruco_gridboard"

        self.marker_length = 0.0335
        self.marker_separation = 0.00675
        self.markers_x = 5
        self.markers_y = 7

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250
        )

        self.board = cv2.aruco.GridBoard(
            (self.markers_x, self.markers_y),
            self.marker_length,
            self.marker_separation,
            self.dictionary,
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            self.detector_params,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/aruco_gridboard/pose",
            10,
        )

        self.debug_pub = self.create_publisher(
            Image,
            "/aruco_gridboard/debug",
            10,
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "/aruco_gridboard/marker",
            10,
        )

        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            10,
        )

        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10,
        )

        self.get_logger().info("ArUco GridBoard detector started")

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        debug_frame = np.ascontiguousarray(frame.copy(), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            self.get_logger().warn("No markers detected")
            self.publish_debug(debug_frame, msg.header)
            return

        self.get_logger().info(
            f"Detected marker IDs: {ids.flatten().tolist()}"
        )

        cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)

        if len(ids) < 4:
            self.publish_debug(debug_frame, msg.header)
            return

        obj_points, img_points = self.board.matchImagePoints(corners, ids)

        if obj_points is None or img_points is None:
            self.publish_debug(debug_frame, msg.header)
            return

        if len(obj_points) < 4:
            self.publish_debug(debug_frame, msg.header)
            return

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
        )

        if not success:
            self.publish_debug(debug_frame, msg.header)
            return

        cv2.drawFrameAxes(
            debug_frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            0.08,
        )

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = msg.header.frame_id

        pose_msg.pose.position.x = float(tvec[0][0])
        pose_msg.pose.position.y = float(tvec[1][0])
        pose_msg.pose.position.z = float(tvec[2][0])

        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        self.pose_pub.publish(pose_msg)

        tf_msg = TransformStamped()
        tf_msg.header = pose_msg.header
        tf_msg.child_frame_id = self.board_frame
        tf_msg.transform.translation.x = pose_msg.pose.position.x
        tf_msg.transform.translation.y = pose_msg.pose.position.y
        tf_msg.transform.translation.z = pose_msg.pose.position.z
        tf_msg.transform.rotation = pose_msg.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)

        self.publish_board_marker(pose_msg.header.stamp)
        self.publish_debug(debug_frame, msg.header)

    def publish_debug(self, frame, header):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        debug_msg = Image()
        debug_msg.header = header
        debug_msg.height = frame.shape[0]
        debug_msg.width = frame.shape[1]
        debug_msg.encoding = "bgr8"
        debug_msg.is_bigendian = False
        debug_msg.step = frame.shape[1] * 3
        debug_msg.data = frame.tobytes()

        self.debug_pub.publish(debug_msg)

    def publish_board_marker(self, stamp):
        board_width = (
            self.markers_x * self.marker_length
            + (self.markers_x - 1) * self.marker_separation
        )
        board_height = (
            self.markers_y * self.marker_length
            + (self.markers_y - 1) * self.marker_separation
        )

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.board_frame
        marker.ns = "aruco_gridboard"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = board_width / 2.0
        marker.pose.position.y = board_height / 2.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = board_width
        marker.scale.y = board_height
        marker.scale.z = 0.003

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.35

        self.marker_pub.publish(marker)

    def rotation_matrix_to_quaternion(self, r):
        q = np.empty((4,))
        trace = np.trace(r)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (r[2, 1] - r[1, 2]) * s
            q[1] = (r[0, 2] - r[2, 0]) * s
            q[2] = (r[1, 0] - r[0, 1]) * s
        else:
            i = np.argmax(np.diag(r))

            if i == 0:
                s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
                q[3] = (r[2, 1] - r[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (r[0, 1] + r[1, 0]) / s
                q[2] = (r[0, 2] + r[2, 0]) / s
            elif i == 1:
                s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
                q[3] = (r[0, 2] - r[2, 0]) / s
                q[0] = (r[0, 1] + r[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (r[1, 2] + r[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
                q[3] = (r[1, 0] - r[0, 1]) / s
                q[0] = (r[0, 2] + r[2, 0]) / s
                q[1] = (r[1, 2] + r[2, 1]) / s
                q[2] = 0.25 * s

        return q


def main(args=None):
    rclpy.init(args=args)
    node = ArucoGridBoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
