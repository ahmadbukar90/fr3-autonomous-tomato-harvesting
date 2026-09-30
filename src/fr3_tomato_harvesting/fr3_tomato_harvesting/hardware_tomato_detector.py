#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import time
import csv
from pathlib import Path

import json
import rclpy
from rclpy.node import Node
from tomato_interfaces.msg import TomatoTarget

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not found. Run: pip install pyrealsense2")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not found. Run: python -m pip install ultralytics")
    sys.exit(1)

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(
    os.environ.get(
        "TOMATO_MODEL_PATH",
        PROJECT_ROOT / "models" / "best_v2.pt",
    )
)

WIDTH, HEIGHT, FPS = 640, 480, 30
# If frames time out, try: WIDTH, HEIGHT, FPS = 424, 240, 15

CONFIDENCE = 0.50
IMG_SIZE = 640
SAVE_DIR = PROJECT_ROOT / "data" / "d405_yolo_captures"
MIN_DEPTH_M, MAX_DEPTH_M = 0.07, 1.00
STATE_TTL_SEC = 10.0  # keep each tomato feature row for this many seconds

# BGR colors for display
BOX_COLORS = {
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "green": (0, 200, 0),
    "unknown": (180, 180, 180),
}


# ---------------------------------------------------------------------------
# Your notebook color-classification logic, adapted for live BGR crops
# ---------------------------------------------------------------------------
def classify_tomato_color(crop: np.ndarray) -> str:
    """
    Classify tomato crop as red, orange, or green based on the dominant
    valid tomato-color pixels in the bounding box.
    """
    if crop is None or crop.size == 0:
        return "unknown"

    crop_blur = cv2.GaussianBlur(crop, (5, 5), 0)
    hsv = cv2.cvtColor(crop_blur, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)

    if w > 20 and h > 20:
        hsv = hsv[margin_y:h - margin_y, margin_x:w - margin_x]

    valid_mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))

    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    orange_mask = cv2.inRange(hsv, (11, 50, 50), (30, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 45, 45), (85, 255, 255))

    red_mask = cv2.bitwise_and(red_mask, valid_mask)
    orange_mask = cv2.bitwise_and(orange_mask, valid_mask)
    green_mask = cv2.bitwise_and(green_mask, valid_mask)

    red_pixels = cv2.countNonZero(red_mask)
    orange_pixels = cv2.countNonZero(orange_mask)
    green_pixels = cv2.countNonZero(green_mask)

    color_counts = {
        "red": red_pixels,
        "orange": orange_pixels,
        "green": green_pixels,
    }

    total_color_pixels = red_pixels + orange_pixels + green_pixels
    if total_color_pixels == 0:
        return "unknown"

    red_ratio = red_pixels / total_color_pixels
    orange_ratio = orange_pixels / total_color_pixels

    if red_ratio >= 0.25:
        return "red"

    if orange_ratio >= 0.30:
        return "orange"

    return max(color_counts, key=color_counts.get)


def depth_at_tomato_center(depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Return robust depth at the tomato center in meters.

    Instead of averaging the inner 60% of the bounding box, this uses a small
    circular neighborhood around the bounding-box center. This makes the depth
    correspond more closely to the tomato center while still rejecting noisy
    single-pixel depth values.
    """
    h, w = depth_m.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return float("nan")

    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))

    # Use a small center disk: 8% of the smaller bbox dimension, at least 3 px.
    radius = max(3, int(0.08 * min(x2 - x1, y2 - y1)))

    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius + 1)
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius + 1)

    roi = depth_m[y_min:y_max, x_min:x_max]
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

    values = roi[mask]
    valid = values[(values >= MIN_DEPTH_M) & (values <= MAX_DEPTH_M)]

    if valid.size < 5:
        return float("nan")

    return float(np.median(valid))


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y + 4), color, -1)
    cv2.putText(img, text, (x + 4, y - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)



# ---------------------------------------------------------------------------
# Motion-compensated tomato ID tracker
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tomato feature storage for robotic system / logging
# ---------------------------------------------------------------------------
def deproject_pixel_to_camera_cm(
    intrinsics,
    cx: float,
    cy: float,
    depth_m: float,
) -> tuple[float, float, float]:
    """Convert 2D pixel center + depth into robot-configured XYZ in cm.

    RealSense camera coordinates are first computed as:
        camera X = horizontal, camera Y = vertical, camera Z = forward/depth.

    Your robotic arm coordinate convention is mapped as:
        robot X = camera Z
        robot Y = camera X
        robot Z = camera Y

    Therefore, the returned (X, Y, Z) values are robot-configured coordinates,
    while depth_cm is still the direct distance from the camera to the tomato.
    """
    if not np.isfinite(depth_m):
        return float("nan"), float("nan"), float("nan")

    cam_z_cm = depth_m * 100.0
    cam_x_cm = (cx - intrinsics.ppx) * depth_m / intrinsics.fx * 100.0
    cam_y_cm = (cy - intrinsics.ppy) * depth_m / intrinsics.fy * 100.0

    robot_x_cm = cam_z_cm
    robot_y_cm = cam_x_cm
    robot_z_cm = cam_y_cm
    return float(robot_x_cm), float(robot_y_cm), float(robot_z_cm)


def update_tomato_state(
    tomato_state: dict[str, dict],
    display_items: list[dict],
    intrinsics,
    now: float,
) -> None:
    """Store the latest feature row for each currently visible tomato ID.

    Each row is kept for STATE_TTL_SEC seconds after it was last seen.
    This avoids unbounded memory growth while keeping recent robot targets.
    """
    for det in display_items:
        track_id = det["display_id"]
        cx = float(det["cx"])
        cy = float(det["cy"])
        depth_m = float(det["dist_m"])
        depth_cm = depth_m * 100.0 if np.isfinite(depth_m) else float("nan")
        x_cm, y_cm, z_cm = deproject_pixel_to_camera_cm(intrinsics, cx, cy, depth_m)

        tomato_state[track_id] = {
            "timestamp": now,
            "track_id": track_id,
            "class": det["color"],
            "confidence": float(det["conf"]),
            "center_2d": (cx, cy),
            "depth_cm": depth_cm,
            "position_3d_cm": (x_cm, y_cm, z_cm),
        }

    # Remove stale rows older than STATE_TTL_SEC.
    stale_ids = [
        tid for tid, row in tomato_state.items()
        if now - row["timestamp"] > STATE_TTL_SEC
    ]
    for tid in stale_ids:
        del tomato_state[tid]



def select_robotic_pick_target(display_items: list[dict], intrinsics, now: float) -> dict | None:
    """Choose the tomato the robotic system should pick next.

    Priority order:
      1) ripeness/color: red > orange > green > unknown,
      2) distance: nearest valid depth first,
      3) confidence: highest confidence first.

    Returns a dictionary containing all robot-ready features for the selected
    tomato, or None if there are no detections. This variable is updated every
    frame and can be passed to the robot controller.
    """
    if not display_items:
        return None

    ripeness_priority = {
        "red": 0,
        "orange": 1,
        "green": 2,
        "unknown": 3,
    }

    def id_number(det: dict) -> int:
        """Extract the numeric part of IDs like #1, #O1, #G1, #U1."""
        raw_id = str(det.get("display_id", ""))
        digits = "".join(ch for ch in raw_id if ch.isdigit())
        return int(digits) if digits else 10**9

    def target_key(det: dict) -> tuple:
        color = det.get("color", "unknown")
        depth_m = det.get("dist_m", float("nan"))
        conf = det.get("conf", 0.0)
        depth_key = depth_m if np.isfinite(depth_m) else float("inf")
        return (
            ripeness_priority.get(color, 3),
            depth_key,
            -float(conf),
            id_number(det),
        )

    # First choose by the main criteria: ripeness, distance, confidence.
    ranked = sorted(display_items, key=target_key)
    best = ranked[0]

    # If multiple tomatoes are almost the same pick quality, force only the
    # smaller Track ID to be selected. This prevents two close targets from
    # visually competing/flickering as the pick target between frames.
    PICK_DEPTH_TIE_M = 0.03      # 3 cm tolerance for "very close" distance
    PICK_CONF_TIE = 0.05         # confidence values within this are treated as tied

    best_color_rank = ripeness_priority.get(best.get("color", "unknown"), 3)
    best_depth = best.get("dist_m", float("nan"))
    best_conf = float(best.get("conf", 0.0))

    tied_candidates = []
    for cand in ranked:
        if ripeness_priority.get(cand.get("color", "unknown"), 3) != best_color_rank:
            continue

        cand_depth = cand.get("dist_m", float("nan"))
        if np.isfinite(best_depth) and np.isfinite(cand_depth):
            if abs(float(cand_depth) - float(best_depth)) > PICK_DEPTH_TIE_M:
                continue
        elif np.isfinite(best_depth) != np.isfinite(cand_depth):
            continue

        cand_conf = float(cand.get("conf", 0.0))
        if abs(cand_conf - best_conf) > PICK_CONF_TIE:
            continue

        tied_candidates.append(cand)

    det = min(tied_candidates, key=id_number) if tied_candidates else best

    cx = float(det["cx"])
    cy = float(det["cy"])
    depth_m = float(det["dist_m"])
    depth_cm = depth_m * 100.0 if np.isfinite(depth_m) else float("nan")
    x_cm, y_cm, z_cm = deproject_pixel_to_camera_cm(intrinsics, cx, cy, depth_m)

    return {
        "timestamp": now,
        "track_id": det["display_id"],
        "class": det["color"],
        "confidence": float(det["conf"]),
        "bbox": (det["x1"], det["y1"], det["x2"], det["y2"]),
        "center_2d": (cx, cy),
        "depth_cm": depth_cm,
        "position_3d_cm": (x_cm, y_cm, z_cm),
    }

def save_tomato_state_csv(tomato_state: dict[str, dict], filename: str) -> None:
    """Save the current 10-second tomato state table to CSV."""
    fieldnames = [
        "timestamp", "track_id", "class", "confidence",
        "cx", "cy", "depth_cm", "X_cm", "Y_cm", "Z_cm",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tomato_state.values():
            cx, cy = row["center_2d"]
            x_cm, y_cm, z_cm = row["position_3d_cm"]
            writer.writerow({
                "timestamp": row["timestamp"],
                "track_id": row["track_id"],
                "class": row["class"],
                "confidence": row["confidence"],
                "cx": cx,
                "cy": cy,
                "depth_cm": row["depth_cm"],
                "X_cm": x_cm,
                "Y_cm": y_cm,
                "Z_cm": z_cm,
            })


def draw_tomato_state_panel(tomato_state: dict[str, dict], width: int, max_rows: int = 12) -> np.ndarray:
    """Create a live table panel showing robot-ready tomato features.

    The table is drawn below the RGB/depth views, so it does not cover the
    detections. It shows the latest state stored in tomato_state.
    """
    row_h = 24
    header_h = 58
    panel_h = header_h + row_h * max_rows + 14
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    title = "Tomato State Table - robot XYZ coordinates, kept for last 10 s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, title, (10, 22), font, 0.58, (230, 230, 230), 1, cv2.LINE_AA)

    columns = [
        (10, "Track ID"),
        (105, "Class"),
        (190, "Conf"),
        (255, "2D center (cx,cy)"),
        (430, "Depth cm"),
        (545, "X cm"),
        (655, "Y cm"),
        (765, "Z cm"),
    ]

    y_header = 48
    for x, text in columns:
        cv2.putText(panel, text, (x, y_header), font, 0.47, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(panel, (8, 55), (width - 8, 55), (90, 90, 90), 1)

    def sort_key(item):
        tid, row = item
        cls = row.get("class", "unknown")
        depth = row.get("depth_cm", float("inf"))
        group_order = {"red": 0, "orange": 1, "green": 2, "unknown": 3}.get(cls, 4)
        depth_key = depth if np.isfinite(depth) else float("inf")
        return (group_order, depth_key, tid)

    rows = sorted(tomato_state.items(), key=sort_key)[:max_rows]

    for i, (tid, row) in enumerate(rows):
        y = header_h + i * row_h
        cls = row.get("class", "unknown")
        bgr = BOX_COLORS.get(cls, BOX_COLORS["unknown"])

        # Subtle row separator and color stripe.
        cv2.line(panel, (8, y + 6), (width - 8, y + 6), (45, 45, 45), 1)
        cv2.rectangle(panel, (10, y + 9), (22, y + 21), bgr, -1)

        cx, cy = row.get("center_2d", (float("nan"), float("nan")))
        x_cm, y_cm, z_cm = row.get("position_3d_cm", (float("nan"), float("nan"), float("nan")))
        depth_cm = row.get("depth_cm", float("nan"))
        conf = row.get("confidence", float("nan"))

        values = [
            (32, str(tid)),
            (105, cls),
            (190, f"{conf:.2f}" if np.isfinite(conf) else "nan"),
            (255, f"({cx:.0f}, {cy:.0f})" if np.isfinite(cx) and np.isfinite(cy) else "(nan,nan)"),
            (430, f"{depth_cm:.1f}" if np.isfinite(depth_cm) else "nan"),
            (545, f"{x_cm:.1f}" if np.isfinite(x_cm) else "nan"),
            (655, f"{y_cm:.1f}" if np.isfinite(y_cm) else "nan"),
            (765, f"{z_cm:.1f}" if np.isfinite(z_cm) else "nan"),
        ]

        for x, text in values:
            cv2.putText(panel, text, (x, y + 23), font, 0.48, (235, 235, 235), 1, cv2.LINE_AA)

    if not rows:
        cv2.putText(panel, "No tomato state available", (10, header_h + 24), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    elif len(tomato_state) > max_rows:
        more = f"... {len(tomato_state) - max_rows} more rows not shown"
        cv2.putText(panel, more, (10, panel_h - 10), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return panel


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def estimate_camera_motion(prev_gray: np.ndarray | None, gray: np.ndarray) -> np.ndarray:
    """Estimate camera motion from previous frame to current frame.

    Returns a 3x3 homography-like transform from the previous image to the
    current image. It first tries ORB feature matching + RANSAC homography,
    which handles stronger camera motion and perspective changes better than a
    simple translation. If that fails, it falls back to Lucas-Kanade optical
    flow + affine estimation. If both fail, it returns identity.
    """
    identity = np.eye(3, dtype=np.float32)
    if prev_gray is None:
        return identity

    # --- Stronger option: ORB feature matching + homography -----------------
    try:
        orb = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8)
        kp0, des0 = orb.detectAndCompute(prev_gray, None)
        kp1, des1 = orb.detectAndCompute(gray, None)

        if des0 is not None and des1 is not None and len(kp0) >= 30 and len(kp1) >= 30:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn = matcher.knnMatch(des0, des1, k=2)
            good = []
            for pair in knn:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) >= 20:
                pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
                if H is not None and mask is not None and int(mask.sum()) >= 15:
                    return H.astype(np.float32)
    except cv2.error:
        pass

    # --- Fallback: optical flow + affine transform --------------------------
    pts0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if pts0 is None or len(pts0) < 20:
        return identity

    pts1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        pts0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if pts1 is None or status is None:
        return identity

    good0 = pts0[status.ravel() == 1]
    good1 = pts1[status.ravel() == 1]
    if len(good0) < 20:
        return identity

    A, _ = cv2.estimateAffinePartial2D(
        good0.reshape(-1, 2),
        good1.reshape(-1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.98,
    )
    if A is None:
        return identity

    H = np.eye(3, dtype=np.float32)
    H[:2, :] = A.astype(np.float32)
    return H


def apply_affine_to_point(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply a 3x3 homography/affine transform to a 2D point."""
    if M.shape == (2, 3):
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return float(nx), float(ny)

    p = M @ np.array([x, y, 1.0], dtype=np.float32)
    if abs(float(p[2])) < 1e-6:
        return float(x), float(y)
    return float(p[0] / p[2]), float(p[1] / p[2])


class MotionCompensatedTomatoTracker:
    """Simple tomato tracker tailored to this RGB-D greenhouse application.

    The tracker does NOT use ByteTrack/BoT-SORT. It keeps a real Track ID only
    when the new detection is consistent with a previous track in:
      1) image-centroid distance,
      2) depth difference,
      3) bounding-box size similarity,
      4) ripeness/color class consistency.

    If no existing track passes these checks, a new real Track ID is created.

    IDs are color-specific and persistent:
      red     -> #1, #2, #3, ...
      orange  -> #O1, #O2, #O3, ...
      green   -> #G1, #G2, #G3, ...
      unknown -> #U1, #U2, #U3, ...
    """
    def __init__(self) -> None:
        self.tracks: dict[str, dict] = {}
        self.frame_index = 0

        # Independent ID counters for each ripeness/color class.
        self.next_id_by_color = {
            "red": 1,
            "orange": 1,
            "green": 1,
            "unknown": 1,
        }

        # Tracks are removed only after being unseen for this many frames.
        # They are not reused, so IDs remain usable for logging/picking/mapping.
        self.max_missed_frames = 90

        # Consistency gates. Tune these if your IDs are too strict/too loose.
        self.max_centroid_distance_px = 140.0
        self.max_depth_difference_m = 0.30
        self.max_size_ratio_difference = 0.60
        self.min_iou_for_bonus = 0.05

        # Matching score weights. Lower score = better match.
        self.depth_weight = 180.0
        self.size_weight = 55.0
        self.iou_bonus = 35.0

    def _new_id(self, color: str) -> str:
        color = color if color in self.next_id_by_color else "unknown"
        number = self.next_id_by_color[color]
        self.next_id_by_color[color] += 1

        if color == "red":
            return f"#{number}"
        if color == "orange":
            return f"#O{number}"
        if color == "green":
            return f"#G{number}"
        return f"#U{number}"

    @staticmethod
    def _area(det_or_track: dict) -> float:
        return float(max(1, (det_or_track["x2"] - det_or_track["x1"]) * (det_or_track["y2"] - det_or_track["y1"])))

    def _make_track_record(self, det: dict, track_id: str) -> dict:
        return {
            "id": track_id,
            "cx": det["cx"],
            "cy": det["cy"],
            "x1": det["x1"],
            "y1": det["y1"],
            "x2": det["x2"],
            "y2": det["y2"],
            "dist_m": det["dist_m"],
            "color": det["color"],
            "area": self._area(det),
            "last_seen": self.frame_index,
        }

    def _remove_stale_tracks(self) -> None:
        stale_ids = [
            tid for tid, tr in self.tracks.items()
            if self.frame_index - tr["last_seen"] > self.max_missed_frames
        ]
        for tid in stale_ids:
            del self.tracks[tid]

    def update(self, detections: list[dict], camera_motion: np.ndarray | None = None) -> list[dict]:
        self.frame_index += 1

        # Predict old track positions into the current frame using camera motion.
        # If camera_motion is unavailable, this becomes identity-like behavior.
        predicted_tracks: dict[str, dict] = {}
        for tid, tr in self.tracks.items():
            if camera_motion is not None:
                pcx, pcy = apply_affine_to_point(camera_motion, tr["cx"], tr["cy"])
            else:
                pcx, pcy = tr["cx"], tr["cy"]

            bw = tr["x2"] - tr["x1"]
            bh = tr["y2"] - tr["y1"]
            predicted_tracks[tid] = {
                **tr,
                "pcx": pcx,
                "pcy": pcy,
                "pbox": (
                    int(pcx - bw / 2), int(pcy - bh / 2),
                    int(pcx + bw / 2), int(pcy + bh / 2),
                ),
            }

        candidates: list[tuple[float, int, str]] = []

        for di, det in enumerate(detections):
            det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
            det_area = self._area(det)

            for tid, tr in predicted_tracks.items():
                # 1) Ripeness/color class consistency: must match.
                if det["color"] != tr["color"]:
                    continue

                # 2) Image-centroid distance.
                centroid_dist = float(np.hypot(det["cx"] - tr["pcx"], det["cy"] - tr["pcy"]))
                if centroid_dist > self.max_centroid_distance_px:
                    continue

                # 3) Depth consistency.
                if np.isfinite(det["dist_m"]) and np.isfinite(tr["dist_m"]):
                    depth_diff = abs(float(det["dist_m"] - tr["dist_m"]))
                    if depth_diff > self.max_depth_difference_m:
                        continue
                else:
                    # If depth is missing, allow matching but penalize it.
                    depth_diff = 0.15

                # 4) Bounding-box size similarity.
                old_area = max(1.0, float(tr.get("area", det_area)))
                size_ratio_diff = abs(det_area - old_area) / max(det_area, old_area)
                if size_ratio_diff > self.max_size_ratio_difference:
                    continue

                # IoU is not mandatory, but overlapping predicted/current boxes
                # make the candidate more likely to be the same tomato.
                iou = box_iou(det_box, tr["pbox"])

                score = (
                    centroid_dist
                    + self.depth_weight * depth_diff
                    + self.size_weight * size_ratio_diff
                    - self.iou_bonus * iou
                )
                candidates.append((score, di, tid))

        candidates.sort(key=lambda x: x[0])
        assigned_dets: set[int] = set()
        assigned_tracks: set[str] = set()

        # Greedy best-score assignment. Simple, transparent, and suitable for
        # a small number of tomatoes in the camera view.
        for _, di, tid in candidates:
            if di in assigned_dets or tid in assigned_tracks:
                continue
            det = detections[di]
            det["id"] = tid
            det["display_id"] = tid
            self.tracks[tid] = self._make_track_record(det, tid)
            assigned_dets.add(di)
            assigned_tracks.add(tid)

        # New tracks for detections that did not match anything consistently.
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            tid = self._new_id(det["color"])
            det["id"] = tid
            det["display_id"] = tid
            self.tracks[tid] = self._make_track_record(det, tid)

        self._remove_stale_tracks()
        detections.sort(key=lambda d: (d["color"], d.get("id", "")))
        return detections



def main() -> int:
    global CONFIDENCE
    
    rclpy.init()
    node = Node("tomato_detector")
    target_pub = node.create_publisher(TomatoTarget, "/tomato_pick_target", 10)

    if not Path(MODEL_PATH).exists():
        print(f"Error: model not found: {MODEL_PATH}")
        print("Set MODEL_PATH near the top of this script to your trained best_v2.onnx file.")
        print("Example: MODEL_PATH = 'runs/detect/train-3/weights/best_v2.onnx'")
        return 1

    print(f"[YOLO] loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    print("[D405] starting pipeline...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start camera: {e}")
        print("Close realsense-viewer/other camera apps, replug the D405, and use USB 3.0+.")
        return 1

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_stream.get_intrinsics()
    align = rs.align(rs.stream.color)
    os.makedirs(SAVE_DIR, exist_ok=True)

    window = "D405 | YOLOv8 tomato color + depth"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps_avg = 0.0
    last_t = time.time()
    snap_id = 0
    save_msg = ""
    save_msg_until = 0.0
    tracker = MotionCompensatedTomatoTracker()
    prev_gray = None
    tomato_state: dict[str, dict] = {}
    robotic_pick_target: dict | None = None

    print("Hotkeys: q/ESC quit | +/- confidence | s save snapshot | c save tomato CSV | l print tomato state")

    try:
        while True:
            try:
                frames = align.process(pipeline.wait_for_frames(10000))
            except RuntimeError as e:
                print(f"Frame error: {e}")
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            image = np.asanyarray(color_frame.get_data())  # BGR
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            disp = image.copy()

            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            camera_motion = estimate_camera_motion(prev_gray, gray)
            prev_gray = gray.copy()

            start_inference = time.perf_counter()
            results = model.predict(source=image, conf=CONFIDENCE, imgsz=IMG_SIZE, verbose=False)
            end_inference = time.perf_counter()
            inference_time_ms = (end_inference - start_inference) * 1000.0

            red_count = 0
            orange_count = 0
            green_count = 0
            unknown_count = 0

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, min(WIDTH - 1, x1))
                    x2 = max(0, min(WIDTH, x2))
                    y1 = max(0, min(HEIGHT - 1, y1))
                    y2 = max(0, min(HEIGHT, y2))

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    color = classify_tomato_color(crop)
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    dist_m = depth_at_tomato_center(depth_m, x1, y1, x2, y2)

                    if color == "red":
                        red_count += 1
                    elif color == "orange":
                        orange_count += 1
                    elif color == "green":
                        green_count += 1
                    else:
                        unknown_count += 1

                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": 0.5 * (x1 + x2), "cy": 0.5 * (y1 + y2),
                        "color": color, "conf": conf, "dist_m": dist_m,
                    })

            detections = tracker.update(detections, camera_motion)

            # Draw detections after counting.
            # IDs are real persistent Track IDs from the custom tracker, not
            # distance-ranked display numbers. New IDs are only created when a
            # detection is not consistent with an existing track in centroid,
            # depth, bounding-box size, and ripeness/color class.
            display_items = list(detections)
            display_items.sort(key=lambda d: (d["x1"], d["y1"]))

            update_tomato_state(tomato_state, display_items, color_intrinsics, now)

            robotic_pick_target = select_robotic_pick_target(
                display_items,
                color_intrinsics,
                now,
            )

                
            if robotic_pick_target is not None:
                x_cm, y_cm, z_cm = robotic_pick_target["position_3d_cm"]

                msg = TomatoTarget()
                msg.track_id = str(robotic_pick_target["track_id"])
                msg.tomato_class = str(robotic_pick_target["class"])
                msg.confidence = float(robotic_pick_target["confidence"])
                msg.depth_cm = float(robotic_pick_target["depth_cm"])
                msg.x_cm = float(x_cm)
                msg.y_cm = float(y_cm)
                msg.z_cm = float(z_cm)
                msg.inference_time_ms = float(inference_time_ms)

                target_pub.publish(msg)
                

            pick_target_id = (
                robotic_pick_target["track_id"]
                if robotic_pick_target is not None
                else None
            )

            flash_pick_box = int(now * 10) % 2 == 0

            for det in display_items:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                color, conf, dist_m = det["color"], det["conf"], det["dist_m"]
                display_id = det["display_id"]
                box_color = BOX_COLORS.get(color, BOX_COLORS["unknown"])
                cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, 2)

                # Flash the selected robotic pick target in yellow.
                if display_id == pick_target_id and flash_pick_box:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 5)

                if np.isfinite(dist_m):
                    label = f"{display_id} {color} {conf:.2f} {dist_m * 100:.1f} cm"
                else:
                    label = f"{display_id} {color} {conf:.2f} no depth"
                draw_label(disp, label, (x1, y1 - 6), box_color)

            total = red_count + orange_count + green_count + unknown_count
            hud1 = f"FPS {fps_avg:.1f} | conf {CONFIDENCE:.2f}"
            hud2 = (
                f"Red {red_count} | Orange {orange_count} | "
                f"Green {green_count} | Unknown {unknown_count} | Total {total}"
            )
            hud3 = f"Inference {inference_time_ms:.2f} ms"
            cv2.putText(disp, hud1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, hud1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, hud2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, hud2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, hud3, (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, hud3, (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

            if robotic_pick_target is not None:
                target_txt = (
                    f"Pick target: {robotic_pick_target['track_id']} "
                    f"{robotic_pick_target['class']} "
                    f"{robotic_pick_target['confidence']:.2f} "
                    f"{robotic_pick_target['depth_cm']:.1f} cm"
                )
            else:
                target_txt = "Pick target: none"
            cv2.putText(disp, target_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, target_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

            if save_msg and now < save_msg_until:
                cv2.putText(disp, save_msg, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 140), 2, cv2.LINE_AA)

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.06),
                cv2.COLORMAP_TURBO,
            )
            top_view = np.hstack((disp, depth_vis))
            state_panel = draw_tomato_state_panel(tomato_state, top_view.shape[1])
            cv2.imshow(window, np.vstack((top_view, state_panel)))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                CONFIDENCE = min(0.95, CONFIDENCE + 0.05)
                print(f"[conf] {CONFIDENCE:.2f}")
            elif key in (ord("-"), ord("_")):
                CONFIDENCE = max(0.05, CONFIDENCE - 0.05)
                print(f"[conf] {CONFIDENCE:.2f}")
            elif key == ord("l"):
                print("[tomato_state] current rows:")
                for tid, row in sorted(tomato_state.items()):
                    x_cm, y_cm, z_cm = row["position_3d_cm"]
                    cx, cy = row["center_2d"]
                    print(
                        f"  {tid}: class={row['class']} conf={row['confidence']:.2f} "
                        f"center=({cx:.1f},{cy:.1f}) depth={row['depth_cm']:.1f}cm "
                        f"RobotXYZ=({x_cm:.1f},{y_cm:.1f},{z_cm:.1f})cm"
                    )
            elif key == ord("c"):
                snap_id += 1
                csv_path = os.path.join(SAVE_DIR, f"tomato_state_{snap_id:03d}.csv")
                save_tomato_state_csv(tomato_state, csv_path)
                save_msg = f"saved {os.path.basename(csv_path)}"
                save_msg_until = now + 2.5
                print(f"[CSV] {csv_path}")
            elif key == ord("s"):
                snap_id += 1
                stem = os.path.join(SAVE_DIR, f"d405_yolo_{snap_id:03d}")
                cv2.imwrite(f"{stem}_rgb.png", image)
                cv2.imwrite(f"{stem}_overlay.png", disp)
                np.save(f"{stem}_depth_meters.npy", depth_m)
                save_msg = f"saved {os.path.basename(stem)}_*"
                save_msg_until = now + 2.5
                print(f"[SAVE] {save_msg}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
