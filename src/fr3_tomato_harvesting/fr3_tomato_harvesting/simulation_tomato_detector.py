#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import time
import csv
from pathlib import Path

import json
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tomato_interfaces.msg import TomatoTarget

import cv2
import numpy as np

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

    total_color_pixels = (
        red_pixels
        + orange_pixels
        + green_pixels
    )

    if total_color_pixels == 0:
        return "unknown"

    # Classify according to the dominant HSV color.
    return max(
        color_counts,
        key=color_counts.get,
    )

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


def estimate_tomato_radius_m(
    intrinsics,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    depth_m: float,
) -> float:
    """Estimate tomato radius from bbox size, depth, and camera intrinsics."""

    if not np.isfinite(depth_m):
        return float("nan")

    bbox_width_px = float(x2 - x1)
    bbox_height_px = float(y2 - y1)

    diameter_x_m = (
        bbox_width_px
        * depth_m
        / intrinsics.fx
    )

    diameter_y_m = (
        bbox_height_px
        * depth_m
        / intrinsics.fy
    )

    estimated_diameter_m = 0.5 * (
        diameter_x_m + diameter_y_m
    )

    return float(
        0.5 * estimated_diameter_m
    )
    

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
    tomato_radius_m: float,
) -> tuple[float, float, float]:
    """
    Estimate tomato-center coordinates in the camera frame.

    The depth measurement corresponds to the visible surface.
    Move one estimated tomato radius farther along the viewing ray.
    """

    if (
        not np.isfinite(depth_m)
        or not np.isfinite(tomato_radius_m)
    ):
        return float("nan"), float("nan"), float("nan")

    surface_x = (
        (cx - intrinsics.ppx)
        * depth_m
        / intrinsics.fx
    )

    surface_y = (
        (cy - intrinsics.ppy)
        * depth_m
        / intrinsics.fy
    )

    surface_z = depth_m

    surface_point = np.array(
        [
            surface_x,
            surface_y,
            surface_z,
        ],
        dtype=float,
    )

    ray_norm = np.linalg.norm(
        surface_point
    )

    if ray_norm <= 1e-9:
        return float("nan"), float("nan"), float("nan")

    ray = surface_point / ray_norm

    center_point = (
        surface_point
        + tomato_radius_m * ray
    )

    return (
        float(center_point[0] * 100.0),
        float(center_point[1] * 100.0),
        float(center_point[2] * 100.0),
    )


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
        tomato_radius_m = float(det["radius_m"])

        depth_cm = (
            depth_m * 100.0
            if np.isfinite(depth_m)
            else float("nan")
        )

        x_cm, y_cm, z_cm = deproject_pixel_to_camera_cm(
            intrinsics,
            cx,
            cy,
            depth_m,
            tomato_radius_m,
        )

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
    tomato_radius_m = float(det["radius_m"])
    

    depth_cm = (
        depth_m * 100.0
        if np.isfinite(depth_m)
        else float("nan")
    )

    x_cm, y_cm, z_cm = deproject_pixel_to_camera_cm(
        intrinsics,
        cx,
        cy,
        depth_m,
        tomato_radius_m,
    )
    
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

def select_robotic_pick_targets(
    display_items: list[dict],
    intrinsics,
    now: float,
) -> list[dict]:
    """
    Return all robot-ready tomato targets in priority order.

    Priority:
      1) red > orange > green > unknown
      2) nearest valid depth first
      3) highest confidence first
      4) smaller track ID first
    """

    if not display_items:
        return []

    ripeness_priority = {
        "red": 0,
        "orange": 1,
        "green": 2,
        "unknown": 3,
    }

    def id_number(det: dict) -> int:
        raw_id = str(det.get("display_id", ""))
        digits = "".join(
            ch for ch in raw_id
            if ch.isdigit()
        )
        return int(digits) if digits else 10**9

    def target_key(det: dict) -> tuple:
        color = det.get("color", "unknown")
        depth_m = det.get(
            "dist_m",
            float("nan"),
        )
        conf = det.get("conf", 0.0)

        depth_key = (
            depth_m
            if np.isfinite(depth_m)
            else float("inf")
        )

        return (
            ripeness_priority.get(color, 3),
            depth_key,
            -float(conf),
            id_number(det),
        )

    ranked = sorted(
        display_items,
        key=target_key,
    )

    targets = []

    for det in ranked:

        cx = float(det["cx"])
        cy = float(det["cy"])
        depth_m = float(det["dist_m"])
        tomato_radius_m = float(
            det["radius_m"]
        )

        depth_cm = (
            depth_m * 100.0
            if np.isfinite(depth_m)
            else float("nan")
        )

        x_cm, y_cm, z_cm = (
            deproject_pixel_to_camera_cm(
                intrinsics,
                cx,
                cy,
                depth_m,
                tomato_radius_m,
            )
        )

        targets.append(
            {
                "timestamp": now,
                "track_id": det["display_id"],
                "class": det["color"],
                "confidence": float(det["conf"]),
                "bbox": (
                    det["x1"],
                    det["y1"],
                    det["x2"],
                    det["y2"],
                ),
                "depth_cm": depth_cm,
                "position_3d_cm": (
                    x_cm,
                    y_cm,
                    z_cm,
                ),
            }
        )

    return targets

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



@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float


class RosTomatoDetector(Node):
    def __init__(self) -> None:
        super().__init__("tomato_detector_ros")

        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

        self.get_logger().info(f"Loading YOLO model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

        self.target_pub = self.create_publisher(
            TomatoTarget,
            "/tomato_pick_target",
            10,
        )

        self.color_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.color_callback,
            qos_profile_sensor_data,
        )
        # Registered depth used for tomato localization.
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            qos_profile_sensor_data,
        )

        # Raw depth from the opposite D405 lens.
        # This is used for visualization only.

        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.latest_color: np.ndarray | None = None

        # Registered depth used by perception.
        self.latest_depth: np.ndarray | None = None

        # Raw depth from the opposite physical lens.
        # Visualization only.


        self.latest_color_stamp_ns: int | None = None
        self.latest_depth_stamp_ns: int | None = None

        self.camera_info: CameraInfo | None = None
        self.processing = False

        self.fps_avg = 0.0
        self.last_t = time.time()
        self.snap_id = 0
        self.save_msg = ""
        self.save_msg_until = 0.0
        self.tracker = MotionCompensatedTomatoTracker()
        self.prev_gray: np.ndarray | None = None
        self.tomato_state: dict[str, dict] = {}

        os.makedirs(SAVE_DIR, exist_ok=True)
        self.window = "MuJoCo ROS camera | YOLOv8 tomato color + depth"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        self.timer = self.create_timer(0.10, self.process_latest_frame)
        self.get_logger().info(
            "ROS tomato detector started. Listening to simulated RGB, depth, and camera_info topics."
        )

    @staticmethod
    def stamp_ns(msg: Image) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    @staticmethod
    def _infer_rgb_shape(msg: Image, channels: int = 3) -> tuple[int, int]:
        expected = int(msg.height) * int(msg.width) * channels
        if msg.height > 1 and msg.width > 1 and expected == len(msg.data):
            return int(msg.height), int(msg.width)

        if len(msg.data) == WIDTH * HEIGHT * channels:
            return HEIGHT, WIDTH

        pixels = len(msg.data) // channels
        inferred_width = int(round(np.sqrt(pixels * 4.0 / 3.0)))
        inferred_height = pixels // inferred_width if inferred_width > 0 else 0
        if inferred_width * inferred_height != pixels:
            raise ValueError(
                f"Cannot infer RGB image shape from {len(msg.data)} bytes "
                f"(reported {msg.width}x{msg.height})."
            )
        return inferred_height, inferred_width

    @staticmethod
    def _infer_depth_shape(msg: Image, item_size: int) -> tuple[int, int]:
        expected = int(msg.height) * int(msg.width) * item_size
        if msg.height > 1 and msg.width > 1 and expected == len(msg.data):
            return int(msg.height), int(msg.width)

        if len(msg.data) == WIDTH * HEIGHT * item_size:
            return HEIGHT, WIDTH

        pixels = len(msg.data) // item_size
        inferred_width = int(round(np.sqrt(pixels * 4.0 / 3.0)))
        inferred_height = pixels // inferred_width if inferred_width > 0 else 0
        if inferred_width * inferred_height != pixels:
            raise ValueError(
                f"Cannot infer depth image shape from {len(msg.data)} bytes "
                f"(reported {msg.width}x{msg.height})."
            )
        return inferred_height, inferred_width

    def decode_color(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        if encoding not in ("rgb8", "bgr8"):
            raise ValueError(f"Unsupported color encoding: {msg.encoding}")

        height, width = self._infer_rgb_shape(msg)
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    def decode_depth(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.upper()
        if encoding == "32FC1":
            height, width = self._infer_depth_shape(msg, 4)
            return np.frombuffer(msg.data, dtype=np.float32).reshape(height, width).copy()
        if encoding in ("16UC1", "MONO16"):
            height, width = self._infer_depth_shape(msg, 2)
            depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
            return depth_mm.astype(np.float32) / 1000.0
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

    def color_callback(self, msg: Image) -> None:
        try:
            self.latest_color = self.decode_color(msg)
            self.latest_color_stamp_ns = self.stamp_ns(msg)
        except Exception as error:
            self.get_logger().error(f"Color image decode failed: {error}")

    def depth_callback(self, msg: Image) -> None:
        try:
            self.latest_depth = self.decode_depth(msg)
            self.latest_depth_stamp_ns = self.stamp_ns(msg)
        except Exception as error:
            self.get_logger().error(f"Depth image decode failed: {error}")
            
    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def scaled_intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        if self.camera_info is None:
            fx = 0.7275045143362225 * width
            fy = 0.7275045143362225 * height
            return CameraIntrinsics(fx=fx, fy=fy, ppx=0.5 * width, ppy=0.5 * height)

        info = self.camera_info
        fx = float(info.k[0])
        fy = float(info.k[4])
        ppx = float(info.k[2])
        ppy = float(info.k[5])

        # The MuJoCo bridge currently publishes normalized intrinsics with
        # CameraInfo width/height equal to 1. Convert them to pixel units.
        if info.width <= 1 or info.height <= 1 or fx < 10.0 or fy < 10.0:
            fx *= width
            fy *= height
            ppx *= width
            ppy *= height
        else:
            sx = width / float(info.width)
            sy = height / float(info.height)
            fx *= sx
            fy *= sy
            ppx *= sx
            ppy *= sy

        return CameraIntrinsics(fx=fx, fy=fy, ppx=ppx, ppy=ppy)

    def process_latest_frame(self) -> None:
        global CONFIDENCE

        if self.processing or self.latest_color is None or self.latest_depth is None:
            return

        if self.latest_color_stamp_ns is not None and self.latest_depth_stamp_ns is not None:
            skew_s = abs(self.latest_color_stamp_ns - self.latest_depth_stamp_ns) / 1e9
            if skew_s > 0.20:
                return

        self.processing = True
        try:
            image = self.latest_color.copy()
            depth_m = self.latest_depth.copy()
            if depth_m.shape[:2] != image.shape[:2]:
                depth_m = cv2.resize(
                    depth_m,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            height, width = image.shape[:2]
            intrinsics = self.scaled_intrinsics(width, height)
            disp = image.copy()

            now = time.time()
            dt = now - self.last_t
            self.last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            self.fps_avg = (
                0.9 * self.fps_avg + 0.1 * inst_fps
                if self.fps_avg > 0
                else inst_fps
            )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            camera_motion = estimate_camera_motion(self.prev_gray, gray)
            self.prev_gray = gray.copy()

            start_inference = time.perf_counter()
            results = self.model.predict(
                source=image,
                conf=CONFIDENCE,
                imgsz=IMG_SIZE,
                verbose=False,
            )
            inference_time_ms = (time.perf_counter() - start_inference) * 1000.0

            counts = {"red": 0, "orange": 0, "green": 0, "unknown": 0}
            detections: list[dict] = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, min(width - 1, x1))
                    x2 = max(0, min(width, x2))
                    y1 = max(0, min(height - 1, y1))
                    y2 = max(0, min(height, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = image[y1:y2, x1:x2]
                    color = classify_tomato_color(crop)
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
         
                    dist_m = depth_at_tomato_center(
                        depth_m,
                        x1,
                        y1,
                        x2,
                        y2,
                    )

                    estimated_radius_m = estimate_tomato_radius_m(
                        intrinsics,
                        x1,
                        y1,
                        x2,
                        y2,
                        dist_m,
                    )
                    
                    print(
                        "SELECTED GEOMETRY RAW: "
                        f"bbox=({x1}, {y1}, {x2}, {y2}), "
                        f"cx={0.5 * (x1 + x2):.2f}, "
                        f"cy={0.5 * (y1 + y2):.2f}, "
                        f"surface_depth={dist_m:.4f} m, "
                        f"estimated_radius={estimated_radius_m * 1000.0:.2f} mm"
                    )

                    counts[
                        color if color in counts else "unknown"
                    ] += 1

                    detections.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": 0.5 * (x1 + x2),
                        "cy": 0.5 * (y1 + y2),
                        "color": color,
                        "conf": conf,
                        "dist_m": dist_m,
                        "radius_m": estimated_radius_m,
                    })

            detections = self.tracker.update(detections, camera_motion)
            display_items = sorted(detections, key=lambda d: (d["x1"], d["y1"]))

            update_tomato_state(self.tomato_state, display_items, intrinsics, now)
            robotic_pick_targets = select_robotic_pick_targets(
                display_items,
                intrinsics,
                now,
            )

            robotic_pick_target = (
                robotic_pick_targets[0]
                if robotic_pick_targets
                else None
            )

            for candidate in robotic_pick_targets:

                x_cm, y_cm, z_cm = (
                    candidate["position_3d_cm"]
                )

                if not all(
                    np.isfinite(v)
                    for v in (
                        x_cm,
                        y_cm,
                        z_cm,
                    )
                ):
                    continue

                msg = TomatoTarget()

                msg.track_id = str(
                    candidate["track_id"]
                )

                msg.tomato_class = str(
                    candidate["class"]
                )

                msg.confidence = float(
                    candidate["confidence"]
                )

                msg.depth_cm = float(
                    candidate["depth_cm"]
                )

                msg.x_cm = float(x_cm)
                msg.y_cm = float(y_cm)
                msg.z_cm = float(z_cm)

                msg.inference_time_ms = float(
                    inference_time_ms
                )

                self.target_pub.publish(msg)

            pick_target_id = (
                robotic_pick_target["track_id"]
                if robotic_pick_target is not None
                else None
            )
            flash_pick_box = int(now * 10) % 2 == 0

            for det in display_items:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                color = det["color"]
                conf = det["conf"]
                dist_m = det["dist_m"]
                display_id = det["display_id"]
                box_color = BOX_COLORS.get(color, BOX_COLORS["unknown"])
                cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, 2)
                if display_id == pick_target_id and flash_pick_box:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (160, 60, 20), 5)

                label = (
                    f"{display_id} {color} {conf:.2f} {dist_m * 100:.1f} cm"
                    if np.isfinite(dist_m)
                    else f"{display_id} {color} {conf:.2f} no depth"
                )
                draw_label(disp, label, (x1, y1 - 6), box_color)

            total = sum(counts.values())
            hud1 = f"FPS {self.fps_avg:.1f} | conf {CONFIDENCE:.2f}"
            hud2 = (
                f"Red {counts['red']} | Orange {counts['orange']} | "
                f"Green {counts['green']} | Unknown {counts['unknown']} | Total {total}"
            )
            hud3 = f"Inference {inference_time_ms:.2f} ms"
            for text, y in ((hud1, 22), (hud2, 46), (hud3, 94)):
                cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

            target_txt = "Pick target: none"
            if robotic_pick_target is not None:
                target_txt = (
                    f"Pick target: {robotic_pick_target['track_id']} "
                    f"{robotic_pick_target['class']} "
                    f"{robotic_pick_target['confidence']:.2f} "
                    f"{robotic_pick_target['depth_cm']:.1f} cm"
                )
            cv2.putText(disp, target_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, target_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

            # --------------------------------------------------------
            # DEPTH VISUALIZATION
            #
            # depth_m comes directly from /camera/depth/image_raw,
            # which is rendered from the opposite D405 lens.
            # --------------------------------------------------------

            depth_for_display = np.nan_to_num(
                depth_m,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(
                    depth_for_display,
                    alpha=255.0 / max(
                        MAX_DEPTH_M,
                        0.001,
                    ),
                ),
                cv2.COLORMAP_TURBO,
            )

            top_view = np.hstack(
                (
                    disp,
                    depth_vis,
                )
            )

            state_panel = draw_tomato_state_panel(
                self.tomato_state,
                top_view.shape[1],
            )

            cv2.imshow(
                self.window,
                np.vstack(
                    (
                        top_view,
                        state_panel,
                    )
                ),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rclpy.shutdown()
            elif key in (ord("+"), ord("=")):
                CONFIDENCE = min(0.95, CONFIDENCE + 0.05)
            elif key in (ord("-"), ord("_")):
                CONFIDENCE = max(0.05, CONFIDENCE - 0.05)
            elif key == ord("l"):
                for tid, row in sorted(self.tomato_state.items()):
                    x_cm, y_cm, z_cm = row["position_3d_cm"]
                    self.get_logger().info(
                        f"{tid}: class={row['class']} conf={row['confidence']:.2f} "
                        f"depth={row['depth_cm']:.1f}cm "
                        f"RobotXYZ=({x_cm:.1f},{y_cm:.1f},{z_cm:.1f})cm"
                    )
            elif key == ord("c"):
                self.snap_id += 1
                csv_path = os.path.join(SAVE_DIR, f"tomato_state_{self.snap_id:03d}.csv")
                save_tomato_state_csv(self.tomato_state, csv_path)
            elif key == ord("s"):
                self.snap_id += 1
                stem = os.path.join(SAVE_DIR, f"mujoco_yolo_{self.snap_id:03d}")
                cv2.imwrite(f"{stem}_rgb.png", image)
                cv2.imwrite(f"{stem}_overlay.png", disp)
                np.save(f"{stem}_depth_meters.npy", depth_m)
        except Exception as error:
            self.get_logger().error(f"Detector processing failed: {error}")
        finally:
            self.processing = False

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> int:
    rclpy.init(args=args)
    try:
        node = RosTomatoDetector()
    except Exception as error:
        print(f"Failed to start ROS tomato detector: {error}")
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
