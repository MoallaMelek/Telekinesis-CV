"""Landmark inference, coordinate math, and a smoothed finger pointer."""

import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from model_assets import ensure_asset

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

# Each pair is an anatomical connection between two of the 21 landmark IDs.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
# OpenCV colors are BGR, not RGB.
HIGHLIGHTS = {
    0: ("wrist", (255, 180, 80)),
    4: ("thumb", (80, 200, 255)),
    8: ("index", (255, 255, 80)),
    12: ("middle", (230, 100, 255)),
}


def ensure_model():
    ensure_asset(MODEL_PATH, MODEL_URL, MODEL_SHA256, max_bytes=16 * 1024 * 1024)


class HandTracker:
    def __init__(self, num_hands=2):
        ensure_model()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            # Sequential VIDEO inference keeps each result aligned with its frame.
            # Tracking reuses the previous hand region instead of redetecting palms
            # on every frame. Webcam frames are still a timestamped video sequence.
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.model = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def detect_all(self, frame, timestamp_seconds):
        # The camera delivers BGR; the model expects RGB channel order.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(self.last_timestamp_ms + 1, int(timestamp_seconds * 1000))
        self.last_timestamp_ms = timestamp_ms
        result = self.model.detect_for_video(image, timestamp_ms)
        height, width = frame.shape[:2]
        # Normalized x/y measure fractions of image width/height. Convert each
        # axis separately: using width for both distorts geometry on a 4:3 frame.
        # Keep floats for later distance, smoothing, and velocity calculations.
        return [
            {"points": np.array([(p.x * width, p.y * height) for p in landmarks],
                                dtype=np.float64),
             "z": np.array([p.z for p in landmarks]),
             "label": result.handedness[i][0].category_name}
            for i, landmarks in enumerate(result.hand_landmarks)
        ]

    def detect(self, frame, timestamp_seconds):
        """Single-hand convenience for the earlier learning checkpoints."""
        hands = self.detect_all(frame, timestamp_seconds)
        return hands[0]["points"] if hands else None

    def close(self):
        self.model.close()


def smooth_pointer(previous, fingertip, dt, smoothing_seconds):
    """Time-based exponential smoothing; None means no tracked hand."""
    if fingertip is None:
        return None
    # Reacquisition starts at the new fingertip, never at a stale screen position.
    # Also reset after a long stall instead of drawing a misleading catch-up path.
    if previous is None or smoothing_seconds <= 0 or dt > 0.25:
        return fingertip.copy()
    # Solve ds/dt = (p - s)/tau over one timestep with p held constant.
    # Time-based alpha keeps the same response time when camera FPS changes.
    alpha = -math.expm1(-max(dt, 0.0) / smoothing_seconds)
    return previous + alpha * (fingertip - previous)


def update_pinch(points, was_pinched, pinch_threshold=0.30, release_threshold=0.45):
    """Return (pinched, tip_distance / palm_size), calculated from pixel geometry."""
    if points is None:
        return False, None
    relevant = points[[0, 4, 5, 8, 9, 17]]
    if not np.isfinite(relevant).all():
        return False, None

    def distance(a, b):
        dx, dy = points[a] - points[b]
        return math.hypot(float(dx), float(dy))  # sqrt(dx**2 + dy**2)

    # Palm length: wrist -> middle knuckle. Width: index -> pinky knuckle.
    # Average both so a single foreshortened axis has less influence.
    palm_size = (distance(0, 9) + distance(5, 17)) / 2.0
    if palm_size < 1e-6:  # Degenerate model geometry, not a gesture threshold.
        return False, None
    ratio = distance(4, 8) / palm_size
    # Both lengths scale together as the hand changes apparent size, so their
    # ratio cancels that scale. Perspective/occlusion still limit this 2D estimate.
    # In the gap between thresholds, remember the old state (hysteresis).
    threshold = release_threshold if was_pinched else pinch_threshold
    return (ratio < threshold if was_pinched else ratio <= threshold), ratio


def palm_geometry(points):
    center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    size = (np.linalg.norm(points[0] - points[9])
            + np.linalg.norm(points[5] - points[17])) / 2
    extended = curled = 0
    for base, joint, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a, b = points[base] - points[joint], points[tip] - points[joint]
        # Dot product gives cos(angle); -1 is a straight 180-degree finger.
        cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
        tip_distance = np.linalg.norm(points[tip] - points[0])
        joint_distance = np.linalg.norm(points[joint] - points[0])
        extended += cosine < -0.80 and tip_distance > joint_distance * 1.10
        curled += cosine > -0.25 and tip_distance < joint_distance * 1.05
    return center, float(size), extended == 4, curled >= 3


def estimate_velocity(history, now, horizon=0.14):
    """Least-squares slope of recent (seconds, x, y), in pixels/second."""
    samples = [sample for sample in history if now - sample[0] <= horizon]
    if len(samples) < 3 or now - samples[-1][0] > 0.10:
        return np.zeros(2)
    values = np.asarray(samples, dtype=float)
    t = values[:, 0] - values[:, 0].mean()
    if np.ptp(values[:, 0]) < 0.025:
        return np.zeros(2)
    # Fit p(t)=v*t+b: centering removes b; several samples reduce two-frame noise.
    return (t[:, None] * (values[:, 1:] - values[:, 1:].mean(axis=0))).sum(axis=0) / np.dot(t, t)


def draw_pinch_indicator(frame, center, state):
    """Outline = open, small inner dot = approaching, filled = pinched."""
    color = {"OPEN": (255, 255, 80), "APPROACHING": (80, 200, 255),
             "PINCHED": (100, 255, 130)}.get(state, (150, 150, 150))
    cv2.circle(frame, center, 10, (15, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(frame, center, 8, color, 2, cv2.LINE_AA)
    if state == "APPROACHING":
        cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)
    elif state == "PINCHED":
        cv2.circle(frame, center, 8, color, -1, cv2.LINE_AA)


def draw_pointer(frame, pointer, raw_tip, debug=False):
    center = tuple(np.rint(pointer).astype(int))
    if debug:
        raw_center = tuple(np.rint(raw_tip).astype(int))
        cv2.line(frame, raw_center, center, (80, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, raw_center, 3, (80, 180, 255), -1, cv2.LINE_AA)
    # Dark backing keeps a small reticle readable on light camera backgrounds.
    for color, thickness in (((15, 20, 20), 4), ((255, 255, 80), 1)):
        cv2.circle(frame, center, 12, color, thickness, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            start = (center[0] + dx * 9, center[1] + dy * 9)
            end = (center[0] + dx * 17, center[1] + dy * 17)
            cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_landmarks(frame, points, debug=False):
    pixels = np.rint(points).astype(int)
    for start, end in CONNECTIONS:
        cv2.line(frame, tuple(pixels[start]), tuple(pixels[end]),
                 (140, 230, 150), 2, cv2.LINE_AA)
    for landmark_id, point in enumerate(pixels):
        highlight = HIGHLIGHTS.get(landmark_id)
        color = highlight[1] if highlight else (235, 235, 235)
        center = tuple(point)
        cv2.circle(frame, center, 7 if highlight else 3, color, -1, cv2.LINE_AA)
        if highlight or debug:
            label = str(landmark_id) if debug else highlight[0]
            cv2.putText(frame, label, (center[0] + 9, center[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (center[0] + 9, center[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
