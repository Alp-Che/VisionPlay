"""Thin wrapper around MediaPipe Tasks HandLandmarker for a live camera loop."""
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from core.geometry import is_hand_closed
from core.paths import resource


def hand_span(hand):
    """Wrist-to-knuckles distance in pixels. Shrinks as the player steps back,
    so it doubles as the game's 'how far away is the player' signal."""
    (ax, ay), (bx, by) = hand.landmarks_px[0], hand.landmarks_px[9]
    return math.hypot(bx - ax, by - ay)


def hand_angle(hand):
    """Direction the hand points, wrist through knuckles, as an angle in the
    usual atan2(dy, dx) image convention."""
    (ax, ay), (bx, by) = hand.landmarks_px[0], hand.landmarks_px[9]
    return math.atan2(by - ay, bx - ax)


def hand_foreshorten(hand):
    """How much of the hand's length actually faces the camera, 0..1.

    1.0 means the hand lies flat across the view; near 0 means it is pointing
    straight at (or away from) the lens, where anything held in it should
    collapse to almost nothing on screen. Uses the metric 3D landmarks, so it
    is the true out-of-plane tilt rather than a guess from the flat picture.
    """
    world = hand.world_landmarks
    if not world:
        return 1.0
    wrist, knuckle = world[0], world[9]
    dx, dy, dz = knuckle.x - wrist.x, knuckle.y - wrist.y, knuckle.z - wrist.z
    full = math.sqrt(dx * dx + dy * dy + dz * dz)
    if full < 1e-6:
        return 1.0
    return min(1.0, math.hypot(dx, dy) / full)


class TrackedHand:
    def __init__(self, label, landmarks_norm, w, h, world_landmarks=None):
        self.label = label  # "Left" or "Right", mirrored/selfie convention
        self.landmarks_norm = landmarks_norm
        self.landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm]
        # metric 3D positions; the flat picture alone can't tell how much the
        # hand is tilted towards the lens
        self.world_landmarks = world_landmarks
        # Fist detection needs evenly scaled coordinates: the metric 3D points
        # when we have them, otherwise pixels. Not the normalised landmarks --
        # those divide x by the frame width and y by its height, which skews
        # every distance on a non-square frame.
        if world_landmarks:
            points = [(lm.x, lm.y, lm.z) for lm in world_landmarks]
        else:
            points = self.landmarks_px
        self.closed = is_hand_closed(points)
        self.index_tip = self.landmarks_px[8]


class HandTracker:
    def __init__(self, model_path=None, num_hands=2,
                 min_detection=0.5, min_tracking=0.3):
        # The tracking threshold is deliberately loose: a hand swung fast
        # enough to blur is exactly when the model's confidence dips, and
        # dropping it mid-swing is worse than holding a slightly shaky one.
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path or resource("models", "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_tracking,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def _next_timestamp_ms(self):
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        h, w = frame_bgr.shape[:2]
        hands = []
        for i, lm_list in enumerate(result.hand_landmarks):
            label = "Unknown"
            if result.handedness and result.handedness[i]:
                label = result.handedness[i][0].category_name
            world = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = result.hand_world_landmarks[i]
            hands.append(TrackedHand(label, lm_list, w, h, world))
        return hands

    def close(self):
        self._landmarker.close()
