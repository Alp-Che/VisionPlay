"""MediaPipe Tasks HandLandmarker, wrapped for a live camera loop.

The wrapper also decides *whose* hands these are. The game is meant to be
played in a room with other people in it, and MediaPipe will report a hand
that wanders into shot as readily as the player's own -- so it is asked for
more hands than the game follows, and the extra ones are then set aside.
"""
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



# Sized by measurement, and deliberately generous. A hand crossing the whole
# 1280px frame in a quarter of a second is only ~5100 px/s, so a real swing --
# even a wild one -- stays under this. It doubles as the furthest a hand could
# plausibly have travelled since the last frame: anything beyond that is a
# different hand, not the same one having teleported.
MAX_HAND_SPEED = 6000.0      # px/s
# How long a place is kept for a hand that has blinked out. Tracking drops for
# a few frames all the time; handing the controls to a stranger over it would
# be worse than waiting.
FOLLOW_GRACE_FRAMES = 20
# The model does occasionally find a hand in a patterned background. Two
# things tell those apart from a real one. They are small -- nothing standing
# at the camera looks this tiny -- and they flicker, where a raised hand stays
# put, so a new hand has to be found in about the same place several frames
# running before it is allowed to take the controls.
MIN_HAND_FRAC = 0.05         # of frame height
CLAIM_FRAMES = 4


def _apparent_size(hand):
    """How big the hand looks, across all its landmarks.

    Used to tell the player from an onlooker: the player is the one standing
    in front of the camera, so theirs is the largest hand in the picture. The
    full spread is steadier here than the wrist-to-knuckles span, which all
    but vanishes when a hand is turned edge-on to the lens.
    """
    xs = [x for x, _ in hand.landmarks_px]
    ys = [y for _, y in hand.landmarks_px]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


class _PlayerHands:
    """Keeps the controls with the hands that already had them.

    Two rules. A hand the game is already following keeps its place for as
    long as it stays in view, so nobody takes over by walking past. A place
    that does fall vacant goes to the largest hand not already spoken for.
    """

    def __init__(self, follow, grace=FOLLOW_GRACE_FRAMES):
        self.follow = follow
        self.grace = grace
        self._slots = []         # [{"at": (x, y), "missing": frames}, ...]
        self._waiting = []       # hands not yet trusted with a place

    def pick(self, hands, dt, frame_h):
        # too small to be a hand held up at the camera
        floor = MIN_HAND_FRAC * frame_h
        hands = [hand for hand in hands if _apparent_size(hand) >= floor]
        centres = [hand.landmarks_px[9] for hand in hands]
        # Deliberately not widened while a hand is missing. A place waiting
        # for its hand back is exactly when a stranger's hand is most likely
        # to be the nearest thing to it, and adopting one is how the controls
        # get stolen. The hand is far likelier to reappear where it vanished.
        reach = MAX_HAND_SPEED * dt
        taken = set()
        chosen = [None] * len(self._slots)

        # every place first tries to keep the hand it had last frame
        for index, slot in enumerate(self._slots):
            best, best_distance = None, None
            for i, centre in enumerate(centres):
                if i in taken:
                    continue
                distance = math.dist(centre, slot["at"])
                if distance <= reach and (best_distance is None or distance < best_distance):
                    best, best_distance = i, distance
            if best is None:
                slot["missing"] += 1
                continue
            taken.add(best)
            chosen[index] = best
            slot["at"] = centres[best]
            slot["missing"] = 0

        # a place is only given up once its hand has been gone a while
        alive = [i for i, slot in enumerate(self._slots) if slot["missing"] <= self.grace]
        self._slots = [self._slots[i] for i in alive]
        chosen = [chosen[i] for i in alive]

        # Anything left over has to hold still for a few frames before it is
        # trusted. Tracking one frame at a time is what let a bit of pattern
        # on the back wall take a place the moment one came free.
        still_waiting, claimed = [], set()
        for i in range(len(hands)):
            if i in taken:
                continue
            seen = 1
            for w, waiting in enumerate(self._waiting):
                if w in claimed:
                    continue
                if math.dist(centres[i], waiting["at"]) <= reach:
                    seen = waiting["seen"] + 1
                    claimed.add(w)
                    break
            still_waiting.append({"at": centres[i], "seen": seen, "hand": i})
        # anything not found again this frame is forgotten: it has to be there
        # frame after frame, not now and then
        self._waiting = still_waiting

        # a free place goes to the steadiest hand nearest the camera
        ready = sorted((c for c in self._waiting if c["seen"] >= CLAIM_FRAMES),
                       key=lambda c: _apparent_size(hands[c["hand"]]), reverse=True)
        for candidate in ready:
            if len(self._slots) >= self.follow:
                break
            self._slots.append({"at": candidate["at"], "missing": 0})
            chosen.append(candidate["hand"])
            self._waiting.remove(candidate)

        return [hands[i] for i in chosen if i is not None]


class HandTracker:
    def __init__(self, model_path=None, num_hands=2, look_for=None,
                 min_detection=0.5, min_tracking=0.3):
        # `num_hands` is how many the game plays with; `look_for` is how many
        # the model is asked to find. Searching wider is what makes it
        # possible to ignore an onlooker rather than mistake them for the
        # player, and it is close to free -- measured at 5.33 ms a frame for
        # two and 5.35 for four.
        self._players = _PlayerHands(num_hands)
        # The tracking threshold is deliberately loose: a hand swung fast
        # enough to blur is exactly when the model's confidence dips, and
        # dropping it mid-swing is worse than holding a slightly shaky one.
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path or resource("models", "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=look_for if look_for is not None else num_hands + 2,
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
        previous_ts = self._last_ts
        timestamp = self._next_timestamp_ms()
        # a stalled frame must not widen the search into the next person
        dt = min((timestamp - previous_ts) / 1000.0, 0.1) if previous_ts >= 0 else 1 / 60.0
        result = self._landmarker.detect_for_video(mp_image, timestamp)

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
        return self._players.pick(hands, dt, h)

    def close(self):
        self._landmarker.close()
