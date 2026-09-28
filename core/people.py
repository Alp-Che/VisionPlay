"""Whose hands are these? The people in the picture, and which of them plays.

The hand model finds hands and nothing else: it cannot say that two hands
belong to one person, so with several people in shot the game could end up
with the player's right hand and a bystander's left. The pose model finds
whole people -- shoulders, elbows, wrists -- and that is what ties a hand to a
body: a hand belongs to whoever's wrist it is on.

The player is whoever stands in front, which in the picture means whoever
looks biggest. For two-player games each half of the picture has its own.

The pose model runs on a thread of its own. It costs about as much as the hand
model, and run in turn with it would halve the frame rate; run alongside it,
the hand model barely slows (measured 10.7 -> 11.5 ms a frame). Who is
standing where changes slowly, so an answer a frame or two old is as good as
a fresh one.
"""
import math
import threading
import time

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from core.paths import resource

# MediaPipe's 33 pose points, the few used here
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
# each hand as the body model sees it: wrist, then the pinky and index knuckles
BODY_HANDS = ((15, 17, 19), (16, 18, 20))
LEFT_HIP, RIGHT_HIP = 23, 24

MAX_PEOPLE = 4
# How often the pose model runs, at most. Who is standing where changes far
# more slowly than hands move, and every run it skips is time the hand model
# gets on a machine with few cores to share.
POSE_INTERVAL = 1 / 20
# A pose answer older than this is no answer: the thread has stalled, and
# going by where people were half a second ago would be worse than not going
# by it at all.
POSE_STALE_SECONDS = 0.6

# How far a hand may be from one of a body's hands and still be that body's,
# as a share of the body's size (about its shoulder width). Generous, because
# the pose answer lags the hands slightly and a swung hand runs ahead of it.
OWNER_REACH = 0.9
OWNER_REACH_MIN_PX = 50
# A hand is only settled as somebody's when it is clearly nearer them than
# anyone else: the next-nearest body must be this much further away. Between
# two people standing close, a hand could be either's, and then it is
# neither's.
CLEAR_MARGIN = 1.5

# Keeping the player. A player the model loses sight of keeps their place for
# a moment, as a hand does; someone else takes over only by standing clearly
# in front of them, for long enough that walking past does not count.
PLAYER_GRACE_SECONDS = 1.0
TAKEOVER_RATIO = 1.3
TAKEOVER_SECONDS = 1.0
FOLLOW_REACH = 1.0           # of the player's size, between two pose answers
FOLLOW_REACH_MIN_PX = 80


class Person:
    """One body, in the coordinates of the picture the game draws on."""

    __slots__ = ("points", "size", "centre", "hands")

    def __init__(self, landmarks, w, h, shift_x=0):
        self.points = [(lm.x * w - shift_x, lm.y * h,
                        lm.visibility if lm.visibility is not None else 1.0)
                       for lm in landmarks]
        left, right = self.points[LEFT_SHOULDER], self.points[RIGHT_SHOULDER]
        self.centre = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        shoulders = math.dist(left[:2], right[:2])
        # Shoulder width shrinks as a body turns side-on; the length of the
        # torso does not, so it stands in when the hips are in view. Facing
        # the camera the two agree: shoulders are about 0.8 of the torso.
        size = shoulders
        hip_l, hip_r = self.points[LEFT_HIP], self.points[RIGHT_HIP]
        if min(hip_l[2], hip_r[2]) > 0.5:
            hips = ((hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2)
            size = max(size, 0.8 * math.dist(self.centre, hips))
        self.size = size
        # each hand as (wrist, middle of the knuckles), to be laid against the
        # same two points of a hand the hand model found
        self.hands = tuple(
            (self.points[wrist][:2],
             ((self.points[pinky][0] + self.points[index][0]) / 2,
              (self.points[pinky][1] + self.points[index][1]) / 2))
            for wrist, pinky, index in BODY_HANDS)


def _hand_distance(hand, body_hand):
    """How far a found hand is from one of a body's hands: wrist to wrist and
    knuckles to knuckles, averaged. Two points agree far more often than one
    -- the body model's wrist alone can sit a hand's width off."""
    points = hand.landmarks_px
    knuckles = ((points[5][0] + points[17][0]) / 2, (points[5][1] + points[17][1]) / 2)
    wrist, middle = body_hand
    return (math.dist(points[0], wrist) + math.dist(knuckles, middle)) / 2


def whose(hand, people):
    """(person, clear) -- whose hand this is.

    The person is the nearest body with a hand within reach, or None. `clear`
    says nobody else comes close: a hand that is almost as near somebody else
    is not settled as anyone's. On a tie the earlier person in `people` wins,
    so the caller lists the players first.
    """
    scored = sorted(((min(_hand_distance(hand, body_hand) for body_hand in person.hands),
                      order, person) for order, person in enumerate(people)),
                    key=lambda item: (item[0], item[1]))
    if not scored:
        return None, False
    nearest, _, person = scored[0]
    if nearest > max(OWNER_REACH * person.size, OWNER_REACH_MIN_PX):
        return None, False
    clear = len(scored) == 1 or scored[1][0] >= nearest * CLEAR_MARGIN
    return person, clear


class Seat:
    """One player's place, and who is in it."""

    def __init__(self):
        self.person = None
        self._last_seen = 0.0
        self._rival_since = None

    def update(self, candidates, now):
        """Returns True when someone new has taken the seat (or it emptied)."""
        changed = False
        if self.person is not None:
            reach = max(FOLLOW_REACH * self.person.size, FOLLOW_REACH_MIN_PX)
            match = min(candidates, key=lambda p: math.dist(p.centre, self.person.centre),
                        default=None)
            if match is not None and math.dist(match.centre, self.person.centre) <= reach:
                self.person, self._last_seen = match, now
            elif now - self._last_seen > PLAYER_GRACE_SECONDS:
                self.person, self._rival_since, changed = None, None, True

        if self.person is None:
            if candidates:
                self.person = max(candidates, key=lambda p: p.size)
                self._last_seen, self._rival_since, changed = now, None, True
            return changed

        rival = max((p for p in candidates if p is not self.person),
                    key=lambda p: p.size, default=None)
        if rival is not None and rival.size > self.person.size * TAKEOVER_RATIO:
            if self._rival_since is None:
                self._rival_since = now
            elif now - self._rival_since >= TAKEOVER_SECONDS:
                self.person, self._last_seen, self._rival_since = rival, now, None
                changed = True
        else:
            self._rival_since = None
        return changed


class PlayerPicker:
    """One seat for the whole picture, or one for each half of it."""

    def __init__(self, seats=1):
        self.seats = [Seat() for _ in range(seats)]

    def update(self, people, width, now):
        """Returns, for each seat, whether its player changed."""
        if len(self.seats) == 1:
            return [self.seats[0].update(people, now)]
        halves = ([p for p in people if p.centre[0] < width / 2],
                  [p for p in people if p.centre[0] >= width / 2])
        return [seat.update(half, now) for seat, half in zip(self.seats, halves)]

    def players(self):
        return [seat.person for seat in self.seats]


class PoseWatcher:
    """The pose model, run on its own thread on the newest frame it is given."""

    def __init__(self, model_path=None, max_people=MAX_PEOPLE):
        with open(model_path or resource("models", "pose_landmarker_lite.task"), "rb") as f:
            model = f.read()
        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=max_people,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            ))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = None
        self._people, self._stamp = [], None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_rgb, shift_x):
        """Hands over a frame. Never waits: a frame not yet started on is
        simply replaced by the newer one."""
        with self._lock:
            self._job = (frame_rgb, shift_x)
        self._wake.set()

    def latest(self):
        """(people, when they were found), or (None, None) before the first."""
        with self._lock:
            return self._people, self._stamp

    def _run(self):
        t0, last_ts = time.monotonic(), -1
        while self._running:
            self._wake.wait(0.2)
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None or not self._running:
                continue
            frame_rgb, shift_x = job
            started = time.monotonic()
            ts = max(int((started - t0) * 1000), last_ts + 1)
            last_ts = ts
            try:
                result = self._landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), ts)
            except Exception:
                continue       # one bad frame must not end the thread
            h, w = frame_rgb.shape[:2]
            people = [Person(lms, w, h, shift_x) for lms in result.pose_landmarks]
            with self._lock:
                self._people, self._stamp = people, time.monotonic()
            # frames keep arriving meanwhile; the newest is the one taken next
            time.sleep(max(0.0, POSE_INTERVAL - (time.monotonic() - started)))

    def close(self):
        self._running = False
        self._wake.set()
        self._thread.join(timeout=1.0)
        self._landmarker.close()
