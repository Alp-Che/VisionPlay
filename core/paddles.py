"""Tracked hands turned into something a ball can bounce off.

Three things a plain per-frame hand position gets wrong, and all of them
matter in any game where the hand is swung:

  - A hand moving fast covers more ground between frames than the ball is
    wide, so testing only where it ended up lets the ball pass straight
    through. Each paddle therefore reports the *segment* it swept.
  - Tracking drops out exactly when the hand is moving fastest and blurs, so
    a lost hand is held for a moment rather than vanishing from under the
    ball -- coasting on along its last path, if the caller wants that.
  - MediaPipe's "Left"/"Right" is a classification it redoes from scratch
    every frame, not an identity. The two labels swap when the hands cross or
    turn, both hands can come back labelled the same, handedness can be
    missing altogether, and a hand alone in the picture -- with no second hand
    to be judged against -- flickers between the two. Keying anything on it
    makes a hand appear to teleport across the body. Hands are therefore
    matched to the previous frame by *position*, which is what actually
    persists, and each keeps a stable id of its own. The label is passed
    through for callers that want a hint, but nothing here depends on it.
"""
import math

from core.hand_tracker import MAX_HAND_SPEED, hand_span

GRACE_SECONDS = 0.3
GRACE_DAMPING = 0.85


class Paddle:
    __slots__ = ("id", "label", "start", "end", "radius", "velocity")

    def __init__(self, id, label, start, end, radius, velocity):
        self.id = id            # stable while the hand is tracked
        self.label = label      # MediaPipe's per-frame guess; a hint, not an id
        self.start = start      # where the hand was last frame
        self.end = end          # where it is now
        self.radius = radius
        self.velocity = velocity


def _match(centers, positions, limit):
    """Pair this frame's hands with the remembered ones by position.

    Returns, for each center, the index of the remembered hand it continues,
    or None if it is new. Two hands is the whole domain here, so both possible
    pairings are compared outright: matching nearest-first can pair the two
    closest and then strand the rest on a far worse match.
    """
    n, m = len(centers), len(positions)
    if n == 0 or m == 0:
        return [None] * n

    if n == 2 and m == 2:
        straight = (math.dist(centers[0], positions[0])
                    + math.dist(centers[1], positions[1]))
        crossed = (math.dist(centers[0], positions[1])
                   + math.dist(centers[1], positions[0]))
        order = (0, 1) if straight <= crossed else (1, 0)
        return [order[i] if math.dist(centers[i], positions[order[i]]) <= limit
                else None for i in (0, 1)]

    pairs = sorted((math.dist(c, p), i, j)
                   for i, c in enumerate(centers)
                   for j, p in enumerate(positions))
    out = [None] * n
    taken_center, taken_position = set(), set()
    for distance, i, j in pairs:
        if distance > limit or i in taken_center or j in taken_position:
            continue
        out[i] = j
        taken_center.add(i)
        taken_position.add(j)
    return out


class HandPaddles:
    def __init__(self, radius_per_span, radius_floor,
                 grace=GRACE_SECONDS, damping=GRACE_DAMPING, coast=True,
                 max_hands=2, landmark=9):
        # which of the hand's 21 points to follow. 9 is the middle-finger
        # knuckle, near enough the centre of the palm to stand for the whole
        # hand; 8 is the index fingertip, for games played with one finger.
        self.landmark = landmark
        self.radius_per_span = radius_per_span
        self.radius_floor = radius_floor
        self.grace = grace
        self.damping = damping
        # coast: a lost hand keeps moving along its last path. Right for
        # sweeping a blade through the air, wrong for anything the player is
        # holding, which should simply stay put until tracking returns.
        self.coast = coast
        # The tracker never reports more than this many hands, so neither
        # should we: a hand that reappears too far from where it was left
        # would otherwise be a third "hand", and a phantom alongside the real
        # one is worse than no phantom at all.
        self.max_hands = max_hands
        self._state = {}
        self._next_id = 0

    def update(self, hands, now, dt):
        centers = [hand.landmarks_px[self.landmark] for hand in hands]
        ids = list(self._state.keys())
        positions = [self._state[i]["end"] for i in ids]
        limit = MAX_HAND_SPEED * max(dt, 1e-3)
        matched = _match(centers, positions, limit)

        seen = set()
        for hand, center, index in zip(hands, centers, matched):
            radius = max(hand_span(hand) * self.radius_per_span, self.radius_floor)
            start, velocity = center, (0.0, 0.0)
            if index is None:
                hand_id = self._next_id
                self._next_id += 1
            else:
                hand_id = ids[index]
                previous = self._state[hand_id]["end"]
                if dt > 0:
                    start = previous
                    velocity = ((center[0] - previous[0]) / dt,
                                (center[1] - previous[1]) / dt)
            self._state[hand_id] = {
                "label": hand.label, "start": start, "end": center,
                "radius": radius, "velocity": velocity, "seen": now,
            }
            seen.add(hand_id)

        for hand_id, state in list(self._state.items()):
            if hand_id in seen:
                continue
            if now - state["seen"] > self.grace:
                self._state.pop(hand_id)
                continue
            if not self.coast:
                state["start"] = state["end"]
                state["velocity"] = (0.0, 0.0)
                continue
            vx, vy = state["velocity"]
            state["start"] = state["end"]
            state["end"] = (state["end"][0] + vx * dt, state["end"][1] + vy * dt)
            state["velocity"] = (vx * self.damping, vy * self.damping)

        # over the limit, the stalest phantom goes first
        while len(self._state) > self.max_hands:
            stalest = min((i for i in self._state if i not in seen),
                          key=lambda i: self._state[i]["seen"], default=None)
            if stalest is None:
                break
            self._state.pop(stalest)

        return [Paddle(i, s["label"], s["start"], s["end"], s["radius"], s["velocity"])
                for i, s in self._state.items()]
