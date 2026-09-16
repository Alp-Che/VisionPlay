"""Landmark geometry helpers shared by every mode."""
import math

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
WRIST = 0

# An extended finger puts its tip further from the wrist than its middle
# joint; curl it and the tip folds back towards the palm. The ratio sits
# near 1.5 when open and below 1.0 when closed, so the line between them is
# well clear of both.
EXTENDED_RATIO = 1.15
CLOSED_FINGERS = 3


def _distance(a, b):
    total = sum((p - q) ** 2 for p, q in zip(a, b))
    return math.sqrt(total)


def is_hand_closed(points) -> bool:
    """True when the hand is a fist.

    `points` must be in a uniform scale -- metric 3D landmarks, or pixel
    coordinates -- because this measures distances. Comparing tip and joint
    *heights* instead, as this used to, reads any hand held fingers-down as
    closed, since an open hand pointing downwards has its tips below its
    knuckles too.
    """
    wrist = points[WRIST]
    curled = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        reach = _distance(points[pip], wrist) * EXTENDED_RATIO
        if _distance(points[tip], wrist) < reach:
            curled += 1
    return curled >= CLOSED_FINGERS


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def closest_on_segment(a, b, point):
    """Nearest point to `point` on the segment a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    span = dx * dx + dy * dy
    if span < 1e-9:
        return a
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / span))
    return (a[0] + t * dx, a[1] + t * dy)
