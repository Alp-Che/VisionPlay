"""The hand rig, drawn over a game while the test switch is on.

It is here rather than in any one screen because it is a setting for the
whole session: the title screen turns it on, every game asks whether to draw
it. Off, `draw_rig` does nothing at all, so games can call it unconditionally.
"""
import cv2

# MediaPipe's 21 points, joined the way the hand is: the wrist out to each
# fingertip, plus the arch across the knuckles that closes the palm.
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # little
    (5, 9), (9, 13), (13, 17),                 # across the knuckles
]

BONE_COLOR = (90, 230, 120)
JOINT_COLOR = (255, 255, 255)
CLOSED_COLOR = (80, 200, 255)

_state = {"on": False}


def set_rig(on):
    _state["on"] = bool(on)


def rig_on():
    return _state["on"]


def draw_rig(frame, hands):
    """Draws every tracked hand's skeleton. Silent when the switch is off."""
    if not _state["on"]:
        return
    for hand in hands:
        points = hand.landmarks_px
        color = CLOSED_COLOR if hand.closed else BONE_COLOR
        for a, b in BONES:
            cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)
        for i, point in enumerate(points):
            radius = 5 if i in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(frame, point, radius, JOINT_COLOR, -1, cv2.LINE_AA)
