"""The hand rig, drawn over a game while the test switch is on.

It is here rather than in any one screen because it is a setting for the
whole session: the title screen turns it on, every game asks whether to draw
it. Off, `draw_rig` does nothing at all, so games can call it unconditionally.

Alongside the hands it draws the bodies the tracker saw, the player's bright
and everyone else's grey -- which is how to see who the game thinks is
playing. The tracker leaves them here each frame (`show_people`), so no game
has to pass them along.
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

# the upper body: the part that matters for whose hands are whose
BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24)]
PLAYER_COLOR = (255, 190, 60)
ONLOOKER_COLOR = (150, 150, 150)
MIN_VISIBILITY = 0.3

_state = {"on": False, "people": [], "players": []}


def set_rig(on):
    _state["on"] = bool(on)


def rig_on():
    return _state["on"]


def show_people(people, players):
    """What the tracker made of this frame: everyone, and who is playing."""
    _state["people"], _state["players"] = people, players


def _draw_body(frame, person, color, thickness):
    points = person.points
    for a, b in BODY:
        if min(points[a][2], points[b][2]) < MIN_VISIBILITY:
            continue
        cv2.line(frame, (int(points[a][0]), int(points[a][1])),
                 (int(points[b][0]), int(points[b][1])), color, thickness, cv2.LINE_AA)
    nose = points[0]
    if nose[2] >= MIN_VISIBILITY:
        cv2.circle(frame, (int(nose[0]), int(nose[1])), max(6, int(person.size * 0.12)),
                   color, thickness, cv2.LINE_AA)


def draw_rig(frame, hands):
    """Draws every tracked hand's skeleton. Silent when the switch is off."""
    if not _state["on"]:
        return
    for person in _state["people"]:
        playing = any(person is player for player in _state["players"])
        _draw_body(frame, person, PLAYER_COLOR if playing else ONLOOKER_COLOR,
                   4 if playing else 2)
    for hand in hands:
        points = hand.landmarks_px
        color = CLOSED_COLOR if hand.closed else BONE_COLOR
        for a, b in BONES:
            cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)
        for i, point in enumerate(points):
            radius = 5 if i in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(frame, point, radius, JOINT_COLOR, -1, cv2.LINE_AA)
