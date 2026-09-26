"""Dans: a wall arrives, and your hands have to be in its holes.

Both holes sit on one side of the picture, and the side swaps with every
wall -- right, then left, then right again -- so getting through means
stepping from side to side, which is the dance. The walls come quicker all
the time, so the stepping does too.

The wall is drawn by dimming the whole picture and leaving the holes clear, so
the only place you can see yourself is where you are supposed to be. That is
the instruction and the feedback at once, which is what this game needs -- it
is played from across the room, where a line of text is no use.

Which hand belongs in which hole is settled by where they are: the leftmost
hand takes the leftmost hole. Nothing keys on MediaPipe's "Left"/"Right",
which is the player's own handedness rather than the side of the picture they
are standing on.
"""
import math
import random
import time

import cv2
import numpy as np

from core.records import RoundRecord
from core.rig import draw_rig
from core.ui import Button, DwellClickController, draw_panel, draw_record, draw_round_timer
from core.window import handle_key

TOOLBAR_H = 90

HOLE_RADIUS_FRAC = 0.105     # of frame height
# Both holes stay on their own half, the nearer one at least this far off the
# middle of the picture (as a share of its width), so neither can be reached
# without moving over.
CENTRE_CLEAR_FRAC = 0.04
MARGIN_X_FRAC = 0.08
# How far apart the two holes may be. They share half the picture, so the far
# end is well short of an arm span.
GAP_MIN_FRAC = 0.20
GAP_MAX_FRAC = 0.36
MARGIN_TOP = 150             # clear of the toolbar and the bar under it
MARGIN_BOTTOM_FRAC = 0.12
WALLS_TO_WIDEST = 14         # the holes spread out to the widest gap by here

# How long there is to get into position. Every wall takes the same share off
# whatever is left above the floor, so it keeps getting quicker for as long as
# the player lasts without ever becoming impossible.
WALL_SECONDS_START = 4.2
WALL_SECONDS_END = 1.2
WALL_SPEEDUP = 0.90
# A beat between walls to see the verdict. It shortens with the walls, so a
# player who is always through early still feels the tempo rise.
PASSED_PAUSE = 0.55
PASSED_PAUSE_MIN = 0.25

LIVES = 3
WALL_DIM = 0.28              # how dark the wall is against the open holes

RING_WAITING = (210, 210, 210)
RING_FILLED = (110, 235, 130)


def _new_wall(w, h, passed, side):
    """Two holes on one side of the picture: -1 for the left, +1 the right."""
    radius = HOLE_RADIUS_FRAC * h
    margin_x = MARGIN_X_FRAC * w
    top = TOOLBAR_H + MARGIN_TOP
    bottom = h - MARGIN_BOTTOM_FRAC * h

    spread = min(passed / WALLS_TO_WIDEST, 1.0)
    gap = w * random.uniform(GAP_MIN_FRAC,
                             GAP_MIN_FRAC + (GAP_MAX_FRAC - GAP_MIN_FRAC) * spread)
    # laid out on the right half, then mirrored if the wall is on the left
    near = random.uniform(w / 2 + CENTRE_CLEAR_FRAC * w, w - margin_x - gap)
    xs = (near, near + gap) if side > 0 else (w - near - gap, w - near)
    holes = [(x, random.uniform(top, bottom)) for x in xs]
    seconds = WALL_SECONDS_END + (WALL_SECONDS_START - WALL_SECONDS_END) * (
        WALL_SPEEDUP ** passed)
    pause = max(PASSED_PAUSE * seconds / WALL_SECONDS_START, PASSED_PAUSE_MIN)
    return {"holes": holes, "radius": radius, "seconds": seconds,
            "pause": pause, "side": side}


def _filled(wall, hands):
    """Which holes have a hand in them, matched left to left by position."""
    if len(hands) < 2:
        return [False, False]
    palms = sorted((hand.landmarks_px[9] for hand in hands), key=lambda p: p[0])
    pairs = zip(sorted(wall["holes"], key=lambda hole: hole[0]),
                (palms[0], palms[-1]))
    return [math.dist(hole, palm) <= wall["radius"] for hole, palm in pairs]


def _draw_wall(frame, wall, filled):
    """Dims everything the wall covers and leaves the holes clear."""
    holes = [(int(x), int(y)) for x, y in wall["holes"]]
    radius = int(wall["radius"])

    open_ground = np.zeros(frame.shape[:2], np.uint8)
    for centre in holes:
        cv2.circle(open_ground, centre, radius, 255, -1, cv2.LINE_AA)

    # copyTo rather than indexing with the mask: same picture, a tenth of the
    # cost, and this runs on every frame of a 1280x720 camera
    walled = cv2.convertScaleAbs(frame, alpha=WALL_DIM)
    cv2.copyTo(frame, open_ground, walled)
    frame[:] = walled

    for centre, taken in zip(sorted(holes), filled):
        cv2.circle(frame, centre, radius, RING_FILLED if taken else RING_WAITING,
                   4, cv2.LINE_AA)


def run_dance_mode(cap, window_name, tracker):
    """Returns 'menu' or 'quit'."""
    dwell = DwellClickController()

    ret, frame = cap.read()
    if not ret:
        return "quit"
    h, w = frame.shape[:2]

    buttons = [
        Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38)),
        Button("restart", "YENİDEN", 120, 10, 150, TOOLBAR_H - 20, color=(26, 46, 30)),
    ]

    passed, lives = 0, LIVES
    record = RoundRecord("dans")
    wall = _new_wall(w, h, passed, random.choice((-1, 1)))
    wall_start = time.time()
    verdict_until, verdict_ok = 0.0, False

    result = None
    while result is None:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        hands = tracker.process(frame)
        clicked, progress_map = dwell.update(hands, buttons)
        if clicked == "menu":
            result = "menu"
            break
        elif clicked == "restart":
            passed, lives = 0, LIVES
            record.reset()
            wall = _new_wall(w, h, passed, random.choice((-1, 1)))
            wall_start = now
            verdict_until = 0.0

        alive = lives > 0
        filled = _filled(wall, hands) if alive else [False, False]
        left = max(wall["seconds"] - (now - wall_start), 0.0)

        if alive and now >= verdict_until:
            if all(filled) or left <= 0:
                # judged the moment both hands are in: getting through
                # early never waits out the clock
                verdict_ok = all(filled)
                if verdict_ok:
                    passed += 1
                else:
                    lives -= 1
                pause = wall["pause"]
                verdict_until = now + pause
                # the next wall is always on the other side: that is the dance
                wall = _new_wall(w, h, passed, -wall["side"])
                wall_start = now + pause

        # ================= draw =================
        if alive:
            _draw_wall(frame, wall, filled)

        for btn in buttons:
            btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                     hovered=dwell.hovered_id() == btn.id)

        draw_panel(frame, f"SKOR: {passed}", (w // 2, 26), scale=3, anchor="center")
        draw_panel(frame, f"CAN: {lives}", (w - 30, 26), scale=2, anchor="topright")
        if alive:
            draw_round_timer(frame, left / wall["seconds"])

        if now < verdict_until:
            draw_panel(frame, "GEÇTİN" if verdict_ok else "ÇARPTIN",
                       (w // 2, h // 2), scale=4, anchor="center",
                       plate=(20, 80, 30) if verdict_ok else (30, 30, 120))
        elif not alive:
            record.finish(passed)
            draw_panel(frame, f"BİTTİ - SKOR: {passed}",
                       (w // 2, h // 2 - 60), scale=3, anchor="center",
                       plate=(0, 60, 130))
            draw_record(frame, record, (w // 2, h // 2))
            draw_panel(frame, "YENİDEN DÜĞMESİNE BAS", (w // 2, h // 2 + 60),
                       scale=2, anchor="center")
        elif len(hands) < 2:
            draw_panel(frame, "İKİ ELİNİ DE GÖSTER", (w // 2, h - 80),
                       scale=2, anchor="center", plate=(0, 90, 140))

        draw_rig(frame, hands)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        handle_key(window_name, key)
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
