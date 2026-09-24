"""Delikten gec: a wall arrives, and your hands have to be in its holes.

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

from core.rig import draw_rig
from core.ui import Button, DwellClickController, draw_panel, draw_round_timer
from core.window import handle_key

TOOLBAR_H = 90

HOLE_RADIUS_FRAC = 0.105     # of frame height
# How far apart the holes may be, across the picture. The far end is an arm
# span: two holes further apart than that cannot both be reached.
GAP_MIN_FRAC = 0.20
GAP_MAX_FRAC = 0.52
MARGIN_X_FRAC = 0.10
MARGIN_TOP = 150             # clear of the toolbar and the bar under it
MARGIN_BOTTOM_FRAC = 0.12

# How long there is to get into position. It closes in as the score grows,
# which is the whole of the difficulty curve.
WALL_SECONDS_START = 3.2
WALL_SECONDS_END = 1.3
WALLS_TO_HARDEST = 14
PASSED_PAUSE = 0.55          # a beat to see the result before the next wall

LIVES = 3
WALL_DIM = 0.28              # how dark the wall is against the open holes

RING_WAITING = (210, 210, 210)
RING_FILLED = (110, 235, 130)


def _new_wall(w, h, passed):
    """Two holes, far enough apart to be a pose and near enough to reach."""
    radius = HOLE_RADIUS_FRAC * h
    margin_x = MARGIN_X_FRAC * w
    top = TOOLBAR_H + MARGIN_TOP
    bottom = h - MARGIN_BOTTOM_FRAC * h

    reach = GAP_MIN_FRAC * w + (GAP_MAX_FRAC - GAP_MIN_FRAC) * w * min(
        passed / WALLS_TO_HARDEST, 1.0) * random.random()
    gap = max(GAP_MIN_FRAC * w, reach)
    left_x = random.uniform(margin_x, w - margin_x - gap)
    holes = [(left_x, random.uniform(top, bottom)),
             (left_x + gap, random.uniform(top, bottom))]
    seconds = WALL_SECONDS_START + (WALL_SECONDS_END - WALL_SECONDS_START) * min(
        passed / WALLS_TO_HARDEST, 1.0)
    return {"holes": holes, "radius": radius, "seconds": seconds}


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


def run_wall_mode(cap, window_name, tracker):
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

    passed, lives, best = 0, LIVES, 0
    wall = _new_wall(w, h, passed)
    wall_start = time.time()
    verdict_until, verdict_ok = 0.0, False

    result = None
    while result is None:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        hands = tracker.process(frame)
        clicked, progress_map = dwell.update(hands, buttons)
        if clicked == "menu":
            result = "menu"
            break
        elif clicked == "restart":
            passed, lives = 0, LIVES
            wall = _new_wall(w, h, passed)
            wall_start = now
            verdict_until = 0.0

        alive = lives > 0
        filled = _filled(wall, hands) if alive else [False, False]
        left = max(wall["seconds"] - (now - wall_start), 0.0)

        if alive and now >= verdict_until:
            if all(filled):
                # through it: no need to wait out the clock
                passed += 1
                best = max(best, passed)
                verdict_ok, verdict_until = True, now + PASSED_PAUSE
                wall = _new_wall(w, h, passed)
                wall_start = now + PASSED_PAUSE
            elif left <= 0:
                lives -= 1
                verdict_ok, verdict_until = False, now + PASSED_PAUSE
                wall = _new_wall(w, h, passed)
                wall_start = now + PASSED_PAUSE

        # ================= draw =================
        if alive:
            _draw_wall(frame, wall, filled)

        for btn in buttons:
            btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                     hovered=dwell.hovered_id() == btn.id)

        draw_panel(frame, f"GEÇTİN: {passed}", (w // 2, 26), scale=3, anchor="center")
        draw_panel(frame, f"CAN: {lives}", (w - 30, 26), scale=2, anchor="topright")
        if alive:
            draw_round_timer(frame, left / wall["seconds"])

        if now < verdict_until:
            draw_panel(frame, "GEÇTİN" if verdict_ok else "ÇARPTIN",
                       (w // 2, h // 2), scale=4, anchor="center",
                       plate=(20, 80, 30) if verdict_ok else (30, 30, 120))
        elif not alive:
            draw_panel(frame, f"BİTTİ - GEÇTİĞİN DUVAR: {best}",
                       (w // 2, h // 2 - 40), scale=3, anchor="center",
                       plate=(0, 60, 130))
            draw_panel(frame, "YENİDEN DÜĞMESİNE BAS", (w // 2, h // 2 + 40),
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
