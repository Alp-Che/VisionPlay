"""Catch the falling things in a basket held between your hands.

The basket hangs at the midpoint of the two palms, a little below them, as
if slung between your hands -- so you carry it by moving both arms together
rather than by pointing at things.

Items drop along fixed straight lines, each with its own speed and drift and
no acceleration, so where one is going can be read the moment it appears.
Colour is worth points; the black ones cost you.
"""
import math
import random
import time

import cv2
import numpy as np

from core.hand_tracker import HandTracker
from core.paddles import HandPaddles
from core.pixel_font import draw_text_with_background
from core.ui import Button, DwellClickController

TOOLBAR_H = 90

BASKET_WIDTH_FRAC = 0.17     # of frame width
BASKET_HEIGHT_FRAC = 0.10    # of frame height
BASKET_DROP_FRAC = 0.09      # how far below the hands it hangs
# Tracking drops the hands exactly when they are swung hardest, and one hand
# often leaves the frame entirely. Neither should make the basket vanish, so
# a lost hand coasts on for a moment and, failing that, the basket is placed
# off the hand that is still there, using the gap the two last had.
BASKET_GRACE = 0.6
SINGLE_HAND_GAP_FRAC = 0.16  # fallback half-gap, of frame width
# When the tracker picks a hand back up, the basket's correct position moves
# although the player's hands did not -- a teleport that reads as the basket
# being thrown across the screen. So the basket is given a speed it cannot
# exceed. This is a ceiling, not smoothing: it sits well above anything two
# hands can do, so in play the basket is pinned exactly between them with no
# lag, and the limit only ever bites on a jump the player did not make. A
# whole-screen teleport becomes a fifth of a second of travel.
MAX_BASKET_SPEED = 4500.0    # px/s

ITEM_RADIUS_FRAC = 0.032     # of frame height
SPAWN_EVERY = (0.55, 1.15)   # seconds between drops
FALL_SPEED = (250.0, 400.0)  # px/s at the start
SPEED_PER_POINT = 2.4        # the longer you last, the quicker they come
MAX_FALL_SPEED = 780.0
DRIFT_MAX = 0.45             # sideways component, as a fraction of fall speed
BLACK_SHARE = 0.28           # how often a penalty item appears

# (name, colour BGR, points)
ITEM_KINDS = [
    ("beyaz", (225, 225, 225), 1),
    ("sari", (60, 220, 240), 2),
    ("yesil", (70, 200, 90), 3),
    ("mor", (220, 110, 90), 5),
]
BLACK_KIND = ("siyah", (28, 28, 28), -5)


def _spawn_item(w, h, radius, score):
    kind = BLACK_KIND if random.random() < BLACK_SHARE else random.choice(ITEM_KINDS)
    speed = min(random.uniform(*FALL_SPEED) + max(score, 0) * SPEED_PER_POINT,
                MAX_FALL_SPEED)
    drift = random.uniform(-DRIFT_MAX, DRIFT_MAX) * speed
    x = random.uniform(radius * 2, w - radius * 2)
    # aim the drift inwards near the edges so nothing leaves immediately
    if x < w * 0.2:
        drift = abs(drift)
    elif x > w * 0.8:
        drift = -abs(drift)
    return {"x": x, "y": -radius, "vx": drift, "vy": speed,
            "name": kind[0], "color": kind[1], "points": kind[2]}


def _draw_item(frame, item, radius):
    center = (int(item["x"]), int(item["y"]))
    b, g, r = item["color"]
    cv2.circle(frame, center, radius, (max(b - 45, 0), max(g - 45, 0), max(r - 45, 0)),
               -1, cv2.LINE_AA)
    cv2.circle(frame, center, max(1, int(radius * 0.74)), item["color"], -1, cv2.LINE_AA)
    if item["points"] < 0:
        # a dark blob on a dark camera picture is easy to miss, so ring it
        cv2.circle(frame, center, radius, (60, 60, 210), 2, cv2.LINE_AA)


def _draw_basket(frame, cx, rim_y, half_w, height):
    """A tapered basket: wide mouth at rim_y, narrower base below it."""
    bottom = rim_y + height
    inset = half_w * 0.26
    body = np.array([
        (cx - half_w, rim_y), (cx + half_w, rim_y),
        (cx + half_w - inset, bottom), (cx - half_w + inset, bottom),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [body], (38, 62, 96), cv2.LINE_AA)
    for i in (1, 2):
        t = i / 3
        y = int(rim_y + (bottom - rim_y) * t)
        cv2.line(frame, (int(cx - half_w + inset * t), y),
                 (int(cx + half_w - inset * t), y), (70, 105, 150), 1, cv2.LINE_AA)
    cv2.line(frame, (int(cx - half_w), int(rim_y)), (int(cx + half_w), int(rim_y)),
             (120, 175, 235), 5, cv2.LINE_AA)


def _draw_legend(frame, x, y):
    draw_text_with_background(frame, "PUAN", (x, y), scale=2, color=(200, 200, 200))
    row = y + 42
    for name, color, points in ITEM_KINDS + [BLACK_KIND]:
        cv2.circle(frame, (x + 10, row + 6), 9, color, -1, cv2.LINE_AA)
        label = f"+{points}" if points > 0 else str(points)
        draw_text_with_background(frame, label, (x + 34, row), scale=2,
                                  color=(200, 200, 200) if points > 0 else (110, 110, 240))
        row += 32


def _solo_offset(palm, remembered_palms, remembered_basket, w, default_gap, drop):
    """Where to hang the basket once only one hand is left.

    Worked out once, at the moment the second hand goes, and then held. Redoing
    it every frame is what threw the basket around: the choice flips as the
    remaining hand wanders across the middle, and each flip jumps the basket by
    twice the gap between the hands.

    Which of the two palms is the one still showing is settled by position.
    MediaPipe's "Left"/"Right" cannot answer it -- with a single hand in the
    picture there is no second hand to judge it against, so the label flickers
    exactly when this decision is being made.
    """
    if remembered_palms and remembered_basket is not None:
        nearest = min(remembered_palms, key=lambda point: math.dist(palm, point))
        return (remembered_basket[0] - nearest[0], remembered_basket[1] - nearest[1])
    # nothing remembered: hang it towards the middle, where the items fall
    toward = default_gap if palm[0] < w / 2 else -default_gap
    return (toward, drop)


def run_catch_mode(cap, window_name):
    """Returns 'menu' or 'quit'."""
    tracker = HandTracker(num_hands=2)
    dwell = DwellClickController()

    ret, frame = cap.read()
    if not ret:
        tracker.close()
        return "quit"
    h, w = frame.shape[:2]

    buttons = [
        Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38)),
        Button("restart", "YENİDEN", 120, 10, 150, TOOLBAR_H - 20, color=(26, 46, 30)),
    ]

    # only the position tracking is wanted here, so the paddle radius, which
    # this mode never collides with, is left at a nominal 1px. coast=False:
    # the basket is carried, not swung, so a hand that blinks out should hold
    # still rather than drift off and take the basket with it.
    hands_tracker = HandPaddles(1.0, 1.0, grace=BASKET_GRACE, coast=False)
    default_gap = SINGLE_HAND_GAP_FRAC * w
    # the last frame in which both hands were tracked, kept so that losing one
    # of them doesn't move the basket
    remembered_palms = None
    remembered_basket = None
    solo_offset = None       # basket centre minus the one palm still showing

    half_w = BASKET_WIDTH_FRAC * w / 2
    basket_h = BASKET_HEIGHT_FRAC * h
    drop = BASKET_DROP_FRAC * h
    radius = int(ITEM_RADIUS_FRAC * h)

    items = []
    basket = None            # (cx, rim_y) this frame
    previous_basket = None
    score = 0
    caught, dropped = 0, 0
    next_spawn = time.time() + 0.8
    popups = []              # (text, x, y, colour, until)
    last_time = time.time()

    result = None
    try:
        while result is None:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = min(now - last_time, 0.05)
            last_time = now

            hands = tracker.process(frame)
            clicked, progress_map = dwell.update(hands, buttons)
            if clicked == "menu":
                result = "menu"
                break
            elif clicked == "restart":
                items.clear()
                score, caught, dropped = 0, 0, 0
                popups.clear()
                next_spawn = now + 0.8

            # --- the basket hangs between the two hands ---
            previous_basket = basket
            target = None
            # a briefly lost hand is held in place for a moment, so the
            # basket doesn't blink out with it
            palms = [p.end for p in hands_tracker.update(hands, now, dt)]

            if len(palms) >= 2:
                first, second = palms[0], palms[1]
                target = ((first[0] + second[0]) / 2.0,
                          (first[1] + second[1]) / 2.0 + drop)
                remembered_palms = (first, second)
                remembered_basket = target
                solo_offset = None      # both hands are back; decide afresh
            elif len(palms) == 1:
                # one hand left the frame: keep the basket where it sat
                # relative to the hand still showing, so it doesn't disappear
                if solo_offset is None:
                    solo_offset = _solo_offset(palms[0], remembered_palms,
                                               remembered_basket, w,
                                               default_gap, drop)
                target = (palms[0][0] + solo_offset[0],
                          palms[0][1] + solo_offset[1])
            else:
                solo_offset = None

            # Pinned straight to the hands -- anything that eases towards them
            # cannot keep up with a fast swipe -- but never moving faster than
            # hands can, so a tracker hiccup slides the basket instead of
            # flinging it.
            if target is None or basket is None:
                basket = target
            else:
                dx, dy = target[0] - basket[0], target[1] - basket[1]
                distance = math.hypot(dx, dy)
                reach = MAX_BASKET_SPEED * dt
                if distance > reach:
                    basket = (basket[0] + dx * reach / distance,
                              basket[1] + dy * reach / distance)
                else:
                    basket = target

            # --- drops ---
            if now >= next_spawn:
                items.append(_spawn_item(w, h, radius, score))
                next_spawn = now + random.uniform(*SPAWN_EVERY)

            for item in items:
                item["x"] += item["vx"] * dt
                item["y"] += item["vy"] * dt

            # --- catching ---
            if basket is not None and previous_basket is not None:
                cx, rim_y = basket
                previous_rim = previous_basket[1]
                for item in items[:]:
                    was_above = (item["y"] - item["vy"] * dt) < previous_rim
                    now_at_or_below = item["y"] >= rim_y
                    if was_above and now_at_or_below and abs(item["x"] - cx) <= half_w:
                        items.remove(item)
                        score += item["points"]
                        caught += 1
                        popups.append((f"+{item['points']}" if item["points"] > 0
                                       else str(item["points"]),
                                       item["x"], rim_y,
                                       (120, 240, 120) if item["points"] > 0 else (110, 110, 240),
                                       now + 0.8))

            for item in items[:]:
                if item["y"] - radius > h or item["x"] < -radius * 3 or item["x"] > w + radius * 3:
                    items.remove(item)
                    if item["points"] > 0:
                        dropped += 1

            popups = [p for p in popups if p[4] > now]

            # ================= draw =================
            for item in items:
                _draw_item(frame, item, radius)

            if basket is not None:
                _draw_basket(frame, basket[0], basket[1], half_w, basket_h)

            for text, px, py, color, _until in popups:
                draw_text_with_background(frame, text, (int(px), int(py) - 30), scale=3,
                                          anchor="center", color=color)

            for btn in buttons:
                btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                         hovered=dwell.hovered_id() == btn.id)

            _draw_legend(frame, 20, TOOLBAR_H + 40)

            draw_text_with_background(frame, f"PUAN: {score}", (w // 2, 26), scale=3,
                                      anchor="center")
            draw_text_with_background(frame, f"KAÇAN: {dropped}", (w - 30, 26), scale=2,
                                      anchor="topright", color=(150, 150, 150))
            if basket is None:
                draw_text_with_background(frame, "İKİ ELİNİ DE GÖSTER",
                                          (w // 2, h // 2), scale=2, anchor="center",
                                          bg_color=(0, 90, 140))

            draw_text_with_background(frame, "SEPET İKİ ELİNİN ARASINDA - SİYAHLARDAN KAÇIN",
                                      (14, h - 44), scale=2, color=(200, 200, 200))

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                result = "quit"
            elif key == 27:
                result = "menu"
    finally:
        tracker.close()

    return result or "quit"
