"""Whack-a-mole: moles pop out of holes along the bottom, swat them back in.

A hit needs an actual swipe, not a parked hand -- otherwise the game would
be won by resting both palms over two holes. The swept path of the hand is
what gets tested, so a fast swat can't slip between frames.

Angry moles cost points, so the round is about picking targets, not flailing.
"""
import math
import random
import time

import cv2

from core.geometry import closest_on_segment
from core.hand_tracker import HandTracker
from core.paddles import HandPaddles
from core.pixel_font import draw_text_with_background
from core.ui import Button, DwellClickController

TOOLBAR_H = 90

ROUND_SECONDS = 30.0

HOLE_COUNT = 4               # more than this and the wide holes would touch
HOLE_LINE_FRAC = 0.94        # where the holes sit, of frame height
HOLE_WIDTH_FRAC = 0.155      # of frame width

# The mole is a capsule standing on end: a domed head on a body whose base
# stays down the hole. Height is in half-widths, and STICK_FRAC is how much
# of that height clears the ground at full stretch -- the rest stays buried,
# which is what keeps it looking like it climbed out rather than floated up.
MOLE_BODY_RADII = 3.2
MOLE_STICK_FRAC = 0.82

RISE_TIME = 0.45             # a mole climbs out unhurriedly
RETREAT_TIME = 0.32
STAY_START = 1.15            # seconds a mole waits up, at the start
STAY_END = 0.5               # ...and by the end of the round
SPAWN_START = 0.85           # gap between moles, at the start
SPAWN_END = 0.35

BAD_SHARE = 0.3
GOOD_POINTS = 2
BAD_POINTS = -3

HAND_RADIUS_PER_SPAN = 1.15
HAND_RADIUS_FLOOR_FRAC = 0.045
MIN_SWIPE_FRAC = 0.35        # of frame height per second

GOOD_BODY = (78, 112, 150)
GOOD_SNOUT = (110, 150, 188)
BAD_BODY = (70, 58, 140)
BAD_SNOUT = (95, 85, 180)


def _hole_centers(w):
    step = w / (HOLE_COUNT + 1)
    return [step * (i + 1) for i in range(HOLE_COUNT)]


def _progress(mole, now):
    """0 = fully underground, 1 = fully out."""
    age = now - mole["born"]
    if mole["hit_at"] is not None:
        gone = (now - mole["hit_at"]) / (RETREAT_TIME * 0.6)
        return max(mole["hit_height"] * (1 - gone), 0.0)
    if age < RISE_TIME:
        return age / RISE_TIME
    if age < RISE_TIME + mole["stay"]:
        return 1.0
    leaving = (age - RISE_TIME - mole["stay"]) / RETREAT_TIME
    return max(1.0 - leaving, 0.0)


def _finished(mole, now):
    if mole["hit_at"] is not None:
        return now - mole["hit_at"] > RETREAT_TIME * 0.6
    return now - mole["born"] > RISE_TIME + mole["stay"] + RETREAT_TIME


def _draw_hole(frame, cx, hole_y, half_w):
    cv2.ellipse(frame, (int(cx), int(hole_y)), (int(half_w), int(half_w * 0.34)),
                0, 0, 360, (18, 20, 24), -1, cv2.LINE_AA)


def _draw_hole_lip(frame, cx, hole_y, half_w):
    """The near rim, drawn over the mole so it looks like it's in the hole."""
    cv2.ellipse(frame, (int(cx), int(hole_y)), (int(half_w), int(half_w * 0.34)),
                0, 0, 180, (34, 38, 44), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (int(cx), int(hole_y)), (int(half_w), int(half_w * 0.34)),
                0, 0, 360, (58, 64, 72), 2, cv2.LINE_AA)


def _mole_head_y(hole_y, out, radius, body_h):
    """Centre of the domed head for a mole that is `out` of the way out."""
    return hole_y - out * body_h * MOLE_STICK_FRAC + radius


def _mole_axis(cx, hole_y, out, radius, body_h):
    """Points down the mole's visible length, for hit testing: the whole body
    counts, not just the head."""
    head = _mole_head_y(hole_y, out, radius, body_h)
    base = hole_y - radius * 0.3
    if base <= head:
        return [(cx, head)]
    return [(cx, head + (base - head) * t) for t in (0.0, 0.5, 1.0)]


def _draw_mole(frame, cx, cy, radius, body_h, bad):
    """Draw into a frame already clipped to the ground above the hole, so the
    body is hidden by the earth rather than floating over it."""
    body = BAD_BODY if bad else GOOD_BODY
    snout = BAD_SNOUT if bad else GOOD_SNOUT
    # trunk below the head; its base runs on past the hole line and is clipped
    cv2.rectangle(frame, (int(cx - radius), int(cy)),
                  (int(cx + radius), int(cy + body_h)), body, -1)
    cv2.circle(frame, (int(cx), int(cy)), radius, body, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (int(cx), int(cy + radius * 0.30)),
                (int(radius * 0.52), int(radius * 0.36)), 0, 0, 360, snout, -1, cv2.LINE_AA)
    eye_dx, eye_dy = int(radius * 0.36), int(radius * 0.22)
    for sign in (-1, 1):
        ex = int(cx + sign * eye_dx)
        cv2.circle(frame, (ex, int(cy - eye_dy)), max(2, radius // 8),
                   (235, 235, 235), -1, cv2.LINE_AA)
        cv2.circle(frame, (ex, int(cy - eye_dy)), max(1, radius // 14),
                   (25, 25, 25), -1, cv2.LINE_AA)
        if bad:
            cv2.line(frame, (int(ex - sign * radius * 0.26), int(cy - radius * 0.56)),
                     (int(ex + sign * radius * 0.16), int(cy - radius * 0.30)),
                     (45, 45, 190), max(2, radius // 10), cv2.LINE_AA)
    cv2.circle(frame, (int(cx), int(cy + radius * 0.26)), max(2, radius // 9),
               (30, 30, 40), -1, cv2.LINE_AA)


def run_mole_mode(cap, window_name):
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
    paddles_for = HandPaddles(HAND_RADIUS_PER_SPAN, HAND_RADIUS_FLOOR_FRAC * h)

    centers = _hole_centers(w)
    hole_y = HOLE_LINE_FRAC * h
    half_w = HOLE_WIDTH_FRAC * w / 2
    mole_r = int(half_w * 0.72)
    body_h = MOLE_BODY_RADII * mole_r
    min_swipe = MIN_SWIPE_FRAC * h

    moles = {}               # hole index -> mole
    score = 0
    hits, misses = 0, 0
    popups = []
    round_start = time.time()
    next_spawn = round_start + 0.6
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
                moles.clear()
                popups.clear()
                score, hits, misses = 0, 0, 0
                round_start = now
                next_spawn = now + 0.6

            elapsed = now - round_start
            remaining = max(ROUND_SECONDS - elapsed, 0.0)
            running = remaining > 0
            ramp = min(elapsed / ROUND_SECONDS, 1.0)

            paddles = paddles_for.update(hands, now, dt)

            # --- spawn ---
            if running and now >= next_spawn:
                free = [i for i in range(HOLE_COUNT) if i not in moles]
                if free:
                    index = random.choice(free)
                    moles[index] = {
                        "born": now,
                        "stay": STAY_START + (STAY_END - STAY_START) * ramp,
                        "bad": random.random() < BAD_SHARE,
                        "hit_at": None, "hit_height": 1.0,
                    }
                next_spawn = now + (SPAWN_START + (SPAWN_END - SPAWN_START) * ramp)

            # --- whacking ---
            for index, mole in list(moles.items()):
                out = _progress(mole, now)
                if mole["hit_at"] is not None or out < 0.45:
                    continue
                axis = _mole_axis(centers[index], hole_y, out, mole_r, body_h)
                head = axis[0]
                for paddle in paddles:
                    speed = math.hypot(*paddle.velocity)
                    if speed < min_swipe:
                        continue
                    if not any(math.dist(closest_on_segment(paddle.start, paddle.end, point),
                                         point) <= paddle.radius + mole_r
                               for point in axis):
                        continue
                    mole["hit_at"] = now
                    mole["hit_height"] = out
                    points = BAD_POINTS if mole["bad"] else GOOD_POINTS
                    score += points
                    hits += 1
                    popups.append((f"+{points}" if points > 0 else str(points),
                                   head[0], head[1],
                                   (120, 240, 120) if points > 0 else (110, 110, 240),
                                   now + 0.7))
                    break

            for index, mole in list(moles.items()):
                if _finished(mole, now):
                    if mole["hit_at"] is None and not mole["bad"]:
                        misses += 1
                    moles.pop(index)

            popups = [p for p in popups if p[4] > now]
            if not running:
                moles.clear()

            # ================= draw =================
            above_ground = frame[:int(hole_y)]
            for index, cx in enumerate(centers):
                _draw_hole(frame, cx, hole_y, half_w)
                mole = moles.get(index)
                if mole is not None:
                    out = _progress(mole, now)
                    if out > 0.01:
                        # the slice is a view, so anything below the hole line
                        # is clipped away for free
                        _draw_mole(above_ground, cx,
                                   _mole_head_y(hole_y, out, mole_r, body_h),
                                   mole_r, body_h, mole["bad"])
                _draw_hole_lip(frame, cx, hole_y, half_w)

            for paddle in paddles:
                cv2.circle(frame, (int(paddle.end[0]), int(paddle.end[1])),
                           int(paddle.radius), (70, 170, 70), 2, cv2.LINE_AA)

            for text, px, py, color, _until in popups:
                draw_text_with_background(frame, text, (int(px), int(py) - 40), scale=3,
                                          anchor="center", color=color)

            for btn in buttons:
                btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                         hovered=dwell.hovered_id() == btn.id)

            draw_text_with_background(frame, f"PUAN: {score}", (w // 2, 26), scale=3,
                                      anchor="center")
            timer_color = (110, 110, 240) if remaining <= 5 else (200, 200, 200)
            draw_text_with_background(frame, f"SÜRE: {remaining:4.1f}", (w - 30, 26),
                                      scale=2, anchor="topright", color=timer_color)

            bar_w = int(w * 0.5)
            bar_x = (w - bar_w) // 2
            cv2.rectangle(frame, (bar_x, 64), (bar_x + bar_w, 76), (55, 55, 55), -1)
            cv2.rectangle(frame, (bar_x, 64),
                          (bar_x + int(bar_w * remaining / ROUND_SECONDS), 76),
                          (90, 190, 90) if remaining > 5 else (70, 70, 220), -1)

            if not running:
                draw_text_with_background(frame, f"SÜRE DOLDU - PUAN: {score}",
                                          (w // 2, h // 2 - 30), scale=3, anchor="center",
                                          bg_color=(0, 60, 130))
                draw_text_with_background(frame, "YENİDEN'E BAS", (w // 2, h // 2 + 30),
                                          scale=2, anchor="center")
            elif not paddles:
                draw_text_with_background(frame, "ELLERİNİ KAMERAYA GÖSTER",
                                          (w // 2, h // 2), scale=2, anchor="center",
                                          bg_color=(0, 90, 140))

            # the holes now run along the bottom edge, so the hint sits up top
            draw_text_with_background(frame, "KÖSTEBEKLERE VUR - KIRMIZI GÖZLÜLERE DOKUNMA",
                                      (14, TOOLBAR_H + 26), scale=2, color=(200, 200, 200))

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                result = "quit"
            elif key == 27:
                result = "menu"
    finally:
        tracker.close()

    return result or "quit"
