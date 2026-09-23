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

from core import assets
from core.geometry import closest_on_segment
from core.paddles import HandPaddles
from core.pixel_font import draw_text
from core.ui import Button, DwellClickController, ROUND_SECONDS, draw_panel, draw_round_timer

TOOLBAR_H = 90

HOLE_COUNT = 4               # more than this and the wide holes would touch
HOLE_LINE_FRAC = 0.94        # where the holes sit, of frame height
HOLE_WIDTH_FRAC = 0.155      # of frame width

# How much of the mole's height clears the ground at full stretch. The rest
# stays down the hole, which is what makes it look like it climbed out rather
# than floated up. Everything else about its shape -- how wide it is next to
# the hole, how tall next to its own width -- is taken from the drawings
# themselves, so redrawing them at other sizes needs no code change.
MOLE_STICK_FRAC = 0.82
# Where the paws grip, as a fraction of the hole sprite's height, measured
# down from the middle of the hole. Positive is towards the near rim.
PAW_GRIP_FRAC = 0.10
# Far enough out to count as a target -- and, because the paws appear at the
# same moment, far enough out to look like it is holding on. Showing them
# earlier leaves two paws gripping the rim above a mole that is barely a nose.
HITTABLE_AT = 0.45

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


def _mole_top_y(hole_y, out, mole_h):
    """Top edge of the mole sprite for a mole that is `out` of the way up."""
    return hole_y - out * MOLE_STICK_FRAC * mole_h


def _mole_axis(cx, hole_y, out, mole_w, mole_h):
    """Points down the mole's visible length, for hit testing: the whole of
    what is above ground counts, not just the head."""
    head = _mole_top_y(hole_y, out, mole_h) + mole_w * 0.5
    base = hole_y - mole_w * 0.15
    if base <= head:
        return [(cx, head)]
    return [(cx, head + (base - head) * t) for t in (0.0, 0.5, 1.0)]


def run_mole_mode(cap, window_name, tracker):
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
    paddles_for = HandPaddles(HAND_RADIUS_PER_SPAN, HAND_RADIUS_FLOOR_FRAC * h)

    hole_img = assets.load("mole/hole.png")
    mole_imgs = {False: assets.load("mole/good.png"), True: assets.load("mole/bad.png")}
    paw_imgs = {False: assets.load("mole/good_paws.png"), True: assets.load("mole/bad_paws.png")}

    centers = _hole_centers(w)
    hole_y = HOLE_LINE_FRAC * h
    min_swipe = MIN_SWIPE_FRAC * h

    # The hole sets the scale; the mole and the paws keep whatever size they
    # were drawn at next to it, so the artwork's own proportions survive.
    hole_w = HOLE_WIDTH_FRAC * w
    sprite_w = hole_img.shape[1] if hole_img is not None else 72
    def _scaled(img, fallback):
        if img is None:
            return fallback
        width = hole_w * img.shape[1] / sprite_w
        return width, width * img.shape[0] / img.shape[1]

    hole_wh = _scaled(hole_img, (hole_w, hole_w * 0.29))
    mole_w, mole_h = _scaled(mole_imgs[False], (hole_w * 0.72, hole_w * 0.90))
    paw_w, paw_h = _scaled(paw_imgs[False], (hole_w * 0.47, hole_w * 0.21))

    moles = {}               # hole index -> mole
    score = 0
    hits, misses = 0, 0
    popups = []
    round_start = time.time()
    next_spawn = round_start + 0.6
    last_time = time.time()

    result = None
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
            if mole["hit_at"] is not None or out < HITTABLE_AT:
                continue
            axis = _mole_axis(centers[index], hole_y, out, mole_w, mole_h)
            head = axis[0]
            for paddle in paddles:
                speed = math.hypot(*paddle.velocity)
                if speed < min_swipe:
                    continue
                if not any(math.dist(closest_on_segment(paddle.start, paddle.end, point),
                                     point) <= paddle.radius + mole_w * 0.5
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
        # A mole is drawn over the hole it is climbing out of, but under
        # its own paws: the paws are the outermost layer, so they read as
        # gripping the near rim rather than being down the hole with it.
        # The frame is sliced at the hole line first -- the slice is a
        # view, so everything below ground is clipped away for free.
        above_ground = frame[:int(hole_y)]
        for index, cx in enumerate(centers):
            assets.overlay_centered(frame, hole_img, int(cx), int(hole_y),
                                    int(hole_wh[0]), int(hole_wh[1]))
            mole = moles.get(index)
            if mole is None:
                continue
            out = _progress(mole, now)
            if out <= 0.01:
                continue
            top = _mole_top_y(hole_y, out, mole_h)
            assets.overlay(above_ground, mole_imgs[mole["bad"]],
                           int(cx - mole_w / 2), int(top),
                           int(mole_w), int(mole_h))
            if out >= HITTABLE_AT:
                assets.overlay_centered(
                    frame, paw_imgs[mole["bad"]], int(cx),
                    int(hole_y + hole_wh[1] * PAW_GRIP_FRAC),
                    int(paw_w), int(paw_h))

        for paddle in paddles:
            cv2.circle(frame, (int(paddle.end[0]), int(paddle.end[1])),
                       int(paddle.radius), (70, 170, 70), 2, cv2.LINE_AA)

        for text, px, py, color, _until in popups:
            # no plate behind these: they are up for less than a second over
            # whatever is moving, and a box drawn to make them readable ends
            # up the loudest thing on screen
            draw_text(frame, text, (int(px), int(py) - 40), scale=3,
                      anchor="center", color=color)

        for btn in buttons:
            btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                     hovered=dwell.hovered_id() == btn.id)

        draw_panel(frame, f"PUAN: {score}", (w // 2, 26), scale=3,
                                  anchor="center")
        draw_round_timer(frame, remaining / ROUND_SECONDS)

        if not running:
            draw_panel(frame, f"SÜRE DOLDU - PUAN: {score}",
                                      (w // 2, h // 2 - 30), scale=3, anchor="center",
                                      plate=(0, 60, 130))
            draw_panel(frame, "YENİDEN DÜĞMESİNE BAS", (w // 2, h // 2 + 30),
                                      scale=2, anchor="center")
        elif not paddles:
            draw_panel(frame, "ELLERİNİ KAMERAYA GÖSTER",
                                      (w // 2, h // 2), scale=2, anchor="center",
                                      plate=(0, 90, 140))

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
