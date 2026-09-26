"""Fruit Ninja.

Cut with your index fingertip. A fingertip is a point rather than a blade, so
two things make up for the reach a sword had: the path the tip swept since the
previous frame is what gets tested, so a fast flick cannot pass clean through a
fruit between frames, and a tip moving too slowly does not cut at all --
without that, holding a finger still in the fruits' path would harvest them.

Both hands cut, each with its own swept path and its own trail. Nothing is
shared between them, so one hand resting still cannot hold the other back and
no slash is ever drawn between the two.
"""
import math
import random
import time

import cv2

from core.paddles import HandPaddles
from core.pixel_font import draw_text
from core.rig import draw_rig
from core.window import handle_key
from core.records import RoundRecord
from core.ui import (PANEL_COLOR, Button, DwellClickController, draw_panel,
                     draw_record, draw_round_timer)

TOOLBAR_H = 90

# Sizes are set for a player at the intended distance -- two to three metres,
# where a hand spans about a sixteenth of the picture's height -- and then held
# there. They used to follow the player's own hand, which made the fruit grow
# as the player walked in: stepping up to the screen was the easy way to win.
FRUIT_RADIUS_FRAC = 0.053     # of frame height
FINGER_RADIUS_FRAC = 0.0075   # forgiveness around the fingertip
# Below this the tip is resting, not cutting. Without it a finger parked in
# the fruits' path would score off every one that fell onto it.
MIN_SWIPE_FRAC = 0.30     # of frame height per second

# Longer than the shared round: a fruit has to be thrown, rise and fall before
# it can be cut, so the same twenty seconds buys fewer chances here.
ROUND_SECONDS = 30.0

GRAVITY = 900.0           # px/s^2
SPAWN_EVERY = (0.7, 1.6)  # seconds between throws
TRAIL_LENGTH = 12

BOMB_SHARE = 0.18        # how often a bomb is thrown instead of fruit
BOMB_POINTS = -5

# A run of fruit, unbroken. Every fruit adds to it; letting one fall or
# cutting a bomb ends it. What the run buys is a bigger score for each fruit,
# which is what makes a long one worth protecting.
COMBO_STEP = 3           # one more point per fruit every this many
COMBO_MAX_BONUS = 9      # so a fruit is worth up to ten
POPUP_SECONDS = 0.6      # how long a fruit's points stay up where it was cut

# A clock, thrown in now and then, that buys more time when it is cut.
TIME_SHARE = 0.07
TIME_BONUS = 4.0         # seconds
TIME_COLORS = ((150, 120, 40), (235, 200, 120))

# Reach this run and everything comes at once for a few seconds -- fruit only,
# no bombs, so it is a free hand rather than a minefield. Ten, not more: the
# run breaks on a missed fruit as well as on a bomb, and a round is thirty
# seconds, so a longer one would be something most players never see.
FRENZY_AT = 10
FRENZY_SECONDS = 5.0
# Four times the usual flow. It started at twenty-two and then nine; both
# filled the screen faster than anyone could swing at it -- a flood rather
# than a feast.
FRENZY_SPAWN_EVERY = (0.28, 0.44)
FRENZY_BURST = (1, 2, 2)
# A breather once it ends: nothing is thrown for this long.
FRENZY_REST = 2.0

# (rind, flesh)
FRUIT_COLORS = [
    ((40, 120, 40), (70, 70, 220)),    # watermelon
    ((30, 130, 240), (90, 190, 250)),  # orange
    ((60, 200, 230), (120, 230, 245)),  # lemon
    ((40, 40, 190), (110, 110, 240)),  # apple
    ((120, 60, 100), (170, 110, 160)),  # plum
]


def _segment_hits_circle(p1, p2, center, radius):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    span = dx * dx + dy * dy
    if span < 1e-9:
        return math.dist(p1, center) <= radius
    t = max(0.0, min(1.0, ((center[0] - p1[0]) * dx + (center[1] - p1[1]) * dy) / span))
    closest = (p1[0] + t * dx, p1[1] + t * dy)
    return math.dist(closest, center) <= radius


def _pick_kind(frenzy):
    """What to throw next. A frenzy throws nothing but fruit; that is the
    whole of what makes it a reward rather than a harder stretch."""
    if frenzy:
        return "fruit"
    roll = random.random()
    if roll < BOMB_SHARE:
        return "bomb"
    if roll < BOMB_SHARE + TIME_SHARE:
        return "time"
    return "fruit"


def _spawn_fruit(w, h, radius, kind="fruit"):
    """Thrown in from below, or in from the bottom-left / bottom-right corner
    so they arc across the screen instead of always rising straight up."""
    if kind == "bomb":
        rind, flesh = (24, 24, 28), (44, 44, 52)
    elif kind == "time":
        rind, flesh = TIME_COLORS
    else:
        rind, flesh = random.choice(FRUIT_COLORS)
    side = random.choice(("bottom", "bottom", "left", "right"))

    if side != "bottom":
        from_left = side == "left"
        rise = random.uniform(h * 0.30, h * 0.55)
        return {
            "x": -radius if from_left else w + radius,
            "y": random.uniform(h * 0.72, h * 0.96),
            "vx": random.uniform(380, 620) * (1 if from_left else -1),
            "vy": -math.sqrt(2 * GRAVITY * rise),
            "r": radius, "rind": rind, "flesh": flesh, "kind": kind,
            "spin": random.uniform(-3, 3), "angle": 0.0,
        }

    x = random.uniform(w * 0.18, w * 0.82)
    # aim the throw at roughly the upper third of the screen
    rise = random.uniform(h * 0.52, h * 0.78)
    return {
        "x": x, "y": h + radius,
        "vx": random.uniform(-140, 140) + (w / 2 - x) * 0.35,
        "vy": -math.sqrt(2 * GRAVITY * rise),
        "r": radius, "rind": rind, "flesh": flesh, "kind": kind,
        "spin": random.uniform(-3, 3), "angle": 0.0,
    }


def _draw_fruit(frame, fruit):
    center = (int(fruit["x"]), int(fruit["y"]))
    radius = int(fruit["r"])
    cv2.circle(frame, center, radius, fruit["rind"], -1, cv2.LINE_AA)
    cv2.circle(frame, center, max(1, int(radius * 0.72)), fruit["flesh"], -1, cv2.LINE_AA)
    highlight = (int(center[0] - radius * 0.3), int(center[1] - radius * 0.32))
    cv2.circle(frame, highlight, max(1, int(radius * 0.16)), (255, 255, 255), -1, cv2.LINE_AA)
    if fruit.get("kind") == "time":
        # a clock face, so it is not mistaken for one more fruit
        cv2.circle(frame, center, radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, center, (center[0], center[1] - int(radius * 0.55)),
                 (40, 40, 50), max(2, radius // 8), cv2.LINE_AA)
        cv2.line(frame, center, (center[0] + int(radius * 0.42), center[1]),
                 (40, 40, 50), max(2, radius // 10), cv2.LINE_AA)
    elif fruit.get("kind") == "bomb":
        # a dark ball against a dark room is easy to swing through by
        # accident, so it gets a fuse and a warning ring
        cv2.circle(frame, center, radius, (60, 60, 210), 2, cv2.LINE_AA)
        fuse_top = (center[0] + int(radius * 0.45), center[1] - int(radius * 1.05))
        cv2.line(frame, (center[0] + int(radius * 0.3), center[1] - int(radius * 0.72)),
                 fuse_top, (90, 110, 140), max(2, radius // 10), cv2.LINE_AA)
        cv2.circle(frame, fuse_top, max(2, radius // 6), (40, 170, 250), -1, cv2.LINE_AA)
        cv2.circle(frame, fuse_top, max(1, radius // 12), (200, 240, 255), -1, cv2.LINE_AA)


def _draw_half(frame, half):
    center = (int(half["x"]), int(half["y"]))
    radius = int(half["r"])
    start = math.degrees(half["angle"])
    cv2.ellipse(frame, center, (radius, radius), start, 0, 180, half["rind"], -1, cv2.LINE_AA)
    cv2.ellipse(frame, center, (int(radius * 0.72), int(radius * 0.72)), start, 0, 180,
                half["flesh"], -1, cv2.LINE_AA)


def _split(fruit, cut_angle):
    halves = []
    normal = (-math.sin(cut_angle), math.cos(cut_angle))
    for side in (1, -1):
        halves.append({
            "x": fruit["x"], "y": fruit["y"],
            "vx": fruit["vx"] + normal[0] * 190 * side,
            "vy": fruit["vy"] + normal[1] * 190 * side - 60,
            "r": fruit["r"], "rind": fruit["rind"], "flesh": fruit["flesh"],
            "angle": cut_angle + (0 if side > 0 else math.pi),
            "spin": fruit["spin"] + side * 2.0,
        })
    return halves


def run_ninja_mode(cap, window_name, tracker):
    """Returns 'menu' or 'quit'."""
    dwell = DwellClickController()
    record = RoundRecord("ninja")
    # landmark 8 is the index fingertip. coast is left on: tracking gives out
    # exactly when a hand is swung hardest, and a cut already under way should
    # carry through rather than stop dead.
    finger_tracker = HandPaddles(1.0, 1.0, landmark=8)

    ret, frame = cap.read()
    if not ret:
        return "quit"
    h, w = frame.shape[:2]

    buttons = [
        Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38)),
        Button("restart", "YENİDEN", 120, 10, 150, TOOLBAR_H - 20, color=(26, 46, 30)),
    ]

    fruits, halves = [], []
    trails = {}          # hand id -> recent fingertip positions, for the swoosh
    score, missed = 0, 0
    combo, best_combo = 0, 0
    frenzy_until = 0.0
    popups = []          # (text, x, y, until): a fruit's points, where it was cut
    message, message_until = "", 0.0
    flash_until = 0.0
    next_spawn = time.time() + 1.0
    round_start = time.time()
    last_time = time.time()

    result = None
    while result is None:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = min(now - last_time, 0.05)
        last_time = now

        hands = tracker.process(frame)

        clicked, progress_map = dwell.update(hands, buttons)
        if clicked == "menu":
            result = "menu"
            break
        elif clicked == "restart":
            fruits.clear()
            halves.clear()
            trails.clear()
            score, missed = 0, 0
            record.reset()
            combo, best_combo = 0, 0
            frenzy_until = 0.0
            popups.clear()
            message, message_until = "", 0.0
            next_spawn = now + 1.0
            round_start = now

        remaining = max(ROUND_SECONDS - (now - round_start), 0.0)
        running = remaining > 0

        fruit_radius = FRUIT_RADIUS_FRAC * h
        finger_r = FINGER_RADIUS_FRAC * h
        min_swipe = MIN_SWIPE_FRAC * h

        frenzy = now < frenzy_until

        resting = frenzy_until > 0 and 0 <= now - frenzy_until < FRENZY_REST

        # ---- spawn ----
        if running and not resting and now >= next_spawn:
            burst = FRENZY_BURST if frenzy else (1, 1, 2)
            for _ in range(random.choice(burst)):
                fruit = _spawn_fruit(w, h, fruit_radius, _pick_kind(frenzy))
                # remembered on the fruit itself, not read off the clock, so a
                # frenzy fruit still counts as one if it is cut after it ends
                fruit["frenzy"] = frenzy
                fruits.append(fruit)
            gap = FRENZY_SPAWN_EVERY if frenzy else SPAWN_EVERY
            next_spawn = now + random.uniform(*gap)

        tips = finger_tracker.update(hands, now, dt)

        # a trail each: sharing one would draw a line from one hand to
        # the other every time they took turns
        live = {tip.id for tip in tips}
        for stale in [i for i in trails if i not in live]:
            del trails[stale]
        for tip in tips:
            path = trails.setdefault(tip.id, [])
            path.append((int(tip.end[0]), int(tip.end[1])))
            del path[:-TRAIL_LENGTH]

        # ---- slicing: the path a tip swept since the last frame is the
        # blade, so a fast flick can't skip over a fruit between frames.
        # Each hand is tested on its own, and a fruit leaves the list the
        # moment it is cut, so two fingers crossing it cannot score twice.
        for tip in tips:
            if math.hypot(*tip.velocity) < min_swipe:
                continue
            angle = math.atan2(tip.velocity[1], tip.velocity[0])
            for fruit in fruits[:]:
                center = (fruit["x"], fruit["y"])
                if not _segment_hits_circle(tip.start, tip.end,
                                            center, fruit["r"] + finger_r):
                    continue
                fruits.remove(fruit)
                halves.extend(_split(fruit, angle))
                kind = fruit.get("kind", "fruit")
                if kind == "bomb":
                    score += BOMB_POINTS
                    combo = 0
                    message, message_until = "BOMBA", now + 1.2
                    flash_until = now + 0.18
                elif kind == "time":
                    round_start += TIME_BONUS
                    message, message_until = f"+{TIME_BONUS:.0f} SANIYE", now + 1.2
                elif fruit.get("frenzy"):
                    # Paid at the run's rate, but not added to it. Counted,
                    # a frenzy's fruit would carry the run straight to the
                    # next frenzy, and that one to the next: a run that never
                    # ends.
                    points = 1 + min(combo // COMBO_STEP, COMBO_MAX_BONUS)
                    score += points
                    popups.append((f"+{points}", fruit["x"], fruit["y"], now + POPUP_SECONDS))
                else:
                    combo += 1
                    best_combo = max(best_combo, combo)
                    points = 1 + min(combo // COMBO_STEP, COMBO_MAX_BONUS)
                    score += points
                    # shown where the fruit was, so the run's worth can be
                    # seen climbing rather than only read off the total
                    popups.append((f"+{points}", fruit["x"], fruit["y"], now + POPUP_SECONDS))
                    if combo % FRENZY_AT == 0 and now >= frenzy_until:
                        frenzy_until = now + FRENZY_SECONDS
                        message, message_until = "CILDIRMA", now + 1.4

        # ---- physics ----
        for item in fruits + halves:
            item["vy"] += GRAVITY * dt
            item["x"] += item["vx"] * dt
            item["y"] += item["vy"] * dt
            item["angle"] = item.get("angle", 0.0) + item["spin"] * dt
        for fruit in fruits[:]:
            gone = (fruit["y"] - fruit["r"] > h
                    or fruit["x"] < -3 * fruit["r"] or fruit["x"] > w + 3 * fruit["r"])
            if gone:
                fruits.remove(fruit)
                # only fruit is owed a cut: a bomb that falls is a good
                # outcome, and a clock missed is an offer not taken
                if fruit.get("kind", "fruit") == "fruit" and not fruit.get("frenzy"):
                    missed += 1
                    combo = 0
        halves = [hf for hf in halves if hf["y"] - hf["r"] <= h]

        # ================= draw =================
        for half in halves:
            _draw_half(frame, half)
        for fruit in fruits:
            _draw_fruit(frame, fruit)

        popups = [p for p in popups if p[3] > now]
        for text, px, py, until in popups:
            rise = (1.0 - (until - now) / POPUP_SECONDS) * 40
            draw_text(frame, text, (int(px), int(py - 30 - rise)), scale=3, anchor="center")

        for path in trails.values():
            for i in range(1, len(path)):
                fade = i / len(path)
                shade = int(90 + 165 * fade)
                cv2.line(frame, path[i - 1], path[i], (shade, shade, shade),
                         max(1, int(1 + 4 * fade)), cv2.LINE_AA)

        for tip in tips:
            # the tip itself, so the player can see exactly what cuts, and
            # a ring around it marking the forgiveness it actually gets
            point = (int(tip.end[0]), int(tip.end[1]))
            cv2.circle(frame, point, max(3, int(finger_r)),
                       (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, max(6, int(finger_r) + 4),
                       (120, 200, 255), 2, cv2.LINE_AA)

        for btn in buttons:
            btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                     hovered=dwell.hovered_id() == btn.id)

        draw_panel(frame, f"SKOR: {score}", (w // 2, 26), scale=3,
                                  anchor="center")
        draw_round_timer(frame, min(remaining / ROUND_SECONDS, 1.0))
        draw_panel(frame, f"KAÇAN: {missed}", (w - 30, 26), scale=2,
                                  anchor="topright")
        if running and combo > 1:
            draw_panel(frame, f"KOMBO x{combo}", (w // 2, TOOLBAR_H + 60),
                       scale=3, anchor="center",
                       plate=(20, 70, 30) if frenzy else PANEL_COLOR)
        if now < flash_until:
            cv2.rectangle(frame, (0, 0), (w, h), (40, 40, 200), 14)
        if frenzy and running:
            # a border while it lasts, so the free hand is unmistakable
            cv2.rectangle(frame, (0, 0), (w, h), (60, 200, 90), 10)
        if not running:
            record.finish(score)
            draw_panel(frame, f"SÜRE DOLDU - SKOR: {score}",
                       (w // 2, h // 2 - 90), scale=3, anchor="center",
                       plate=(0, 60, 130))
            draw_record(frame, record, (w // 2, h // 2 - 30))
            draw_panel(frame, f"EN UZUN KOMBO: {best_combo}", (w // 2, h // 2 + 30),
                       scale=2, anchor="center")
            draw_panel(frame, "YENİDEN DÜĞMESİNE BAS", (w // 2, h // 2 + 90),
                       scale=2, anchor="center")
        if now < message_until:
            draw_panel(frame, message, (w // 2, 96), scale=3,
                                      anchor="center", plate=(0, 0, 130))
        draw_rig(frame, hands)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        handle_key(window_name, key)
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
