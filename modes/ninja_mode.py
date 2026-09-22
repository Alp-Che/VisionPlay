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

from core.hand_tracker import hand_span
from core.paddles import HandPaddles
from core.pixel_font import draw_text_with_background
from core.ui import Button, DwellClickController

TOOLBAR_H = 90

# A little forgiveness around the tip, measured in hand spans so it shrinks
# with the player as they step back -- the same way everything else here does.
FINGER_RADIUS_PER_HAND = 0.12
# Below this the tip is resting, not cutting. Without it a finger parked in
# the fruits' path would score off every one that fell onto it.
MIN_SWIPE_FRAC = 0.30     # of frame height per second

GRAVITY = 900.0           # px/s^2
SPAWN_EVERY = (0.7, 1.6)  # seconds between throws
FRUIT_RADIUS_PER_HAND = 0.85
TRAIL_LENGTH = 12

BOMB_SHARE = 0.18        # how often a bomb is thrown instead of fruit
BOMB_POINTS = -5

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


def _spawn_fruit(w, h, radius):
    """Thrown in from below, or in from the bottom-left / bottom-right corner
    so they arc across the screen instead of always rising straight up."""
    bomb = random.random() < BOMB_SHARE
    rind, flesh = ((24, 24, 28), (44, 44, 52)) if bomb else random.choice(FRUIT_COLORS)
    side = random.choice(("bottom", "bottom", "left", "right"))

    if side != "bottom":
        from_left = side == "left"
        rise = random.uniform(h * 0.30, h * 0.55)
        return {
            "x": -radius if from_left else w + radius,
            "y": random.uniform(h * 0.72, h * 0.96),
            "vx": random.uniform(380, 620) * (1 if from_left else -1),
            "vy": -math.sqrt(2 * GRAVITY * rise),
            "r": radius, "rind": rind, "flesh": flesh, "bomb": bomb,
            "spin": random.uniform(-3, 3), "angle": 0.0,
        }

    x = random.uniform(w * 0.18, w * 0.82)
    # aim the throw at roughly the upper third of the screen
    rise = random.uniform(h * 0.52, h * 0.78)
    return {
        "x": x, "y": h + radius,
        "vx": random.uniform(-140, 140) + (w / 2 - x) * 0.35,
        "vy": -math.sqrt(2 * GRAVITY * rise),
        "r": radius, "rind": rind, "flesh": flesh, "bomb": bomb,
        "spin": random.uniform(-3, 3), "angle": 0.0,
    }


def _draw_fruit(frame, fruit):
    center = (int(fruit["x"]), int(fruit["y"]))
    radius = int(fruit["r"])
    cv2.circle(frame, center, radius, fruit["rind"], -1, cv2.LINE_AA)
    cv2.circle(frame, center, max(1, int(radius * 0.72)), fruit["flesh"], -1, cv2.LINE_AA)
    highlight = (int(center[0] - radius * 0.3), int(center[1] - radius * 0.32))
    cv2.circle(frame, highlight, max(1, int(radius * 0.16)), (255, 255, 255), -1, cv2.LINE_AA)
    if fruit.get("bomb"):
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
    # landmark 8 is the index fingertip. coast is left on: tracking gives out
    # exactly when a hand is swung hardest, and a cut already under way should
    # carry through rather than stop dead.
    finger_tracker = HandPaddles(1.0, 1.0, landmark=8)

    ret, frame = cap.read()
    if not ret:
        return "quit"
    h, w = frame.shape[:2]

    buttons = [Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38))]

    fruits, halves = [], []
    trails = {}          # hand id -> recent fingertip positions, for the swoosh
    hand_px = 45.0
    score, missed = 0, 0
    message, message_until = "", 0.0
    flash_until = 0.0
    next_spawn = time.time() + 1.0
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
        if hands:
            measured = max(hand_span(hd) for hd in hands)
            if measured > 5:
                hand_px += (measured - hand_px) * 0.15

        clicked, progress_map = dwell.update(hands, buttons)
        if clicked == "menu":
            result = "menu"
            break

        fruit_radius = FRUIT_RADIUS_PER_HAND * hand_px
        finger_r = FINGER_RADIUS_PER_HAND * hand_px
        min_swipe = MIN_SWIPE_FRAC * h

        # ---- spawn ----
        if now >= next_spawn:
            for _ in range(random.choice((1, 1, 2))):
                fruits.append(_spawn_fruit(w, h, fruit_radius))
            next_spawn = now + random.uniform(*SPAWN_EVERY)

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
                if fruit.get("bomb"):
                    score += BOMB_POINTS
                    message, message_until = "BOMBA!", now + 1.2
                    flash_until = now + 0.18
                else:
                    score += 1

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
                if not fruit.get("bomb"):
                    missed += 1
        halves = [hf for hf in halves if hf["y"] - hf["r"] <= h]

        # ================= draw =================
        for half in halves:
            _draw_half(frame, half)
        for fruit in fruits:
            _draw_fruit(frame, fruit)

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

        draw_text_with_background(frame, f"SKOR: {score}", (w // 2, 26), scale=3,
                                  anchor="center")
        draw_text_with_background(frame, f"KAÇAN: {missed}", (w - 30, 26), scale=2,
                                  anchor="topright", color=(150, 150, 150))
        if now < flash_until:
            cv2.rectangle(frame, (0, 0), (w, h), (40, 40, 200), 14)
        if now < message_until:
            draw_text_with_background(frame, message, (w // 2, 96), scale=3,
                                      anchor="center", bg_color=(0, 0, 130))
        draw_text_with_background(frame, "İKİ İŞARET PARMAĞINLA DA KESEBİLİRSİN",
                                  (14, h - 44), scale=2, color=(200, 200, 200))

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
