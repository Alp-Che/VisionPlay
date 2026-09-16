"""Fruit Ninja.

A single sword hangs off the player's hand -- the right one, or the left if
the right isn't in frame -- pointing the way that hand points and sized by its
span, so it shrinks along with the player as they move back from the camera.
The whole cutting edge slices on contact, and the area the edge swept since
the previous frame counts too, so a fast swing cannot pass straight through a
fruit between frames.
"""
import math
import random
import time

import cv2

from core import assets
from core.hand_tracker import HandTracker, hand_angle, hand_foreshorten, hand_span
from core.pixel_font import draw_text_with_background
from core.transform import integer_scale_for, place_rotated
from core.ui import Button, DwellClickController

TOOLBAR_H = 90

# sword.png is 13x58 and drawn point-up: blade tip at the very top, crossguard
# at y=45-48, handle below it. Anchored at the handle so the fist holds it.
SWORD_GRIP_FRAC = (0.5, 0.922)
SWORD_TIP_FRAC = (0.5, 0.009)
# place_rotated aligns the sprite's +x axis with the angle it is given, but
# this blade points up (local -y), so it is turned a quarter turn further.
SWORD_ANGLE_OFFSET = math.pi / 2

SWORD_PER_HAND = 4.6      # whole sprite height, in hand spans
# The cutting edge runs from just above the crossguard to the tip, measured
# out from the grip as a fraction of the sprite's height. The whole edge
# cuts, not only the point.
EDGE_NEAR_FRAC = SWORD_GRIP_FRAC[1] - 0.776
EDGE_FAR_FRAC = SWORD_GRIP_FRAC[1] - SWORD_TIP_FRAC[1]
EDGE_SAMPLES = 6
SQUASH_SMOOTHING = 0.35   # blends the depth estimate, ~3 frames to settle

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


def run_ninja_mode(cap, window_name):
    """Returns 'menu' or 'quit'."""
    tracker = HandTracker(num_hands=2)
    dwell = DwellClickController(dwell_seconds=2.0)
    sword_img = assets.load("sword.png")

    ret, frame = cap.read()
    if not ret:
        tracker.close()
        return "quit"
    h, w = frame.shape[:2]

    buttons = [Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38))]

    fruits, halves = [], []
    trail = []           # recent blade-tip positions, for the swoosh
    last_edge = None     # where the edge was last frame
    hand_px = 45.0
    squash = 1.0         # smoothed, so a jittery depth estimate can't make
    score, missed = 0, 0  # the blade pulse while the hand is held still
    message, message_until = "", 0.0
    flash_until = 0.0
    next_spawn = time.time() + 1.0
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
            if hands:
                measured = max(hand_span(hd) for hd in hands)
                if measured > 5:
                    hand_px += (measured - hand_px) * 0.15

            clicked, progress_map = dwell.update(hands, buttons)
            if clicked == "menu":
                result = "menu"
                break

            sword_size = SWORD_PER_HAND * hand_px
            if sword_img is not None:
                sword_size = integer_scale_for(sword_img, sword_size) * max(sword_img.shape[:2])
            fruit_radius = FRUIT_RADIUS_PER_HAND * hand_px

            # ---- spawn ----
            if now >= next_spawn:
                for _ in range(random.choice((1, 1, 2))):
                    fruits.append(_spawn_fruit(w, h, fruit_radius))
                next_spawn = now + random.uniform(*SPAWN_EVERY)

            # ---- one sword: the right hand carries it, the left only if the
            # right isn't in frame, so a left-hander can just drop it out ----
            sword_hand = next((hd for hd in hands if hd.label == "Right"), None)
            if sword_hand is None:
                sword_hand = next((hd for hd in hands if hd.label == "Left"), None)

            blade = None
            if sword_hand is not None:
                angle = hand_angle(sword_hand)
                grip = sword_hand.landmarks_px[9]
                # point the hand at the lens and the blade all but disappears,
                # so the drawn length and the cutting edge both collapse with it
                squash += (hand_foreshorten(sword_hand) - squash) * SQUASH_SMOOTHING
                near = EDGE_NEAR_FRAC * sword_size * squash
                far = EDGE_FAR_FRAC * sword_size * squash
                step = (math.cos(angle), math.sin(angle))
                edge = [(int(grip[0] + step[0] * d), int(grip[1] + step[1] * d))
                        for d in (near + (far - near) * i / (EDGE_SAMPLES - 1)
                                  for i in range(EDGE_SAMPLES))]
                blade = (grip, angle, edge, squash)

            # ---- slicing: the whole edge cuts, and the path it swept since
            # the last frame counts too so fast swings can't pass through ----
            if blade is None:
                trail.clear()
                last_edge = None
            else:
                _grip, angle, edge, _squash = blade
                trail.append(edge[-1])
                del trail[:-TRAIL_LENGTH]

                for fruit in fruits[:]:
                    center, radius = (fruit["x"], fruit["y"]), fruit["r"]
                    hit = _segment_hits_circle(edge[0], edge[-1], center, radius)
                    if not hit and last_edge is not None:
                        hit = any(_segment_hits_circle(last_edge[i], edge[i], center, radius)
                                  for i in range(EDGE_SAMPLES))
                    if hit:
                        fruits.remove(fruit)
                        halves.extend(_split(fruit, angle))
                        if fruit.get("bomb"):
                            score += BOMB_POINTS
                            message, message_until = "BOMBA!", now + 1.2
                            flash_until = now + 0.18
                        else:
                            score += 1

                last_edge = edge

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

            for i in range(1, len(trail)):
                fade = i / len(trail)
                shade = int(90 + 165 * fade)
                cv2.line(frame, trail[i - 1], trail[i], (shade, shade, shade),
                         max(1, int(1 + 4 * fade)), cv2.LINE_AA)

            if blade is not None:
                grip, angle, edge, squash = blade
                if sword_img is not None:
                    place_rotated(frame, sword_img, grip, angle + SWORD_ANGLE_OFFSET,
                                  sword_size, pivot=SWORD_GRIP_FRAC, squash=squash)
                else:
                    cv2.line(frame, edge[0], edge[-1], (220, 220, 220), 6, cv2.LINE_AA)

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
            draw_text_with_background(frame, "ELİNİ SALLA, MEYVELERİ KES",
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
