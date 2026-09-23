"""Keepy-uppy: bounce the ball on your hands and keep it off the floor.

Each hand is a circle the ball reflects off, so where on the palm it lands
decides which way it flies, and swiping upwards as it arrives puts extra pace
on it -- the hand's own velocity is carried into the bounce. Both hands are
live at once.

No face tracking, so the player doesn't have to look at the camera at all.
"""
import math
import random
import time

import cv2

from core import assets
from core.geometry import closest_on_segment
from core.paddles import HandPaddles
from core.transform import integer_scale_for, place_rotated
from core.ui import Button, DwellClickController, draw_panel

TOOLBAR_H = 90

# The ball keeps one size on screen however close the player stands.
BALL_DIAMETER_FRAC = 0.13   # of frame height

# The hitbox follows the hand's real size, with a floor so the game stays
# playable from across the room where the hand is only a few dozen pixels.
HAND_RADIUS_PER_SPAN = 1.15
HAND_RADIUS_FLOOR_FRAC = 0.045   # of frame height

# A bounce rises about a quarter of the frame and hangs for roughly three
# quarters of a second.
RISE_FRAC = 0.26
RISE_TIME = 0.72
MAX_SPEED_FACTOR = 2.2
TOP_MARGIN_FRAC = 0.06      # never let the arc peak above this much of frame
MIN_BOUNCE_FACTOR = 0.32    # ...but never damp it into a stall either
RESTITUTION = 0.92
WALL_RESTITUTION = 1.0      # the sides give everything back, so a ball driven
WALL_MIN_SPEED_FRAC = 0.22  # into them comes off lively instead of dying there
HAND_ASSIST = 0.55          # how much of the hand's own motion the ball takes
BOUNCE_COOLDOWN = 0.2       # stops one touch counting several times
SPIN_PER_VX = 0.012


def _respawn(w, ball_r, near_x=None):
    x = (near_x if near_x is not None else w / 2) + random.uniform(-1.0, 1.0) * ball_r
    return {
        "x": min(max(x, ball_r), w - ball_r), "y": -ball_r,
        "vx": random.uniform(-0.3, 0.3) * ball_r,
        "vy": 0.0, "angle": 0.0,
    }


def run_juggle_mode(cap, window_name, tracker):
    """Returns 'menu' or 'quit'."""
    dwell = DwellClickController()
    ball_img = assets.load("ball.png")

    ret, frame = cap.read()
    if not ret:
        return "quit"
    h, w = frame.shape[:2]

    buttons = [Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38))]

    ball_r = BALL_DIAMETER_FRAC * h / 2
    if ball_img is not None:
        ball_r = integer_scale_for(ball_img, ball_r * 2) * max(ball_img.shape[:2]) / 2
    rise = RISE_FRAC * h
    bounce_speed = 2 * rise / RISE_TIME
    gravity = bounce_speed / RISE_TIME
    max_speed = bounce_speed * MAX_SPEED_FACTOR
    radius_floor = HAND_RADIUS_FLOOR_FRAC * h

    hand_paddles = HandPaddles(HAND_RADIUS_PER_SPAN, radius_floor)
    ball = None
    streak, best = 0, 0
    last_bounce = 0.0
    message, message_until = "", 0.0
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

        paddles = hand_paddles.update(hands, now, dt)

        if ball is None:
            ball = _respawn(w, ball_r, paddles[0].end[0] if paddles else None)

        # ---- physics ----
        ball["vy"] += gravity * dt
        ball["x"] += ball["vx"] * dt
        ball["y"] += ball["vy"] * dt
        ball["angle"] += ball["vx"] * SPIN_PER_VX * dt

        wall_min = bounce_speed * WALL_MIN_SPEED_FRAC
        if ball["x"] < ball_r:
            ball["x"] = ball_r
            ball["vx"] = max(abs(ball["vx"]) * WALL_RESTITUTION, wall_min)
        elif ball["x"] > w - ball_r:
            ball["x"] = w - ball_r
            ball["vx"] = -max(abs(ball["vx"]) * WALL_RESTITUTION, wall_min)
        if ball["y"] < ball_r and ball["vy"] < 0:
            ball["y"] = ball_r
            ball["vy"] = abs(ball["vy"]) * 0.5

        # ---- bounce off a hand ----
        if now - last_bounce > BOUNCE_COOLDOWN:
            for paddle in paddles:
                contact = paddle.radius + ball_r
                velocity = paddle.velocity
                # test the path the hand swept, not just where it ended up
                touch = closest_on_segment(paddle.start, paddle.end,
                                           (ball["x"], ball["y"]))
                dx, dy = ball["x"] - touch[0], ball["y"] - touch[1]
                distance = math.hypot(dx, dy)
                if distance > contact:
                    continue
                if distance < 1e-6:
                    nx, ny = 0.0, -1.0
                else:
                    nx, ny = dx / distance, dy / distance
                # reflect in the hand's frame, so a rising hand hits harder
                rvx = ball["vx"] - velocity[0] * HAND_ASSIST
                rvy = ball["vy"] - velocity[1] * HAND_ASSIST
                approaching = rvx * nx + rvy * ny
                if approaching >= 0:
                    continue
                rvx -= 2 * approaching * nx
                rvy -= 2 * approaching * ny
                vx = rvx * RESTITUTION + velocity[0] * HAND_ASSIST
                vy = min(rvy * RESTITUTION + velocity[1] * HAND_ASSIST, -bounce_speed)
                speed = math.hypot(vx, vy)
                if speed > max_speed:
                    vx, vy = vx / speed * max_speed, vy / speed * max_speed
                # lift it clear so it can't re-trigger next frame
                ball["x"] = touch[0] + nx * (contact + 1)
                ball["y"] = touch[1] + ny * (contact + 1)
                # cap the arc so it always peaks inside the frame, measured
                # from where the ball now sits rather than where it was
                headroom = max(ball["y"] - TOP_MARGIN_FRAC * h, 0.0)
                reachable = math.sqrt(2 * gravity * headroom)
                vy = -max(min(-vy, reachable), bounce_speed * MIN_BOUNCE_FACTOR)
                ball["vx"], ball["vy"] = vx, vy
                last_bounce = now
                streak += 1
                best = max(best, streak)
                if streak and streak % 10 == 0:
                    message, message_until = f"{streak} SERİ!", now + 1.4
                break

        # ---- dropped ----
        if ball["y"] - ball_r > h:
            if streak:
                message, message_until = f"DÜŞTÜ - {streak}", now + 1.6
            streak = 0
            ball = None

        # ================= draw =================
        for paddle in paddles:
            cv2.circle(frame, (int(paddle.end[0]), int(paddle.end[1])),
                       int(paddle.radius), (70, 170, 70), 2, cv2.LINE_AA)

        if ball is not None:
            center = (ball["x"], ball["y"])
            if ball_img is not None:
                place_rotated(frame, ball_img, center, ball["angle"],
                              int(ball_r * 2), pivot=(0.5, 0.5))
            else:
                cv2.circle(frame, (int(center[0]), int(center[1])), int(ball_r),
                           (240, 240, 240), -1, cv2.LINE_AA)

        for btn in buttons:
            btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                     hovered=dwell.hovered_id() == btn.id)

        draw_panel(frame, f"SERİ: {streak}", (w // 2, 26), scale=3,
                                  anchor="center")
        draw_panel(frame, f"REKOR: {best}", (w - 30, 26), scale=2,
                                  anchor="topright", color=(150, 150, 150))
        if now < message_until:
            draw_panel(frame, message, (w // 2, 92), scale=3,
                                      anchor="center", plate=(0, 110, 0))
        if not paddles:
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
