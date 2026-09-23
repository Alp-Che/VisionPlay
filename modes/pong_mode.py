"""Pong, for two people standing side by side.

Each player moves the bat on their own side with one hand. Which bat a hand
gets is decided by where it is: with two hands in view the leftmost takes the
left bat and the rightmost the right, so two players simply stand where they
want to play. Nothing keys on MediaPipe's "Left"/"Right" -- that is the
player's own handedness, not the side of the picture they are standing on.

The ball is stepped in small pieces rather than moved all at once. At full
speed it crosses more than a bat's thickness between frames, and a bat is
thin, so a single big step would let it pass clean through.
"""
import math
import random
import time

import cv2

from core.rig import draw_rig
from core.ui import ROUND_SECONDS, Button, DwellClickController, draw_panel, draw_round_timer

TOOLBAR_H = 90

BAT_INSET_FRAC = 0.055       # how far the bats sit in from the edges
BAT_WIDTH_FRAC = 0.016       # of frame width
BAT_HEIGHT_FRAC = 0.24       # of frame height
BALL_RADIUS_FRAC = 0.022     # of frame height

BALL_START_SPEED_FRAC = 0.62  # of frame height per second
BALL_SPEEDUP = 1.04          # each return makes it a little quicker
BALL_MAX_SPEED_FRAC = 1.5
# How much of the bounce angle the contact point decides: hit the end of the
# bat and the ball leaves at a steeper angle than off the middle.
MAX_BOUNCE_ANGLE = math.radians(52)
SERVE_DELAY = 0.7            # a pause after a point, so it can be seen

BALL_COLOR = (235, 235, 235)
BAT_COLORS = ((120, 200, 120), (120, 160, 240))
COURT_COLOR = (90, 90, 90)


def _serve(w, h, towards):
    """A ball at the middle, aimed at the player who just conceded."""
    angle = random.uniform(-0.35, 0.35)
    speed = BALL_START_SPEED_FRAC * h
    return {
        "x": w / 2.0, "y": h / 2.0,
        "vx": math.cos(angle) * speed * towards,
        "vy": math.sin(angle) * speed,
        "speed": speed,
    }


def _bat_targets(hands, w):
    """Which hand drives which bat, by where the hands are, not what they are.

    Two or more hands: the leftmost takes the left bat, the rightmost the
    right. One hand: it takes the bat on the side of the picture it is in.
    """
    if not hands:
        return None, None
    ordered = sorted(hands, key=lambda hd: hd.landmarks_px[9][0])
    if len(ordered) == 1:
        only = ordered[0].landmarks_px[9]
        return (only[1], None) if only[0] < w / 2 else (None, only[1])
    return ordered[0].landmarks_px[9][1], ordered[-1].landmarks_px[9][1]


def _draw_court(frame, w, h):
    for y in range(TOOLBAR_H + 10, h - 10, 34):
        cv2.line(frame, (w // 2, y), (w // 2, min(y + 18, h - 10)), COURT_COLOR, 3)


def run_pong_mode(cap, window_name, tracker):
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

    bat_w = max(6, int(BAT_WIDTH_FRAC * w))
    bat_h = BAT_HEIGHT_FRAC * h
    ball_r = int(BALL_RADIUS_FRAC * h)
    max_speed = BALL_MAX_SPEED_FRAC * h
    inset = BAT_INSET_FRAC * w
    top, bottom = TOOLBAR_H + ball_r, h - ball_r

    bats = [h / 2.0, h / 2.0]                 # centre y of each bat
    bat_x = (inset, w - inset - bat_w)
    scores = [0, 0]
    ball = _serve(w, h, random.choice((-1, 1)))
    serve_at = time.time() + SERVE_DELAY
    round_start = time.time()
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
            scores = [0, 0]
            ball = _serve(w, h, random.choice((-1, 1)))
            serve_at = now + SERVE_DELAY
            round_start = now

        remaining = max(ROUND_SECONDS - (now - round_start), 0.0)
        running = remaining > 0

        # --- bats follow their hands ---
        left_y, right_y = _bat_targets(hands, w)
        for i, target in enumerate((left_y, right_y)):
            if target is not None:
                bats[i] = min(max(float(target), TOOLBAR_H + bat_h / 2), h - bat_h / 2)

        # --- the ball, in small steps so a bat cannot be passed through ---
        if running and now >= serve_at:
            travel = math.hypot(ball["vx"], ball["vy"]) * dt
            steps = max(1, int(travel / (bat_w * 0.5)) + 1)
            for _ in range(steps):
                ball["x"] += ball["vx"] * dt / steps
                ball["y"] += ball["vy"] * dt / steps

                if ball["y"] <= top and ball["vy"] < 0:
                    ball["y"] = top
                    ball["vy"] = -ball["vy"]
                elif ball["y"] >= bottom and ball["vy"] > 0:
                    ball["y"] = bottom
                    ball["vy"] = -ball["vy"]

                for i in (0, 1):
                    going_in = ball["vx"] < 0 if i == 0 else ball["vx"] > 0
                    if not going_in:
                        continue
                    face = bat_x[i] + bat_w if i == 0 else bat_x[i]
                    reached = ball["x"] - ball_r <= face if i == 0 else ball["x"] + ball_r >= face
                    if not reached:
                        continue
                    offset = (ball["y"] - bats[i]) / (bat_h / 2)
                    if abs(offset) > 1.15:
                        continue
                    angle = max(-1.0, min(1.0, offset)) * MAX_BOUNCE_ANGLE
                    ball["speed"] = min(ball["speed"] * BALL_SPEEDUP, max_speed)
                    towards = 1 if i == 0 else -1
                    ball["vx"] = math.cos(angle) * ball["speed"] * towards
                    ball["vy"] = math.sin(angle) * ball["speed"]
                    ball["x"] = face + ball_r * towards

                if ball["x"] < -ball_r * 3 or ball["x"] > w + ball_r * 3:
                    conceded = 0 if ball["x"] < 0 else 1
                    scores[1 - conceded] += 1
                    ball = _serve(w, h, -1 if conceded == 1 else 1)
                    serve_at = now + SERVE_DELAY
                    break

        # ================= draw =================
        _draw_court(frame, w, h)
        for i in (0, 1):
            y0 = int(bats[i] - bat_h / 2)
            cv2.rectangle(frame, (int(bat_x[i]), y0),
                          (int(bat_x[i]) + bat_w, int(y0 + bat_h)),
                          BAT_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(frame, (int(ball["x"]), int(ball["y"])), ball_r,
                   BALL_COLOR, -1, cv2.LINE_AA)

        for btn in buttons:
            btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                     hovered=dwell.hovered_id() == btn.id)

        draw_round_timer(frame, remaining / ROUND_SECONDS)
        # each score on its own player's side, and below the bar rather than
        # beside it -- the bar is wide enough to reach the middle of the top
        draw_panel(frame, str(scores[0]), (int(w * 0.28), TOOLBAR_H + 60),
                   scale=5, anchor="center")
        draw_panel(frame, str(scores[1]), (int(w * 0.72), TOOLBAR_H + 60),
                   scale=5, anchor="center")

        if not running:
            if scores[0] == scores[1]:
                sonuc = f"BERABERE {scores[0]} - {scores[1]}"
            else:
                kazanan = "SOL" if scores[0] > scores[1] else "SAĞ"
                sonuc = f"{kazanan} KAZANDI {max(scores)} - {min(scores)}"
            draw_panel(frame, sonuc, (w // 2, h // 2 - 30), scale=3, anchor="center",
                       plate=(0, 60, 130))
            draw_panel(frame, "YENİDEN DÜĞMESİNE BAS", (w // 2, h // 2 + 30),
                       scale=2, anchor="center")
        elif not hands:
            draw_panel(frame, "ELLERİNİZİ KAMERAYA GÖSTERİN", (w // 2, h // 2),
                       scale=2, anchor="center", plate=(0, 90, 140))

        draw_rig(frame, hands)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
