"""Mode 3: archery.

- Bow hand closed (fist)  = holds the bow.
- Draw hand closed (fist) = an arrow appears in that hand and follows it.
  Bring it close to the bow (while the bow hand is closed) and it snaps
  onto the string -- only then does pulling away draw the bow. Pull
  distance -> launch power. Opening the draw hand while nocked releases
  the shot.
- A "YON" button up top swaps which physical hand is the bow hand, and
  flips the target to the other side of the screen to match.
- Flight is a real projectile arc (constant gravity), not a straight line.
- Target hit detection is a single point (its center pixel): once the
  arrow's flight crosses the target's x-position, only the vertical (y)
  offset from that center point decides the score -- not how far forward/
  back it is.
"""
import math
import time

import cv2

from core import assets
from core.hand_tracker import hand_span
from core.transform import integer_scale_for, place_rotated
from core.pixel_font import draw_text
from core.rig import draw_rig
from core.window import handle_key
from core.ui import Button, DwellClickController, draw_panel

TOOLBAR_H = 90

# Gear is drawn at a fixed size, as a fraction of the frame, so it neither
# grows nor shrinks as the player moves about.
BOW_FRAC = 0.24
ARROW_FRAC = 0.18
TARGET_FRAC = 0.34

# How far the bow has to be drawn, though, still follows the hand: at a few
# metres back the whole arm span is only a couple of hundred pixels, and a
# fixed pixel threshold there is simply unreachable.
#
# The snap reaches three hand spans, not the one and three-quarters it was.
# At that the fists had to all but touch, and two fists touching is exactly
# what the tracker handles worst -- it tends to lose one of them, and the snap
# went with it. Reaching further, the arrow locks on while the hands are still
# clearly apart and still both being followed.
SNAP_PER_HAND = 3.0
MIN_PULL_PER_HAND = 0.9
MAX_PULL_PER_HAND = 4.2

DEFAULT_HAND_PX = 45.0
HAND_SMOOTHING = 0.12  # per-frame blend, keeps the draw length steady

ARROWS_PER_ROUND = 10

# A foul line between archer and target. Reach past it and the bow won't
# form, so there's no creeping up on the target for an easy ten. Measured as
# a gap from the target rather than a fixed screen position, so the distance
# it actually enforces doesn't drift if the target moves or changes size.
FOUL_GAP_FRAC = 0.48      # of frame width, from the target

# The target slides up and down its side of the screen.
TARGET_TRAVEL_TOP = 0.20     # of frame height
TARGET_TRAVEL_BOTTOM = 0.74
TARGET_PERIOD = 3.6          # seconds for a full up-and-down

# Anchor points as fractions of each sprite's own (width, height) -- see
# core/transform.place_rotated's `pivot`. Tuned to the current bow.png
# (16x54: tips near the left edge, belly bulges right) and arrow.png
# (26x5: fletching left, arrowhead right).
BOW_GRIP_FRAC = (0.094, 0.5)
# The string is strung symmetrically about the grip, so only one tip is
# needed: the sprite's two ends sit 0.472 and 0.481 from the grip, a
# difference of about a pixel and a half on screen.
BOW_BOTTOM_TIP_FRAC = (0.094, 0.972)
ARROW_NOCK_FRAC = (0.08, 0.5)
ARROW_TIP_FRAC = (0.90, 0.5)
ARROW_CENTER_FRAC = (0.5, 0.5)

STRING_COLOR = (15, 15, 15)
STRING_THICKNESS = 3

MIN_SPEED = 520
MAX_SPEED = 1650
GRAVITY = 1150  # px/s^2

# Nocking forces the hands together, and overlapping hands is exactly when
# MediaPipe drops one of them -- so a hand that vanishes keeps its last
# position instead of cancelling the shot. While the bow is actually drawn
# the hold is much longer: that is precisely when the draw hand is sitting
# on top of the bow hand, and losing the bow mid-pull ruins the shot.
HAND_GRACE_SECONDS = 0.4
HAND_GRACE_DRAWING = 2.5

# Where the bullseye actually sits inside target.png (24x36): the rings are
# left of centre, the brown depth of the butt is on the right. The sprite is
# placed by this point so the single-point collider lines up with the drawn
# bullseye.
TARGET_BULLSEYE_FRAC = (0.375, 0.5)
# Vertical hit zones from the bullseye, as fractions of the target's drawn
# height, read off the artwork's own rings -- the outermost is the painted
# face's edge, so the bands keep matching the picture at any size.
TARGET_RING_FRACS = (0.056, 0.167, 0.278, 0.486)
TARGET_RING_SCORES = (10, 7, 4, 1)
TARGET_RING_COLORS = [(30, 215, 255), (40, 40, 220), (220, 220, 220), (230, 230, 230)]
TARGET_MARGIN = 130


def _palm(hand):
    return hand.landmarks_px[9]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ring_radii(target_size):
    return tuple(f * target_size for f in TARGET_RING_FRACS)


def _score_for_distance(d, target_size):
    for radius, score in zip(_ring_radii(target_size), TARGET_RING_SCORES):
        if d <= radius:
            return score
    return 0


def _draw_target(frame, pos, target_img, target_size, mirror=False):
    """Draws the target with its bullseye exactly on `pos`. When the shooting
    side is flipped the sprite is mirrored so the face keeps pointing at the
    archer."""
    if target_img is not None:
        img = cv2.flip(target_img, 1) if mirror else target_img
        fx, fy = TARGET_BULLSEYE_FRAC
        pivot = (1.0 - fx, fy) if mirror else (fx, fy)
        place_rotated(frame, img, pos, 0.0, target_size, pivot=pivot)
        return
    radii = _ring_radii(target_size)
    for radius, color in zip(reversed(radii + (radii[-1] * 1.3,)),
                              reversed(TARGET_RING_COLORS + [(90, 90, 90)])):
        cv2.circle(frame, pos, int(radius), color, -1, cv2.LINE_AA)
    cv2.circle(frame, pos, 4, (0, 0, 0), -1)


class _StickyHand:
    """Keeps a hand's last known position for HAND_GRACE_SECONDS after
    tracking drops it, so an occluded hand doesn't cancel a drawn shot."""

    def __init__(self):
        self.point = None
        self.closed = False
        self._seen_at = 0.0

    def update(self, hand, now, grace=HAND_GRACE_SECONDS):
        if hand is not None:
            self.point = _palm(hand)
            self.closed = hand.closed
            self._seen_at = now
        elif now - self._seen_at > grace:
            self.point = None
            self.closed = False

    @property
    def present(self):
        return self.point is not None


def run_archery_mode(cap, window_name, tracker):
    """Returns 'menu' or 'quit'."""
    dwell = DwellClickController()

    ret, frame = cap.read()
    if not ret:
        return "quit"
    h, w = frame.shape[:2]

    bow_img = assets.load("archery/bow.png")
    arrow_img = assets.load("archery/arrow.png")
    target_img = assets.load("archery/target.png")

    buttons = [
        Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38)),
        Button("swap", "YÖN", 120, 10, 100, TOOLBAR_H - 20, color=(62, 38, 22)),
        Button("restart", "YENİDEN", 230, 10, 150, TOOLBAR_H - 20, color=(26, 46, 30)),
    ]

    bow_size = int(BOW_FRAC * h)
    arrow_size = int(ARROW_FRAC * h)
    target_size = int(TARGET_FRAC * h)
    target_margin = max(TARGET_MARGIN, target_size // 2 + 40)
    travel_top = TARGET_TRAVEL_TOP * h
    travel_bottom = TARGET_TRAVEL_BOTTOM * h
    foul_gap = int(FOUL_GAP_FRAC * w)

    bow_is_left = True
    holding_arrow = False   # draw hand is closed -> an arrow exists in it
    nocked = False          # that arrow has snapped onto the bowstring
    pending_shot = None     # (bow_pt, draw_pt, pull_dist, angle) while nocked
    nock_flash_until = 0.0
    flying_arrows = []      # each: dict(x,y,vx,vy,angle,landed,score)
    total_score = 0
    arrows_left = ARROWS_PER_ROUND
    round_started = time.time()
    message, message_until = "", 0.0
    bow_sticky, draw_sticky = _StickyHand(), _StickyHand()
    hand_px = DEFAULT_HAND_PX

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
        if hands:
            measured = max(hand_span(hd) for hd in hands)
            if measured > 5:
                hand_px += (measured - hand_px) * HAND_SMOOTHING

        snap_radius = SNAP_PER_HAND * hand_px
        min_pull = MIN_PULL_PER_HAND * hand_px
        max_pull = MAX_PULL_PER_HAND * hand_px

        clicked, progress_map = dwell.update(hands, buttons)
        if clicked == "menu":
            result = "menu"
            break
        elif clicked == "swap":
            bow_is_left = not bow_is_left
        elif clicked == "restart":
            total_score, arrows_left = 0, ARROWS_PER_ROUND
            flying_arrows.clear()
            nocked = holding_arrow = False
            pending_shot = None
            round_started = now
            message, message_until = "YENİ TUR", now + 1.2

        # the target slides up and down; a full cycle is TARGET_PERIOD
        phase = math.sin((now - round_started) / TARGET_PERIOD * 2 * math.pi)
        target_y = int((travel_top + travel_bottom) / 2
                       + (travel_bottom - travel_top) / 2 * phase)
        target_pos = (w - target_margin if bow_is_left else target_margin, target_y)

        # the line sits between archer and target, and flips with them
        foul_x = (target_pos[0] - foul_gap if bow_is_left
                  else target_pos[0] + foul_gap)
        past_line = any(
            (hd.landmarks_px[9][0] > foul_x) if bow_is_left
            else (hd.landmarks_px[9][0] < foul_x)
            for hd in hands)

        if bow_is_left:
            bow_hand = next((hd for hd in hands if hd.label == "Left"), None)
            draw_hand = next((hd for hd in hands if hd.label == "Right"), None)
        else:
            bow_hand = next((hd for hd in hands if hd.label == "Right"), None)
            draw_hand = next((hd for hd in hands if hd.label == "Left"), None)

        grace = HAND_GRACE_DRAWING if nocked else HAND_GRACE_SECONDS
        bow_sticky.update(bow_hand, now, grace)
        draw_sticky.update(draw_hand, now, grace)
        bow_pt, draw_pt = bow_sticky.point, draw_sticky.point
        bow_closed = bow_sticky.present and bow_sticky.closed
        draw_closed = draw_sticky.present and draw_sticky.closed

        # --- aim angle: bow/held-arrow track the other hand whenever both are visible ---
        shoot_angle = 0.0
        if bow_pt is not None and draw_pt is not None:
            shoot_angle = math.atan2(bow_pt[1] - draw_pt[1], bow_pt[0] - draw_pt[0])

        # --- arrow-in-hand / nock / draw / release state machine ---
        was_nocked = nocked
        if past_line:
            # stepping over the line disarms everything, fist or not
            holding_arrow = nocked = False
            pending_shot = None
        elif draw_closed and arrows_left > 0:
            holding_arrow = True
            if not nocked and bow_closed and bow_pt is not None and draw_pt is not None:
                if _dist(draw_pt, bow_pt) < snap_radius:
                    nocked = True
                    nock_flash_until = now + 0.25
            if nocked and not bow_closed:
                nocked = False  # bow let go -- un-nock, arrow stays in hand
        else:
            holding_arrow = False
            nocked = False

        if nocked and bow_pt is not None and draw_pt is not None:
            pull_dist = _dist(bow_pt, draw_pt)
            pending_shot = (bow_pt, draw_pt, pull_dist, shoot_angle)

        if was_nocked and not nocked:
            if pending_shot is not None and bow_closed and draw_sticky.present and not draw_closed:
                p_bow, _p_draw, pull_dist, angle = pending_shot
                if pull_dist >= min_pull and arrows_left > 0:
                    arrows_left -= 1
                    power = min((pull_dist - min_pull) / max(max_pull - min_pull, 1.0), 1.0)
                    speed = MIN_SPEED + power * (MAX_SPEED - MIN_SPEED)
                    flying_arrows.append({
                        "x": float(p_bow[0]), "y": float(p_bow[1]),
                        "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                        "angle": angle, "landed": False, "score": None,
                        "stuck_dy": None,
                    })
            pending_shot = None

        # --- simulate flying arrows (gravity, no straight line) ---
        for a in flying_arrows:
            if a["landed"]:
                continue
            prev_x, prev_y = a["x"], a["y"]
            a["vy"] += GRAVITY * dt
            a["x"] += a["vx"] * dt
            a["y"] += a["vy"] * dt
            a["angle"] = math.atan2(a["vy"], a["vx"])

            crossed = (prev_x < target_pos[0] <= a["x"]) or (prev_x > target_pos[0] >= a["x"])
            if crossed and a["vx"] != 0:
                t = (target_pos[0] - prev_x) / (a["x"] - prev_x)
                cross_y = prev_y + (a["y"] - prev_y) * t
                dist_to_center = abs(cross_y - target_pos[1])
                a["score"] = _score_for_distance(dist_to_center, target_size)
                if a["score"] > 0:
                    a["x"], a["y"] = float(target_pos[0]), cross_y
                    a["landed"] = True
                    # ride with the target from now on: it slides up and
                    # down, and an arrow left at fixed screen coordinates
                    # would slip off the face it is supposed to be stuck in
                    a["stuck_dy"] = cross_y - target_pos[1]
                    total_score += a["score"]
                    message = f"İSABET +{a['score']}"
                else:
                    # A miss is not stuck in anything, so it does not stop:
                    # it flies on past the target and out of the picture.
                    # Pinned where it crossed, it hung in mid-air beside the
                    # target and rode up and down with it.
                    message = "ISKA"
                message_until = now + 1.6
            elif a["y"] > h + 40:
                a["landed"] = True
                a["score"] = 0
        for a in flying_arrows:
            if a.get("stuck_dy") is not None:
                a["x"], a["y"] = float(target_pos[0]), target_pos[1] + a["stuck_dy"]
        # gone once it has left the picture, over the side or off the bottom
        flying_arrows = [a for a in flying_arrows
                         if -60 <= a["x"] <= w + 60 and a["y"] <= h + 40]

        round_over = arrows_left == 0 and not any(not a["landed"] for a in flying_arrows)

        # ================= draw =================
        line_color = (60, 60, 220) if past_line else (90, 90, 90)
        for y0 in range(TOOLBAR_H, h, 30):
            cv2.line(frame, (foul_x, y0), (foul_x, min(y0 + 16, h)), line_color, 3)

        _draw_target(frame, target_pos, target_img, target_size, mirror=not bow_is_left)

        for a in flying_arrows:
            pivot = ARROW_TIP_FRAC if a["landed"] else ARROW_CENTER_FRAC
            if arrow_img is not None:
                place_rotated(frame, arrow_img, (a["x"], a["y"]), a["angle"], arrow_size, pivot=pivot)
            else:
                tip = (int(a["x"] + 15 * math.cos(a["angle"])), int(a["y"] + 15 * math.sin(a["angle"])))
                cv2.line(frame, (int(a["x"]), int(a["y"])), tip, (0, 200, 255), 3)

        if bow_pt is not None and bow_closed and bow_img is not None:
            bh = bow_img.shape[0]
            zoom = integer_scale_for(bow_img, bow_size)
            half_span = (BOW_BOTTOM_TIP_FRAC[1] - BOW_GRIP_FRAC[1]) * bh * zoom
            perp = (-math.sin(shoot_angle), math.cos(shoot_angle))
            tip_a = (int(bow_pt[0] + perp[0] * half_span), int(bow_pt[1] + perp[1] * half_span))
            tip_b = (int(bow_pt[0] - perp[0] * half_span), int(bow_pt[1] - perp[1] * half_span))

            place_rotated(frame, bow_img, bow_pt, shoot_angle, bow_size, pivot=BOW_GRIP_FRAC)

            if nocked and draw_pt is not None:
                cv2.line(frame, tip_a, draw_pt, STRING_COLOR, STRING_THICKNESS, cv2.LINE_AA)
                cv2.line(frame, draw_pt, tip_b, STRING_COLOR, STRING_THICKNESS, cv2.LINE_AA)
            else:
                cv2.line(frame, tip_a, tip_b, STRING_COLOR, STRING_THICKNESS, cv2.LINE_AA)

            if now < nock_flash_until:
                cv2.circle(frame, bow_pt, int(snap_radius * 0.5), (0, 255, 120), 2, cv2.LINE_AA)

        # the held arrow: shown in the draw hand whether or not it's nocked yet
        if holding_arrow and draw_pt is not None and arrow_img is not None:
            angle = shoot_angle if bow_pt is not None else 0.0
            place_rotated(frame, arrow_img, draw_pt, angle, arrow_size, pivot=ARROW_NOCK_FRAC)

        for sticky, is_bow in ((bow_sticky, True), (draw_sticky, False)):
            if not sticky.present:
                continue
            pt = sticky.point
            color = (0, 255, 0) if sticky.closed else (0, 165, 255)
            cv2.circle(frame, pt, 8, color, -1)
            draw_text(frame, "YAY" if is_bow else "OK", (pt[0] + 14, pt[1] - 26), scale=2)

        for btn in buttons:
            hovered = dwell.hovered_id() == btn.id
            progress = progress_map.get(btn.id, 0.0)
            btn.draw(frame, progress=progress, hovered=hovered)

        # centred: the target sits at one edge or the other, never here
        draw_panel(frame, f"SKOR: {total_score}", (w // 2, 26),
                                  scale=3, anchor="center")
        draw_panel(frame, f"OK: {arrows_left}", (w - 30, 26), scale=2,
                                  anchor="topright" if arrows_left else (80, 80, 255))
        if now < message_until:
            draw_panel(frame, message, (w // 2, 90), scale=3, anchor="center",
                                      plate=(0, 130, 0) if "İSABET" in message else (60, 60, 60))

        if past_line:
            draw_panel(frame, "ÇİZGİYİ GEÇTİN - GERİ ÇEKİL",
                                      (w // 2, h // 2), scale=2, anchor="center",
                                      plate=(0, 0, 130))

        if round_over:
            draw_panel(frame, f"BİTTİ - SKOR: {total_score}",
                                      (w // 2, h // 2 - 30), scale=3, anchor="center",
                                      plate=(0, 60, 130))
            draw_panel(frame, "YENİDEN'E BAS", (w // 2, h // 2 + 30),
                                      scale=2, anchor="center")

        draw_rig(frame, hands)


        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        handle_key(window_name, key)
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
