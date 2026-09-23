"""Buttons and the two ways of pressing one.

By hand: hold a fingertip over a button and close your fist for a second.
That is how the game is meant to be played, from across the room.

By mouse: a plain left click, for whoever is setting the thing up and is
standing at the keyboard anyway.

Also here: how long a timed round lasts and the countdown bar every such
screen draws, so a round is the same length anywhere and looks the same too.
"""
import math
import time

import cv2
import numpy as np

from core import assets
from core.hand_tracker import hand_span
from core.pixel_font import FONT


# assets/ui/timebar.png is one drawn bar per step of the countdown, laid out
# left to right: the first is full and green, the last is empty, and the
# colour walks through yellow to red on the way. Using the drawn states
# rather than squashing one sprite keeps the pixel art exactly as drawn and
# gets the colour change for free.
TIME_BAR_STEPS = 22
# How long a timed round lasts, anywhere in the game.
ROUND_SECONDS = 20.0


def draw_time_bar(frame, remaining, x, y, width, height):
    """Draw the countdown bar with its top-left corner at (x, y).

    `remaining` runs from 1.0 at the start of a round down to 0.0 when the
    time is up.
    """
    sheet = assets.load("ui/timebar.png")
    if sheet is None:
        return
    step_w = sheet.shape[1] // TIME_BAR_STEPS
    step = int(round((1.0 - min(max(remaining, 0.0), 1.0)) * (TIME_BAR_STEPS - 1)))
    assets.overlay(frame, sheet[:, step * step_w:(step + 1) * step_w],
                   x, y, width, height)


def draw_round_timer(frame, remaining):
    """The countdown bar where every timed screen puts it: across the top,
    just under the score. `remaining` is 1.0 at the start of a round and 0.0
    when it runs out.

    Sized in whole multiples of the drawing, so its pixels stay square.
    """
    sheet = assets.load("ui/timebar.png")
    if sheet is None:
        return
    w = frame.shape[1]
    step_w = sheet.shape[1] // TIME_BAR_STEPS
    zoom = max(1, round(w * 0.34 / step_w))
    bar_w, bar_h = step_w * zoom, sheet.shape[0] * zoom
    draw_time_bar(frame, remaining, (w - bar_w) // 2, 50, bar_w, bar_h)



# how dark the strip under a button goes, against its own colour
SHADOW_TONE = 0.45

_ART_CACHE = {}


def _button_art(width, height, color):
    """The button drawn from its three pieces: the cap on the left, the same
    cap mirrored on the right, and the middle repeated between them.

    Repeating the middle rather than stretching it keeps whatever is drawn in
    it intact at any width, and the pieces are enlarged by whole numbers so
    the pixels stay square. The middle's fill is then recoloured to the
    button's own colour -- that is what lets the menu tell its six games
    apart -- while the drawn edges are left exactly as drawn.
    """
    key = (width, height, color)
    cached = _ART_CACHE.get(key)
    if cached is not None:
        return cached

    cap = assets.load("ui/button_cap.png")
    middle = assets.load("ui/button_middle.png")
    if cap is None or middle is None or width <= 0 or height <= 0:
        return None, None

    zoom = max(1, round(height / cap.shape[0]))
    strip_h = cap.shape[0] * zoom
    cap_w = cap.shape[1] * zoom
    # the cap is drawn as the right-hand end -- it narrows towards its own
    # right edge -- so the left end is the same piece mirrored
    right = cv2.resize(cap, (cap_w, strip_h), interpolation=cv2.INTER_NEAREST)
    left = right[:, ::-1]
    body = cv2.resize(middle, (middle.shape[1] * zoom, strip_h),
                      interpolation=cv2.INTER_NEAREST)

    span = max(1, width - 2 * cap_w)
    repeats = span // body.shape[1] + 1
    stretch = np.hstack([body] * repeats)[:, :span]
    art = np.hstack([left, stretch, right])

    # the fill is whatever the middle is mostly made of
    opaque = body[body[:, :, 3] > 0][:, :3]
    if len(opaque):
        tones, counts = np.unique(opaque, axis=0, return_counts=True)
        fill = tones[counts.argmax()]
        art = art.copy()
        is_fill = (art[:, :, :3] == fill).all(axis=2) & (art[:, :, 3] > 0)

        # The drawing carries a second band of fill below its bottom edge.
        # Painted the same as the face it reads as a stray stripe; darkened it
        # reads as the button sitting above its own shadow. It is found rather
        # than measured: it is the fill that lies below the last drawn edge.
        edge = (~is_fill) & (art[:, :, 3] > 0)
        rows = np.arange(art.shape[0])[:, None]
        lowest_edge = np.where(edge.any(axis=0), (rows * edge).max(axis=0), art.shape[0])
        shadow = is_fill & (rows > lowest_edge[None, :])
        face = is_fill & ~shadow

        art[face, 0], art[face, 1], art[face, 2] = color[0], color[1], color[2]
        art[shadow, 0] = int(color[0] * SHADOW_TONE)
        art[shadow, 1] = int(color[1] * SHADOW_TONE)
        art[shadow, 2] = int(color[2] * SHADOW_TONE)
        # where the button's face stops and its shadow begins. A label centred
        # on the whole strip sits low, because the shadow is not part of the
        # face the eye reads.
        # The box the label lives in: the part actually painted the button's
        # colour. Not the whole strip -- that takes in the drawn frame and the
        # shadow under it, and a label fitted to those sits low and small.
        rows = np.flatnonzero(face.any(axis=1))
        cols = np.flatnonzero(face.any(axis=0))
        face_box = ((int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
                    if len(rows) and len(cols) else (0, 0, art.shape[1] - 1, art.shape[0] - 1))
    else:
        face_box = (0, 0, art.shape[1] - 1, art.shape[0] - 1)

    if len(_ART_CACHE) > 64:
        _ART_CACHE.clear()
    _ART_CACHE[key] = (art, face_box)
    return art, face_box


# The only room left around a label inside the coloured face: just enough
# that it does not touch the drawn edge. The label is otherwise as large
# as that area will take.
LABEL_PAD_X = 8
LABEL_PAD_Y = 2

# the plate a score or a notice sits on when it has no colour of its own
PANEL_COLOR = (46, 41, 35)


def draw_panel(frame, text, org, scale=2, plate=PANEL_COLOR, anchor="topleft"):
    """Text on the same drawn plate the buttons use, sized around the text.

    Scores and notices sat on plain black rectangles before. Putting them on
    the button's own frame means the whole screen is made of one thing, and
    the plate is built at a whole multiple of the drawing so its pixels stay
    square whatever the text measures.

    `org` and `anchor` place the *text*, exactly as they did before; the plate
    is then drawn around wherever the text landed.
    """
    text_w, text_h = FONT.measure(text, scale)
    x, y = int(org[0]), int(org[1])
    if anchor == "center":
        x -= text_w // 2
        y -= text_h // 2
    elif anchor == "topright":
        x -= text_w
    elif anchor == "bottomleft":
        y -= text_h

    cap = assets.load("ui/button_cap.png")
    unit_h = cap.shape[0] if cap is not None else 23
    zoom = max(1, -(-(text_h + 12) // unit_h))
    height = unit_h * zoom
    pad = (cap.shape[1] if cap is not None else 6) * zoom + 6
    art, face_box = _button_art(text_w + 2 * pad, height, plate)
    if art is None:
        FONT.draw(frame, text, (x, y), scale=scale)
        return
    fx0, fy0, fx1, fy1 = face_box
    art_y = y + text_h // 2 - (fy0 + fy1) // 2
    assets.overlay(frame, art, x - pad, art_y, art.shape[1], art.shape[0])
    FONT.draw(frame, text, (x + text_w // 2, art_y + (fy0 + fy1) // 2),
              scale=scale, anchor="center")


class _MouseClicks:
    """Left clicks on the game window, held until a screen asks for them.

    The window belongs to the whole app rather than to any one screen, so the
    callback is attached once, in main, and each screen in turn reads from
    here. Clicks that land on nothing are simply dropped.
    """

    def __init__(self):
        self._pending = []

    def attach(self, window_name):
        try:
            cv2.setMouseCallback(window_name, self._record)
        except cv2.error:
            pass         # no window (headless test): nothing to listen to

    def _record(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._pending.append((x, y))

    def take(self):
        clicks, self._pending = self._pending, []
        return clicks


MOUSE = _MouseClicks()


def attach_mouse(window_name):
    """Start listening for clicks on `window_name`. Call once, after the
    window is created."""
    MOUSE.attach(window_name)


class Button:
    def __init__(self, id, label, x, y, w, h, color=(70, 70, 70), icon=None, text_scale=8):
        self.id = id
        self.label = label
        self.x, self.y, self.w, self.h = x, y, w, h
        self.color = color
        self.text_scale = text_scale
        # icon: asset filename under assets/ui/. Defaults to "icon_<id>.png"
        # so buttons pick up art automatically once it's dropped in; pass
        # icon="" to force the plain color/label look for this button.
        self.icon = f"ui/icon_{id}.png" if icon is None else icon

    def contains(self, point):
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def draw(self, frame, progress=0.0, hovered=False, selected=False):
        icon_img = assets.load(self.icon) if self.icon else None

        art, face_box = _button_art(self.w, self.h, self.color)
        label_at = (self.x + self.w // 2, self.y + self.h // 2)
        room = (self.w - LABEL_PAD_X, self.h - LABEL_PAD_Y)
        if icon_img is None:
            if art is None:
                cv2.rectangle(frame, (self.x, self.y),
                              (self.x + self.w, self.y + self.h), self.color, -1)
            else:
                art_y = self.y + (self.h - art.shape[0]) // 2
                assets.overlay(frame, art, self.x, art_y, art.shape[1], art.shape[0])
                fx0, fy0, fx1, fy1 = face_box
                label_at = (self.x + (fx0 + fx1) // 2, art_y + (fy0 + fy1) // 2)
                room = (fx1 - fx0 + 1 - LABEL_PAD_X, fy1 - fy0 + 1 - LABEL_PAD_Y)

        # The artwork draws its own edge, so a border is only added as
        # feedback -- green for the one in use, white while a hand is over it.
        if selected:
            cv2.rectangle(frame, (self.x, self.y),
                          (self.x + self.w, self.y + self.h), (0, 255, 120), 3)
        elif hovered:
            cv2.rectangle(frame, (self.x, self.y),
                          (self.x + self.w, self.y + self.h), (255, 255, 255), 2)

        if icon_img is not None:
            pad = 6
            assets.overlay(frame, icon_img, self.x + pad, self.y + pad, self.w - 2 * pad, self.h - 2 * pad)
        elif self.label:
            # As large as the face will take, rather than a size guessed per
            # button: the labels are different lengths, and a fixed size
            # leaves the short ones looking shrunken next to the long ones.
            # text_scale is the ceiling, not the starting point.
            scale = 1
            while scale < self.text_scale:
                width, height = FONT.ink_size(self.label, scale + 1)
                if width > room[0] or height > room[1]:
                    break
                scale += 1
            FONT.draw(frame, self.label, label_at, scale=scale, anchor="center")

        if hovered and progress > 0:
            self._draw_progress_border(frame, progress)

    def _draw_progress_border(self, frame, progress):
        """Traces the dwell progress around the button's own edge, clockwise
        from the top-left, so it never crosses the label."""
        x, y, w, h = self.x, self.y, self.w, self.h
        edges = [((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                 ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))]
        remaining = 2 * (w + h) * min(progress, 1.0)
        for (x0, y0), (x1, y1) in edges:
            if remaining <= 0:
                break
            length = abs(x1 - x0) + abs(y1 - y0)
            t = min(remaining / length, 1.0)
            end = (int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t))
            cv2.line(frame, (x0, y0), end, (0, 255, 120), 5)
            remaining -= length


# A hand that is being swung is playing, not pointing. While it is moving
# faster than this the dwell will not start, so a swat that happens to pass
# over a button with a closed fist cannot arm a press. Measured in the hand's
# own widths per second, so it means the same at any distance from the camera.
STEADY_SPANS_PER_SECOND = 6.0

# How long a fist must be held to count as a click. Long enough that a hand
# passing over a button doesn't press it, short enough not to be a chore --
# every screen takes it from here, so this is the only line to change.
DWELL_SECONDS = 1.0


class DwellClickController:
    """Fires a click when a hand's index fingertip hovers a button AND the
    hand stays closed (fist) continuously for `dwell_seconds`."""

    def __init__(self, dwell_seconds=DWELL_SECONDS):
        self.dwell_seconds = dwell_seconds
        self._target_id = None
        self._start_time = None
        self._latched = False
        self._was_at = None
        self._was_when = None
        # a click made before this screen existed is not meant for it
        MOUSE.take()

    def _settled(self, hand):
        """Is this hand being held still enough to mean it?"""
        now = time.time()
        at = hand.index_tip
        was_at, was_when = self._was_at, self._was_when
        self._was_at, self._was_when = at, now
        if was_at is None or was_when is None or now <= was_when:
            return False              # nothing to compare against yet
        moved = math.dist(at, was_at) / (now - was_when)
        return moved <= STEADY_SPANS_PER_SECOND * max(hand_span(hand), 1.0)

    def update(self, hands, buttons):
        progress_map = {}

        # A mouse click is an outright press -- no hovering, no waiting. It
        # also clears any dwell in progress, so a click cannot be followed by
        # a stale fist finishing a second press on its own.
        for point in MOUSE.take():
            for btn in buttons:
                if btn.contains(point):
                    self._target_id = None
                    self._start_time = None
                    self._latched = False
                    return btn.id, progress_map

        hover_id = None
        hover_closed = False
        hover_hand = None
        for hand in hands:
            for btn in buttons:
                if btn.contains(hand.index_tip):
                    hover_id = btn.id
                    hover_closed = hover_closed or hand.closed
                    hover_hand = hand
                    break
            if hover_id is not None:
                break

        if hover_id is None:
            self._target_id = None
            self._start_time = None
            self._latched = False
            self._was_at = None
            return None, progress_map

        steady = self._settled(hover_hand)
        if not steady:
            self._start_time = None
            self._latched = False
            progress_map[hover_id] = 0.0
            return None, progress_map

        if hover_id != self._target_id:
            self._target_id = hover_id
            self._start_time = None
            self._latched = False

        if not hover_closed:
            self._start_time = None
            self._latched = False
            progress_map[hover_id] = 0.0
            return None, progress_map

        if self._latched:
            progress_map[hover_id] = 1.0
            return None, progress_map

        if self._start_time is None:
            self._start_time = time.time()

        progress = min((time.time() - self._start_time) / self.dwell_seconds, 1.0)
        progress_map[hover_id] = progress

        if progress >= 1.0:
            self._latched = True
            return hover_id, progress_map

        return None, progress_map

    def hovered_id(self):
        return self._target_id
