"""Buttons and the two ways of pressing one.

By hand: hold a fingertip over a button and close your fist for a second.
That is how the game is meant to be played, from across the room.

By mouse: a plain left click, for whoever is setting the thing up and is
standing at the keyboard anyway.
"""
import time

import cv2

from core import assets
from core.pixel_font import BUTTON


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
    def __init__(self, id, label, x, y, w, h, color=(70, 70, 70), icon=None, text_scale=2):
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

        if icon_img is None:
            cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)

        border_color = (255, 255, 255) if hovered else (30, 30, 30)
        border_thickness = 3 if selected else 2
        if selected:
            border_color = (0, 255, 120)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), border_color, border_thickness)

        if icon_img is not None:
            pad = 6
            assets.overlay(frame, icon_img, self.x + pad, self.y + pad, self.w - 2 * pad, self.h - 2 * pad)
        elif self.label:
            scale = self.text_scale
            while scale > 1 and BUTTON.measure(self.label, scale)[0] > self.w - 14:
                scale -= 1
            BUTTON.draw(frame, self.label, self.center(), scale=scale, anchor="center")

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
        for hand in hands:
            for btn in buttons:
                if btn.contains(hand.index_tip):
                    hover_id = btn.id
                    hover_closed = hover_closed or hand.closed
                    break
            if hover_id is not None:
                break

        if hover_id is None:
            self._target_id = None
            self._start_time = None
            self._latched = False
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
