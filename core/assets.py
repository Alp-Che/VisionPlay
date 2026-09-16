"""Loads optional PNG art from assets/ and alpha-composites it onto frames.
Every lookup is safe: a missing file returns None so callers keep using
their current code-drawn fallback until real art is dropped in.

Pass paths relative to assets/, e.g. load("ui/icon_menu.png") or
load("archery/bow.png")."""
import os

import cv2
import numpy as np

from core.paths import resource

ASSETS_DIR = resource("assets")

_cache = {}


def load(name):
    """Returns a BGRA numpy array for assets/<name>, or None if missing."""
    if name in _cache:
        return _cache[name]

    path = os.path.join(ASSETS_DIR, name)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None

    if img is not None and img.ndim == 3 and img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])

    _cache[name] = img
    return img


def _resize(img, w, h):
    """Nearest-neighbour when enlarging so pixel art keeps hard edges; area
    averaging when shrinking, which is what that direction wants."""
    growing = w >= img.shape[1] or h >= img.shape[0]
    interp = cv2.INTER_NEAREST if growing else cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interp)


def overlay(frame, icon_bgra, x, y, w, h):
    """Alpha-composites icon_bgra (resized to w x h) onto frame with its
    top-left corner at (x, y). Clips safely against frame bounds."""
    if icon_bgra is None or w <= 0 or h <= 0:
        return

    resized = _resize(icon_bgra, w, h)
    fg = resized[:, :, :3].astype(np.float32)
    alpha = (resized[:, :, 3].astype(np.float32) / 255.0)[..., None]

    y0, y1 = max(y, 0), min(y + h, frame.shape[0])
    x0, x1 = max(x, 0), min(x + w, frame.shape[1])
    if y0 >= y1 or x0 >= x1:
        return

    roi = frame[y0:y1, x0:x1].astype(np.float32)
    fg_crop = fg[y0 - y:y1 - y, x0 - x:x1 - x]
    a_crop = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    frame[y0:y1, x0:x1] = (roi * (1 - a_crop) + fg_crop * a_crop).astype(np.uint8)


def overlay_centered(frame, icon_bgra, cx, cy, w, h):
    overlay(frame, icon_bgra, cx - w // 2, cy - h // 2, w, h)


def overlay_fill(frame, icon_bgra, x, y, w, h, opacity=1.0):
    """Like overlay(), but also lets you dial down a fully-opaque background
    image (e.g. a toolbar/panel backdrop) instead of it needing its own
    alpha channel."""
    if icon_bgra is None or w <= 0 or h <= 0:
        return
    resized = _resize(icon_bgra, w, h)
    fg = resized[:, :, :3].astype(np.float32)
    alpha = (resized[:, :, 3].astype(np.float32) / 255.0 * opacity)[..., None]

    y0, y1 = max(y, 0), min(y + h, frame.shape[0])
    x0, x1 = max(x, 0), min(x + w, frame.shape[1])
    if y0 >= y1 or x0 >= x1:
        return

    roi = frame[y0:y1, x0:x1].astype(np.float32)
    fg_crop = fg[y0 - y:y1 - y, x0 - x:x1 - x]
    a_crop = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    frame[y0:y1, x0:x1] = (roi * (1 - a_crop) + fg_crop * a_crop).astype(np.uint8)
