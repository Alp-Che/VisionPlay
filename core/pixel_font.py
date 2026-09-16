"""Pixel-font rendering for the two hand-drawn fonts in assets/font/.

INFO  -- one PNG per glyph (assets/font/<name>.png). Used for hint/status
         text. The files are tightly cropped and carry no baseline, so the
         metrics below supply it: cap height 9, x-height 5, accents 3 rows
         above the cap line, cedillas 3 rows below the baseline.

BUTTON -- a single 8x14 sheet (assets/font/genel.png), a taller condensed
         face used on buttons. Its grid and per-row baselines are measured
         from the image itself at load time, so nothing is hand-tuned; it is
         halved on load (its strokes are 4px wide) to a 19px cap height.

Both fonts rasterise into a stencil mask, upscale by whole numbers with
nearest-neighbour so edges stay hard, then tint to any colour. Rendered
strings are cached, so redrawing static labels every frame costs nothing.
"""
import os
from collections import Counter

import cv2
import numpy as np

from core.paths import resource

FONT_DIR = resource("assets", "font")
SHEET_PATH = os.path.join(FONT_DIR, "genel.png")

# ---------------------------------------------------------------- INFO font

_INFO_BASELINE = 12
_INFO_CAP_TOP = 3

_INFO_NAMED = {
    ":": "colon", ",": "comma", ".": "period", "-": "dash", "/": "slash",
    "(": "paren_open", ")": "paren_close", "+": "plus", "*": "times",
    "$": "dollar",
}

# Top row of each glyph inside the line box; anything absent falls back to the
# rule for its category in _info_top().
_INFO_TOPS = {
    "Ö_upper": 0, "Ü_upper": 0, "İ_upper": 0, "Ğ_upper": 0,
    "ö_lower": 4, "ü_lower": 5, "ğ_lower": 5, "i_lower": 4, "j_lower": 4,
    "g_lower": 8, "p_lower": 7, "q_lower": 7, "y_lower": 8,
    "ç_lower": 8, "ş_lower": 7, "b_lower": 4,
    "period": 11, "comma": 11, "colon": 6, "dash": 9, "slash": 4,
    "paren_open": _INFO_CAP_TOP, "paren_close": _INFO_CAP_TOP,
    "plus": 8, "times": 8, "dollar": 2,
}


def _info_top(name, height):
    if name in _INFO_TOPS:
        return _INFO_TOPS[name]
    if name.endswith("_upper") or name[0].isdigit():
        return _INFO_CAP_TOP
    if name.endswith("_lower"):
        return _INFO_CAP_TOP if height >= 9 else _INFO_BASELINE - height
    return _INFO_BASELINE - height


def _load_info_glyphs():
    glyphs = {}
    if not os.path.isdir(FONT_DIR):
        return glyphs
    for filename in sorted(os.listdir(FONT_DIR)):
        if not filename.endswith(".png") or filename == "genel.png":
            continue
        name = filename[:-4]
        img = cv2.imread(os.path.join(FONT_DIR, filename), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            glyph = img
        else:
            bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            opaque = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8) * 255
            glyph = np.dstack([bgr, opaque])

        # U_upper.png currently holds the Ü artwork (umlaut + U); crop the
        # accent off until it is redrawn. A correct 9-row file skips this.
        if name == "U_upper" and glyph.shape[0] == 12:
            glyph = glyph[3:, :]

        if name.endswith(("_upper", "_lower")):
            ch = name[0]
        elif name.isdigit():
            ch = name
        else:
            ch = next((c for c, n in _INFO_NAMED.items() if n == name), None)
            if ch is None:
                continue
        glyphs[ch] = (glyph, _info_top(name, glyph.shape[0]) - _INFO_BASELINE)
    return glyphs


# -------------------------------------------------------------- BUTTON font

_SHEET_LAYOUT = [
    "ABCÇDEFG",
    "ĞHIİJKLM",
    "NOÖPQRSŞ",
    "TUÜVWXYZ",
    "abcçdefg",
    "ğhıijklm",
    "noöpqrsş",
    "tuüvwxyz",
    "12345678",
    "90/:;()₺",
    "&@\".,?!'",
    "[]{}#%^×",
    "-+=_\\~<>",
    "€$£.",
]


# How far below its own content a sheet row may reach to pick up descenders
# and cedillas, in sheet pixels.
DESCENDER_ALLOWANCE = 14


def _runs(flags, gap):
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    out = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - out[-1][1] <= gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def _load_sheet_glyphs():
    """Carves genel.png into glyphs. Columns come from the transparent
    gutters; rows from the letter columns, which hold one clean glyph each.

    A row's band is its own content plus a small allowance for descenders --
    NOT the midpoint to the next row. The letter columns (A/B/C, Ğ/H/I, ...)
    never descend, so a midpoint band would reach far enough up to swallow
    the cedilla of the row above: Ç's tail landed inside İ, Ş's inside Z.
    """
    img = cv2.imread(SHEET_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {}
    alpha = img[:, :, 3]
    solid = alpha > 10

    col_ranges = [(a, b + 1) for a, b in _runs(solid.sum(axis=0) > 0, 8)]
    if len(col_ranges) < 8:
        return {}

    letters = solid[:, col_ranges[0][0]:col_ranges[2][1]]
    row_runs = _runs(letters.sum(axis=1) > 0, 12)

    glyphs = {}
    for r, (top, bottom) in enumerate(row_runs):
        if r >= len(_SHEET_LAYOUT):
            break
        lo = top
        if r == len(row_runs) - 1:
            hi = solid.shape[0]
        else:
            hi = bottom + 1 + min(DESCENDER_ALLOWANCE, max(0, row_runs[r + 1][0] - bottom - 1))

        cells = []
        for c, (cx0, cx1) in enumerate(col_ranges):
            if c >= len(_SHEET_LAYOUT[r]):
                break
            cell = solid[lo:hi, cx0:cx1]
            ys, xs = np.flatnonzero(cell.sum(axis=1) > 0), np.flatnonzero(cell.sum(axis=0) > 0)
            if len(ys) == 0:
                continue
            glyph = img[lo + ys[0]:lo + ys[-1] + 1, cx0 + xs[0]:cx0 + xs[-1] + 1]
            cells.append((_SHEET_LAYOUT[r][c], glyph, lo + ys[0], lo + ys[-1]))

        if not cells:
            continue
        baseline = Counter(c[3] for c in cells).most_common(1)[0][0] + 1
        for ch, glyph, top, _ in cells:
            # halve it: the art is drawn with 4px strokes, so this lands on a
            # 19px cap height with clean 2px strokes and no detail lost
            half = cv2.resize(glyph, (max(1, glyph.shape[1] // 2), max(1, glyph.shape[0] // 2)),
                              interpolation=cv2.INTER_NEAREST)
            glyphs[ch] = (half, (top - baseline) // 2)
    return glyphs


_CIRCUMFLEX = {"Â": "A", "â": "a", "Î": "İ", "î": "i", "Û": "U", "û": "u"}


# ------------------------------------------------------------------ engine

class PixelFont:
    def __init__(self, loader, tracking, space_width, preserve_color=False):
        self._loader = loader
        self.tracking = tracking
        self.space_width = space_width
        # preserve_color: draw the artwork's own pixels instead of using it as
        # a stencil. genel.png is shaded in three greens, so tinting it flat
        # would throw that away.
        self.preserve_color = preserve_color
        self._glyphs = None
        self._top = 0
        self._height = 1
        self._cache = {}

    @property
    def glyphs(self):
        if self._glyphs is None:
            self._glyphs = self._loader()
            if self._glyphs:
                self._top = min(off for _, off in self._glyphs.values())
                bottom = max(off + m.shape[0] for m, off in self._glyphs.values())
                self._height = bottom - self._top
        return self._glyphs

    def _lookup(self, ch):
        g = self.glyphs
        entry = g.get(ch) or g.get(ch.upper()) or g.get(ch.lower())
        if entry is None:
            # neither font draws the circumflex vowels (hâlâ, kâğıt), so fall
            # back to the bare letter rather than dropping a blank
            plain = _CIRCUMFLEX.get(ch)
            if plain:
                entry = g.get(plain) or g.get(plain.lower())
        return entry

    def _compose(self, text):
        """Lays the glyphs out on one baseline into a BGRA bitmap."""
        pieces, width = [], 0
        for ch in text:
            entry = None if ch == " " else self._lookup(ch)
            pieces.append(entry)
            width += (entry[0].shape[1] if entry else self.space_width) + self.tracking

        canvas = np.zeros((self._height, max(width - self.tracking, 1), 4), np.uint8)
        x = 0
        for entry in pieces:
            if entry is None:
                x += self.space_width + self.tracking
                continue
            glyph, off = entry
            gh, gw = glyph.shape[:2]
            y = off - self._top
            region = canvas[y:y + gh, x:x + gw]
            visible = glyph[:, :, 3] > 0
            region[visible] = glyph[visible]
            x += gw + self.tracking
        return canvas

    def render(self, text, scale, color):
        key = (text, scale, None if self.preserve_color else color)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        bgra = self._compose(text)
        if scale != 1:
            bgra = cv2.resize(bgra, (bgra.shape[1] * scale, bgra.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        if not self.preserve_color:
            bgra = bgra.copy()
            bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2] = color[0], color[1], color[2]

        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = bgra
        return bgra

    def measure(self, text, scale=1):
        self.glyphs  # ensure metrics are loaded
        bitmap = self._compose(text) if text else np.zeros((self._height, 1, 4), np.uint8)
        return bitmap.shape[1] * scale, bitmap.shape[0] * scale

    def draw(self, frame, text, org, scale=1, color=(255, 255, 255), anchor="topleft"):
        if not text:
            return
        bgra = self.render(text, scale, tuple(int(c) for c in color))
        th, tw = bgra.shape[:2]
        x, y = int(org[0]), int(org[1])
        if anchor == "center":
            x -= tw // 2
            y -= th // 2
        elif anchor == "topright":
            x -= tw
        elif anchor == "bottomleft":
            y -= th

        H, W = frame.shape[:2]
        x0, x1 = max(x, 0), min(x + tw, W)
        y0, y1 = max(y, 0), min(y + th, H)
        if x0 >= x1 or y0 >= y1:
            return
        crop = bgra[y0 - y:y1 - y, x0 - x:x1 - x]
        alpha = (crop[:, :, 3].astype(np.float32) / 255.0)[..., None]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (roi * (1 - alpha) + crop[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)


INFO = PixelFont(_load_info_glyphs, tracking=1, space_width=3)
BUTTON = PixelFont(_load_sheet_glyphs, tracking=2, space_width=5, preserve_color=True)


def draw_text(frame, text, org, scale=2, color=(255, 255, 255), anchor="topleft", font=INFO):
    font.draw(frame, text, org, scale=scale, color=color, anchor=anchor)


def draw_text_with_background(frame, text, org, scale=2, color=(255, 255, 255),
                               bg_color=(0, 0, 0), padding=6, anchor="topleft", font=INFO):
    tw, th = font.measure(text, scale)
    x, y = int(org[0]), int(org[1])
    if anchor == "center":
        x -= tw // 2
        y -= th // 2
    elif anchor == "topright":
        x -= tw
    elif anchor == "bottomleft":
        y -= th
    cv2.rectangle(frame, (x - padding, y - padding), (x + tw + padding, y + th + padding),
                  bg_color, -1)
    font.draw(frame, text, (x, y), scale=scale, color=color)
