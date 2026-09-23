"""Pixel-font rendering for the hand-drawn face in assets/font/fonts.png.

One sheet, one font, used for everything: labels, scores, notices. The rows
below say what is drawn where; everything else -- where each glyph starts and
ends, and which line it sits on -- is measured from the image at load time, so
redrawing the sheet needs no code change.

Glyphs rasterise into a stencil mask, upscale by whole numbers with
nearest-neighbour so edges stay hard, then tint to any colour. Rendered
strings are cached, so redrawing static labels every frame costs nothing.
"""
import cv2
import numpy as np

from core.paths import resource


def _imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread that works with non-ASCII paths on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


SHEET_PATH = resource("assets", "font", "fonts.png")

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

# The sheet carries a few stray pixels at an alpha of 3 or so -- invisible on
# screen, but enough to be read as a glyph of their own and throw a whole row
# out of step. Anything this faint is not ink.
_INK = 16


def _runs(flags):
    """Start and end of each unbroken run of True."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    out = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def _load_glyphs():
    """Carves fonts.png into glyphs, one row of the layout at a time.

    Rows and glyphs are found from the gaps between them. The line each row
    sits on is taken to be wherever most of its glyphs end: in any row of
    eight only a couple descend, so the majority is the baseline. Each glyph
    then remembers how far it sits above or below that line, which is what
    keeps a 'g' hanging and an 'İ' dotted when the text is laid out.
    """
    img = _imread_unicode(SHEET_PATH)
    if img is None:
        return {}
    ink = img[:, :, 3] > _INK

    glyphs = {}
    for row, (top, bottom) in enumerate(_runs(ink.sum(axis=1) > 0)):
        if row >= len(_SHEET_LAYOUT):
            break
        band = ink[top:bottom + 1]
        cells = []
        for column, (x0, x1) in enumerate(_runs(band.sum(axis=0) > 0)):
            if column >= len(_SHEET_LAYOUT[row]):
                break
            ys = np.flatnonzero(band[:, x0:x1 + 1].sum(axis=1) > 0)
            cells.append((_SHEET_LAYOUT[row][column],
                          img[top + ys[0]:top + ys[-1] + 1, x0:x1 + 1],
                          top + ys[0], top + ys[-1]))
        if not cells:
            continue
        bottoms = [c[3] for c in cells]
        baseline = max(set(bottoms), key=bottoms.count) + 1
        for ch, glyph, glyph_top, _ in cells:
            glyphs[ch] = (glyph, glyph_top - baseline)
    return glyphs


_CIRCUMFLEX = {"Â": "A", "â": "a", "Î": "İ", "î": "i", "Û": "U", "û": "u"}


# ------------------------------------------------------------------ engine

class PixelFont:
    def __init__(self, loader, tracking, space_width):
        self._loader = loader
        self.tracking = tracking
        self.space_width = space_width
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
        key = (text, scale, color)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        bgra = self._compose(text)
        if scale != 1:
            bgra = cv2.resize(bgra, (bgra.shape[1] * scale, bgra.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
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


FONT = PixelFont(_load_glyphs, tracking=1, space_width=3)


def draw_text(frame, text, org, scale=2, color=(255, 255, 255), anchor="topleft", font=FONT):
    font.draw(frame, text, org, scale=scale, color=color, anchor=anchor)


def draw_text_with_background(frame, text, org, scale=2, color=(255, 255, 255),
                               bg_color=(0, 0, 0), padding=6, anchor="topleft", font=FONT):
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
