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

# The letters whose tail is meant to fall below the line. Every other glyph is
# put on the line exactly: the sheet is drawn a pixel out here and there -- A
# and B reach one row lower than C and D, for instance -- and left alone that
# shows up as text that will not sit straight.
DESCENDERS = set("gjpqyçşğÇŞ,;()")


def _is_face_tone(img, ink):
    """True where the sheet is the lettering rather than its shadow.

    The face is simply whichever tone there is most of; the shadow is drawn in
    a darker one and always covers less ground.
    """
    tones, counts = np.unique(img[ink][:, :3], axis=0, return_counts=True)
    if not len(tones):
        return ink
    face = tones[counts.argmax()]
    return (img[:, :, :3] == face).all(axis=2)


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

    Rows and glyphs are found from the gaps between them. The line a row sits
    on is wherever most of its letters end: in a row of eight only a couple
    descend, so the majority is the baseline. Each glyph then remembers how
    far above or below that line it starts, which is what keeps a 'g' hanging
    and an 'İ' dotted once the text is laid out.

    The letters are drawn with a shadow a pixel under them, in a darker tone.
    The baseline is measured from the lettering alone: counting the shadow as
    part of the letter puts every glyph that has one a pixel lower than every
    glyph that does not, which is exactly the wobble it looks like.
    """
    img = _imread_unicode(SHEET_PATH)
    if img is None:
        return {}
    ink = img[:, :, 3] > _INK
    lettering = ink & _is_face_tone(img, ink)

    glyphs = {}
    for row, (top, bottom) in enumerate(_runs(ink.sum(axis=1) > 0)):
        if row >= len(_SHEET_LAYOUT):
            break
        band, faces = ink[top:bottom + 1], lettering[top:bottom + 1]
        cells = []
        for column, (x0, x1) in enumerate(_runs(band.sum(axis=0) > 0)):
            if column >= len(_SHEET_LAYOUT[row]):
                break
            ys = np.flatnonzero(band[:, x0:x1 + 1].sum(axis=1) > 0)
            face_ys = np.flatnonzero(faces[:, x0:x1 + 1].sum(axis=1) > 0)
            sits_on = int(face_ys[-1]) if len(face_ys) else int(ys[-1])
            cells.append((_SHEET_LAYOUT[row][column],
                          img[top + ys[0]:top + ys[-1] + 1, x0:x1 + 1],
                          top + int(ys[0]), top + sits_on))
        if not cells:
            continue
        bottoms = [c[3] for c in cells]
        baseline = max(set(bottoms), key=bottoms.count) + 1
        for ch, glyph, glyph_top, sits_on in cells:
            if ch in DESCENDERS:
                glyphs[ch] = (glyph, glyph_top - baseline)
            else:
                glyphs[ch] = (glyph, glyph_top - sits_on - 1)
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

    def render(self, text, scale):
        key = (text, scale)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        bgra = self._compose(text)
        if scale != 1:
            bgra = cv2.resize(bgra, (bgra.shape[1] * scale, bgra.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = bgra
        return bgra

    def ink_size(self, text, scale=1):
        """Width, and the height of the inked rows alone.

        The line box keeps room above every line for the tallest accent and
        below it for the deepest tail, which most strings never use. Fitting a
        label to the box instead of to its ink leaves it needlessly small.
        """
        bgra = self.render(text, scale)
        rows = np.flatnonzero(bgra[:, :, 3].max(axis=1) > 0)
        height = int(rows[-1] - rows[0]) + 1 if len(rows) else 0
        return bgra.shape[1], height

    def measure(self, text, scale=1):
        self.glyphs  # ensure metrics are loaded
        bitmap = self._compose(text) if text else np.zeros((self._height, 1, 4), np.uint8)
        return bitmap.shape[1] * scale, bitmap.shape[0] * scale

    def draw(self, frame, text, org, scale=1, anchor="topleft"):
        if not text:
            return
        bgra = self.render(text, scale)
        th, tw = bgra.shape[:2]
        x, y = int(org[0]), int(org[1])
        if anchor == "center":
            # Centred on the ink, not on the line box. The box keeps room
            # above every line for the tallest accent and below it for the
            # deepest tail; centring that pushes ordinary text downwards until
            # a cedilla drops out of the bottom of a button and reads as a
            # mark of its own.
            rows = np.flatnonzero(bgra[:, :, 3].max(axis=1) > 0)
            top, bottom = (int(rows[0]), int(rows[-1])) if len(rows) else (0, th - 1)
            x -= tw // 2
            y -= (top + bottom) // 2
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


def draw_text(frame, text, org, scale=2, anchor="topleft", font=FONT):
    font.draw(frame, text, org, scale=scale, anchor=anchor)
