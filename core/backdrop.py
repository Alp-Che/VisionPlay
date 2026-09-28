"""How bright the picture is behind something drawn on it.

The games draw over a camera picture of a room, not over a chosen background,
so anything in one fixed colour vanishes against a wall that happens to match
it. What has to stay visible -- Pong's ball, the bowstring -- goes light or
dark by what is actually behind it.

The reading is eased over time, so someone walking past cannot make it
flicker, and light and dark swap only once it is clearly past the middle, so
a wall of just that brightness cannot make it flicker either.
"""
MIDDLE = 128.0
HYSTERESIS = 12.0


class Backdrop:
    def __init__(self, smoothing=0.06):
        self.smoothing = smoothing
        self.level = MIDDLE
        self.dark = True

    def measure(self, frame, box=None, step=8):
        """Eases towards the brightness of `frame`, or of the part of it in
        `box` = (x0, y0, x1, y1). Returns whether the backdrop is dark."""
        if box is not None:
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = (int(v) for v in box)
            x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
            y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
            frame = frame[y0:y1, x0:x1]
        # every eighth pixel is plenty to tell dark from bright
        patch = frame[::step, ::step]
        if patch.size:
            measured = float(patch[..., 0].mean() * 0.114 + patch[..., 1].mean() * 0.587
                             + patch[..., 2].mean() * 0.299)
            self.level += (measured - self.level) * self.smoothing
        if self.level < MIDDLE - HYSTERESIS:
            self.dark = True
        elif self.level > MIDDLE + HYSTERESIS:
            self.dark = False
        return self.dark
