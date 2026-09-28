"""Does the game run on Windows? Run by the Windows build, without a camera.

Loads the hand model, asks the screen for its shape, then drives every screen
for a few hundred frames on a fake camera with fake hands moving about and
making fists -- enough to go through scoring, popups and end screens, which
is where errors hide. Any exception fails the build before a broken .exe is
handed to anyone.
"""
import os
import random
import sys
import tempfile
import traceback

# records are written somewhere disposable, not into the machine's own folder
os.environ["APPDATA"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

W, H = 1280, 720
FRAMES = 400

# nothing is shown: there is no one to show it to
for name in ("namedWindow", "destroyAllWindows", "destroyWindow", "imshow",
             "setWindowProperty", "setMouseCallback"):
    setattr(cv2, name, lambda *a, **k: None)
cv2.getWindowProperty = lambda *a, **k: 0.0
_frames = [0]


def _wait_key(delay=0):
    _frames[0] += 1
    return ord("q") if _frames[0] >= FRAMES else -1


cv2.waitKey = _wait_key


class FakeHand:
    def __init__(self, x, y, span, closed):
        cx, cy = int(x), int(y)
        self.landmarks_px = [(cx, cy)] * 21
        self.landmarks_px[0] = (cx, cy + span)
        self.landmarks_px[4] = (cx - span // 2, cy)
        self.landmarks_px[8] = (cx, cy - span // 2)
        self.landmarks_px[12] = (cx + span // 3, cy - span // 2)
        self.index_tip = self.landmarks_px[8]
        self.closed = closed
        self.label = "Right"


class FakeTracker:
    """Two hands wandering the picture, now and then closing into a fist."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.pos = [[400.0, 400.0], [880.0, 400.0]]
        self.vel = [[0.0, 0.0], [0.0, 0.0]]

    def process(self, frame):
        hands = []
        for i in range(2):
            if self.rng.random() < 0.10:
                self.vel[i] = [self.rng.uniform(-60, 60), self.rng.uniform(-60, 60)]
            self.pos[i][0] = min(max(self.pos[i][0] + self.vel[i][0], 10), W - 10)
            self.pos[i][1] = min(max(self.pos[i][1] + self.vel[i][1], 10), H - 10)
            if self.rng.random() < 0.92:
                hands.append(FakeHand(*self.pos[i], span=48,
                                      closed=self.rng.random() < 0.55))
        return hands


class NoHands:
    def process(self, frame):
        return []


class FakeCamera:
    camera_index = 0

    def next_camera(self):
        return 0

    def isOpened(self):
        return True

    def read(self):
        return True, np.full((H, W, 3), 60, np.uint8)

    def release(self):
        pass


def main():
    failed = []

    def check(name, fn):
        try:
            detail = fn()
            print(f"  ok    {name}" + (f"  ({detail})" if detail else ""))
        except Exception:
            failed.append(name)
            print(f"  HATA  {name}")
            traceback.print_exc()

    def tracker():
        from core.hand_tracker import HandTracker
        t = HandTracker(num_hands=2)
        found = t.process(np.zeros((H, W, 3), np.uint8))
        t.close()
        return f"bos karede {len(found)} el"

    def aspect():
        from core.window import screen_aspect
        return f"ekran orani {screen_aspect()}"

    def records():
        from core import records as rec
        rec.submit("deneme", 7)
        rec._cache = None                 # read back from the file itself
        assert rec.best("deneme") == 7, rec.best("deneme")
        return rec.data_folder()

    check("el modeli", tracker)
    check("ekran", aspect)
    check("rekor dosyasi", records)

    import core.rig as rig
    from modes.start import run_start
    from modes.menu import run_menu
    from modes.draw_mode import run_draw_mode
    from modes.archery_mode import run_archery_mode
    from modes.ninja_mode import run_ninja_mode
    from modes.juggle_mode import run_juggle_mode
    from modes.mole_mode import run_mole_mode
    from modes.pong_mode import run_pong_mode
    from modes.dance_mode import run_dance_mode
    import main as app

    screens = [("baslangic", run_start), ("menu", run_menu), ("cizim", run_draw_mode),
               ("okculuk", run_archery_mode), ("ninja", run_ninja_mode),
               ("sektirme", run_juggle_mode), ("kostebek", run_mole_mode),
               ("pong", run_pong_mode), ("dans", run_dance_mode)]
    for rig_on, make in ((False, lambda: FakeTracker(4)), (True, lambda: FakeTracker(9)),
                         (True, NoHands)):
        rig.set_rig(rig_on)
        for name, run in screens:
            def drive(run=run, make=make):
                _frames[0] = 0
                result = run(FakeCamera(), "VisionPlay", make())
                return f"{_frames[0]} kare -> {result}"
            check(f"{name} (test {'acik' if rig_on else 'kapali'}, "
                  f"{'el yok' if make is NoHands else 'eller'})", drive)

    check("kamera yok ekrani", lambda: app._camera_missing())

    print()
    if failed:
        print(f"{len(failed)} HATA: " + ", ".join(failed))
        sys.exit(1)
    print("Windows'ta her sey calisti.")


if __name__ == "__main__":
    main()
