"""Camera reader that runs on its own thread.

cv2.VideoCapture.read() blocks until the driver hands over a frame, and the
driver keeps a small backlog -- so a game loop that is slower than the camera
ends up working through stale frames, which looks like lag even when the
frame rate is fine. Draining the camera on a background thread and always
handing the game the newest frame keeps the picture current.

Exposes the few VideoCapture methods the game uses, so modes keep calling
cap.read() exactly as before.
"""
import sys
import threading
import time

import cv2


def _open_capture(index):
    """Windows' default Media Foundation backend can take several seconds to
    open a webcam and often ignores the requested resolution; DirectShow does
    neither. On every other platform the default backend is the right one."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


# how many camera numbers are tried when looking for the next one: a laptop's
# own camera, a phone, a capture card and a virtual camera or two
MAX_CAMERAS = 5


class CameraStream:
    def __init__(self, index=0, width=1280, height=720, fps=60):
        self.index = index
        self.size = (width, height, fps)
        self._cap = _open_capture(index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            # keep the driver's backlog as short as it will allow
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._seq = 0
        self._delivered = -1
        self._lock = threading.Lock()
        self._running = self._cap.isOpened()
        self._thread = None
        if self._running:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()

    def _pump(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def isOpened(self):
        return self._cap.isOpened()

    def read(self, timeout=0.5):
        """Blocks until a frame the caller hasn't seen yet is ready. Never
        returns the same frame twice, so no work is spent re-processing one."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._frame is not None and self._seq != self._delivered:
                    self._delivered = self._seq
                    return True, self._frame
                stale = self._frame
            if time.monotonic() >= deadline:
                return stale is not None, stale
            time.sleep(0.001)

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()


class ScreenView:
    """The camera as the game shows it: mirrored, and cut to the screen's
    shape.

    A 16:9 camera on a screen of any other shape leaves bars once full screen.
    Cutting the picture to the screen's own shape here, before any game sees
    it, means every game lays itself out in exactly the area that will be on
    screen -- nothing is placed where the cut would lose it.

    What is cut off is not lost to the tracker, though. The tracker is handed
    the whole mirrored picture (`full`) and told where the shown part starts
    (`x0`), so a hand that strays just past the edge of the screen is still
    followed: the camera sees wider than the game shows.

    Mirroring happens here and nowhere else, so that the picture the tracker
    reads and the picture the game draws on can never disagree about it.
    """

    def __init__(self, cap, aspect=None):
        self.cap = cap
        self.aspect = aspect      # width over height to show; None = as is
        self.full = None          # the whole mirrored picture, last read
        self.x0 = 0               # where the shown part begins in `full`

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()

    @property
    def camera_index(self):
        return self.cap.index

    def next_camera(self):
        """Moves on to the next camera that works, wrapping round.

        Returns the number now in use, which is unchanged if no other camera
        answers. The new one is opened and made to deliver a frame before the
        old one is let go, so trying a camera that is not there costs nothing
        but the wait.
        """
        current = self.cap.index
        width, height, fps = self.cap.size
        for step in range(1, MAX_CAMERAS):
            index = (current + step) % MAX_CAMERAS
            candidate = CameraStream(index, width, height, fps)
            # a camera can take a moment to deliver its first frame
            if candidate.isOpened() and candidate.read(timeout=3.0)[0]:
                self.cap.release()
                self.cap = candidate
                return index
            candidate.release()
        return current

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return ret, frame
        frame = cv2.flip(frame, 1)
        self.full = frame
        h, w = frame.shape[:2]
        keep = w if not self.aspect else min(w, int(round(h * self.aspect)))
        self.x0 = (w - keep) // 2
        if keep == w:
            return ret, frame.copy()
        return ret, frame[:, self.x0:self.x0 + keep].copy()
