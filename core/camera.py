"""Camera reader that runs on its own thread.

cv2.VideoCapture.read() blocks until the driver hands over a frame, and the
driver keeps a small backlog -- so a game loop that is slower than the camera
ends up working through stale frames, which looks like lag even when the
frame rate is fine. Draining the camera on a background thread and always
handing the game the newest frame keeps the picture current.

Exposes the few VideoCapture methods the game uses, so modes keep calling
cap.read() exactly as before.
"""
import threading
import time

import cv2


class CameraStream:
    def __init__(self, index=0, width=1280, height=720, fps=60):
        self._cap = cv2.VideoCapture(index)
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
