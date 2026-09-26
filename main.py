"""VisionPlay - kamera ve el hareketleriyle oynanan mini oyunlar.

Akis: BASLA -> mod secimi -> OKCULUK / CIZIM / FRUIT NINJA / TOP SEKTIRME / PONG / KOSTEBEK / DANS.

Butonlar: elinizi butonun uzerine getirip yumruk yaparak 1 saniye bekleyin.
"""
import sys

import cv2
import numpy as np

from core.camera import CameraStream, ScreenView
from core.hand_tracker import HandTracker
from core.ui import attach_mouse, draw_panel
from core.window import create_window, screen_aspect
from modes.start import run_start
from modes.menu import run_menu
from modes.draw_mode import run_draw_mode
from modes.archery_mode import run_archery_mode
from modes.ninja_mode import run_ninja_mode
from modes.juggle_mode import run_juggle_mode
from modes.catch_mode import run_catch_mode
from modes.mole_mode import run_mole_mode
from modes.pong_mode import run_pong_mode
from modes.dance_mode import run_dance_mode

WINDOW_NAME = "VisionPlay"


def _camera_missing():
    """Says so in a window. The packaged app has no console to print to, and
    one that simply vanished would look like it had crashed -- when the usual
    cause is only the system's camera permission being off."""
    frame = np.full((360, 760, 3), 30, np.uint8)
    draw_panel(frame, "KAMERA AÇILAMADI", (380, 140), scale=3, anchor="center",
               plate=(30, 30, 120))
    draw_panel(frame, "KAMERA İZNİNİ KONTROL ET", (380, 230), scale=2, anchor="center")
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(10000)          # any key closes it sooner


def main():
    # the picture is cut to the screen's shape once, here, and every game
    # lays itself out in that; the tracker still sees the whole of it
    cap = ScreenView(CameraStream(0, width=1280, height=720), screen_aspect())
    if not cap.isOpened():
        print("Kamera acilamadi.")
        _camera_missing()
        sys.exit(1)

    # One hand tracker for the whole session, handed to each screen in turn.
    # Building one is cheap but tearing one down is not -- around a quarter of
    # a second -- so a screen that made its own would freeze the picture every
    # time the player left it.
    tracker = HandTracker(num_hands=2, view=cap)

    create_window(WINDOW_NAME)
    # one frame, only to learn the picture's size: a click has to be put back
    # into these coordinates once the window is resized or made full screen
    ret, first = cap.read()
    size = (first.shape[1], first.shape[0]) if ret else None
    attach_mouse(WINDOW_NAME, size)

    state = "start"
    try:
        while state != "quit":
            if state == "start":
                state = run_start(cap, WINDOW_NAME, tracker)
            elif state == "menu":
                state = run_menu(cap, WINDOW_NAME, tracker)
            elif state == "draw":
                state = run_draw_mode(cap, WINDOW_NAME, tracker)
            elif state == "archery":
                state = run_archery_mode(cap, WINDOW_NAME, tracker)
            elif state == "ninja":
                state = run_ninja_mode(cap, WINDOW_NAME, tracker)
            elif state == "juggle":
                state = run_juggle_mode(cap, WINDOW_NAME, tracker)
            elif state == "catch":
                state = run_catch_mode(cap, WINDOW_NAME, tracker)
            elif state == "mole":
                state = run_mole_mode(cap, WINDOW_NAME, tracker)
            elif state == "pong":
                state = run_pong_mode(cap, WINDOW_NAME, tracker)
            elif state == "dance":
                state = run_dance_mode(cap, WINDOW_NAME, tracker)
            else:
                state = "quit"
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
