"""VisionPlay - kamera ve el hareketleriyle oynanan mini oyunlar.

Akis: BASLA -> mod secimi -> OKCULUK / CIZIM / FRUIT NINJA / TOP SEKTIRME / PONG / KOSTEBEK / DELIKTEN GEC.

Butonlar: elinizi butonun uzerine getirip yumruk yaparak 1 saniye bekleyin.
"""
import sys

import cv2

from core.camera import CameraStream, ScreenView
from core.hand_tracker import HandTracker
from core.ui import attach_mouse
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
from modes.wall_mode import run_wall_mode

WINDOW_NAME = "VisionPlay"


def main():
    # the picture is cut to the screen's shape once, here, and every game
    # lays itself out in that; the tracker still sees the whole of it
    cap = ScreenView(CameraStream(0, width=1280, height=720), screen_aspect())
    if not cap.isOpened():
        print("Kamera acilamadi.")
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
            elif state == "wall":
                state = run_wall_mode(cap, WINDOW_NAME, tracker)
            else:
                state = "quit"
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
