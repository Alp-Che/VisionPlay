"""VisionPlay - kamera ve el hareketleriyle oynanan mini oyunlar.

Akis: BASLA -> mod secimi -> OKCULUK / CIZIM / FRUIT NINJA / TOP SEKTIRME / YAKALA / KOSTEBEK.

Butonlar: elinizi butonun uzerine getirip yumruk yaparak 1 saniye bekleyin.
"""
import sys

import cv2

from core.camera import CameraStream
from modes.start import run_start
from modes.menu import run_menu
from modes.draw_mode import run_draw_mode
from modes.archery_mode import run_archery_mode
from modes.ninja_mode import run_ninja_mode
from modes.juggle_mode import run_juggle_mode
from modes.catch_mode import run_catch_mode
from modes.mole_mode import run_mole_mode

WINDOW_NAME = "VisionPlay"


def main():
    cap = CameraStream(0, width=1280, height=720)
    if not cap.isOpened():
        print("Kamera acilamadi.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME)

    state = "start"
    try:
        while state != "quit":
            if state == "start":
                state = run_start(cap, WINDOW_NAME)
            elif state == "menu":
                state = run_menu(cap, WINDOW_NAME)
            elif state == "draw":
                state = run_draw_mode(cap, WINDOW_NAME)
            elif state == "archery":
                state = run_archery_mode(cap, WINDOW_NAME)
            elif state == "ninja":
                state = run_ninja_mode(cap, WINDOW_NAME)
            elif state == "juggle":
                state = run_juggle_mode(cap, WINDOW_NAME)
            elif state == "catch":
                state = run_catch_mode(cap, WINDOW_NAME)
            elif state == "mole":
                state = run_mole_mode(cap, WINDOW_NAME)
            else:
                state = "quit"
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
