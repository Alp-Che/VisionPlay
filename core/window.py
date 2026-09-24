"""The one window the game is shown in, and what can be done to it.

It is created able to resize, and that is the whole point: a window fixed to
the size of its picture cannot go full screen at all -- the system's own full
screen button does nothing on one. Keeping the ratio is asked for at the same
time, so a 16:9 camera picture is fitted inside a differently shaped screen
with bars rather than stretched across it.
"""
import cv2

FULLSCREEN_KEY = ord('f')


def create_window(name):
    """Opens the game window, able to resize and to go full screen."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)


def is_fullscreen(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
    except cv2.error:
        return False


def set_fullscreen(name, on):
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
    except cv2.error:
        pass             # no window (headless test): nothing to set


def handle_key(name, key):
    """Deals with the keys that belong to the window rather than to a game.

    Returns True when the key was its business, so a screen can ignore it.
    """
    if key == FULLSCREEN_KEY:
        set_fullscreen(name, not is_fullscreen(name))
        return True
    return False
