"""The one window the game is shown in, and what can be done to it.

It is created able to resize, and that is the whole point: a window fixed to
the size of its picture cannot go full screen at all -- the system's own full
screen button does nothing on one. Keeping the ratio is asked for at the same
time, so a 16:9 camera picture is fitted inside a differently shaped screen
with bars rather than stretched across it.
"""
import ctypes
import sys

import cv2

FULLSCREEN_KEY = ord('f')


def screen_aspect():
    """Width over height of the main display, or None if it cannot be told.

    Asked of the operating system directly rather than of the window: OpenCV
    keeps reporting a full-screen window at the picture's own size, so it
    cannot be used to find out what shape the screen is.
    """
    try:
        if sys.platform == "darwin":
            cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
            cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
            display = cg.CGMainDisplayID()
            w, h = cg.CGDisplayPixelsWide(display), cg.CGDisplayPixelsHigh(display)
        elif sys.platform == "win32":
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        else:
            return None
        return w / h if w and h else None
    except Exception:
        return None


def create_window(name):
    """Opens the game window, able to resize and to go full screen."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)


def is_fullscreen(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
    except cv2.error:
        return False


def set_fullscreen(name, on):
    # Full screen fills the whole screen, stretching if it must. The picture
    # has already been cut to the screen's shape, so there is nothing to
    # stretch unless the system holds back a strip of the screen for itself;
    # a window keeps its proportions, since there it can be dragged any shape.
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO,
                              cv2.WINDOW_FREERATIO if on else cv2.WINDOW_KEEPRATIO)
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
