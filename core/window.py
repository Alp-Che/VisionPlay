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
QUIT_KEY = ord('q')


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


class _Cocoa:
    """Just enough of the Objective-C runtime to reach the window OpenCV made.

    OpenCV opens its macOS window without a working close button: the style it
    asks for leaves "closable" out, so the red button sits greyed out. There
    is no OpenCV setting for it, so the button is switched on here, directly.
    """
    CLOSABLE = 1 << 1          # NSWindowStyleMaskClosable

    def __init__(self):
        self.objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self._calls = {}
        self.app = self.send(self.objc.objc_getClass(b"NSApplication"), "sharedApplication")

    def send(self, receiver, selector, restype=ctypes.c_void_p, argtypes=(), *args):
        # objc_msgSend has to be called through a prototype of the exact
        # method being called; on Apple silicon a loosely typed call passes
        # its arguments in the wrong registers.
        key = (restype, tuple(argtypes))
        if key not in self._calls:
            proto = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
            self._calls[key] = proto(("objc_msgSend", self.objc))
        return self._calls[key](receiver, self.objc.sel_registerName(selector.encode()), *args)

    def find_window(self, title):
        windows = self.send(self.app, "windows")
        for i in range(self.send(windows, "count", ctypes.c_ulong)):
            window = self.send(windows, "objectAtIndex:", ctypes.c_void_p, (ctypes.c_ulong,), i)
            name = self.send(self.send(window, "title"), "UTF8String", ctypes.c_char_p)
            if name is not None and name.decode("utf-8", "replace") == title:
                return window
        return None


_mac = {}                      # window name -> (cocoa, NSWindow)


def _mac_make_closable(name):
    try:
        cocoa = _Cocoa()
        window = cocoa.find_window(name)
        if window is None:
            return
        mask = cocoa.send(window, "styleMask", ctypes.c_ulong)
        cocoa.send(window, "setStyleMask:", None, (ctypes.c_ulong,), mask | _Cocoa.CLOSABLE)
        # Closed, it is only hidden, not freed: OpenCV still holds on to it,
        # and would crash drawing to a window that no longer existed.
        cocoa.send(window, "setReleasedWhenClosed:", None, (ctypes.c_bool,), False)
        _mac[name] = (cocoa, window)
    except Exception:
        pass                   # the button stays as OpenCV left it


def create_window(name):
    """Opens the game window, able to resize and to go full screen."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if sys.platform == "darwin":
        _mac_make_closable(name)


def window_closed(name):
    """Whether the player closed the window with its close button.

    Left to itself the game would never notice: the next frame drawn opens a
    new window of the same name, so closing it just made it blink.
    """
    if sys.platform == "darwin":
        if name not in _mac:
            return False
        cocoa, window = _mac[name]
        try:
            # out of sight is not the same as closed: in the Dock, or the app
            # hidden with Cmd-H, it is only put away
            if cocoa.send(window, "isMiniaturized", ctypes.c_bool) or \
                    cocoa.send(cocoa.app, "isHidden", ctypes.c_bool):
                return False
            return not cocoa.send(window, "isVisible", ctypes.c_bool)
        except Exception:
            return False
    try:
        # Windows forgets a closed window, and then reports it as not visible
        # (0); where the question is not supported the answer is -1, which is
        # no reason to quit
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 0
    except cv2.error:
        return False


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


def wait_key(name, delay=1):
    """cv2.waitKey for the game window, with the window's own business done:
    the full-screen key, and a closed window turned into the quit key."""
    key = cv2.waitKey(delay) & 0xFF
    handle_key(name, key)
    if window_closed(name):
        return QUIT_KEY
    return key


def handle_key(name, key):
    """Deals with the keys that belong to the window rather than to a game.

    Returns True when the key was its business, so a screen can ignore it.
    """
    if key == FULLSCREEN_KEY:
        set_fullscreen(name, not is_fullscreen(name))
        return True
    return False
