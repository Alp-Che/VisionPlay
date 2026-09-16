"""Locates bundled files whether running from source or from VisionPlay.app.

PyInstaller unpacks data next to the executable and points sys._MEIPASS at
it, so paths must not be resolved against the working directory -- a
double-clicked .app starts with the working directory set to /.
"""
import os
import sys


def base_dir():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource(*parts):
    return os.path.join(base_dir(), *parts)
