"""Choices that should outlast a session -- which camera to use, for one.

Kept in the same folder as the records, and with the same rule: nothing here
may stop the game. A file that cannot be read means the defaults, and one that
cannot be written means the choice lasts until the app is closed.
"""
import json
import os

from core.records import data_folder

_FILE_NAME = "ayarlar.json"
_cache = None


def _load():
    global _cache
    if _cache is None:
        try:
            with open(os.path.join(data_folder(), _FILE_NAME), encoding="utf-8") as f:
                data = json.load(f)
            _cache = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _cache = {}
    return _cache


def get(name, default=None):
    return _load().get(name, default)


def put(name, value):
    _load()[name] = value
    folder = data_folder()
    path = os.path.join(folder, _FILE_NAME)
    try:
        os.makedirs(folder, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=1)
        os.replace(path + ".tmp", path)
    except OSError:
        pass
