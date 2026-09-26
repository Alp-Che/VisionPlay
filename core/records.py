"""Best scores, kept between runs.

They live in the user's own data folder rather than next to the code: a
packaged VisionPlay.app is read-only, and a record that vanished every time
the app was closed would not be much of a record.

Nothing here is allowed to stop a game. A file that cannot be read starts the
records afresh, and one that cannot be written just means this record is kept
for the rest of the session only.
"""
import json
import os
import sys

_FILE_NAME = "rekorlar.json"
_cache = None


def data_folder():
    """Where VisionPlay keeps what it remembers between runs."""
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/VisionPlay")
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"),
                            "VisionPlay")
    return os.path.join(os.environ.get("XDG_DATA_HOME")
                        or os.path.expanduser("~/.local/share"), "VisionPlay")


def _load():
    global _cache
    if _cache is None:
        try:
            with open(os.path.join(data_folder(), _FILE_NAME), encoding="utf-8") as f:
                data = json.load(f)
            _cache = {k: int(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            _cache = {}
    return _cache


def _save():
    folder = data_folder()
    path = os.path.join(folder, _FILE_NAME)
    try:
        os.makedirs(folder, exist_ok=True)
        # written aside and swapped in, so a crash mid-write cannot leave a
        # half-written file that loses every record at once
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=1)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def best(game):
    return _load().get(game, 0)


def submit(game, score):
    """Records the score if it beats the best. Returns whether it did."""
    records = _load()
    if score <= records.get(game, 0):
        return False
    records[game] = int(score)
    _save()
    return True


class RoundRecord:
    """A game's record, checked once when a round ends.

    finish() can be called on every frame of the end screen; only the first
    call counts, so the round's own score never beats itself.
    """

    def __init__(self, game):
        self.game = game
        self.reset()

    def reset(self):
        self.done = False
        self.is_new = False

    def finish(self, score):
        if not self.done:
            self.is_new = submit(self.game, score)
            self.done = True

    @property
    def best(self):
        return best(self.game)
