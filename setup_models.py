"""Downloads the MediaPipe Tasks model files VisionPlay needs into models/."""
import os
import shutil
import ssl
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def _https():
    """Where to find the certificates that vouch for the download server.

    Python from python.org on macOS does not use the system's own, so an
    https download fails there until "Install Certificates.command" has been
    run. certifi -- which MediaPipe installs anyway -- carries the same list.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[ok] {filename} zaten var, atlaniyor.")
            continue
        print(f"[indiriliyor] {filename} ...")
        with urllib.request.urlopen(url, context=_https()) as response, \
                open(dest + ".part", "wb") as out:
            shutil.copyfileobj(response, out)
        # only a finished download takes the real name, so a broken one is
        # not mistaken for the model next time
        os.replace(dest + ".part", dest)
        print(f"[tamam] {filename} indirildi ({os.path.getsize(dest)} bytes)")


if __name__ == "__main__":
    main()
