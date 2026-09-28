"""Downloads the MediaPipe Tasks model files VisionPlay needs into models/."""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    # the bodies, to tell whose hands are whose when several people are in shot
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[ok] {filename} zaten var, atlaniyor.")
            continue
        print(f"[indiriliyor] {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[tamam] {filename} indirildi ({os.path.getsize(dest)} bytes)")


if __name__ == "__main__":
    main()
