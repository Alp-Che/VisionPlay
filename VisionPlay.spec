# PyInstaller build recipe.
#
#   macOS:   source venv/bin/activate && pyinstaller VisionPlay.spec --noconfirm
#            -> dist/VisionPlay.app
#   Windows: built by .github/workflows/windows.yml on every push
#            -> dist/VisionPlay/VisionPlay.exe (the whole folder is needed)
#
# PyInstaller cannot build for one system on another, which is why the
# Windows build runs on GitHub's Windows machines rather than here.
#
# NSCameraUsageDescription below is not optional: without it macOS denies the
# camera to the bundled app and the window opens to a black frame. BUNDLE
# does nothing on Windows.
import sys

from PyInstaller.utils.hooks import collect_all

# PyInstaller packs whatever the Python running it can import, and only warns
# about the rest -- so run from outside the venv it turns out an .exe that
# dies on its first line with "No module named 'mediapipe'". Stop here instead.
try:
    import mediapipe  # noqa: F401
except ImportError:
    raise SystemExit(
        "\nHATA: bu Python'da mediapipe kurulu degil, yani sanal ortam etkin degil.\n"
        "Once:  venv\\Scripts\\Activate.ps1   (macOS: source venv/bin/activate)\n"
        "Sonra: python -m pip install -r requirements.txt pyinstaller\n"
        "       python -m PyInstaller VisionPlay.spec --noconfirm --clean\n")

mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=mp_binaries,
    datas=[("assets", "assets"), ("models", "models")] + mp_datas,
    hiddenimports=mp_hidden,
    hookspath=[],
    runtime_hooks=[],
    # matplotlib must stay: mediapipe.tasks.python.vision imports its
    # drawing_utils at load time, which pulls in pyplot.
    excludes=["pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VisionPlay",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/icon.ico" if sys.platform == "win32" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VisionPlay",
)

app = BUNDLE(
    coll,
    name="VisionPlay.app",
    icon="assets/icon.icns",
    bundle_identifier="com.alpche.visionplay",
    info_plist={
        "NSCameraUsageDescription":
            "VisionPlay el hareketlerinizi takip etmek icin kamerayi kullanir.",
        "CFBundleName": "VisionPlay",
        "CFBundleDisplayName": "VisionPlay",
        "CFBundleShortVersionString": "1.0",
        "NSHighResolutionCapable": True,
    },
)
