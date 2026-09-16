# PyInstaller build recipe -- produces dist/VisionPlay.app
#
#   source venv/bin/activate && pyinstaller VisionPlay.spec --noconfirm
#
# NSCameraUsageDescription below is not optional: without it macOS denies the
# camera to the bundled app and the window opens to a black frame.
from PyInstaller.utils.hooks import collect_all

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
