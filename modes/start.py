"""Title screen: one BAŞLA button, opened with the usual fist dwell."""
import cv2

from core import assets, settings
from core.pixel_font import draw_text
from core.transform import integer_scale_for
from core.rig import rig_on, set_rig
from core.window import handle_key
from core.ui import Button, DwellClickController


def run_start(cap, window_name, tracker):
    """Returns 'menu', 'quit', or 'start' to be laid out again."""
    dwell = DwellClickController()

    ret, frame = cap.read()
    h, w = frame.shape[:2] if ret else (720, 1280)

    start_btn = Button("menu", "BAŞLA", w // 2 - 170, h // 2 - 40, 340, 130,
                       color=(30, 52, 30))
    quit_btn = Button("quit", "ÇIKIŞ", w - 140, 20, 120, 50, color=(38, 38, 38))
    # Leaves the hand rig drawn over every game, for checking what the camera
    # is actually making of a player. It is a session-wide switch, so it is
    # set here and read by the games themselves.
    test_btn = Button("test", "TEST", 20, 20, 120, 50, color=(38, 38, 38))
    # Steps through the cameras attached -- a phone used as a webcam shows up
    # as one more. The number is on the button so it is clear which is on.
    camera_btn = Button("camera", f"KAMERA {cap.camera_index + 1}", 150, 20, 170, 50,
                        color=(38, 38, 38))
    buttons = [start_btn, quit_btn, test_btn, camera_btn]

    result = None
    while result is None:
        ret, frame = cap.read()
        if not ret:
            break

        hands = tracker.process(frame)
        clicked, progress_map = dwell.update(hands, buttons)

        logo = assets.load("ui/logo.png")
        if logo is not None:
            zoom = integer_scale_for(logo, 200)
            lw, lh = logo.shape[1] * zoom, logo.shape[0] * zoom
            assets.overlay(frame, logo, (w - lw) // 2, h // 2 - 60 - lh, lw, lh)
        else:
            draw_text(frame, "VISIONPLAY", (w // 2, h // 2 - 140), scale=5, anchor="center")

        for btn in buttons:
            btn.draw(frame, progress=progress_map.get(btn.id, 0.0),
                     hovered=dwell.hovered_id() == btn.id,
                     selected=btn.id == "test" and rig_on())

        for hand in hands:
            cv2.circle(frame, hand.index_tip, 10,
                       (0, 255, 0) if hand.closed else (0, 200, 255), -1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        handle_key(window_name, key)
        if key == ord('q'):
            result = "quit"
        elif clicked == "test":
            set_rig(not rig_on())
        elif clicked == "camera":
            settings.put("kamera", cap.next_camera())
            # a different camera can give a picture of a different size, so
            # the screen is laid out again from its first frame
            result = "start"
        elif clicked:
            result = clicked
    return result or "quit"
