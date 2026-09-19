"""Main menu: pick a mode with a hand hover + 2s fist dwell click."""
import cv2

from core import assets
from core.hand_tracker import HandTracker
from core.pixel_font import draw_text
from core.transform import integer_scale_for
from core.ui import Button, DwellClickController


def run_menu(cap, window_name):
    """Returns a mode id ('archery', 'draw', ...), 'start' or 'quit'."""
    tracker = HandTracker(num_hands=2)
    dwell = DwellClickController()

    ret, frame = cap.read()
    h, w = frame.shape[:2] if ret else (480, 640)

    # Two rows: the longest labels need more width than five across can give.
    # Backgrounds stay dark and low-saturation, because the button font keeps
    # its own mid-green artwork and that vanishes on anything lighter.
    rows = [
        [("archery", "OKÇULUK", (26, 30, 55), 3),
         ("draw", "ÇİZİM", (62, 38, 22), 3),
         ("ninja", "FRUIT NINJA", (30, 52, 30), 2)],
        [("juggle", "TOP SEKTİRME", (52, 30, 46), 2),
         ("catch", "YAKALA", (46, 30, 26), 3),
         ("mole", "KÖSTEBEK", (30, 44, 56), 3)],
    ]
    btn_w, btn_h, gap = 330, 120, 30
    top = h // 2 - (btn_h * len(rows) + gap * (len(rows) - 1)) // 2

    buttons = []
    for r, row in enumerate(rows):
        row_x = (w - (btn_w * len(row) + gap * (len(row) - 1))) // 2
        y = top + r * (btn_h + gap)
        for c, (ident, label, color, scale) in enumerate(row):
            buttons.append(Button(ident, label, row_x + c * (btn_w + gap), y,
                                  btn_w, btn_h, color=color, text_scale=scale))

    quit_btn = Button("start", "GERİ", w - 140, 20, 120, 50, color=(38, 38, 38))
    all_buttons = buttons + [quit_btn]

    result = None
    try:
        while result is None:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)
            clicked, progress_map = dwell.update(hands, all_buttons)

            logo = assets.load("ui/logo.png")
            if logo is not None:
                zoom = integer_scale_for(logo, 155)
                assets.overlay(frame, logo, 30, 20,
                               logo.shape[1] * zoom, logo.shape[0] * zoom)
            else:
                draw_text(frame, "VisionPlay", (30, 30), scale=4)
            draw_text(frame, "ELİNİ BİR MODA GETİR, YUMRUK YAP VE 1 SANİYE BEKLE",
                      (30, h - 50), scale=2, color=(200, 200, 200))

            for btn in all_buttons:
                hovered = dwell.hovered_id() == btn.id
                progress = progress_map.get(btn.id, 0.0)
                btn.draw(frame, progress=progress, hovered=hovered)

            for hand in hands:
                cursor = assets.load("ui/cursor_closed.png" if hand.closed else "ui/cursor_open.png")
                if cursor is not None:
                    assets.overlay_centered(frame, cursor, *hand.index_tip, 50, 50)
                else:
                    cv2.circle(frame, hand.index_tip, 10,
                               (0, 255, 0) if hand.closed else (0, 200, 255), -1)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                result = "quit"
            elif clicked:
                result = clicked
    finally:
        tracker.close()

    return result or "quit"
