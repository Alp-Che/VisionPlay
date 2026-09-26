"""Mode 1: draw with your hands.

- One hand is the pen: its index fingertip is the brush.
- The other hand is the clutch: closed (fist) = pen down, open = pen up.
  Same for the eraser.
- The pen starts in the right hand. YÖN DEĞİŞTİR swaps the two, for anyone
  who draws with their left.
- Top toolbar (colors / eraser / clear / swap / menu) uses the generic hover +
  fist dwell click from core.ui, with either hand.

Which hand is which is settled by where they are in the picture, never by
MediaPipe's "Left"/"Right": that label is redone from scratch every frame and
flickers, and the pen jumping between hands mid-stroke draws a line straight
across the page.
"""
import cv2
import numpy as np

from core import assets
from core.rig import draw_rig
from core.window import handle_key
from core.ui import Button, DwellClickController, draw_panel

TOOLBAR_H = 90
BRUSH_THICKNESS = 10
ERASER_THICKNESS = 45

COLORS = [
    ("KIRMIZI", (0, 0, 255)),
    ("SARI", (0, 255, 255)),
    ("YESIL", (0, 255, 0)),
    ("MAVI", (255, 0, 0)),
    ("SIYAH", (20, 20, 20)),
    ("BEYAZ", (255, 255, 255)),
]


def _build_toolbar(w):
    buttons = [Button("menu", "MENÜ", 10, 10, 100, TOOLBAR_H - 20, color=(38, 38, 38))]
    x = 120
    for label, color in COLORS:
        buttons.append(Button(f"color_{label}", "", x, 10, 70, TOOLBAR_H - 20, color=color))
        x += 80
    buttons.append(Button("eraser", "SİLGİ", x, 10, 100, TOOLBAR_H - 20, color=(48, 48, 48)))
    x += 110
    buttons.append(Button("clear", "TEMİZLE", x, 10, 110, TOOLBAR_H - 20, color=(26, 26, 58)))
    x += 120
    buttons.append(Button("swap", "YÖN DEĞİŞTİR", x, 10, min(200, w - 10 - x),
                          TOOLBAR_H - 20, color=(52, 36, 26)))
    return buttons


def _pen_and_clutch(hands, w, swapped):
    """(pen hand, clutch hand), either of which may be None.

    The picture is mirrored, so the player's right hand is the one further
    right on screen. A hand on its own is placed by which half it is in.
    """
    by_x = sorted(hands, key=lambda hand: hand.landmarks_px[9][0])
    if len(by_x) >= 2:
        left, right = by_x[0], by_x[-1]
    elif by_x and by_x[0].landmarks_px[9][0] >= w / 2:
        left, right = None, by_x[0]
    elif by_x:
        left, right = by_x[0], None
    else:
        left, right = None, None
    return (left, right) if swapped else (right, left)


def run_draw_mode(cap, window_name, tracker):
    """Returns 'menu' or 'quit'."""
    dwell = DwellClickController()

    ret, frame = cap.read()
    if not ret:
        return "quit"
    h, w = frame.shape[:2]

    buttons = _build_toolbar(w)
    color_lookup = {f"color_{label}": color for label, color in COLORS}

    canvas_color = np.zeros((h, w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((h, w), dtype=np.uint8)

    active_mode = "brush"
    active_color = COLORS[0][1]
    active_button_id = "color_KIRMIZI"
    swapped = False
    prev_point = None

    result = None
    while result is None:
        ret, frame = cap.read()
        if not ret:
            break

        hands = tracker.process(frame)
        clicked, progress_map = dwell.update(hands, buttons)

        if clicked == "menu":
            result = "menu"
            break
        elif clicked == "clear":
            canvas_mask[:] = 0
        elif clicked == "eraser":
            active_mode = "eraser"
            active_button_id = "eraser"
        elif clicked in color_lookup:
            active_mode = "brush"
            active_color = color_lookup[clicked]
            active_button_id = clicked
        elif clicked == "swap":
            swapped = not swapped
            prev_point = None

        pen_hand, clutch_hand = _pen_and_clutch(hands, w, swapped)

        pen_down = (
            pen_hand is not None
            and clutch_hand is not None
            and clutch_hand.closed
            and pen_hand.index_tip[1] > TOOLBAR_H
        )

        if pen_down:
            pt = pen_hand.index_tip
            if prev_point is None:
                prev_point = pt
            thickness = ERASER_THICKNESS if active_mode == "eraser" else BRUSH_THICKNESS
            if active_mode == "eraser":
                cv2.line(canvas_mask, prev_point, pt, 0, thickness)
            else:
                cv2.line(canvas_color, prev_point, pt, active_color, thickness)
                cv2.line(canvas_mask, prev_point, pt, 255, thickness)
            prev_point = pt
        else:
            prev_point = None

        display = frame.copy()
        mask_bool = canvas_mask > 0
        display[mask_bool] = canvas_color[mask_bool]

        toolbar_bg = assets.load("ui/toolbar_bg.png")
        if toolbar_bg is not None:
            assets.overlay_fill(display, toolbar_bg, 0, 0, w, TOOLBAR_H)

        for btn in buttons:
            hovered = dwell.hovered_id() == btn.id
            progress = progress_map.get(btn.id, 0.0)
            selected = btn.id == active_button_id
            btn.draw(display, progress=progress, hovered=hovered, selected=selected)

        if pen_hand:
            brush_cursor = assets.load("ui/cursor_brush.png")
            if brush_cursor is not None:
                assets.overlay_centered(display, brush_cursor, *pen_hand.index_tip, 50, 50)
            else:
                cv2.circle(display, pen_hand.index_tip, 12, (255, 255, 255), 2)
        if clutch_hand:
            clutch_cursor = assets.load("ui/cursor_closed.png" if clutch_hand.closed else "ui/cursor_open.png")
            if clutch_cursor is not None:
                assets.overlay_centered(display, clutch_cursor, *clutch_hand.index_tip, 50, 50)
            else:
                marker_color = (0, 255, 0) if clutch_hand.closed else (0, 165, 255)
                cv2.circle(display, clutch_hand.index_tip, 10, marker_color, -1)

        status = f"MOD: {'SİLGİ' if active_mode == 'eraser' else 'FIRÇA'}"
        draw_panel(display, status, (14, h - 44), scale=2)
        draw_rig(display, hands)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        handle_key(window_name, key)
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
