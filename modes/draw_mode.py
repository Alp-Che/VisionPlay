"""Mode 1: draw with your hands.

- Right hand index fingertip = brush position.
- Left hand closed (fist) = pen down (draw). Left hand open = pen up.
- Same clutch logic applies to the eraser.
- Top toolbar (colors / eraser / clear / menu) uses the generic hover + 2s
  fist dwell click from core.ui, with either hand.
"""
import cv2
import numpy as np

from core import assets
from core.rig import draw_rig
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
    return buttons


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
    prev_point = None

    result = None
    while result is None:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

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

        right_hand = next((hd for hd in hands if hd.label == "Right"), None)
        left_hand = next((hd for hd in hands if hd.label == "Left"), None)

        pen_down = (
            right_hand is not None
            and left_hand is not None
            and left_hand.closed
            and right_hand.index_tip[1] > TOOLBAR_H
        )

        if pen_down:
            pt = right_hand.index_tip
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

        if right_hand:
            brush_cursor = assets.load("ui/cursor_brush.png")
            if brush_cursor is not None:
                assets.overlay_centered(display, brush_cursor, *right_hand.index_tip, 50, 50)
            else:
                cv2.circle(display, right_hand.index_tip, 12, (255, 255, 255), 2)
        if left_hand:
            left_cursor = assets.load("ui/cursor_closed.png" if left_hand.closed else "ui/cursor_open.png")
            if left_cursor is not None:
                assets.overlay_centered(display, left_cursor, *left_hand.index_tip, 50, 50)
            else:
                marker_color = (0, 255, 0) if left_hand.closed else (0, 165, 255)
                cv2.circle(display, left_hand.index_tip, 10, marker_color, -1)

        status = f"MOD: {'SİLGİ' if active_mode == 'eraser' else 'FIRÇA'}"
        draw_panel(display, status, (14, h - 44), scale=2)
        draw_rig(display, hands)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            result = "quit"
        elif key == 27:
            result = "menu"
    return result or "quit"
