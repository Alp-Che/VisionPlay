"""Scale + rotate small pixel-art sprites around an arbitrary pivot, then
alpha-composite them onto a frame. Used by archery_mode.py to draw the bow
and arrow at any angle without blurring the pixel art."""
import math

import cv2
import numpy as np


def integer_scale_for(icon_bgra, target_size):
    """Largest whole-number zoom that lands near target_size. Pixel art must
    be enlarged by an integer factor, otherwise some source pixels become 3
    screen pixels wide and their neighbours 4, which reads as blur."""
    return max(1, round(target_size / max(icon_bgra.shape[:2])))


def scale_and_rotate(icon_bgra, target_size, angle_rad, pivot=(0.5, 0.5), squash=1.0):
    """Upscales icon_bgra (nearest-neighbor, keeps pixel-art edges crisp) so
    its larger side equals target_size, rotates it by angle_rad around
    `pivot` (a fraction of the ORIGINAL sprite's width/height), and returns
    (rotated_bgra, pivot_xy_in_output) so the caller can align pivot_xy with
    a target screen point.

    angle_rad is measured in image coordinates (x right, y down): rotating
    the local vector (1, 0) by angle_rad=pi/2 points it to local (0, 1),
    i.e. straight down on screen.

    `squash` compresses the sprite along its own y axis only, leaving the
    width alone -- that is what foreshortening looks like when an object is
    turned towards the camera, rather than it simply getting smaller.
    """
    ih, iw = icon_bgra.shape[:2]
    zoom = integer_scale_for(icon_bgra, target_size)
    nw = iw * zoom
    nh = max(1, int(round(ih * zoom * squash)))
    upscaled = cv2.resize(icon_bgra, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Rotate about the sprite's own centre so the canvas only has to hold the
    # sprite's diagonal. Padding out to the pivot instead makes it far bigger
    # -- for a sword gripped at the hilt it more than triples the area warped
    # every frame -- and the pivot is recovered from the matrix below anyway.
    diag = int(np.ceil(math.hypot(nw, nh))) + 2
    canvas = np.zeros((diag, diag, 4), dtype=np.uint8)
    ox, oy = (diag - nw) // 2, (diag - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = upscaled

    # cv2.getRotationMatrix2D's positive angle is counter-clockwise in a
    # y-up math sense, which is clockwise on an y-down image -- negate the
    # degrees so angle_rad follows the atan2(dy, dx) image convention.
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), -np.degrees(angle_rad), 1.0)
    rotated = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    px, py = ox + pivot[0] * nw, oy + pivot[1] * nh
    return rotated, (M[0, 0] * px + M[0, 1] * py + M[0, 2],
                     M[1, 0] * px + M[1, 1] * py + M[1, 2])


def paste_alpha(frame, img_bgra, x, y):
    """Alpha-composites img_bgra onto frame with its top-left at (x, y).
    x/y may be fractional or extend past frame bounds; safely clipped."""
    ih, iw = img_bgra.shape[:2]
    H, W = frame.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, x1 = max(x, 0), min(x + iw, W)
    y0, y1 = max(y, 0), min(y + ih, H)
    if x0 >= x1 or y0 >= y1:
        return
    crop = img_bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    fg = crop[:, :, :3].astype(np.float32)
    alpha = (crop[:, :, 3].astype(np.float32) / 255.0)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - alpha) + fg * alpha).astype(np.uint8)


def place_rotated(frame, icon_bgra, anchor_xy, angle_rad, target_size, pivot=(0.5, 0.5),
                  squash=1.0):
    """Scales icon_bgra to target_size, rotates it around `pivot` to
    angle_rad, and pastes it so that pivot lands exactly at anchor_xy."""
    rotated, (pcx, pcy) = scale_and_rotate(icon_bgra, target_size, angle_rad, pivot, squash)
    paste_alpha(frame, rotated, anchor_xy[0] - pcx, anchor_xy[1] - pcy)
