"""Steps 4-5: skeleton, length, width per crack, and pixel -> mm conversion."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

SQRT2 = math.sqrt(2.0)
TABLE_COLUMNS = ["crack_id", "length", "mean_width", "max_width", "unit", "anchor_x", "anchor_y"]


def skeleton_length(skel: np.ndarray) -> float:
    """Length of a 1-pixel-wide skeleton in pixels.

    Walks every 8-connected pixel pair once: horizontal/vertical steps count 1,
    diagonal steps count sqrt(2). A diagonal step is skipped when the two pixels
    are already linked through an orthogonal neighbour (avoids double counting
    the corner of a staircase).
    """
    s = np.pad(skel.astype(bool), 1)
    c = s[1:-1, 1:-1]
    right = c & s[1:-1, 2:]
    down = c & s[2:, 1:-1]
    # diagonal down-right: skip if the pixel to the right or below is also set
    dr = c & s[2:, 2:] & ~s[1:-1, 2:] & ~s[2:, 1:-1]
    # diagonal down-left: skip if the pixel to the left or below is also set
    dl = c & s[2:, :-2] & ~s[1:-1, :-2] & ~s[2:, 1:-1]
    return float(right.sum() + down.sum() + SQRT2 * (dr.sum() + dl.sum()))


def width_map(mask: np.ndarray, offset: float = -1.0) -> np.ndarray:
    """Per-pixel crack width = 2 * distance_transform + offset (read on the skeleton).

    The distance transform measures from a pixel centre to the nearest background
    pixel centre, so a 5-px wide line gives dt=3 at its centre; ``offset=-1``
    turns 2*3 into the true 5.
    """
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return np.maximum(2.0 * dt + offset, 1.0)


def measure_cracks(
    mask: np.ndarray,
    mm_per_pixel: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Measure every crack (8-connected component) in ``mask``.

    Returns:
        table:    DataFrame [crack_id, length, mean_width, max_width, unit,
                  anchor_x, anchor_y], longest crack first. The anchor is a
                  skeleton pixel used to place the text label.
        labels:   int32 image holding the crack id per mask pixel (0 = background)
        skeleton: bool image of the skeleton pixels
    """
    m = cfg["measure"] if cfg else {}
    offset = m.get("width_offset_px", -1.0)
    binary = mask > 0
    skel = skeletonize(binary)
    widths = width_map(binary, offset)
    n, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=m.get("connectivity", 8))

    scale = mm_per_pixel if mm_per_pixel else 1.0
    unit = "mm" if mm_per_pixel else "px"
    rows: List[Dict[str, Any]] = []
    for lbl in range(1, n):
        comp_skel = skel & (labels == lbl)
        if not comp_skel.any():
            continue
        ys, xs = np.nonzero(comp_skel)
        w = widths[comp_skel]
        mid = len(xs) // 2
        rows.append({
            "_label": lbl,
            "length": skeleton_length(comp_skel) * scale,
            "mean_width": float(w.mean()) * scale,
            "max_width": float(w.max()) * scale,
            "unit": unit,
            "anchor_x": int(xs[mid]),
            "anchor_y": int(ys[mid]),
        })
    rows.sort(key=lambda r: r["length"], reverse=True)
    relabel = np.zeros(n, dtype=np.int32)
    for new_id, r in enumerate(rows, start=1):
        r["crack_id"] = new_id
        relabel[r.pop("_label")] = new_id
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return table, relabel[labels], skel


def scale_from_points(p1: Tuple[float, float], p2: Tuple[float, float], real_length_mm: float) -> float:
    """mm-per-pixel from two points on a reference object of known length."""
    pixel_dist = math.dist(p1, p2)
    if pixel_dist <= 0 or real_length_mm <= 0:
        raise ValueError("Points must be distinct and the real length must be positive.")
    return real_length_mm / pixel_dist


def pick_scale_interactively(image: np.ndarray, real_length_mm: float, max_display: int = 1200) -> float:
    """Open a window; the user clicks both ends of a reference object of known size.

    Returns mm-per-pixel for the *given* image. Press Esc to cancel.
    """
    zoom = min(1.0, max_display / max(image.shape[:2]))
    view = cv2.resize(image, None, fx=zoom, fy=zoom) if zoom < 1.0 else image.copy()
    points: List[Tuple[float, float]] = []
    title = "Click both ends of the reference object (Esc to cancel)"

    def on_mouse(event: int, x: int, y: int, *_: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x / zoom, y / zoom))
            cv2.circle(view, (x, y), 4, (0, 255, 0), -1)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while len(points) < 2:
        cv2.imshow(title, view)
        if cv2.waitKey(30) == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Scale calibration cancelled.")
    cv2.imshow(title, view)
    cv2.waitKey(400)
    cv2.destroyAllWindows()
    return scale_from_points(points[0], points[1], real_length_mm)
