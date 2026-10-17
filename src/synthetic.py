"""Synthetic concrete + crack images, used for tests and demos when no dataset is present."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def concrete_texture(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Grey, grainy, unevenly lit background that looks vaguely like concrete (BGR)."""
    h, w = shape
    base = np.full((h, w), 150.0)
    for sigma, amp in ((60, 18), (12, 10), (2, 8)):
        noise = rng.normal(0, 1, (h, w)).astype(np.float32)
        base += amp * cv2.GaussianBlur(noise, (0, 0), sigma) * (sigma ** 0.5) * 0.9
    yy, xx = np.mgrid[0:h, 0:w]
    base += 15 * (xx / w - 0.5)  # lighting gradient
    gray = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def wavy_polyline(start: Tuple[int, int], end: Tuple[int, int], rng: np.random.Generator,
                  segments: int = 25, wobble: float = 12.0) -> np.ndarray:
    """Jagged line between two points, as an int32 array of (x, y) points."""
    t = np.linspace(0, 1, segments)
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    d = np.array([end[0] - start[0], end[1] - start[1]], float)
    normal = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    offset = np.cumsum(rng.normal(0, wobble / 3, segments))
    offset -= np.linspace(offset[0], offset[-1], segments)  # pin both ends
    x = x + normal[0] * offset
    y = y + normal[1] * offset
    return np.stack([x, y], axis=1).round().astype(np.int32)


def make_synthetic_crack_image(size: Tuple[int, int] = (480, 640), seed: Optional[int] = 0,
                               with_stain: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (image_bgr, ground_truth_mask) with one main crack, one branch and a stain."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = concrete_texture(size, rng)
    gt = np.zeros((h, w), np.uint8)

    main = wavy_polyline((int(w * 0.1), int(h * 0.2)), (int(w * 0.9), int(h * 0.8)), rng)
    branch = wavy_polyline(tuple(main[10]), (int(w * 0.5), int(h * 0.95)), rng, segments=12, wobble=8)
    for pts, width in ((main, 4), (branch, 2)):
        cv2.polylines(gt, [pts.reshape(-1, 1, 2)], False, 255, width, cv2.LINE_8)

    if with_stain:  # dark round blotch; a good detector must reject it
        stain = np.zeros((h, w), np.float32)
        cv2.circle(stain, (int(w * 0.8), int(h * 0.2)), 28, 1.0, -1)
        stain = cv2.GaussianBlur(stain, (0, 0), 8)
        img = (img * (1 - 0.35 * stain[..., None])).astype(np.uint8)

    darkening = cv2.GaussianBlur(gt.astype(np.float32) / 255, (0, 0), 0.8)
    img = (img * (1 - 0.65 * darkening[..., None])).astype(np.uint8)
    return img, gt
