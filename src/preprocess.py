"""Step 1: load an image, grayscale it, boost contrast (CLAHE) and denoise."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def load_image(path: str | Path) -> Optional[np.ndarray]:
    """Read an image as BGR. Returns None if the file is missing or unreadable.

    ``np.fromfile`` + ``imdecode`` keeps Windows paths with non-ASCII characters working.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, ValueError):
        return None


def resize_if_large(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Shrink ``image`` so its longest side is at most ``max_side``; returns (image, scale)."""
    longest = max(image.shape[:2])
    if max_side <= 0 or longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (or already gray) image to single-channel uint8."""
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, method: str, ksize: int) -> np.ndarray:
    """Light blur that removes grain without erasing thin cracks."""
    if method == "none" or ksize <= 1:
        return gray
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    if method == "median":
        return cv2.medianBlur(gray, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)
    raise ValueError(f"Unknown blur method: {method!r}")


def preprocess(image: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Grayscale -> CLAHE -> blur. Returns every stage as {"gray", "clahe", "blur"}."""
    p = cfg["preprocess"]
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(p["clahe_clip_limit"], (p["clahe_tile_grid"],) * 2).apply(gray)
    return {"gray": gray, "clahe": clahe, "blur": denoise(clahe, p["blur"], p["blur_ksize"])}
