"""Steps 2-3: black-hat highlighting, thresholding and mask cleanup."""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from skimage.morphology import skeletonize


def blackhat(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Black-hat = closing(gray) - gray: bright where thin dark lines are."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def threshold_blackhat(bh: np.ndarray, d: Dict[str, Any]) -> np.ndarray:
    """Binarise the black-hat image (Otsu, adaptive or fixed) -> uint8 0/255.

    A floor (``min_threshold``) is always enforced: Otsu always splits *something*,
    so without it a crack-free image would be filled with noise detections.
    """
    method = d["threshold_method"]
    floor = d["min_threshold"]
    # Texture on clean concrete also lights up after CLAHE; cap the foreground share so
    # the threshold rises until only the strongest (crack-like) responses remain.
    cap = d.get("max_foreground_fraction")
    if cap:
        floor = max(floor, float(np.quantile(bh, 1.0 - cap)))
    if method == "otsu":
        t, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(bh, max(t, floor), 255, cv2.THRESH_BINARY)
    elif method == "fixed":
        _, mask = cv2.threshold(bh, max(d["fixed_threshold"], floor), 255, cv2.THRESH_BINARY)
    elif method == "adaptive":
        block = int(d["adaptive_block_size"]) | 1
        local = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, block, -d["adaptive_c"])
        mask = np.where(bh >= floor, local, 0)
    else:
        raise ValueError(f"Unknown threshold method: {method!r}")
    return mask.astype(np.uint8)


def close_mask(mask: np.ndarray, ksize: int, iterations: int) -> np.ndarray:
    """Morphological closing to bridge small gaps in a broken crack."""
    if ksize <= 1 or iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def filter_components(mask: np.ndarray, d: Dict[str, Any]) -> np.ndarray:
    """Drop tiny specks and blob-like stains; keep long thin components.

    A component is kept only if all hold:
      * area >= min_area
      * skeleton pixel count >= min_skeleton_length
      * skeleton_length / sqrt(area) >= min_elongation   (long vs. fat)
      * area / skeleton_length <= max_mean_width         (not a thick stain)
    """
    binary = mask > 0
    if not binary.any():
        return np.zeros_like(mask)
    n, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    skel = skeletonize(binary)
    area = np.bincount(labels.ravel(), minlength=n).astype(float)
    skel_len = np.bincount(labels[skel], minlength=n).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        elongation = skel_len / np.sqrt(area)
        mean_width = area / skel_len
    keep = (
        (area >= d["min_area"])
        & (skel_len >= d["min_skeleton_length"])
        & (elongation >= d["min_elongation"])
        & (mean_width <= d["max_mean_width"])
    )
    keep[0] = False  # background label
    return (keep[labels] * 255).astype(np.uint8)


def detect_cracks(gray: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Run black-hat -> threshold -> close -> filter. Returns a 0/255 uint8 mask."""
    d = cfg["detect"]
    bh = blackhat(gray, d["kernel_size"])
    mask = threshold_blackhat(bh, d)
    mask = close_mask(mask, d["close_kernel"], d["close_iterations"])
    return filter_components(mask, d)
