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

    ``min_threshold`` is always enforced as a floor: Otsu always splits *something*, so
    without it a crack-free image would be filled with texture detections.
    """
    method, floor = d["threshold_method"], d["min_threshold"]
    if method == "otsu":
        t, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(bh, max(t, floor), 255, cv2.THRESH_BINARY)
    elif method == "fixed":
        _, mask = cv2.threshold(bh, max(d["fixed_threshold"], floor), 255, cv2.THRESH_BINARY)
    elif method == "adaptive":
        local = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                      int(d["adaptive_block_size"]) | 1, -d["adaptive_c"])
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
    """Drop specks and blob-like stains; keep long thin components.

    A component is kept only if: area >= min_area, skeleton length >= min_skeleton_length,
    skeleton_length / sqrt(area) >= min_elongation, and area / skeleton_length <= max_mean_width.
    """
    binary = mask > 0
    if not binary.any():
        return np.zeros_like(mask)
    n, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    skel = skeletonize(binary)
    area = np.bincount(labels.ravel(), minlength=n).astype(float)
    skel_len = np.bincount(labels[skel], minlength=n).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = ((area >= d["min_area"]) & (skel_len >= d["min_skeleton_length"])
                & (skel_len / np.sqrt(area) >= d["min_elongation"])
                & (area / skel_len <= d["max_mean_width"]))
    keep[0] = False  # background label
    return (keep[labels] * 255).astype(np.uint8)


def cleanup(mask: np.ndarray, d: Dict[str, Any]) -> np.ndarray:
    """Closing followed by the component filter (shared by the method and the baselines)."""
    return filter_components(close_mask(mask, d["close_kernel"], d["close_iterations"]), d)


def detect_cracks(blurred: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Black-hat -> threshold -> cleanup. Returns {"blackhat", "threshold", "closed", "mask"}."""
    d = cfg["detect"]
    bh = blackhat(blurred, d["kernel_size"])
    thresh = threshold_blackhat(bh, d)
    closed = close_mask(thresh, d["close_kernel"], d["close_iterations"])
    return {"blackhat": bh, "threshold": thresh, "closed": closed, "mask": filter_components(closed, d)}
