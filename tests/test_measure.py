"""Unit tests for src/measure.py using synthetic masks with known geometry."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.measure import measure_cracks, scale_from_points, skeleton_length  # noqa: E402


def line_mask(p1, p2, thickness, shape=(200, 300)):
    m = np.zeros(shape, np.uint8)
    cv2.line(m, p1, p2, 255, thickness)
    return m


def bar_mask(x0, x1, y_center, width, shape=(200, 300)):
    """Axis-aligned bar of EXACT pixel width (cv2.line draws thickness+2 wide)."""
    m = np.zeros(shape, np.uint8)
    half = width // 2
    m[y_center - half:y_center + half + 1, x0:x1] = 255
    return m


def test_horizontal_line_length_and_width():
    mask = bar_mask(20, 221, 100, 5)  # 201 px long, exactly 5 px wide
    table, labels, skel = measure_cracks(mask)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["unit"] == "px"
    # Skeleton shrinks by ~width/2 at each end, so allow a few pixels
    assert row["length"] == pytest.approx(200, abs=6)
    assert row["max_width"] == pytest.approx(5, abs=1)
    assert row["mean_width"] == pytest.approx(5, abs=1)


def test_diagonal_line_uses_sqrt2_steps():
    mask = line_mask((20, 20), (120, 120), 3)  # true length 100*sqrt(2) = 141.4
    table, _, _ = measure_cracks(mask)
    assert table.iloc[0]["length"] == pytest.approx(100 * math.sqrt(2), abs=8)


def test_skeleton_length_exact_cases():
    horizontal = np.zeros((5, 12), bool)
    horizontal[2, 1:11] = True  # 10 pixels -> 9 steps
    assert skeleton_length(horizontal) == pytest.approx(9)
    diagonal = np.eye(10, dtype=bool)  # 10 pixels -> 9 diagonal steps
    assert skeleton_length(diagonal) == pytest.approx(9 * math.sqrt(2))
    staircase = np.zeros((4, 4), bool)
    staircase[0, 0] = staircase[0, 1] = staircase[1, 1] = True  # corner counted once
    assert skeleton_length(staircase) == pytest.approx(2)


def test_wider_line_reports_larger_width():
    thin = measure_cracks(bar_mask(20, 251, 60, 3))[0].iloc[0]
    thick = measure_cracks(bar_mask(20, 251, 60, 9))[0].iloc[0]
    assert thick["max_width"] > thin["max_width"]
    assert thick["max_width"] == pytest.approx(9, abs=1.5)


def test_two_cracks_get_separate_ids_sorted_by_length():
    mask = line_mask((10, 30), (110, 30), 3) | line_mask((10, 150), (260, 150), 3)
    table, labels, _ = measure_cracks(mask)
    assert list(table["crack_id"]) == [1, 2]
    assert table.iloc[0]["length"] > table.iloc[1]["length"]
    assert set(np.unique(labels)) == {0, 1, 2}


def test_mm_conversion():
    mask = bar_mask(20, 221, 100, 5)
    px = measure_cracks(mask)[0].iloc[0]
    mm = measure_cracks(mask, mm_per_pixel=0.5)[0].iloc[0]
    assert mm["unit"] == "mm"
    assert mm["length"] == pytest.approx(px["length"] * 0.5)
    assert mm["max_width"] == pytest.approx(px["max_width"] * 0.5)


def test_empty_mask_gives_empty_table():
    table, labels, skel = measure_cracks(np.zeros((50, 50), np.uint8))
    assert table.empty and not skel.any() and labels.max() == 0


def test_scale_from_points():
    assert scale_from_points((0, 0), (300, 400), 100.0) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        scale_from_points((5, 5), (5, 5), 10.0)
