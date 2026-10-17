"""End-to-end sanity tests on a synthetic image and edge cases."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config  # noqa: E402
from src.pipeline import process_array, process_image  # noqa: E402
from src.synthetic import make_synthetic_crack_image  # noqa: E402


def test_synthetic_crack_found_and_stain_rejected():
    img, gt = make_synthetic_crack_image(seed=0)
    res = process_array(img, get_config())
    assert res.n_cracks >= 1
    inter = ((res.mask > 0) & (gt > 0)).sum()
    assert inter / max(1, (res.mask > 0).sum()) > 0.5           # precision
    stain_zone = res.mask[:120, 450:]                            # stain is top right
    assert (stain_zone > 0).sum() == 0


def test_blank_image_reports_no_cracks():
    flat = np.full((200, 200, 3), 140, np.uint8)
    res = process_array(flat, get_config())
    assert res.n_cracks == 0 and "No cracks" in res.message


def test_unreadable_file(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    res = process_image(bad, get_config())
    assert not res.ok and "Could not read" in res.message
    assert not process_image(tmp_path / "missing.jpg", get_config()).ok
