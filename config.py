"""Central configuration for the Concrete Crack Detector.

Every tunable value lives in CONFIG. Paths are derived from this file's
location (or from environment variables), never hardcoded.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent

CONFIG: Dict[str, Any] = {
    "paths": {
        "data_dir": Path(os.environ.get("CRACK_DATA_DIR", ROOT_DIR / "data")),
        "results_dir": Path(os.environ.get("CRACK_RESULTS_DIR", ROOT_DIR / "results")),
        # Sub-folders inside data_dir (see data/README.md)
        "metu_subdir": "metu",
        "cfd_subdir": "cfd",
    },
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
    "preprocess": {
        "max_side": 2000,        # larger images are shrunk so the longest side is this
        "clahe_clip_limit": 1.0,
        "clahe_tile_grid": 8,
        "blur": "median",        # "median" | "gaussian" | "none"
        "blur_ksize": 3,         # must be odd
    },
    "detect": {
        "kernel_size": 15,       # black-hat kernel: must be wider than the widest crack
        "threshold_method": "otsu",   # "otsu" | "adaptive" | "fixed"
        "fixed_threshold": 20,
        "min_threshold": 60,     # absolute black-hat contrast floor; Otsu alone fires on clean concrete grain
        "max_foreground_fraction": None,  # optional: raise threshold until at most this share of pixels is foreground
        "adaptive_block_size": 31,
        "adaptive_c": 5,
        "close_kernel": 3,       # closing kernel size that joins broken crack pieces
        "close_iterations": 2,
        "min_area": 50,          # connected components smaller than this are removed (px)
        "min_skeleton_length": 15,   # component skeleton must be at least this long (px)
        "min_elongation": 1.5,   # skeleton_length / sqrt(area); low values mean blob-like stains
        "max_mean_width": 20.0,  # area / skeleton_length above this means a stain, not a crack
    },
    "measure": {
        "connectivity": 8,
        "min_crack_length": 15,  # shortest crack (px) that counts as a "detection"
        "width_offset_px": -1.0, # width = 2 * dt + offset; -1 makes odd pixel widths exact
    },
    "visualize": {
        "crack_color_bgr": (0, 0, 255),       # red
        "skeleton_color_bgr": (0, 255, 255),  # yellow
        "mask_alpha": 0.45,
        "font_scale_base": 0.5,
    },
    "evaluate": {
        "metu_sample_size": 2000,
        "random_seed": 42,
        "cfd_tolerance_px": 2,
        "min_crack_length_metu": 15,
    },
}


def get_config(overrides: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Return a deep copy of CONFIG with optional per-section overrides applied.

    Example: ``get_config({"detect": {"kernel_size": 21}})``.
    """
    cfg = copy.deepcopy(CONFIG)
    for section, values in (overrides or {}).items():
        if isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg
