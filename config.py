"""All tunable values for the Concrete Crack Detector live in CONFIG."""
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
        "metu_subdir": "metu",   # data/metu/Positive and data/metu/Negative
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
        "min_threshold": 60,     # absolute black-hat contrast floor (Otsu alone fires on clean grain)
        "adaptive_block_size": 31,
        "adaptive_c": 5,
        "close_kernel": 3,       # closing joins broken crack pieces
        "close_iterations": 2,
        "min_area": 50,          # components smaller than this are removed (px)
        "min_skeleton_length": 15,
        "min_elongation": 1.5,   # skeleton_length / sqrt(area); low = blob-like stain
        "max_mean_width": 20.0,  # area / skeleton_length; high = thick stain
    },
    "measure": {
        "connectivity": 8,
        "min_crack_length": 15,  # shortest crack (px) that counts as a detection
        "width_offset_px": -1.0, # width = 2 * distance + offset (-1 makes odd widths exact)
    },
    "visualize": {
        "crack_color_bgr": (0, 0, 255),       # red
        "skeleton_color_bgr": (0, 255, 255),  # yellow
        "mask_alpha": 0.45,
        "font_scale_base": 0.5,
    },
    "evaluate": {
        "sample_size": 2000,     # evaluation images (random, seeded)
        "val_size": 400,         # separate images used only for tuning
        "random_seed": 42,
    },
    # Result of `python main.py --tune` on the validation images; used by `--evaluate metu`.
    "metu_tuned": {
        "detect": {"kernel_size": 31, "min_threshold": 100},
        "canny_thresholds": (130, 390),   # baseline, tuned the same way
    },
}


def get_config(overrides: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Return a deep copy of CONFIG with optional per-section overrides applied."""
    cfg = copy.deepcopy(CONFIG)
    for section, values in (overrides or {}).items():
        if isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg
