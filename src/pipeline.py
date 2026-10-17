"""Runs preprocess -> detect -> measure -> visualize on one image."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from src.detect import detect_cracks
from src.measure import measure_cracks
from src.preprocess import load_image, preprocess, resize_if_large
from src.visualize import draw_overlay, save_side_by_side


@dataclass
class PipelineResult:
    """Everything produced for one image. ``ok`` is False if it could not be read."""
    name: str
    ok: bool
    message: str = ""
    image: Optional[np.ndarray] = None       # BGR, after safe resize
    mask: Optional[np.ndarray] = None        # 0/255 uint8
    skeleton: Optional[np.ndarray] = None    # bool
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlay: Optional[np.ndarray] = None
    resize_scale: float = 1.0
    unit: str = "px"

    @property
    def n_cracks(self) -> int:
        return len(self.table)


def process_array(image: np.ndarray, cfg: Dict[str, Any], mm_per_pixel: Optional[float] = None,
                  name: str = "image", resize_scale: float = 1.0) -> PipelineResult:
    """Run the pipeline on an in-memory BGR image (already resized if needed).

    ``mm_per_pixel`` refers to the *original* resolution; it is corrected for
    ``resize_scale`` so measurements stay in true units after a safe downscale.
    """
    effective_mm = mm_per_pixel / resize_scale if mm_per_pixel else None
    gray = preprocess(image, cfg)
    mask = detect_cracks(gray, cfg)
    table, _labels, skeleton = measure_cracks(mask, effective_mm, cfg)
    overlay = draw_overlay(image, mask, skeleton, table, cfg)
    msg = "" if len(table) else "No cracks found."
    return PipelineResult(name=name, ok=True, message=msg, image=image, mask=mask,
                          skeleton=skeleton, table=table, overlay=overlay,
                          resize_scale=resize_scale, unit="mm" if mm_per_pixel else "px")


def process_image(path: str | Path, cfg: Dict[str, Any], mm_per_pixel: Optional[float] = None,
                  out_dir: Optional[Path] = None) -> PipelineResult:
    """Load ``path``, run the pipeline and (if ``out_dir`` is given) save overlay + figure."""
    path = Path(path)
    image = load_image(path)
    if image is None:
        return PipelineResult(name=path.name, ok=False, message=f"Could not read image: {path}")
    image, scale = resize_if_large(image, cfg["preprocess"]["max_side"])
    result = process_array(image, cfg, mm_per_pixel, name=path.name, resize_scale=scale)
    if scale != 1.0:
        result.message = (result.message + f" (image downscaled by {scale:.2f} for processing)").strip()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), result.overlay)
        save_side_by_side(result.image, result.mask, result.overlay,
                          out_dir / f"{path.stem}_figure.png", title=path.name)
    return result


def results_to_csv(results: list[PipelineResult], csv_path: Path) -> pd.DataFrame:
    """Write columns image, crack_id, length, mean_width, max_width, unit for all results."""
    frames = []
    for r in results:
        if r.ok and r.n_cracks:
            t = r.table[["crack_id", "length", "mean_width", "max_width", "unit"]].copy()
            t.insert(0, "image", r.name)
            frames.append(t)
    cols = ["image", "crack_id", "length", "mean_width", "max_width", "unit"]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df = df.round({"length": 2, "mean_width": 2, "max_width": 2})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
