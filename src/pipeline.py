"""Runs preprocess -> detect -> measure on one image and draws the step-by-step figure."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.detect import detect_cracks  # noqa: E402
from src.measure import measure_cracks  # noqa: E402
from src.preprocess import load_image, preprocess, resize_if_large  # noqa: E402

CSV_COLUMNS = ["image", "crack_id", "length", "mean_width", "max_width", "unit"]


@dataclass
class PipelineResult:
    """Everything produced for one image. ``ok`` is False if the file could not be read."""
    name: str
    ok: bool
    message: str = ""
    steps: Dict[str, np.ndarray] = field(default_factory=dict)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_cracks(self) -> int:
        return len(self.table)


def draw_overlay(image: np.ndarray, mask: np.ndarray, skeleton: np.ndarray,
                 table: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    """Cracks in red, skeleton in yellow, and a '#id length x max-width' label per crack."""
    v = cfg["visualize"]
    out = image.copy()
    red = np.zeros_like(out)
    red[:] = v["crack_color_bgr"]
    blend = cv2.addWeighted(out, 1 - v["mask_alpha"], red, v["mask_alpha"], 0)
    out[mask > 0] = blend[mask > 0]
    out[skeleton] = v["skeleton_color_bgr"]
    scale = v["font_scale_base"] * float(np.clip(max(image.shape[:2]) / 800, 0.5, 3.0))
    thick = max(1, round(scale * 2))
    for _, r in table.iterrows():
        text = f"#{int(r['crack_id'])} {r['length']:.0f}x{r['max_width']:.1f}{r['unit']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = min(max(int(r["anchor_x"]) + 6, 0), max(0, out.shape[1] - tw - 2))
        y = min(max(int(r["anchor_y"]) - 6, th + 2), out.shape[0] - 2)
        cv2.rectangle(out, (x - 1, y - th - 2), (x + tw + 1, y + 3), (0, 0, 0), -1)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def run_steps(image: np.ndarray, cfg: Dict[str, Any], mm_per_pixel: Optional[float] = None) -> PipelineResult:
    """Run the full pipeline on a BGR image and keep every intermediate result."""
    pre = preprocess(image, cfg)
    det = detect_cracks(pre["blur"], cfg)
    table, _, skeleton = measure_cracks(det["mask"], mm_per_pixel, cfg)
    skel_img = np.zeros((*det["mask"].shape, 3), np.uint8)
    skel_img[det["mask"] > 0] = (60, 60, 60)
    skel_img[skeleton] = cfg["visualize"]["skeleton_color_bgr"]
    steps = {"original": image, **pre, **det, "skeleton": skel_img,
             "overlay": draw_overlay(image, det["mask"], skeleton, table, cfg)}
    return PipelineResult(name="", ok=True, message="" if len(table) else "No cracks found.",
                          steps=steps, table=table)


def process_image(path: str | Path, cfg: Dict[str, Any], mm_per_pixel: Optional[float] = None) -> PipelineResult:
    """Load ``path`` (resizing very large images) and run the pipeline.

    ``mm_per_pixel`` refers to the original resolution; it is corrected for any downscale.
    """
    path = Path(path)
    image = load_image(path)
    if image is None:
        return PipelineResult(name=path.name, ok=False, message=f"Could not read image: {path}")
    image, scale = resize_if_large(image, cfg["preprocess"]["max_side"])
    res = run_steps(image, cfg, mm_per_pixel / scale if mm_per_pixel else None)
    res.name = path.name
    if scale != 1.0:
        res.message = f"{res.message} (image downscaled by {scale:.2f} for processing)".strip()
    return res


def save_steps_figure(res: PipelineResult, path: Path, title: str, cfg: Dict[str, Any]) -> None:
    """One figure showing all ten processing steps plus the measured result."""
    s, d = res.steps, cfg["detect"]
    rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # noqa: E731
    panels = [
        ("1. Original", rgb(s["original"]), None), ("2. Grayscale", s["gray"], "gray"),
        ("3. CLAHE (contrast)", s["clahe"], "gray"), ("4. Median blur (denoise)", s["blur"], "gray"),
        (f"5. Black-hat {d['kernel_size']}x{d['kernel_size']}\n(dark thin lines glow)", s["blackhat"], "gray"),
        (f"6. Threshold (Otsu, floor {d['min_threshold']})", s["threshold"], "gray"),
        ("7. Closing (join gaps)", s["closed"], "gray"), ("8. Remove specks / stains", s["mask"], "gray"),
        ("9. Skeleton (yellow)", rgb(s["skeleton"]), None), ("10. Final overlay", rgb(s["overlay"]), None)]
    fig, axes = plt.subplots(2, 5, figsize=(20, 9.4))
    for ax, (name, data, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        ax.set_title(name, fontsize=12)
        ax.axis("off")
    t = res.table
    result = ("Result: no cracks found" if t.empty else
              f"Result: {len(t)} crack(s), total length {t['length'].sum():.0f} {t['unit'].iloc[0]}, "
              f"max width {t['max_width'].max():.1f} {t['unit'].iloc[0]}")
    fig.suptitle(f"{title}\n{result}", fontsize=15)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.86, wspace=0.04, hspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def measurements_to_csv(results: List[PipelineResult], csv_path: Path) -> pd.DataFrame:
    """Write one row per crack: image, crack_id, length, mean_width, max_width, unit."""
    frames = []
    for r in results:
        if r.ok and r.n_cracks:
            t = r.table[CSV_COLUMNS[1:]].copy()
            t.insert(0, "image", r.name)
            frames.append(t)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(csv_path, index=False)
    return df
