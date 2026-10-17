"""Save one figure showing every processing step for a single image.

Usage:
    python make_step_figure.py --image path.jpg --out results/pipeline_cracked.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import get_config  # noqa: E402
from src.detect import blackhat, close_mask, filter_components, threshold_blackhat  # noqa: E402
from src.evaluate import load_tuned_config
from src.measure import measure_cracks  # noqa: E402
from src.preprocess import apply_clahe, denoise, load_image, resize_if_large, to_grayscale  # noqa: E402
from src.visualize import draw_overlay  # noqa: E402


def run_steps(image: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run the pipeline and keep every intermediate result."""
    p, d = cfg["preprocess"], cfg["detect"]
    gray = to_grayscale(image)
    clahe = apply_clahe(gray, p["clahe_clip_limit"], p["clahe_tile_grid"])
    blur = denoise(clahe, p["blur"], p["blur_ksize"])
    bh = blackhat(blur, d["kernel_size"])
    thresh = threshold_blackhat(bh, d)
    closed = close_mask(thresh, d["close_kernel"], d["close_iterations"])
    final = filter_components(closed, d)
    table, _, skeleton = measure_cracks(final, None, cfg)
    overlay = draw_overlay(image, final, skeleton, table, cfg)
    skel_img = np.zeros((*final.shape, 3), np.uint8)
    skel_img[final > 0] = (60, 60, 60)
    skel_img[skeleton] = cfg["visualize"]["skeleton_color_bgr"]
    return {"image": image, "gray": gray, "clahe": clahe, "blur": blur, "bh": bh,
            "thresh": thresh, "closed": closed, "final": final, "skel": skel_img,
            "overlay": overlay, "table": table}


def save_step_figure(steps: Dict[str, Any], path: Path, title: str, cfg: Dict[str, Any]) -> None:
    """2 x 5 grid, one panel per step, with the measured result as the figure subtitle."""
    d = cfg["detect"]
    rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # noqa: E731
    n_px = lambda a: int((a > 0).sum())  # noqa: E731
    panels = [
        ("1. Original", rgb(steps["image"]), None),
        ("2. Grayscale", steps["gray"], "gray"),
        ("3. CLAHE (contrast)", steps["clahe"], "gray"),
        ("4. Median blur (denoise)", steps["blur"], "gray"),
        (f"5. Black-hat {d['kernel_size']}x{d['kernel_size']}\n(dark thin lines glow)", steps["bh"], "gray"),
        (f"6. Threshold (Otsu, floor {d['min_threshold']})", steps["thresh"], "gray"),
        ("7. Closing (join gaps)", steps["closed"], "gray"),
        ("8. Remove specks / stains", steps["final"], "gray"),
        ("9. Skeleton (yellow)", rgb(steps["skel"]), None),
        ("10. Final overlay", rgb(steps["overlay"]), None),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(20, 9.4))
    for ax, (name, data, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        ax.set_title(name, fontsize=12)
        ax.axis("off")
    table = steps["table"]
    if table.empty:
        result = "Result: no cracks found"
    else:
        result = f"Result: {len(table)} crack(s), total length {table['length'].sum():.0f} px, " \
                 f"max width {table['max_width'].max():.1f} px"
    fig.suptitle(f"{title}\n{result}", fontsize=15)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.86, wspace=0.04, hspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Step-by-step figure for one image.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--kernel", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_tuned_config(get_config(), Path(get_config()["paths"]["results_dir"]) / "metu")
    if args.kernel:
        cfg["detect"]["kernel_size"] = args.kernel
    image = load_image(args.image)
    if image is None:
        print(f"Could not read image: {args.image}")
        return 1
    image, _ = resize_if_large(image, cfg["preprocess"]["max_side"])
    save_step_figure(run_steps(image, cfg), args.out, args.title or args.image.name, cfg)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
