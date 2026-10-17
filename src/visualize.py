"""Step 6: overlay and side-by-side figure."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def draw_overlay(image: np.ndarray, mask: np.ndarray, skeleton: np.ndarray,
                 table: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    """Cracks in red, skeleton in yellow, ``#id L=.. W=..`` label next to each crack."""
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
        unit = r["unit"]
        text = f"#{int(r['crack_id'])} {r['length']:.0f}x{r['max_width']:.1f}{unit}"
        x, y = int(r["anchor_x"]), int(r["anchor_y"])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = min(max(x + 6, 0), max(0, out.shape[1] - tw - 2))
        y = min(max(y - 6, th + 2), out.shape[0] - 2)
        cv2.rectangle(out, (x - 1, y - th - 2), (x + tw + 1, y + 3), (0, 0, 0), -1)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def save_side_by_side(image: np.ndarray, mask: np.ndarray, overlay: np.ndarray,
                      path: Path, title: str = "") -> None:
    """Save the figure: original | mask | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [("Original", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None),
              ("Crack mask", mask, "gray"),
              ("Overlay", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), None)]
    for ax, (name, data, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
