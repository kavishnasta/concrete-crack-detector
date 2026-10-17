"""Baseline comparison and ablation study on METU (image-level cracked vs clean).

All methods share the same decision rule (longest skeleton >= min length after the
same cleanup) and differ only in the *detection* stage. Baseline parameters are
tuned on the same validation split as the proposed method and scored on the same
held-out evaluation sample. Writes:

    results/comparison.csv   proposed vs Otsu vs Canny (with 95% Wilson CIs, McNemar p)
    results/ablation.csv     proposed method with one component removed at a time

    python experiments.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from config import get_config
from src.detect import blackhat, close_mask, filter_components, threshold_blackhat
from src.evaluate import BEST_PARAMS_FILE, classification_metrics, split_metu
from src.measure import measure_cracks
from src.preprocess import apply_clahe, denoise, load_image, to_grayscale

Predictor = Callable[[np.ndarray], bool]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def decide(mask: np.ndarray, cfg: Dict[str, Any], use_filter: bool = True) -> bool:
    """Shared decision rule: cleanup, then 'cracked' iff the longest crack >= min length."""
    d = cfg["detect"]
    mask = close_mask(mask, d["close_kernel"], d["close_iterations"])
    if use_filter:
        mask = filter_components(mask, d)
    else:  # area filter only
        n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
        area = np.bincount(lab.ravel(), minlength=n)
        keep = area >= d["min_area"]
        keep[0] = False
        mask = (keep[lab] * 255).astype(np.uint8)
    table, _, _ = measure_cracks(mask, None, cfg)
    return len(table) > 0 and table["length"].max() >= cfg["evaluate"]["min_crack_length_metu"]


def make_proposed(cfg: Dict[str, Any], clahe: bool = True, blur: bool = True,
                  closing: bool = True, comp_filter: bool = True, otsu: bool = True) -> Predictor:
    """Proposed pipeline, optionally with one component switched off (ablation)."""
    p, d = cfg["preprocess"], dict(cfg["detect"])
    if not otsu:
        d["threshold_method"] = "fixed"
        d["fixed_threshold"] = d["min_threshold"]
    cfg = {**cfg, "detect": {**d, **({} if closing else {"close_kernel": 1, "close_iterations": 0})}}

    def predict(img: np.ndarray) -> bool:
        g = to_grayscale(img)
        if clahe:
            g = apply_clahe(g, p["clahe_clip_limit"], p["clahe_tile_grid"])
        if blur:
            g = denoise(g, p["blur"], p["blur_ksize"])
        mask = threshold_blackhat(blackhat(g, cfg["detect"]["kernel_size"]), cfg["detect"])
        return decide(mask, cfg, comp_filter)
    return predict


def make_otsu_baseline(cfg: Dict[str, Any]) -> Predictor:
    """Plain global Otsu on the blurred gray image; dark pixels are 'crack'."""
    def predict(img: np.ndarray) -> bool:
        g = denoise(to_grayscale(img), cfg["preprocess"]["blur"], cfg["preprocess"]["blur_ksize"])
        _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return decide(mask, cfg)
    return predict


def make_canny_baseline(cfg: Dict[str, Any], low: int, high: int) -> Predictor:
    """Canny edges on the blurred gray image, then the shared cleanup."""
    def predict(img: np.ndarray) -> bool:
        g = denoise(to_grayscale(img), cfg["preprocess"]["blur"], cfg["preprocess"]["blur_ksize"])
        return decide(cv2.Canny(g, low, high), cfg)
    return predict


def run(predictor: Predictor, items: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    return np.array([predictor(img) for img, _ in items], dtype=bool)


def score(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    c = {"TP": int((pred & y).sum()), "FP": int((pred & ~y).sum()),
         "TN": int((~pred & ~y).sum()), "FN": int((~pred & y).sum())}
    m = classification_metrics(c)
    lo, hi = wilson(c["TP"] + c["TN"], len(y))
    return {**c, **m, "acc_ci_low": lo, "acc_ci_high": hi}


def main() -> int:
    cfg = get_config()
    out = Path(cfg["paths"]["results_dir"])
    best = json.loads((out / "metu" / BEST_PARAMS_FILE).read_text())
    cfg["detect"].update(best)
    val, ev = split_metu(cfg, 400)
    load = lambda items: [(load_image(p), y) for p, y in items]  # noqa: E731
    val_imgs, ev_imgs = load(val), load(ev)
    yv = np.array([y for _, y in val_imgs], bool)
    ye = np.array([y for _, y in ev_imgs], bool)
    print(f"validation {len(val_imgs)} | evaluation {len(ev_imgs)} | tuned {best}")

    # tune Canny thresholds on validation (Otsu has no free parameter)
    best_canny, best_f1 = (50, 150), -1.0
    for low in (20, 30, 40, 60, 80, 100, 130):
        for ratio in (2, 3):
            f1 = score(run(make_canny_baseline(cfg, low, low * ratio), val_imgs), yv)["f1"]
            print(f"canny low={low} high={low * ratio}: val F1={f1:.3f}", flush=True)
            if f1 > best_f1:
                best_canny, best_f1 = (low, low * ratio), f1
    print("best Canny", best_canny, best_f1)

    methods: Dict[str, Predictor] = {
        "Proposed (black-hat)": make_proposed(cfg),
        "Global Otsu": make_otsu_baseline(cfg),
        f"Canny ({best_canny[0]}/{best_canny[1]})": make_canny_baseline(cfg, *best_canny),
    }
    preds: Dict[str, np.ndarray] = {}
    rows = []
    for name, fn in methods.items():
        t0 = time.perf_counter()
        preds[name] = run(fn, ev_imgs)
        ms = (time.perf_counter() - t0) / len(ev_imgs) * 1000
        rows.append({"method": name, **score(preds[name], ye), "ms_per_image": ms})
    ref = preds["Proposed (black-hat)"]
    for r in rows:
        p = preds[r["method"]]
        r["mcnemar_p_vs_proposed"] = (np.nan if p is ref else
                                      mcnemar_exact(int(((ref == ye) & (p != ye)).sum()),
                                                    int(((ref != ye) & (p == ye)).sum())))
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(out / "comparison.csv", index=False)
    print(comp[["method", "accuracy", "precision", "recall", "f1", "ms_per_image"]].round(3).to_string(index=False))

    # ablations: remove one component at a time from the tuned proposed method
    abl = {"Full method": {}, "- CLAHE": {"clahe": False}, "- median blur": {"blur": False},
           "- morphological closing": {"closing": False}, "- stain/elongation filter": {"comp_filter": False},
           "- Otsu (fixed floor only)": {"otsu": False}}
    arows = []
    for name, kw in abl.items():
        pr = run(make_proposed(cfg, **kw), ev_imgs)
        arows.append({"variant": name, **score(pr, ye)})
    k15 = get_config()
    k15["detect"].update({"kernel_size": 15, "min_threshold": best["min_threshold"]})
    arows.append({"variant": "kernel 15 instead of tuned", **score(run(make_proposed(k15), ev_imgs), ye)})
    ab = pd.DataFrame(arows)
    ab.round(4).to_csv(out / "ablation.csv", index=False)
    print(ab[["variant", "accuracy", "precision", "recall", "f1"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
