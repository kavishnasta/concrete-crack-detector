"""Grid-search the black-hat kernel size and threshold floor on a validation subset.

The validation images are disjoint from the evaluation sample (see split_metu).
Writes results/metu/tuning.csv and results/metu/best_params.json, which
``python main.py --evaluate metu`` then uses automatically.

    python tune.py [--val-size 400]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from config import get_config
from src.detect import blackhat, close_mask, filter_components, threshold_blackhat
from src.evaluate import BEST_PARAMS_FILE, classification_metrics, split_metu
from src.measure import measure_cracks
from src.preprocess import load_image, preprocess

KERNELS = [15, 21, 25, 31]
FLOORS = [50, 60, 70, 80, 90, 100, 110, 120]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-size", type=int, default=400)
    args = ap.parse_args()
    cfg = get_config()
    val, _ = split_metu(cfg, args.val_size)
    d = cfg["detect"]
    min_len = cfg["evaluate"]["min_crack_length_metu"]

    grays = [(preprocess(load_image(p), cfg), y) for p, y in val if load_image(p) is not None]
    print(f"Tuning on {len(grays)} validation images")
    rows = []
    for k in KERNELS:
        bhs = [(blackhat(g, k), y) for g, y in grays]
        for floor in FLOORS:
            dd = {**d, "kernel_size": k, "min_threshold": floor}
            c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
            for bh, y in bhs:
                mask = filter_components(close_mask(threshold_blackhat(bh, dd), d["close_kernel"],
                                                    d["close_iterations"]), dd)
                t, _, _ = measure_cracks(mask, None, cfg)
                pred = len(t) > 0 and t["length"].max() >= min_len
                c[("T" if pred == bool(y) else "F") + ("P" if pred else "N")] += 1
            rows.append({"kernel_size": k, "min_threshold": floor, **c, **classification_metrics(c)})
            print(f"kernel={k:2d} floor={floor}  F1={rows[-1]['f1']:.3f}  "
                  f"recall={rows[-1]['recall']:.3f}  precision={rows[-1]['precision']:.3f}", flush=True)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out = Path(cfg["paths"]["results_dir"]) / "metu"
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "tuning.csv", index=False)
    best = df.iloc[0]
    (out / BEST_PARAMS_FILE).write_text(json.dumps(
        {"kernel_size": int(best.kernel_size), "min_threshold": int(best.min_threshold)}))
    print(f"\nBest: kernel={int(best.kernel_size)} floor={int(best.min_threshold)} (F1={best.f1:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
