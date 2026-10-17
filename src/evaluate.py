"""METU evaluation: parameter tuning, baseline comparison and result files.

Cracked is the positive class. An image counts as "cracked" if it contains at least one
crack at least ``measure.min_crack_length`` pixels long.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.detect import blackhat, cleanup, detect_cracks, threshold_blackhat  # noqa: E402
from src.measure import measure_cracks  # noqa: E402
from src.pipeline import run_steps, save_steps_figure  # noqa: E402
from src.preprocess import denoise, load_image, preprocess, to_grayscale  # noqa: E402

CLASS = {1: "cracked", 0: "clean"}
METHODS = {"predicted": "Proposed (black-hat)", "otsu_predicted": "Global Otsu", "canny_predicted": "Canny"}
TUNE_KERNELS = [15, 21, 25, 31]
TUNE_FLOORS = [50, 60, 70, 80, 90, 100, 110, 120]
TUNE_CANNY_LOW = [20, 30, 40, 60, 80, 100, 130]


# ---------------------------------------------------------------- data
def split_metu(cfg: Dict[str, Any]) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Reproducible (validation, evaluation) split of METU with no image in both sets."""
    root = Path(cfg["paths"]["data_dir"]) / cfg["paths"]["metu_subdir"]
    items: List[Tuple[Path, int]] = []
    for folder, label in (("Positive", 1), ("Negative", 0)):
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"METU folder not found: {d} (see data/README.md)")
        items += [(p, label) for p in sorted(d.iterdir()) if p.suffix.lower() in cfg["image_extensions"]]
    e = cfg["evaluate"]
    random.Random(e["random_seed"]).shuffle(items)
    return items[:e["val_size"]], items[e["val_size"]:e["val_size"] + e["sample_size"]]


# ---------------------------------------------------------------- prediction
def is_cracked(mask: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, int, float]:
    """Return (cracked?, number of cracks, longest crack in px) for a cleaned mask."""
    table, _, _ = measure_cracks(mask, None, cfg)
    longest = float(table["length"].max()) if len(table) else 0.0
    return longest >= cfg["measure"]["min_crack_length"], len(table), longest


def baseline_masks(image: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline detection masks: global Otsu (dark = crack) and Canny edges, then shared cleanup."""
    p, d = cfg["preprocess"], cfg["detect"]
    gray = denoise(to_grayscale(image), p["blur"], p["blur_ksize"])
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    low, high = cfg["metu_tuned"]["canny_thresholds"]
    return cleanup(otsu, d), cleanup(cv2.Canny(gray, low, high), d)


def predict_all(items: List[Tuple[Path, int]], cfg: Dict[str, Any]) -> pd.DataFrame:
    """Classify every image with the proposed method and both baselines."""
    rows = []
    for path, label in items:
        img = load_image(path)
        if img is None:
            print(f"[SKIP] unreadable: {path}")
            continue
        pred, n, longest = is_cracked(detect_cracks(preprocess(img, cfg)["blur"], cfg)["mask"], cfg)
        otsu_mask, canny_mask = baseline_masks(img, cfg)
        rows.append({"image": f"{path.parent.name}/{path.name}", "actual": CLASS[label],
                     "predicted": CLASS[int(pred)], "correct": pred == bool(label),
                     "cracks_found": n, "longest_crack_px": round(longest, 1),
                     "otsu_predicted": CLASS[int(is_cracked(otsu_mask, cfg)[0])],
                     "canny_predicted": CLASS[int(is_cracked(canny_mask, cfg)[0])]})
    return pd.DataFrame(rows)


def scores(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """Confusion counts and accuracy/precision/recall/F1 for one prediction column."""
    a, p = df["actual"] == "cracked", df[col] == "cracked"
    tp, fp, tn, fn = int((a & p).sum()), int((~a & p).sum()), int((~a & ~p).sum()), int((a & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "accuracy": (tp + tn) / len(df),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


# ---------------------------------------------------------------- tuning
def tune(cfg: Dict[str, Any]) -> None:
    """Grid-search kernel size and contrast floor (and the Canny baseline) on the validation images."""
    val, _ = split_metu(cfg)
    imgs = [(load_image(p), y) for p, y in val]
    d, min_len = cfg["detect"], cfg["measure"]["min_crack_length"]
    blurred = [(preprocess(i, cfg)["blur"], i, y) for i, y in imgs]
    print(f"Tuning on {len(imgs)} validation images")

    def f1_of(preds: List[bool]) -> float:
        df = pd.DataFrame({"actual": [CLASS[y] for _, y in imgs], "p": [CLASS[int(x)] for x in preds]})
        return scores(df, "p")["f1"]

    best = (-1.0, 0, 0)
    for k in TUNE_KERNELS:
        bhs = [blackhat(b, k) for b, _, _ in blurred]
        for floor in TUNE_FLOORS:
            dd = {**d, "kernel_size": k, "min_threshold": floor}
            preds = [is_cracked(cleanup(threshold_blackhat(bh, dd), dd), cfg)[0] for bh in bhs]
            f1 = f1_of(preds)
            print(f"  black-hat kernel={k:2d} floor={floor:3d}  F1={f1:.3f}")
            best = max(best, (f1, k, floor))
    print(f"Best black-hat: kernel={best[1]} floor={best[2]} (F1={best[0]:.3f})")

    p = cfg["preprocess"]
    grays = [denoise(to_grayscale(i), p["blur"], p["blur_ksize"]) for i, _ in imgs]
    best_c = (-1.0, 0, 0)
    for low in TUNE_CANNY_LOW:
        for ratio in (2, 3):
            preds = [is_cracked(cleanup(cv2.Canny(g, low, low * ratio), d), cfg)[0] for g in grays]
            best_c = max(best_c, (f1_of(preds), low, low * ratio))
    print(f"Best Canny thresholds: {best_c[1]}/{best_c[2]} (F1={best_c[0]:.3f})")
    print("Copy the best values into 'metu_tuned' in config.py.")


# ---------------------------------------------------------------- outputs
def contact_sheet(df: pd.DataFrame, root: Path, path: Path, title: str, cols: int = 8, rows: int = 5) -> None:
    """Thumbnail page; each tile shows actual vs predicted (red border = wrong, green = right)."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        ax.imshow(cv2.cvtColor(load_image(root / r["image"]), cv2.COLOR_BGR2RGB))
        colour = "green" if r["correct"] else "red"
        for s in ax.spines.values():
            s.set_edgecolor(colour)
            s.set_linewidth(3)
        ax.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{r['image']}\nactual: {r['actual']}\npredicted: {r['predicted']}", fontsize=7, color=colour)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.08, hspace=0.55)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def summary_plot(df: pd.DataFrame, path: Path) -> None:
    """Confusion matrix of the proposed method + metric comparison with the two baselines."""
    s = scores(df, "predicted")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    cm = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]])
    a1.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a1.text(j, i, str(v), ha="center", va="center", fontsize=18, color="white" if v > cm.max() / 2 else "black")
    a1.set_xticks([0, 1], ["clean", "cracked"])
    a1.set_yticks([0, 1], ["clean", "cracked"])
    a1.set_xlabel("Predicted")
    a1.set_ylabel("Actual")
    a1.set_title("Confusion matrix (proposed method)")
    metrics, w = ["accuracy", "precision", "recall", "f1"], 0.26
    for n, (col, name) in enumerate(METHODS.items()):
        m = scores(df, col)
        bars = a2.bar(np.arange(4) + (n - 1) * w, [m[k] for k in metrics], w, label=name)
        for b, k in zip(bars, metrics):
            a2.text(b.get_x() + b.get_width() / 2, m[k] + 0.01, f"{m[k]:.2f}", ha="center", fontsize=7)
    a2.set_xticks(range(4), metrics)
    a2.set_ylim(0, 1.12)
    a2.set_title(f"Method comparison ({len(df)} images)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_metu(cfg: Dict[str, Any]) -> int:
    """Evaluate on the METU sample and write results.csv plus a few figures to results/."""
    cfg["detect"].update(cfg["metu_tuned"]["detect"])
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["paths"]["data_dir"]) / cfg["paths"]["metu_subdir"]
    _, sample = split_metu(cfg)
    print(f"Evaluating {len(sample)} METU images (kernel {cfg['detect']['kernel_size']}, "
          f"floor {cfg['detect']['min_threshold']}) ...")
    df = predict_all(sample, cfg)

    # wrong first (false alarms, then misses), then the right ones
    rank = {("clean", "cracked"): 0, ("cracked", "clean"): 1, ("cracked", "cracked"): 2, ("clean", "clean"): 3}
    df["_r"] = [rank[(a, p)] for a, p in zip(df["actual"], df["predicted"])]
    df = df.sort_values(["_r", "image"]).drop(columns="_r").reset_index(drop=True)
    df.to_csv(out / "results.csv", index=False)

    wrong, right = df[~df["correct"]], df[df["correct"]]
    for old in out.glob("wrong_predictions_p*.png"):
        old.unlink()
    per_page = 40
    for n, start in enumerate(range(0, len(wrong), per_page), start=1):
        contact_sheet(wrong.iloc[start:start + per_page], root, out / f"wrong_predictions_p{n}.png",
                      f"WRONG predictions: false alarms first, then misses (page {n})")
    ok = pd.concat([right[right.actual == "cracked"].head(20), right[right.actual == "clean"].head(20)])
    contact_sheet(ok, root, out / "correct_examples.png", "CORRECT predictions: 20 cracked, then 20 clean")
    summary_plot(df, out / "summary.png")

    for label, name, title in (("cracked", "pipeline_cracked.png", "Cracked concrete"),
                               ("clean", "pipeline_clean.png", "Clean concrete")):
        row = right[right.actual == label].iloc[0]
        res = run_steps(load_image(root / row["image"]), cfg)
        save_steps_figure(res, out / name, f"{title} (METU {row['image']}, correctly classified)", cfg)

    print(f"\nActual: {(df.actual == 'cracked').sum()} cracked, {(df.actual == 'clean').sum()} clean")
    for col, name in METHODS.items():
        s = scores(df, col)
        print(f"{name:22s} TP={s['TP']} FN={s['FN']} FP={s['FP']} TN={s['TN']}  acc={s['accuracy']:.3f} "
              f"prec={s['precision']:.3f} rec={s['recall']:.3f} F1={s['f1']:.3f}")
    print(f"Results written to {out}")
    return 0


def run_evaluation(name: str, cfg: Dict[str, Any]) -> int:
    """Dispatch for ``main.py --evaluate``."""
    if name == "metu":
        return run_metu(cfg)
    print("CrackForest (cfd) evaluation is not implemented.")
    return 1
