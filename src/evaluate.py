"""Evaluation on the METU / Ozgenel dataset (image-level cracked vs clean)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.detect import detect_cracks  # noqa: E402
from src.measure import measure_cracks  # noqa: E402
from src.preprocess import load_image, preprocess  # noqa: E402

CLASS_NAMES = {1: "cracked", 0: "clean"}
BEST_PARAMS_FILE = "best_params.json"


# ---------------------------------------------------------------- data
def list_metu(cfg: Dict[str, Any]) -> List[Tuple[Path, int]]:
    """All METU images as (path, label); label 1 = Positive (cracked), 0 = Negative (clean)."""
    root = Path(cfg["paths"]["data_dir"]) / cfg["paths"]["metu_subdir"]
    items: List[Tuple[Path, int]] = []
    for folder, label in (("Positive", 1), ("Negative", 0)):
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"METU folder not found: {d} (see data/README.md)")
        items += [(p, label) for p in sorted(d.iterdir()) if p.suffix.lower() in cfg["image_extensions"]]
    return items


def split_metu(cfg: Dict[str, Any], val_size: int) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Reproducible (validation, evaluation) split with NO image in both.

    The validation set is used only for tuning; the evaluation sample comes from the rest.
    """
    items = list_metu(cfg)
    rng = random.Random(cfg["evaluate"]["random_seed"])
    rng.shuffle(items)
    val, rest = items[:val_size], items[val_size:]
    return val, rest[: cfg["evaluate"]["metu_sample_size"]]


# ---------------------------------------------------------------- prediction
def predict_image(image: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, int, float]:
    """Return (is_cracked, n_cracks, longest_crack_px). Cracked = any crack >= min length."""
    mask = detect_cracks(preprocess(image, cfg), cfg)
    table, _, _ = measure_cracks(mask, None, cfg)
    longest = float(table["length"].max()) if len(table) else 0.0
    return longest >= cfg["evaluate"]["min_crack_length_metu"], len(table), longest


def predict_many(items: Sequence[Tuple[Path, int]], cfg: Dict[str, Any]) -> pd.DataFrame:
    """Run the detector on every (path, label); unreadable files are skipped with a warning."""
    rows = []
    for path, label in items:
        img = load_image(path)
        if img is None:
            print(f"[SKIP] unreadable: {path}")
            continue
        pred, n, longest = predict_image(img, cfg)
        rows.append({"image": f"{path.parent.name}/{path.name}", "path": str(path),
                     "actual": CLASS_NAMES[label], "predicted": CLASS_NAMES[int(pred)],
                     "correct": bool(pred) == bool(label), "cracks_found": n,
                     "longest_crack_px": round(longest, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- metrics
def confusion_counts(df: pd.DataFrame) -> Dict[str, int]:
    """TP/FP/TN/FN with 'cracked' as the positive class."""
    a, p = df["actual"] == "cracked", df["predicted"] == "cracked"
    return {"TP": int((a & p).sum()), "FP": int((~a & p).sum()),
            "TN": int((~a & ~p).sum()), "FN": int((a & ~p).sum())}


def classification_metrics(c: Dict[str, int]) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 from a confusion-count dict."""
    total = sum(c.values())
    prec = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
    rec = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (c["TP"] + c["TN"]) / total if total else 0.0,
            "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------- outputs
def contact_sheets(df: pd.DataFrame, out_prefix: Path, title: str, cols: int = 8, rows: int = 5) -> List[Path]:
    """Thumbnail pages; each tile shows actual vs predicted (red border = wrong, green = right)."""
    per_page = cols * rows
    paths = []
    for page, start in enumerate(range(0, len(df), per_page), start=1):
        chunk = df.iloc[start:start + per_page]
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), chunk.iterrows()):
            img = load_image(r["path"])
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            colour = "green" if r["correct"] else "red"
            for s in ax.spines.values():
                s.set_edgecolor(colour)
                s.set_linewidth(3)
            ax.set_visible(True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r['image']}\nactual: {r['actual']}\npredicted: {r['predicted']}",
                         fontsize=7, color=colour)
        n_pages = -(-len(df) // per_page)
        fig.suptitle(f"{title}  (page {page}/{n_pages})", fontsize=12)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.08, hspace=0.55)
        path = out_prefix.with_name(f"{out_prefix.name}_p{page:02d}.png")
        fig.savefig(path, dpi=80)
        plt.close(fig)
        paths.append(path)
    return paths


def summary_plot(c: Dict[str, int], m: Dict[str, float], path: Path, n: int) -> None:
    """Confusion matrix + metric bars in one figure."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cm = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    a1.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a1.text(j, i, str(v), ha="center", va="center", fontsize=18,
                color="white" if v > cm.max() / 2 else "black")
    a1.set_xticks([0, 1], ["clean", "cracked"]); a1.set_yticks([0, 1], ["clean", "cracked"])
    a1.set_xlabel("Predicted"); a1.set_ylabel("Actual"); a1.set_title("Confusion matrix")
    names = list(m)
    bars = a2.bar(names, [m[k] for k in names], color="#4c78a8")
    a2.set_ylim(0, 1.05); a2.set_title(f"Metrics ({n} images)")
    for b, k in zip(bars, names):
        a2.text(b.get_x() + b.get_width() / 2, m[k] + 0.01, f"{m[k]:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- entry points
def load_tuned_config(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Apply results/metu/best_params.json (from tune.py) to cfg if it exists."""
    f = out_dir / BEST_PARAMS_FILE
    if f.is_file():
        best = json.loads(f.read_text())
        cfg["detect"].update(best)
        print(f"Using tuned parameters from {f}: {best}")
    return cfg


def run_metu(cfg: Dict[str, Any], val_size: int = 400) -> int:
    """Evaluate on the METU sample, write predictions.csv, metrics.csv and the pictures."""
    out = Path(cfg["paths"]["results_dir"]) / "metu"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_tuned_config(cfg, out)
    _, sample = split_metu(cfg, val_size)
    print(f"Evaluating {len(sample)} METU images ...")
    df = predict_many(sample, cfg)

    # wrong first (false alarms, then misses), then right ones (cracked, then clean)
    order = {("clean", "cracked"): 0, ("cracked", "clean"): 1, ("cracked", "cracked"): 2, ("clean", "clean"): 3}
    df["_o"] = [order[(a, p)] for a, p in zip(df["actual"], df["predicted"])]
    df = df.sort_values(["_o", "image"]).drop(columns="_o").reset_index(drop=True)
    df.drop(columns="path").to_csv(out / "predictions.csv", index=False)

    c = confusion_counts(df)
    m = classification_metrics(c)
    n_clean, n_cracked = int((df.actual == "clean").sum()), int((df.actual == "cracked").sum())
    pd.DataFrame(
        [("images_evaluated", len(df)), ("actual_cracked", n_cracked), ("actual_clean", n_clean)]
        + list(c.items()) + [(k, round(v, 4)) for k, v in m.items()]
        + [("kernel_size", cfg["detect"]["kernel_size"]), ("min_threshold", cfg["detect"]["min_threshold"])],
        columns=["metric", "value"]).to_csv(Path(cfg["paths"]["results_dir"]) / "metrics.csv", index=False)

    summary_plot(c, m, out / "summary.png", len(df))
    wrong, right = df[~df["correct"]], df[df["correct"]]
    for old in out.glob("*_p[0-9][0-9].png"):
        old.unlink()
    contact_sheets(wrong, out / "1_wrong", "WRONG predictions (false alarms first, then misses)")
    contact_sheets(right[right.actual == "cracked"].head(40), out / "2_right_cracked", "CORRECT: cracked images")
    contact_sheets(right[right.actual == "clean"].head(40), out / "3_right_clean", "CORRECT: clean images")

    print(f"\nActual:  {n_cracked} cracked, {n_clean} clean")
    print(f"Confusion: TP={c['TP']} FN={c['FN']} FP={c['FP']} TN={c['TN']}")
    print("  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print(f"Results in {out}")
    return 0


def run_evaluation(name: str, cfg: Dict[str, Any]) -> int:
    """Dispatch for ``main.py --evaluate``."""
    if name == "metu":
        return run_metu(cfg)
    print("CrackForest (cfd) evaluation is not implemented yet: dataset not downloaded.")
    return 1
