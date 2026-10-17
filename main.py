"""Command-line interface for the Concrete Crack Detector.

    python main.py --image wall.jpg [--mm-per-pixel 0.2] [--kernel 15] [--floor 60] [--min-area 50]
    python main.py --folder photos/
    python main.py --image wall.jpg --calibrate --ref-mm 100     (click a reference object)
    python main.py --evaluate metu                               (also: cfd, not implemented)
    python main.py --tune                                        (grid search on validation images)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from config import get_config
from src.evaluate import run_evaluation, tune
from src.measure import pick_scale_interactively
from src.pipeline import PipelineResult, measurements_to_csv, process_image, save_steps_figure
from src.preprocess import load_image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classical (no-ML) concrete crack detector and measurer.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="path to one image")
    mode.add_argument("--folder", type=Path, help="folder of images (batch mode)")
    mode.add_argument("--evaluate", choices=["cfd", "metu"], help="evaluate on a dataset")
    mode.add_argument("--tune", action="store_true", help="grid-search kernel size and threshold floor")
    p.add_argument("--mm-per-pixel", type=float, help="scale; omit to report pixels")
    p.add_argument("--calibrate", action="store_true", help="click two points on a reference object (needs --ref-mm)")
    p.add_argument("--ref-mm", type=float, help="real length of the reference object in mm")
    p.add_argument("--kernel", type=int, help="black-hat kernel size")
    p.add_argument("--floor", type=int, help="black-hat contrast floor for thresholding")
    p.add_argument("--min-area", type=int, help="minimum component area in px")
    p.add_argument("--results-dir", type=Path, help="output folder (default: results/)")
    return p


def report(res: PipelineResult) -> None:
    """Print a human-readable summary for one image."""
    if not res.ok:
        print(f"[SKIP] {res.message}")
    elif res.n_cracks == 0:
        print(f"[{res.name}] No cracks found.")
    else:
        print(f"[{res.name}] {res.n_cracks} crack(s) found {res.message}".rstrip())
        for _, r in res.table.iterrows():
            u = r["unit"]
            print(f"   #{int(r['crack_id'])}: length={r['length']:.1f}{u}  "
                  f"mean width={r['mean_width']:.2f}{u}  max width={r['max_width']:.2f}{u}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = get_config()
    for key, section, name in ((args.kernel, "detect", "kernel_size"), (args.floor, "detect", "min_threshold"),
                               (args.min_area, "detect", "min_area"), (args.results_dir, "paths", "results_dir")):
        if key is not None:
            cfg[section][name] = key
    out_dir = Path(cfg["paths"]["results_dir"])

    if args.evaluate:
        return run_evaluation(args.evaluate, cfg)
    if args.tune:
        tune(cfg)
        return 0

    if args.image:
        paths = [args.image]
    elif args.folder.is_dir():
        paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in cfg["image_extensions"])
    else:
        paths = []
    if not paths:
        print("No images to process.")
        return 1

    mm = args.mm_per_pixel
    if args.calibrate:
        img = load_image(paths[0])
        if img is None or args.ref_mm is None:
            sys.exit("--calibrate needs a readable --image and --ref-mm.")
        mm = pick_scale_interactively(img, args.ref_mm)
        print(f"Calibrated scale: {mm:.4f} mm/pixel")

    results = []
    for path in paths:
        res = process_image(path, cfg, mm)
        report(res)
        if res.ok:
            save_steps_figure(res, out_dir / f"{path.stem}_steps.png", path.name, cfg)
        results.append(res)
    df = measurements_to_csv(results, out_dir / "measurements.csv")
    print(f"\n{len(df)} crack row(s) saved to {out_dir / 'measurements.csv'}; step figures in {out_dir}")
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
