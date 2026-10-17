"""Command-line interface for the Concrete Crack Detector.

Examples:
    python main.py --image wall.jpg --mm-per-pixel 0.2
    python main.py --folder photos/
    python main.py --evaluate metu
    python main.py --evaluate cfd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from config import get_config
from src.evaluate import run_evaluation
from src.measure import pick_scale_interactively
from src.pipeline import PipelineResult, process_image, results_to_csv
from src.preprocess import load_image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classical (no-ML) concrete crack detector and measurer.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="path to one image")
    src.add_argument("--folder", type=Path, help="folder of images (batch mode)")
    src.add_argument("--evaluate", choices=["cfd", "metu"], help="evaluate on a dataset")
    p.add_argument("--mm-per-pixel", type=float, default=None, help="scale; omit to report pixels")
    p.add_argument("--calibrate", action="store_true",
                   help="click two points on a reference object to compute the scale (needs --ref-mm)")
    p.add_argument("--ref-mm", type=float, default=None, help="real length of the reference object in mm")
    p.add_argument("--kernel", type=int, default=None, help="black-hat kernel size (default from config)")
    p.add_argument("--min-area", type=int, default=None, help="minimum component area in px")
    p.add_argument("--threshold", choices=["otsu", "adaptive", "fixed"], default=None)
    p.add_argument("--results-dir", type=Path, default=None, help="output folder (default: results/)")
    p.add_argument("--sample-size", type=int, default=None, help="images sampled for --evaluate metu")
    return p


def apply_cli_overrides(args: argparse.Namespace) -> dict:
    """Build the config, applying any command-line overrides."""
    detect = {}
    if args.kernel is not None:
        detect["kernel_size"] = args.kernel
    if args.min_area is not None:
        detect["min_area"] = args.min_area
    if args.threshold is not None:
        detect["threshold_method"] = args.threshold
    overrides = {"detect": detect}
    if args.results_dir is not None:
        overrides["paths"] = {"results_dir": args.results_dir}
    if args.sample_size is not None:
        overrides["evaluate"] = {"metu_sample_size": args.sample_size}
    return get_config(overrides)


def report(result: PipelineResult) -> None:
    """Print a human-readable summary for one image."""
    if not result.ok:
        print(f"[SKIP] {result.message}")
        return
    if result.n_cracks == 0:
        print(f"[{result.name}] No cracks found.")
        return
    note = f"  ({result.message})" if result.message else ""
    print(f"[{result.name}] {result.n_cracks} crack(s) found{note}")
    for _, r in result.table.iterrows():
        u = r["unit"]
        print(f"   #{int(r['crack_id'])}: length={r['length']:.1f}{u}  "
              f"mean width={r['mean_width']:.2f}{u}  max width={r['max_width']:.2f}{u}")


def list_images(folder: Path, extensions: List[str]) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)


def resolve_scale(args: argparse.Namespace, cfg: dict) -> Optional[float]:
    """Return mm-per-pixel (original resolution) from --mm-per-pixel or --calibrate."""
    if args.calibrate:
        if args.image is None or args.ref_mm is None:
            sys.exit("--calibrate needs --image and --ref-mm (real length of the reference object).")
        img = load_image(args.image)
        if img is None:
            sys.exit(f"Could not read image: {args.image}")
        scale = pick_scale_interactively(img, args.ref_mm)
        print(f"Calibrated scale: {scale:.4f} mm/pixel")
        return scale
    return args.mm_per_pixel


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_cli_overrides(args)
    out_dir: Path = cfg["paths"]["results_dir"]

    if args.evaluate:
        return run_evaluation(args.evaluate, cfg)

    mm = resolve_scale(args, cfg)

    if args.image:
        return run_paths([args.image], cfg, mm, out_dir, csv_name=f"{args.image.stem}_cracks.csv")

    if not args.folder.is_dir():
        print(f"Folder not found: {args.folder}")
        return 1
    paths = list_images(args.folder, cfg["image_extensions"])
    if not paths:
        print(f"No images found in {args.folder}")
        return 1
    return run_paths(paths, cfg, mm, out_dir, csv_name="measurements.csv")


def run_paths(paths: List[Path], cfg: dict, mm: Optional[float], out_dir: Path, csv_name: str) -> int:
    """Process each path, print summaries and save the CSV."""
    results = []
    for path in paths:
        res = process_image(path, cfg, mm, out_dir)
        report(res)
        results.append(res)
    df = results_to_csv(results, out_dir / csv_name)
    print(f"\nSaved {len(df)} crack row(s) to {out_dir / csv_name}")
    print(f"Overlays and figures saved in {out_dir}")
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
