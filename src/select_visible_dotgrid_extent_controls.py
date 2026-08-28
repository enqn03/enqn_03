#!/usr/bin/env python3
"""Collect four human-reviewed outer dot centers on the immutable DotGrid image.

Click exactly four **visible outer dot centers** in this order:
1. top-left, 2. top-right, 3. bottom-right, 4. bottom-left.

The preview is downsampled only for display; click positions are converted back to
raw camera pixels before the compact JSON is written. This selector never alters
raw TIFFs, grid-size/coverage policy, calibration, transform rank/orientation,
model/target data, or quality candidate reporting.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from audit_independent_metrology_fiducials_refined import grayscale, read_tiff


POINT_ORDER = ["top_left_outer_dot_center", "top_right_outer_dot_center", "bottom_right_outer_dot_center", "bottom_left_outer_dot_center"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid TIFF; opened via tifffile.memmap(mode='r').")
    parser.add_argument("--output-json", required=True, type=Path, help="New compact ignored JSON for four human-review control clicks.")
    parser.add_argument("--preview-max-side", type=int, default=1000, help="Preview maximum side in display pixels; raw positions are preserved. Default: 1000.")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace only an existing controls JSON after review.")
    return parser.parse_args()


def prepare_output_path(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output control JSON already exists: {path}. Review it or use --overwrite deliberately.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def preview_stride(shape: tuple[int, int], max_side: int) -> int:
    if max_side <= 0:
        raise ValueError("--preview-max-side must be positive.")
    return max(1, int(np.ceil(max(shape) / float(max_side))))


def collect_points(gray: np.ndarray, stride: int) -> list[tuple[float, float]]:
    display = gray[::stride, ::stride]
    figure, axis = plt.subplots(figsize=(10, 10), dpi=120)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    axis.set_title("Click visible OUTER dot centers in order: 1 TL, 2 TR, 3 BR, 4 BL\nPreview only; raw camera pixels are saved")
    axis.set_xlabel("display x")
    axis.set_ylabel("display y")
    figure.tight_layout()
    print("\nClick exactly four visible OUTER dot centers in this order:")
    for index, name in enumerate(POINT_ORDER, start=1):
        print(f"  {index}. {name}")
    print("Close the window only after four points are visible. Press Esc to cancel.")
    points = plt.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    plt.close(figure)
    if len(points) != 4:
        raise RuntimeError(f"Expected exactly four control points, received {len(points)}. No JSON was written.")
    return [(float(x), float(y)) for x, y in points]


def main() -> None:
    args = parse_args()
    if not args.dot_grid.is_file():
        raise FileNotFoundError(f"Required immutable DotGrid TIFF not found: {args.dot_grid}")
    prepare_output_path(args.output_json, args.overwrite)
    channels, metadata = read_tiff(args.dot_grid)
    gray = grayscale(channels)
    stride = preview_stride(gray.shape, args.preview_max_side)
    clicked_display = collect_points(gray, stride)
    controls: list[dict[str, Any]] = []
    for index, (name, (display_x, display_y)) in enumerate(zip(POINT_ORDER, clicked_display), start=1):
        raw_x = float(display_x * stride)
        raw_y = float(display_y * stride)
        controls.append({
            "selection_order": index,
            "semantic_name": name,
            "display_preview_xy_px": [display_x, display_y],
            "raw_camera_xy_px": [raw_x, raw_y],
        })
    payload = {
        "audit_type": "human-reviewed visible DotGrid outer-extent controls; not a calibration control-point set",
        "purpose": "Record only four visible outer dot centers for physical-panel extent validation against V3 image-lattice coverage.",
        "input": metadata,
        "preview": {
            "max_side_px": int(args.preview_max_side),
            "downsample_stride": int(stride),
            "display_shape_yx": [int(gray[::stride, ::stride].shape[0]), int(gray[::stride, ::stride].shape[1])],
            "important_limit": "Preview downsampling is display-only. Raw coordinates save display coordinate times stride; validation later snaps only after explicit tolerance checks.",
        },
        "required_click_order": POINT_ORDER,
        "controls": controls,
        "prohibitions": [
            "Does not write raw TIFF/CSV.",
            "Does not change nominal grid size, coverage gate, calibration_v1.yaml, transform/rank/orientation, or machine origin.",
            "Does not access A/B TIFF, XCT, weak target/support, model, checkpoint, training, decoder, or candidate output.",
            "Does not change camera-primary XCT-derived continuous quality candidate reporting.",
        ],
        "next_required_validation": "Run audit_visible_dotgrid_extent_controls.py; do not treat raw click coordinates as a calibration or policy decision.",
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Visible DotGrid extent controls recorded. No raw TIFF/CSV, calibration, coverage gate, model, target, checkpoint, decoder, or candidate output was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
