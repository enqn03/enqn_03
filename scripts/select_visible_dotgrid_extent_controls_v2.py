#!/usr/bin/env python3
"""Collect four human-reviewed outer DotGrid dot centres using a macOS GUI backend.

This V2 selector is intentionally independent of batch-QC modules that configure
Matplotlib ``Agg``. Before importing ``matplotlib.pyplot``, it selects the
``MacOSX`` backend and falls back to ``TkAgg`` only if ``MacOSX`` is unavailable.
It reads the immutable TIFF directly via ``tifffile.memmap(..., mode='r')`` and
writes a compact four-click control JSON only after all points are collected.

Click exactly four visible outer dot centres in visual order: top-left,
top-right, bottom-right, bottom-left. The click coordinates are converted from
downsampled display pixels to raw camera pixels. This is visible-panel evidence,
not calibration control points or a grid/gate/config decision.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import tifffile


POINT_ORDER = ["top_left_outer_dot_center", "top_right_outer_dot_center", "bottom_right_outer_dot_center", "bottom_left_outer_dot_center"]


def load_gui_pyplot() -> tuple[Any, str, list[str]]:
    """Select a GUI backend before importing pyplot; never accept Agg for clicks."""
    attempted: list[str] = []
    last_error: Exception | None = None
    for backend in ("MacOSX", "TkAgg"):
        try:
            attempted.append(backend)
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as pyplot  # Imported only after GUI backend selection.
            selected = str(matplotlib.get_backend())
            if selected.lower() == "agg":
                raise RuntimeError("Matplotlib selected noninteractive Agg despite GUI backend request.")
            return pyplot, selected, attempted
        except Exception as error:
            last_error = error
            # pyplot is not imported on a failed backend selection path; retry fallback safely.
            continue
    raise RuntimeError(
        "No interactive Matplotlib backend is available. Tried "
        f"{attempted}. Install/enable the macOS Python GUI backend or Tk, then rerun. "
        f"Last error: {last_error}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid TIFF opened with tifffile.memmap(..., series=0, mode='r').")
    parser.add_argument("--output-json", required=True, type=Path, help="New compact ignored JSON for exactly four human review controls.")
    parser.add_argument("--preview-max-side", type=int, default=1000, help="Maximum display preview side in pixels. Default: 1000.")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace only the specified existing control JSON after review.")
    return parser.parse_args()


def verify_output_path_before_selection(path: Path, overwrite: bool) -> None:
    """Fail before opening GUI if output needs review; never delete or create anything."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output control JSON already exists: {path}. Review it or use --overwrite deliberately.")
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Output path is a directory, not a JSON file: {path}")


def write_controls_json_after_four_valid_clicks(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    """Persist JSON only after successful selection; explicit overwrite is applied here."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output control JSON appeared during selection: {path}. No replacement was made.")
        if path.is_dir():
            raise IsADirectoryError(f"Output path is a directory, not a JSON file: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_grayscale_memmap(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return immutable YX DotGrid pixels; intentionally reject unknown axes."""
    with tifffile.TiffFile(path) as tiff:
        if not tiff.series:
            raise ValueError(f"No TIFF series found: {path}")
        series = tiff.series[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        dtype = str(series.dtype)
        is_imagej = bool(tiff.is_imagej)
    if axes != "YX" or len(shape) != 2:
        raise ValueError(f"V2 selector requires a grayscale YX DotGrid TIFF; observed axes={axes}, shape={shape}.")
    pixels = tifffile.memmap(path, series=0, mode="r")
    if pixels.shape != shape:
        raise RuntimeError(f"Read-only memmap shape mismatch: expected {shape}, got {pixels.shape}.")
    if not np.issubdtype(pixels.dtype, np.number):
        raise ValueError(f"DotGrid pixels must be numeric; observed dtype={pixels.dtype}.")
    metadata = {
        "path": str(path),
        "series_axes": axes,
        "series_shape": list(shape),
        "dtype": dtype,
        "is_imagej": is_imagej,
        "tiff_access": "tifffile.memmap(..., series=0, mode='r')",
        "raw_height_px": int(shape[0]),
        "raw_width_px": int(shape[1]),
    }
    return pixels, metadata


def preview_stride(shape: tuple[int, int], max_side: int) -> int:
    if max_side <= 0:
        raise ValueError("--preview-max-side must be positive.")
    return max(1, int(math.ceil(max(shape) / float(max_side))))


def collect_points(pyplot: Any, gray: np.ndarray, stride: int) -> list[tuple[float, float]]:
    display = gray[::stride, ::stride]
    figure, axis = pyplot.subplots(figsize=(10, 10), dpi=120)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    axis.set_title("Click OUTER visible dot centres: 1 TL, 2 TR, 3 BR, 4 BL\nPreview only; raw camera pixels are saved")
    axis.set_xlabel("display x")
    axis.set_ylabel("display y")
    figure.tight_layout()
    print("\nClick exactly four visible OUTER dot centres in this order:")
    for index, name in enumerate(POINT_ORDER, start=1):
        print(f"  {index}. {name}")
    print("The selector returns automatically after the fourth left-click. Right-click removes the latest point; press Esc to cancel.")
    points = pyplot.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    pyplot.close(figure)
    if len(points) != 4:
        raise RuntimeError(f"Expected exactly four control points, received {len(points)}. No JSON was written.")
    return [(float(x), float(y)) for x, y in points]


def main() -> None:
    args = parse_args()
    if not args.dot_grid.is_file():
        raise FileNotFoundError(f"Required immutable DotGrid TIFF not found: {args.dot_grid}")
    # Select GUI backend before any pyplot import and before altering output paths.
    pyplot, backend, attempted_backends = load_gui_pyplot()
    verify_output_path_before_selection(args.output_json, args.overwrite)
    gray, metadata = read_grayscale_memmap(args.dot_grid)
    stride = preview_stride(gray.shape, args.preview_max_side)
    clicked_display = collect_points(pyplot, gray, stride)
    controls: list[dict[str, Any]] = []
    for index, (name, (display_x, display_y)) in enumerate(zip(POINT_ORDER, clicked_display), start=1):
        controls.append({
            "selection_order": index,
            "semantic_name": name,
            "display_preview_xy_px": [display_x, display_y],
            "raw_camera_xy_px": [float(display_x * stride), float(display_y * stride)],
        })
    payload = {
        "audit_type": "human-reviewed visible DotGrid outer-extent controls V2; interactive GUI selector, not calibration control points",
        "purpose": "Record four visible outer dot centres in fixed visual order for later image-space extent validation against V3 footprint.",
        "input": metadata,
        "interactive_backend": {
            "selected": backend,
            "attempted_in_order": attempted_backends,
            "policy": "MacOSX requested first, TkAgg fallback only; noninteractive Agg is rejected.",
        },
        "preview": {
            "max_side_px": int(args.preview_max_side),
            "downsample_stride": int(stride),
            "display_shape_yx": [int(gray[::stride, ::stride].shape[0]), int(gray[::stride, ::stride].shape[1])],
            "important_limit": "Preview downsampling is display-only; raw coordinates are display coordinates times stride, before later snap validation.",
        },
        "required_click_order": POINT_ORDER,
        "controls": controls,
        "prohibitions": [
            "Does not write raw TIFF/CSV.",
            "Does not change nominal grid size, coverage gate, calibration_v1.yaml, transform/rank/orientation, or machine origin.",
            "Does not access A/B TIFF, XCT, weak target/support, model, checkpoint, training, decoder, or candidate output.",
            "Does not change camera-primary XCT-derived continuous quality candidate reporting.",
        ],
        "next_required_validation": "Run audit_visible_dotgrid_extent_controls.py using this V2 controls JSON; controls alone are not a calibration or policy decision.",
    }
    write_controls_json_after_four_valid_clicks(args.output_json, payload, args.overwrite)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Visible DotGrid extent controls V2 recorded. No raw TIFF/CSV, calibration, coverage gate, model, target, checkpoint, decoder, or candidate output was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
