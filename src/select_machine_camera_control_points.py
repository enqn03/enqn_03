#!/usr/bin/env python3
"""Interactively select raw-camera pixel control points for AMMT calibration.

This tool does not estimate a transform. It only records user-selected raw
camera pixel locations that correspond to declared machine-XY part corners.
Run it locally in an interactive desktop session; it opens a matplotlib window.

The default uses Burned (B), layer 125, LED 3 because LED 3 has substantially
less full-scale saturation than LED 1/2 in the prior audit. The selected points
are saved as an auditable JSON file for the separate calibration audit tool.

Example
-------
cd ~/ammt_project
/usr/local/bin/python3 src/select_machine_camera_control_points.py \
  --tiff raw_original/layer_camera/LayerCameraBurned.tif \
  --output-json processed/calibration/camera_control_points.json

For each requested point, click the matching visible *outer part corner* on the
raw camera image. The console and figure title identify the machine coordinate
of the requested corner. Avoid reflected, saturated or obstructed edges.
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
import tifffile

# Rectangle boundaries observed from registered command XY support in the X4 CSVs.
# Coordinate order is exact: x_min/y_min, x_max/y_min, x_max/y_max, x_min/y_max.
PART_MACHINE_RECTANGLES_MM: dict[str, tuple[float, float, float, float]] = {
    "part01": (-6.0, 3.0, 11.0, 16.0),
    "part02": (-2.0, 7.0, 2.0, 7.0),
    "part03": (2.0, 11.0, -7.0, -2.0),
    "part04": (6.0, 15.0, -16.0, -11.0),
}
CORNER_ORDER = ("x_min_y_min", "x_max_y_min", "x_max_y_max", "x_min_y_max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively record machine-XY to raw-camera control points")
    parser.add_argument("--tiff", required=True, type=Path, help="LayerCameraBurned.tif recommended")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--layer-z", type=int, default=125)
    parser.add_argument("--led", type=int, default=3)
    parser.add_argument("--percentiles", nargs=2, type=float, default=[1.0, 99.5], metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--display-max-px",
        type=int,
        default=1000,
        help="Maximum displayed image dimension in screen coordinates; clicks are converted back to raw pixels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inspect_stack(path: Path) -> tuple[str, tuple[int, ...]]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return str(series.axes), tuple(int(value) for value in series.shape)


def read_frame(path: Path, axes: str, shape: tuple[int, ...], layer_z: int, led: int) -> np.ndarray:
    if not {"T", "Z", "Y", "X"}.issubset(set(axes)):
        raise ValueError(f"Expected TZYX-compatible stack, got axes={axes!r}")
    max_layers = shape[axes.index("Z")]
    max_leds = shape[axes.index("T")]
    if not 1 <= layer_z <= max_layers:
        raise ValueError(f"--layer-z must be 1..{max_layers}")
    if not 1 <= led <= max_leds:
        raise ValueError(f"--led must be 1..{max_leds}")
    data = tifffile.memmap(path, series=0, mode="r")
    index: list[Any] = []
    for axis in axes:
        if axis == "T":
            index.append(led - 1)
        elif axis == "Z":
            index.append(layer_z - 1)
        elif axis == "C":
            index.append(0)
        elif axis in {"Y", "X"}:
            index.append(slice(None))
        else:
            raise ValueError(f"Unsupported TIFF axis: {axis}")
    frame = np.asarray(data[tuple(index)])
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got {frame.shape}")
    return frame


def requested_machine_corners() -> list[dict[str, float | str]]:
    requested: list[dict[str, float | str]] = []
    for part, (x_min, x_max, y_min, y_max) in PART_MACHINE_RECTANGLES_MM.items():
        coordinates = {
            "x_min_y_min": (x_min, y_min),
            "x_max_y_min": (x_max, y_min),
            "x_max_y_max": (x_max, y_max),
            "x_min_y_max": (x_min, y_max),
        }
        for corner_name in CORNER_ORDER:
            x_mm, y_mm = coordinates[corner_name]
            requested.append({"part": part, "corner": corner_name, "machine_x_mm": x_mm, "machine_y_mm": y_mm})
    return requested


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Review it or use --overwrite deliberately.")
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    tiff_path = args.tiff.resolve()
    output_path = args.output_json.resolve()
    if not tiff_path.is_file():
        raise FileNotFoundError(f"Missing TIFF: {tiff_path}")
    if not 0 <= args.percentiles[0] < args.percentiles[1] <= 100:
        raise ValueError("--percentiles must satisfy 0 <= low < high <= 100")
    if args.display_max_px <= 0:
        raise ValueError("--display-max-px must be positive")
    ensure_output_path(output_path, args.overwrite)

    axes, shape = inspect_stack(tiff_path)
    frame = read_frame(tiff_path, axes, shape, args.layer_z, args.led)
    raw_height, raw_width = frame.shape
    display_scale = min(1.0, float(args.display_max_px) / float(max(raw_height, raw_width)))
    display_width = raw_width * display_scale
    display_height = raw_height * display_scale
    low, high = np.percentile(frame, args.percentiles)
    requested = requested_machine_corners()

    fig, axis = plt.subplots(figsize=(8.2, 8.2))
    axis.imshow(
        frame,
        cmap="gray",
        vmin=low,
        vmax=high,
        origin="upper",
        extent=(0.0, display_width, display_height, 0.0),
        interpolation="nearest",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0.0, display_width)
    axis.set_ylim(display_height, 0.0)
    axis.set_xlabel(f"Display x (scale={display_scale:.4f} × raw pixel)")
    axis.set_ylabel(f"Display y (scale={display_scale:.4f} × raw pixel)")
    selected: list[dict[str, float | str]] = []

    print("Control-point selection started. Click one visible OUTER PART CORNER per prompt.")
    print("Use the same physical part and machine corner label shown in the title. Do not click label text, glare, or obstruction edges.")
    for number, request in enumerate(requested, start=1):
        part = str(request["part"])
        corner = str(request["corner"])
        x_mm = float(request["machine_x_mm"])
        y_mm = float(request["machine_y_mm"])
        axis.set_title(
            f"Click {number}/16: {part} | {corner} | machine XY=({x_mm:.3f}, {y_mm:.3f}) mm\n"
            "Left-click one outer part corner. Display is scaled; saved values remain raw camera pixels."
        )
        fig.canvas.draw_idle()
        print(
            f"[{number}/16] Click {part} {corner}, machine XY=({x_mm:.3f}, {y_mm:.3f}) mm "
            f"on a {display_width:.0f}×{display_height:.0f} displayed image (scale={display_scale:.4f})."
        )
        points = plt.ginput(1, timeout=-1, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
        if len(points) != 1:
            plt.close(fig)
            raise RuntimeError("Selection cancelled before all 16 control points were recorded")
        display_x, display_y = (float(points[0][0]), float(points[0][1]))
        pixel_x = min(max(display_x / display_scale, 0.0), float(raw_width - 1))
        pixel_y = min(max(display_y / display_scale, 0.0), float(raw_height - 1))
        selected.append({**request, "raw_camera_x_px": pixel_x, "raw_camera_y_px": pixel_y})
        axis.plot(display_x, display_y, marker="x", color="cyan", markersize=8, markeredgewidth=2)
        axis.text(display_x + 4, display_y + 4, f"{part[-2:]}-{corner}", color="cyan", fontsize=7)

    axis.set_title("All 16 control points selected. Close this window to write JSON.")
    fig.canvas.draw_idle()
    plt.show(block=True)
    plt.close(fig)

    payload = {
        "purpose": "User-selected control points for provisional machine XY to raw layer-camera pixel calibration.",
        "raw_input_policy": "Source TIFF was read with tifffile.memmap(mode='r') and was not modified.",
        "reference_frame": {
            "tiff_path": str(tiff_path),
            "axes": axes,
            "shape": list(shape),
            "stage": "B",
            "layer_z": args.layer_z,
            "led": args.led,
            "display_percentiles": [float(args.percentiles[0]), float(args.percentiles[1])],
            "raw_dimensions_px": [raw_width, raw_height],
            "display_dimensions_px": [display_width, display_height],
            "display_scale_to_raw_pixel": display_scale,
        },
        "machine_rectangle_source": "Registered XCT command XY ranges observed per part; values in millimetres.",
        "corner_order": list(CORNER_ORDER),
        "control_points": selected,
        "status": "provisional_manual_selection_requires_residual_and_overlay_audit",
        "notes": [
            "Control points represent part outer-corner correspondences, not defect locations.",
            "A later calibration audit must evaluate residuals and visual overlay before this mapping is used for weak heatmaps.",
            "Do not treat an uncalibrated raw camera pixel as machine XY supervision.",
        ],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(selected)} control points to: {output_path}")
    print("Next: run src/audit_machine_camera_calibration.py with this JSON file.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
