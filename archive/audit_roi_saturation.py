#!/usr/bin/env python3
"""Read-only ROI and saturation audit for AMMT A/B layer-camera TIFF data.
Purpose
-------
Before defining model normalization or fusion inputs, this script measures
where 16-bit full-scale saturation occurs and evaluates one candidate ROI.
It does NOT modify either raw TIFF file.
Input
-----
* LayerCameraAfterSpreading.tif (A)
* LayerCameraBurned.tif (B)
Output (only under --output-dir)
--------------------------------
* roi_saturation_summary.csv : per sample frame and aggregate ROI statistics
* roi_candidate_qc.png       : candidate ROI and A/B saturation frequency maps
Example
-------
cd ~/ammt_project
/usr/local/bin/python3 src/audit_roi_saturation.py \
  --tiff-a raw_original/layer_camera/LayerCameraAfterSpreading.tif \
  --tiff-b raw_original/layer_camera/LayerCameraBurned.tif \
  --output-dir processed/roi_audit \
  --candidate-roi 250 250 1750 1750 \
  --z-values 1 10 20 125 230 250
"""
from __future__ import annotations
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import tifffile
FULL_SCALE = np.iinfo(np.uint16).max
@dataclass(frozen=True)
class StackInfo:
    axes: str
    shape: tuple[int, ...]
    dtype: np.dtype
    height: int
    width: int
    layers: int
    leds: int
def inspect(path: Path) -> StackInfo:
    """Read TIFF/ImageJ metadata without decoding the stack."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(v) for v in series.shape)
        imagej = tif.imagej_metadata or {}
    required = {"T", "Z", "Y", "X"}
    if not required.issubset(set(axes)):
        raise ValueError(f"Expected TZYX-compatible data, got axes={axes}")
    return StackInfo(
        axes=axes,
        shape=shape,
        dtype=np.dtype(series.dtype),
        height=shape[axes.index("Y")],
        width=shape[axes.index("X")],
        layers=int(imagej.get("slices", 1)),
        leds=int(imagej.get("frames", 1)),
    )
def check_pair(a: StackInfo, b: StackInfo) -> None:
    for field in ("axes", "shape", "dtype", "height", "width", "layers", "leds"):
        if getattr(a, field) != getattr(b, field):
            raise ValueError(f"A/B mismatch at {field}: {getattr(a, field)!r} != {getattr(b, field)!r}")
    if a.dtype != np.dtype(np.uint16):
        raise ValueError(f"This audit expects uint16 TIFF data, got {a.dtype}")
def read_frame(data: np.memmap, info: StackInfo, z: int, led: int) -> np.ndarray:
    """Read exactly one 2D frame. z and led are 1-based external indices."""
    if not 1 <= z <= info.layers:
        raise ValueError(f"z must be 1..{info.layers}, got {z}")
    if not 1 <= led <= info.leds:
        raise ValueError(f"led must be 1..{info.leds}, got {led}")
    index: list[Any] = []
    for axis in info.axes:
        if axis == "T":
            index.append(led - 1)
        elif axis == "Z":
            index.append(z - 1)
        elif axis == "C":
            index.append(0)
        elif axis in {"Y", "X"}:
            index.append(slice(None))
        else:
            raise ValueError(f"Unsupported axis {axis} in {info.axes}")
    frame = np.asarray(data[tuple(index)])
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got {frame.shape}")
    return frame
def display_scale(frame: np.ndarray) -> np.ndarray:
    """For PNG visualization only; never used as a training transformation."""
    lo, hi = np.percentile(frame, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1)
def ensure_new(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite after review.")
def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def draw_roi(axis: plt.Axes, roi: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = roi
    rectangle = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="#FF3B30", facecolor="none", linewidth=2.0)
    axis.add_patch(rectangle)
def save_qc(
    path: Path,
    a_data: np.memmap,
    b_data: np.memmap,
    info: StackInfo,
    roi: tuple[int, int, int, int],
    sat_frequency_a: np.ndarray,
    sat_frequency_b: np.ndarray,
    reference_z: int,
    reference_led: int,
    sample_count: int,
    overwrite: bool,
) -> None:
    ensure_new(path, overwrite)
    a = read_frame(a_data, info, reference_z, reference_led)
    b = read_frame(b_data, info, reference_z, reference_led)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    panels = (
        (axes[0, 0], display_scale(a), f"A frame | z={reference_z}, LED={reference_led}"),
        (axes[0, 1], display_scale(b), f"B frame | z={reference_z}, LED={reference_led}"),
        (axes[1, 0], sat_frequency_a / sample_count, "A saturation frequency"),
        (axes[1, 1], sat_frequency_b / sample_count, "B saturation frequency"),
    )
    for axis, image, title in panels:
        im = axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        draw_roi(axis, roi)
        axis.set_title(title)
        axis.set_axis_off()
        if "frequency" in title:
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="fraction at 65535")
    fig.suptitle("AMMT ROI candidate and saturation audit (red = candidate model ROI)", fontsize=14)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only AMMT ROI and saturation audit")
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--tiff-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-roi", required=True, nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--z-values", nargs="+", type=int, default=[1, 10, 20, 125, 230, 250])
    parser.add_argument("--reference-z", type=int, default=125, help="QC panel frame z index")
    parser.add_argument("--reference-led", type=int, default=1, help="QC panel LED index")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    path_a, path_b = args.tiff_a.resolve(), args.tiff_b.resolve()
    output_dir = args.output_dir.resolve()
    if not path_a.is_file() or not path_b.is_file():
        raise FileNotFoundError(f"Missing TIFF input. A={path_a}, B={path_b}")
    info_a, info_b = inspect(path_a), inspect(path_b)
    check_pair(info_a, info_b)
    info = info_a
    x0, y0, x1, y1 = tuple(args.candidate_roi)
    if not (0 <= x0 < x1 <= info.width and 0 <= y0 < y1 <= info.height):
        raise ValueError(f"ROI must fit within 0..{info.width} x 0..{info.height}: {args.candidate_roi}")
    roi = (x0, y0, x1, y1)
    z_values = sorted(set(args.z_values))
    if any(z < 1 or z > info.layers for z in z_values):
        raise ValueError(f"z-values must be within 1..{info.layers}")
    summary_csv = output_dir / "roi_saturation_summary.csv"
    qc_png = output_dir / "roi_candidate_qc.png"
    for output in (summary_csv, qc_png):
        ensure_new(output, args.overwrite)
    print("[1/3] Opening A/B TIFF through read-only memmap.")
    a_data = tifffile.memmap(path_a, series=0, mode="r")
    b_data = tifffile.memmap(path_b, series=0, mode="r")
    print(f"A/B shape={a_data.shape}; ROI={roi}; sampled z={z_values}")
    saturation_a = np.zeros((info.height, info.width), dtype=np.uint16)
    saturation_b = np.zeros((info.height, info.width), dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    roi_slice = np.s_[y0:y1, x0:x1]
    print("[2/3] Measuring sampled A/B frames; raw TIFF files remain unchanged.")
    for z in z_values:
        for led in range(1, info.leds + 1):
            frame_a = read_frame(a_data, info, z, led)
            frame_b = read_frame(b_data, info, z, led)
            saturation_a += frame_a == FULL_SCALE
            saturation_b += frame_b == FULL_SCALE
            a_roi, b_roi = frame_a[roi_slice], frame_b[roi_slice]
            diff_roi = np.abs(b_roi.astype(np.float32) - a_roi.astype(np.float32))
            rows.extend([
                {
                    "stage": "A", "layer_z": z, "led_t": led,
                    "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
                    "roi_mean": float(a_roi.mean()), "roi_std": float(a_roi.std()),
                    "roi_full_scale_fraction": float(np.mean(a_roi == FULL_SCALE)),
                    "pair_mean_abs_difference": float(diff_roi.mean()),
                },
                {
                    "stage": "B", "layer_z": z, "led_t": led,
                    "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
                    "roi_mean": float(b_roi.mean()), "roi_std": float(b_roi.std()),
                    "roi_full_scale_fraction": float(np.mean(b_roi == FULL_SCALE)),
                    "pair_mean_abs_difference": float(diff_roi.mean()),
                },
            ])
    print("[3/3] Writing two small audit outputs.")
    write_csv(summary_csv, rows, args.overwrite)
    sample_count = len(z_values) * info.leds
    save_qc(qc_png, a_data, b_data, info, roi, saturation_a, saturation_b, args.reference_z, args.reference_led, sample_count, args.overwrite)
    print("Done. Raw TIFF files were never modified.")
    print(f"- {summary_csv}")
    print(f"- {qc_png}")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
