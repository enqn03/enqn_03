utf-8
#!/usr/bin/env python3
"""Estimate AMMT layer-camera normalization statistics from train data only.
The script reads the causal manifest, collects the *unique* layer indices used
by train histories, and samples valid pixels from a provisional working ROI.
It computes separate robust statistics for A/B stage and LED 1/2/3.
Raw TIFF files are opened only through ``tifffile.memmap(..., mode="r")`` and
are never modified. The script does not save cropped frames, image tensors, or
per-layer saturation masks.
Saturation policy
-----------------
A uint16 full-scale pixel (65535 by default) is treated as invalid for
normalization statistics. During later Dataset loading, use the same on-the-fly
rule instead of saving masks to disk:
    valid_mask = raw_frame < 65535
The policy is intentionally not final model configuration. Review the summary
before creating a tracked normalization config.
Example
-------
cd ~/ammt_project
/usr/local/bin/python3 src/estimate_train_normalization.py \
  --tiff-a raw_original/layer_camera/LayerCameraAfterSpreading.tif \
  --tiff-b raw_original/layer_camera/LayerCameraBurned.tif \
  --manifest manifests/causal_sequence_manifest.csv \
  --roi 250 250 1750 1750 \
  --pixel-stride 8 \
  --output-dir processed/normalization_v1
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import tifffile
@dataclass(frozen=True)
class StackInfo:
    """Metadata required to index an ImageJ logical hyperstack."""
    axes: str
    shape: tuple[int, ...]
    dtype: str
    width: int
    height: int
    layers: int
    leds: int
@dataclass(frozen=True)
class Roi:
    """Raw camera-pixel rectangle with an exclusive x1/y1 boundary."""
    x0: int
    y0: int
    x1: int
    y1: int
    @property
    def width(self) -> int:
        return self.x1 - self.x0
    @property
    def height(self) -> int:
        return self.y1 - self.y0
    @property
    def area_pixels(self) -> int:
        return self.width * self.height
    def as_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
def inspect_stack(path: Path) -> StackInfo:
    """Read TIFF metadata without decoding or changing pixel data."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
        imagej = tif.imagej_metadata or {}
    required_axes = {"T", "Z", "Y", "X"}
    if not required_axes.issubset(set(axes)):
        raise ValueError(f"Expected TZYX-compatible stack, got axes={axes!r}")
    if dtype != np.dtype(np.uint16):
        raise ValueError(f"Expected uint16 TIFF data, got {dtype}")
    return StackInfo(
        axes=axes,
        shape=shape,
        dtype=str(dtype),
        width=shape[axes.index("X")],
        height=shape[axes.index("Y")],
        layers=int(imagej.get("slices", 1)),
        leds=int(imagej.get("frames", 1)),
    )
def validate_pair(a: StackInfo, b: StackInfo) -> None:
    """Ensure A/B frames have a compatible logical `(layer, LED)` mapping."""
    for field in ("axes", "shape", "dtype", "width", "height", "layers", "leds"):
        if getattr(a, field) != getattr(b, field):
            raise ValueError(f"A/B mismatch at {field}: {getattr(a, field)!r} != {getattr(b, field)!r}")
def read_frame(data: np.memmap, info: StackInfo, z: int, led: int) -> np.ndarray:
    """Read exactly one 2D frame using 1-based manufacturing layer and LED."""
    if not 1 <= z <= info.layers:
        raise ValueError(f"z must be 1..{info.layers}, got {z}")
    if not 1 <= led <= info.leds:
        raise ValueError(f"LED must be 1..{info.leds}, got {led}")
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
            raise ValueError(f"Unsupported axis {axis!r} in {info.axes!r}")
    frame = np.asarray(data[tuple(index)])
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape={frame.shape}")
    return frame
def read_train_history_layers(manifest_path: Path) -> tuple[list[int], int]:
    """Extract unique train-history layers only; validation/test rows are ignored."""
    required = {"split", "history_layer_z"}
    layers: set[int] = set()
    train_rows = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = required.difference(headers)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for row in reader:
            if row.get("split") != "train":
                continue
            train_rows += 1
            history = str(row["history_layer_z"]).strip()
            if not history:
                raise ValueError("A train manifest row has empty history_layer_z")
            try:
                layers.update(int(value) for value in history.split(";"))
            except ValueError as error:
                raise ValueError(f"Invalid history_layer_z value: {history!r}") from error
    if train_rows == 0:
        raise ValueError("Manifest contains no train rows")
    if not layers:
        raise ValueError("Manifest train rows produced no history layers")
    return sorted(layers), train_rows
def validate_roi(roi: Roi, info: StackInfo) -> None:
    if not (0 <= roi.x0 < roi.x1 <= info.width):
        raise ValueError(f"ROI x bounds must fit width={info.width}, got {roi}")
    if not (0 <= roi.y0 < roi.y1 <= info.height):
        raise ValueError(f"ROI y bounds must fit height={info.height}, got {roi}")
def ensure_new(path: Path, overwrite: bool) -> None:
    """Prevent accidental replacement of a prior normalization audit."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}. Review it or use --overwrite.")
def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def write_json(path: Path, content: dict[str, Any], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
def float_or_none(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)
def summarize_group(
    stage: str,
    led: int,
    chunks: list[np.ndarray],
    full_scale_count: int,
    sampled_count: int,
    frame_count: int,
    roi: Roi,
    pixel_stride: int,
    full_scale_value: int,
) -> dict[str, Any]:
    """Calculate robust statistics from valid sampled pixels only."""
    if sampled_count <= 0:
        raise ValueError(f"No sampled pixels for {stage}, LED {led}")
    if not chunks:
        raise ValueError(f"No valid pixels for {stage}, LED {led}; all samples are full-scale")
    values = np.concatenate(chunks).astype(np.float32, copy=False)
    p01, p05, p50, p95, p99 = np.percentile(values, [1.0, 5.0, 50.0, 95.0, 99.0])
    denominator = float(p99 - p01)
    if denominator <= 0:
        raise ValueError(f"Non-positive p99-p01 range for {stage}, LED {led}")
    return {
        "stage": stage,
        "led_t": led,
        "unique_train_history_layer_count": frame_count,
        "roi_x0": roi.x0,
        "roi_y0": roi.y0,
        "roi_x1": roi.x1,
        "roi_y1": roi.y1,
        "roi_area_pixels": roi.area_pixels,
        "pixel_stride": pixel_stride,
        "sampled_pixel_count": sampled_count,
        "valid_pixel_count": int(values.size),
        "full_scale_pixel_count": full_scale_count,
        "full_scale_fraction": float(full_scale_count / sampled_count),
        "valid_fraction": float(values.size / sampled_count),
        "full_scale_value": full_scale_value,
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "recommended_clip_low_p01": float(p01),
        "recommended_clip_high_p99": float(p99),
        "recommended_scale_p99_minus_p01": denominator,
        "normalization_formula_candidate": "clip((x - p01) / (p99 - p01), 0, 1)",
        "validity_mask_candidate": f"raw_pixel < {full_scale_value}",
    }
def save_qc(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    """Write one deterministic QC chart from calculated summary statistics."""
    ensure_new(path, overwrite)
    labels = [f"{row['stage']}-L{row['led_t']}" for row in rows]
    indices = np.arange(len(rows))
    p01 = np.asarray([float(row["p01"]) for row in rows])
    p50 = np.asarray([float(row["p50"]) for row in rows])
    p99 = np.asarray([float(row["p99"]) for row in rows])
    saturation = 100.0 * np.asarray([float(row["full_scale_fraction"]) for row in rows])
    scale = float(np.iinfo(np.uint16).max)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    width = 0.24
    axes[0].bar(indices - width, 100.0 * p01 / scale, width=width, label="p01", color="#5ab4ac")
    axes[0].bar(indices, 100.0 * p50 / scale, width=width, label="p50", color="#74a9cf")
    axes[0].bar(indices + width, 100.0 * p99 / scale, width=width, label="p99", color="#ef8a62")
    axes[0].set_title("Train-only valid-pixel intensity percentiles")
    axes[0].set_ylabel("uint16 full scale (%)")
    axes[0].set_xticks(indices, labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    colors = ["#2a6fbb" if str(row["stage"]) == "A" else "#d55e00" for row in rows]
    axes[1].bar(indices, saturation, color=colors)
    axes[1].set_title("Train-only sampled full-scale saturation")
    axes[1].set_ylabel("sampled ROI pixels at 65535 (%)")
    axes[1].set_xticks(indices, labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(
        0.5,
        -0.20,
        "A=AfterSpreading, B=Burned; colors distinguish stage only",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.suptitle("AMMT normalization screening — train history layers only", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate train-only A/B LED normalization statistics")
    parser.add_argument("--tiff-a", required=True, type=Path, help="AfterSpreading A TIFF")
    parser.add_argument("--tiff-b", required=True, type=Path, help="Burned B TIFF")
    parser.add_argument("--manifest", required=True, type=Path, help="Causal sequence manifest CSV")
    parser.add_argument("--roi", required=True, nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--pixel-stride", type=int, default=8, help="Deterministic within-ROI sampling step; must be >= 1")
    parser.add_argument("--full-scale-value", type=int, default=int(np.iinfo(np.uint16).max))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output after review")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    if args.pixel_stride < 1:
        raise ValueError("--pixel-stride must be at least 1")
    if not 0 <= args.full_scale_value <= np.iinfo(np.uint16).max:
        raise ValueError("--full-scale-value must be within uint16 range")
    path_a = args.tiff_a.resolve()
    path_b = args.tiff_b.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if not path_a.is_file() or not path_b.is_file():
        raise FileNotFoundError(f"Missing TIFF input. A={path_a}, B={path_b}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    info_a = inspect_stack(path_a)
    info_b = inspect_stack(path_b)
    validate_pair(info_a, info_b)
    info = info_a
    roi = Roi(*tuple(int(value) for value in args.roi))
    validate_roi(roi, info)
    train_layers, train_row_count = read_train_history_layers(manifest_path)
    if train_layers[0] < 1 or train_layers[-1] > info.layers:
        raise ValueError(f"Train history layers {train_layers[0]}..{train_layers[-1]} exceed TIFF bounds 1..{info.layers}")
    summary_csv = output_dir / "train_stage_led_summary.csv"
    summary_json = output_dir / "train_stage_led_summary.json"
    qc_png = output_dir / "normalization_qc.png"
    for output in (summary_csv, summary_json, qc_png):
        ensure_new(output, args.overwrite)
    y_slice = slice(roi.y0, roi.y1, args.pixel_stride)
    x_slice = slice(roi.x0, roi.x1, args.pixel_stride)
    grid_height = len(range(roi.y0, roi.y1, args.pixel_stride))
    grid_width = len(range(roi.x0, roi.x1, args.pixel_stride))
    print("[1/3] Opening A/B TIFF through read-only memmap.")
    a_data = tifffile.memmap(path_a, series=0, mode="r")
    b_data = tifffile.memmap(path_b, series=0, mode="r")
    print(
        f"train rows={train_row_count}; unique train history layers={len(train_layers)} "
        f"({train_layers[0]}..{train_layers[-1]}); sampled grid={grid_height}x{grid_width}"
    )
    chunks: dict[tuple[str, int], list[np.ndarray]] = {
        (stage, led): [] for stage in ("A", "B") for led in range(1, info.leds + 1)
    }
    sampled_counts = {(stage, led): 0 for stage in ("A", "B") for led in range(1, info.leds + 1)}
    full_scale_counts = {(stage, led): 0 for stage in ("A", "B") for led in range(1, info.leds + 1)}
    print("[2/3] Sampling valid train-history pixels by stage and LED. Raw TIFF files remain unchanged.")
    for layer_index, z in enumerate(train_layers, start=1):
        for led in range(1, info.leds + 1):
            frame_a = read_frame(a_data, info, z, led)
            frame_b = read_frame(b_data, info, z, led)
            for stage, frame in (("A", frame_a), ("B", frame_b)):
                grid = frame[y_slice, x_slice]
                sampled_counts[(stage, led)] += int(grid.size)
                valid_mask = grid < args.full_scale_value
                full_scale_counts[(stage, led)] += int(np.count_nonzero(~valid_mask))
                valid_values = grid[valid_mask]
                if valid_values.size:
                    chunks[(stage, led)].append(np.asarray(valid_values, dtype=np.uint16))
        if layer_index % 25 == 0 or layer_index == len(train_layers):
            print(f"  processed {layer_index}/{len(train_layers)} train-history layers")
    rows: list[dict[str, Any]] = []
    for stage in ("A", "B"):
        for led in range(1, info.leds + 1):
            rows.append(
                summarize_group(
                    stage=stage,
                    led=led,
                    chunks=chunks[(stage, led)],
                    full_scale_count=full_scale_counts[(stage, led)],
                    sampled_count=sampled_counts[(stage, led)],
                    frame_count=len(train_layers),
                    roi=roi,
                    pixel_stride=args.pixel_stride,
                    full_scale_value=args.full_scale_value,
                )
            )
    summary: dict[str, Any] = {
        "audit_type": "train-only stage/LED normalization screening; not final model configuration",
        "raw_input_policy": "A/B TIFF opened only with tifffile.memmap(mode='r'); raw bytes are not modified.",
        "raw_inputs": {"after_spreading": str(path_a), "burned": str(path_b)},
        "manifest": str(manifest_path),
        "split_policy": {
            "source_rows_used": "split=train only",
            "train_manifest_row_count": train_row_count,
            "unique_train_history_layers": train_layers,
            "unique_train_history_layer_count": len(train_layers),
            "validation_and_test_policy": "No validation or test manifest row or layer is read for these statistics.",
        },
        "stack": asdict(info),
        "working_roi_raw_pixels": roi.as_dict(),
        "deterministic_sampling": {
            "pixel_stride": args.pixel_stride,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "grid_pixels_per_frame": grid_height * grid_width,
        },
        "saturation_validity_mask_policy": {
            "candidate_expression": f"raw_pixel < {args.full_scale_value}",
            "full_scale_value": args.full_scale_value,
            "storage": "Do not write a per-frame mask image. Recompute the boolean mask on-the-fly in the Dataset.",
            "normalization_use": "Exclude invalid pixels from percentile and moment estimation.",
            "model_use": "Expose the mask to loss/heatmap logic after final configuration review.",
        },
        "normalization_candidate": {
            "formula": "clip((x - p01) / (p99 - p01), 0, 1)",
            "scope": "Use a separate p01/p99 pair per stage and LED.",
            "status": "Candidate only. Review summary values before writing a config file.",
        },
        "stage_led_statistics": rows,
    }
    print("[3/3] Writing small summary outputs only; no image tensors or masks are saved.")
    write_csv(summary_csv, rows, args.overwrite)
    write_json(summary_json, summary, args.overwrite)
    save_qc(qc_png, rows, args.overwrite)
    print("Done. Raw TIFF files were opened read-only and were not modified.")
    for output in (summary_csv, summary_json, qc_png):
        print(f"- {output}")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
