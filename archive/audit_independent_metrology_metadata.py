#!/usr/bin/env python3
"""Audit existing independent layer-camera metrology metadata without refitting calibration.

This read-only pre-audit inspects the NIST-provided dot-grid, checkerboard, and
secondary-camera laser-origin TIFF artifacts. It reports TIFF schema, robust
intensity/edge structure, and a *candidate* red-laser centroid where RGB data
permit. It does not fit or replace a homography, select rank 1/2, modify
calibration_v1.yaml, train a model, or create a target/label.

Only small CSV/JSON metrics and one deterministic QC contact sheet are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("dot_grid", "dot_grid", "layer-camera dot-grid geometry"),
    ("secondary_laser_origin", "secondary_camera", "secondary-camera machine-origin laser reference"),
    ("checkerboard", "checkerboard", "layer-camera checkerboard geometry"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid_2000x2000 TIFF.")
    parser.add_argument("--secondary-camera", required=True, type=Path, help="Immutable SecondaryCamera_Laser00 TIFF.")
    parser.add_argument("--checkerboard", required=True, type=Path, help="Immutable Checkerboard_2000x2000 TIFF.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for compact CSV/JSON/QC PNG output.")
    parser.add_argument("--max-display-pixels", type=int, default=1000, help="Maximum side length of QC contact-sheet images.")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace an existing audit output directory only after review.")
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def robust_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Image has no finite values.")
    p01, p50, p99 = (float(np.percentile(finite, value)) for value in (1, 50, 99))
    if not math.isfinite(p01) or not math.isfinite(p99) or p99 <= p01:
        raise ValueError(f"Invalid robust range p01={p01}, p99={p99}.")
    normalized = np.clip((values - p01) / (p99 - p01), 0.0, 1.0)
    return normalized, {"p01": p01, "p50": p50, "p99": p99}


def downsample(image: np.ndarray, max_display_pixels: int) -> tuple[np.ndarray, int]:
    if max_display_pixels < 128:
        raise ValueError("--max-display-pixels must be at least 128.")
    height, width = image.shape[-2:]
    stride = max(1, int(math.ceil(max(height, width) / max_display_pixels)))
    return image[..., ::stride, ::stride], stride


def to_channels_yx(data: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Select non-spatial singleton/page dimensions and return C,Y,X float-compatible array."""
    axes = str(axes)
    if data.ndim != len(axes):
        raise ValueError(f"Series axes {axes!r} do not match array shape {data.shape}.")
    selectors: list[int | slice] = []
    retained_axes: list[str] = []
    for axis, size in zip(axes, data.shape, strict=True):
        if axis in {"Y", "X", "C", "S"}:
            selectors.append(slice(None))
            retained_axes.append(axis)
        else:
            if size < 1:
                raise ValueError(f"Axis {axis!r} has unusable size {size}.")
            selectors.append(0)
    selected = np.asarray(data[tuple(selectors)])
    if "Y" not in retained_axes or "X" not in retained_axes:
        raise ValueError(f"Metadata TIFF must retain Y and X axes, got {axes!r}.")
    channel_axis = next((index for index, axis in enumerate(retained_axes) if axis in {"C", "S"}), None)
    y_axis = retained_axes.index("Y")
    x_axis = retained_axes.index("X")
    if channel_axis is None:
        if selected.ndim != 2:
            raise ValueError(f"Expected 2D grayscale after selection, got {selected.shape} from axes {axes!r}.")
        return selected[None, ...], "grayscale"
    if selected.ndim != 3:
        raise ValueError(f"Expected CYX-equivalent image after selection, got {selected.shape} from axes {axes!r}.")
    return np.transpose(selected, (channel_axis, y_axis, x_axis)), "multi_channel"


def gray_from_channels(channels: np.ndarray) -> np.ndarray:
    if channels.shape[0] == 1:
        return channels[0]
    if channels.shape[0] >= 3:
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return channels.mean(axis=0)


def image_structure_metrics(gray: np.ndarray, robust: dict[str, float | None]) -> dict[str, float | int | None]:
    normalized, _ = robust_normalize(gray)
    gy, gx = np.gradient(normalized)
    magnitude = np.hypot(gx, gy)
    edge_threshold = float(np.percentile(magnitude, 99.0))
    edge_mask = magnitude >= edge_threshold
    profile_x = magnitude.mean(axis=0)
    profile_y = magnitude.mean(axis=1)
    def coefficient_of_variation(values: np.ndarray) -> float | None:
        mean = float(np.mean(values))
        return None if abs(mean) <= 1.0e-12 else float(np.std(values) / abs(mean))
    return {
        "robust_p01": robust["p01"],
        "robust_p50": robust["p50"],
        "robust_p99": robust["p99"],
        "normalized_gradient_mean": float(np.mean(magnitude)),
        "normalized_gradient_p99": edge_threshold,
        "top_one_percent_edge_fraction": float(edge_mask.mean()),
        "edge_profile_x_cv": coefficient_of_variation(profile_x),
        "edge_profile_y_cv": coefficient_of_variation(profile_y),
    }


def red_laser_candidate(channels: np.ndarray) -> dict[str, Any]:
    if channels.shape[0] < 3:
        return {
            "rgb_channels_available": False,
            "candidate_detected": False,
            "reason": "fewer_than_three_channels",
            "raw_camera_x_px": None,
            "raw_camera_y_px": None,
            "top_fraction": None,
            "red_dominance_p999": None,
            "weighted_spread_px": None,
        }
    rgb = np.asarray(channels[:3], dtype=np.float32)
    per_channel = []
    for channel in rgb:
        normalized, _ = robust_normalize(channel)
        per_channel.append(normalized)
    red, green, blue = per_channel
    dominance = np.clip(red - 0.5 * (green + blue), 0.0, None)
    positive = dominance[dominance > 0.0]
    if positive.size == 0:
        return {
            "rgb_channels_available": True,
            "candidate_detected": False,
            "reason": "no_positive_red_dominance",
            "raw_camera_x_px": None,
            "raw_camera_y_px": None,
            "top_fraction": None,
            "red_dominance_p999": None,
            "weighted_spread_px": None,
        }
    threshold = float(np.percentile(positive, 99.9))
    mask = dominance >= threshold
    weights = np.where(mask, dominance, 0.0)
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return {
            "rgb_channels_available": True,
            "candidate_detected": False,
            "reason": "invalid_top_red_weight",
            "raw_camera_x_px": None,
            "raw_camera_y_px": None,
            "top_fraction": None,
            "red_dominance_p999": threshold,
            "weighted_spread_px": None,
        }
    yy, xx = np.indices(dominance.shape, dtype=np.float64)
    centroid_x = float((xx * weights).sum() / total_weight)
    centroid_y = float((yy * weights).sum() / total_weight)
    variance = float((((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2) * weights).sum() / total_weight)
    total_positive = float(dominance.sum())
    return {
        "rgb_channels_available": True,
        "candidate_detected": True,
        "reason": "top_0.1_percent_red_dominance_weighted_centroid; visual confirmation_required",
        "raw_camera_x_px": centroid_x,
        "raw_camera_y_px": centroid_y,
        "top_fraction": float(mask.mean()),
        "red_dominance_p999": threshold,
        "weighted_spread_px": float(math.sqrt(max(variance, 0.0))),
        "top_to_all_positive_red_weight_fraction": None if total_positive <= 0.0 else float(total_weight / total_positive),
    }


def inspect_tiff(name: str, path: Path, role: str, max_display_pixels: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None, dict[str, Any] | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata artifact not found: {path}")
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        axes = str(series.axes)
        series_shape = [int(value) for value in series.shape]
        dtype = str(series.dtype)
        is_imagej = bool(handle.is_imagej)
    data = tifffile.memmap(path, series=0, mode="r")
    channels, channel_mode = to_channels_yx(data, axes)
    gray = gray_from_channels(channels)
    robust_gray, robust_values = robust_normalize(gray)
    display_gray, display_stride = downsample(robust_gray, max_display_pixels)
    structure = image_structure_metrics(gray, robust_values)
    display_rgb: np.ndarray | None = None
    laser: dict[str, Any] | None = None
    if channels.shape[0] >= 3:
        rgb_channels = []
        for channel in channels[:3]:
            channel_norm, _ = robust_normalize(channel)
            rgb_channels.append(channel_norm)
        rgb = np.stack(rgb_channels, axis=-1)
        display_rgb = downsample(np.moveaxis(rgb, -1, 0), max_display_pixels)[0]
        display_rgb = np.moveaxis(display_rgb, 0, -1)
    if name == "secondary_laser_origin":
        laser = red_laser_candidate(channels)
    dtype_info = np.dtype(data.dtype)
    saturated_fraction = None
    if np.issubdtype(dtype_info, np.integer):
        saturated_fraction = float((channels == np.iinfo(dtype_info).max).mean())
    record: dict[str, Any] = {
        "artifact": name,
        "role": role,
        "path": str(path),
        "tiff_access": "tifffile.memmap(..., series=0, mode='r')",
        "series_axes": axes,
        "series_shape": series_shape,
        "dtype": dtype,
        "is_imagej": is_imagej,
        "canonical_channel_mode": channel_mode,
        "channel_count": int(channels.shape[0]),
        "raw_height_px": int(channels.shape[1]),
        "raw_width_px": int(channels.shape[2]),
        "saturated_fraction": saturated_fraction,
        "display_stride": display_stride,
        "display_height_px": int(display_gray.shape[0]),
        "display_width_px": int(display_gray.shape[1]),
        **structure,
    }
    if laser is not None:
        record["laser_origin_candidate"] = laser
    return record, display_gray, display_rgb, laser


def plot_qc(items: list[tuple[dict[str, Any], np.ndarray, np.ndarray | None, dict[str, Any] | None]], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150, constrained_layout=True)
    panels = list(axes.ravel())
    for panel in panels:
        panel.set_axis_off()
    for index, (record, display_gray, display_rgb, laser) in enumerate(items):
        panel = panels[index]
        image = display_rgb if display_rgb is not None else display_gray
        panel.imshow(image, cmap=None if display_rgb is not None else "gray", origin="upper", interpolation="nearest")
        panel.set_axis_on()
        panel.set_title(f"{record['artifact']}\n{record['series_axes']} {record['series_shape']} | display stride={record['display_stride']}", fontsize=10)
        panel.set_xlabel("display x [pixel]")
        panel.set_ylabel("display y [pixel]")
        if laser is not None and bool(laser["candidate_detected"]):
            x = float(laser["raw_camera_x_px"]) / int(record["display_stride"])
            y = float(laser["raw_camera_y_px"]) / int(record["display_stride"])
            panel.scatter([x], [y], marker="+", s=180, linewidths=2.0, color="cyan", label="red-dominance candidate")
            panel.legend(loc="lower right", fontsize=8, framealpha=0.85)
    panels[3].text(0.5, 0.72, "Independent metrology pre-audit", ha="center", va="center", fontsize=15, fontweight="bold")
    panels[3].text(0.5, 0.48, "Read-only metadata TIFF inspection\nNo homography fit, rank selection, or config change", ha="center", va="center", fontsize=11)
    panels[3].text(0.5, 0.25, "Laser marker = red-dominance candidate only;\nvisual/metrology confirmation is required.", ha="center", va="center", fontsize=10)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    for required_path in (args.dot_grid, args.secondary_camera, args.checkerboard):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required metadata TIFF not found: {required_path}")
    if args.max_display_pixels < 128:
        raise ValueError("--max-display-pixels must be at least 128.")
    output_dir = args.output_dir
    prepare_output_directory(output_dir, args.overwrite)
    inspected: list[tuple[dict[str, Any], np.ndarray, np.ndarray | None, dict[str, Any] | None]] = []
    for name, cli_attribute, role in ARTIFACTS:
        inspected.append(inspect_tiff(name, getattr(args, cli_attribute), role, args.max_display_pixels))
    records = [item[0] for item in inspected]
    inventory_path = output_dir / "metrology_metadata_artifact_inventory.csv"
    fieldnames = [key for key in records[0] if key != "laser_origin_candidate"]
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames}
            row["series_shape"] = ";".join(str(value) for value in row["series_shape"])
            writer.writerow(row)
    qc_path = output_dir / "metrology_metadata_pre_audit_qc.png"
    plot_qc(inspected, qc_path)
    laser_record = next(record for record in records if record["artifact"] == "secondary_laser_origin")
    laser = laser_record.get("laser_origin_candidate")
    summary = {
        "audit_type": "read-only independent layer-camera metrology metadata pre-audit; not calibration refit, rank selection, or defect labeling",
        "inputs": {
            "dot_grid": str(args.dot_grid),
            "secondary_camera": str(args.secondary_camera),
            "checkerboard": str(args.checkerboard),
            "artifact_count": len(records),
            "tiff_access": "tifffile.memmap(..., series=0, mode='r')",
        },
        "artifacts": records,
        "pre_audit_checks": {
            "all_artifacts_have_yx_geometry": bool(all(record["raw_height_px"] > 1 and record["raw_width_px"] > 1 for record in records)),
            "dot_grid_structure_metrics_recorded": True,
            "checkerboard_structure_metrics_recorded": True,
            "secondary_rgb_available": bool(laser and laser["rgb_channels_available"]),
            "secondary_red_laser_candidate_recorded": bool(laser and laser["candidate_detected"]),
            "independent_anchor_ready_for_next_fit_audit": bool(laser and laser["candidate_detected"]),
        },
        "required_hold": "This audit reports image schema and candidate reference features only. It does not prove feature identity, establish machine origin, refit a transform, select a mirror rank, modify calibration_v1.yaml, or validate physical candidate coordinates.",
        "storage_policy": "writes one compact deterministic QC PNG plus CSV/JSON metrics only; no metadata TIFF, raw manufacturing TIFF/CSV, target, model, checkpoint, calibration config, dense crop, mask, or heatmap is modified",
        "outputs": {
            "artifact_inventory_csv": str(inventory_path),
            "qc_png": str(qc_path),
            "summary_json": str(output_dir / "independent_metrology_pre_audit_summary.json"),
        },
    }
    summary_path = output_dir / "independent_metrology_pre_audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Independent metrology metadata pre-audit complete. No raw TIFF/CSV, calibration config, target, model, checkpoint, or dense output was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
