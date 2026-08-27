#!/usr/bin/env python3
"""Create deterministic layer-125 orientation QC overlays for existing calibration ranks.

This read-only audit overlays layer-125 laser-on XYPT command paths on one
B-stage layer-camera frame for two pre-existing calibration hypotheses. It
supports visual comparison against the authoritative NIST layer-125 layout:
Part 1–4 numbering, cylindrical cavity at -X/left, and overhang at +X/right.
It does not select a rank, modify calibration, train a model, or make labels.
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
import yaml

from audit_machine_camera_calibration import PARTS, RECT, build_candidates

PART_COLORS = {
    "part01": "#d62728",
    "part02": "#1f77b4",
    "part03": "#2ca02c",
    "part04": "#9467bd",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_from_working_directory(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiff-b", required=True, type=Path)
    parser.add_argument("--xypt", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--layer-z", type=int, default=125)
    parser.add_argument("--led", type=int, default=3)
    parser.add_argument("--compare-ranks", nargs=2, type=int, default=[1, 2], metavar=("ALTERNATIVE_RANK", "SELECTED_RANK"))
    parser.add_argument("--max-display-pixels", type=int, default=1000)
    parser.add_argument("--max-points-per-part", type=int, default=30000)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def point_part_ids(xy: np.ndarray) -> np.ndarray:
    result = np.full(shape=(len(xy),), fill_value="outside_known_part_rectangles", dtype=object)
    for part in PARTS:
        x_min, x_max, y_min, y_max = RECT[part]
        inside = (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) & (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
        result[inside] = str(part)
    return result


def read_xypt_laser_on(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.loadtxt(path, delimiter=",")
    except ValueError:
        values = np.genfromtxt(path, delimiter=",", skip_header=1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"Expected at least four XYPT columns X,Y,P,T: {path}")
    finite = np.isfinite(values[:, :4]).all(axis=1)
    laser_on = finite & (values[:, 2] > 0.0)
    xy = values[laser_on, :2]
    if xy.size == 0:
        raise ValueError("No finite laser-on XYPT positions were found.")
    return xy, point_part_ids(xy)


def project_points(homography: np.ndarray, machine_xy: np.ndarray, offset_x: float, offset_y: float) -> tuple[np.ndarray, np.ndarray]:
    homogeneous_input = np.column_stack([machine_xy, np.ones(len(machine_xy), dtype=np.float64)])
    projected = homogeneous_input @ homography.T
    denominators = projected[:, 2]
    valid = np.isfinite(denominators) & (np.abs(denominators) > 1.0e-12)
    raw_xy = np.full((len(machine_xy), 2), np.nan, dtype=np.float64)
    raw_xy[valid] = projected[valid, :2] / denominators[valid, None]
    raw_xy[valid, 0] += offset_x
    raw_xy[valid, 1] += offset_y
    return raw_xy, valid


def downsample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=maximum, dtype=np.int64)


def finite_percentile_display(frame: np.ndarray, max_display_pixels: int) -> tuple[np.ndarray, dict[str, float | int]]:
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D layer frame, got {frame.shape}")
    valid = frame < np.iinfo(frame.dtype).max
    finite = frame[valid]
    if finite.size == 0:
        raise ValueError("Selected B-stage frame has no non-saturated pixels for display normalization.")
    p01, p99 = (float(np.percentile(finite, percentile)) for percentile in (1, 99))
    if not math.isfinite(p01) or not math.isfinite(p99) or p99 <= p01:
        raise ValueError("Display percentile range is invalid.")
    normalized = np.clip((frame.astype(np.float32) - p01) / (p99 - p01), 0.0, 1.0)
    stride = max(1, int(math.ceil(max(frame.shape) / max_display_pixels)))
    return normalized[::stride, ::stride], {
        "p01": p01,
        "p99": p99,
        "saturated_fraction": float(1.0 - valid.mean()),
        "display_stride": stride,
        "display_height": int(normalized[::stride, ::stride].shape[0]),
        "display_width": int(normalized[::stride, ::stride].shape[1]),
    }


def metadata(rank: int, hypothesis: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "orientation": str(hypothesis["orientation"]),
        "screen_A_to_D_machine_parts": list(hypothesis["assignment"]),
        "corner_index_order": list(hypothesis["corner_index_order"]),
        "fit_rmse_px": float(hypothesis["fit_rmse"]),
        "loo_rmse_px": float(hypothesis["loo_rmse"]),
    }


def plot_overlay(display_image: np.ndarray, frame_shape: tuple[int, int], rank_info: dict[str, Any], raw_by_part: dict[str, np.ndarray], output_path: Path) -> None:
    height, width = frame_shape
    figure, axis = plt.subplots(figsize=(10, 10), dpi=160)
    axis.imshow(display_image, cmap="gray", origin="upper", extent=(0, width, height, 0), interpolation="nearest")
    for part in PARTS:
        points = raw_by_part.get(str(part), np.empty((0, 2), dtype=np.float64))
        if len(points) == 0:
            continue
        axis.scatter(points[:, 0], points[:, 1], s=0.18, color=PART_COLORS[str(part)], alpha=0.80, linewidths=0, label=str(part))
        center = np.median(points, axis=0)
        axis.text(float(center[0]), float(center[1]), str(part), color="white", fontsize=9, fontweight="bold", ha="center", va="center", bbox={"facecolor": PART_COLORS[str(part)], "alpha": 0.72, "edgecolor": "none", "pad": 1.5})
    axis.set_title(f"Layer 125 B-stage LED overlay — rank {rank_info['rank']} ({rank_info['orientation']})\nVisual QC only: compare cavity (-X/left) and overhang (+X/right) with NIST Fig. 2")
    axis.set_xlabel("raw layer-camera x [pixel]")
    axis.set_ylabel("raw layer-camera y [pixel]")
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.legend(loc="upper right", framealpha=0.85, title="projected XYPT part")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rank_a, rank_b = (int(value) for value in args.compare_ranks)
    if rank_a == rank_b or min(rank_a, rank_b) < 1:
        raise ValueError("--compare-ranks requires two distinct positive ranks.")
    if args.layer_z != 125:
        raise ValueError("This authoritative NIST visual criterion is defined for layer 125; use --layer-z 125.")
    if args.led < 1 or args.led > 3:
        raise ValueError("--led must be 1, 2, or 3.")
    if args.max_display_pixels < 128 or args.max_points_per_part < 1:
        raise ValueError("Display and point limits must be positive and usable.")
    for path in (args.tiff_b, args.xypt, args.calibration_config):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    calibration = load_yaml(args.calibration_config)
    selected_rank = int(calibration["geometry_candidate"]["rank"])
    if rank_b != selected_rank:
        raise ValueError(f"Second compare rank {rank_b} must equal configured selected rank {selected_rank}.")
    controls_path = resolve_from_working_directory(str(calibration["control_points"]["path"]))
    controls = load_json(controls_path).get("control_points")
    if not isinstance(controls, list):
        raise ValueError("Calibration control JSON has no control_points list.")
    hypotheses = build_candidates(controls)
    if max(rank_a, rank_b) > len(hypotheses):
        raise ValueError("Requested rank is outside available existing calibration hypotheses.")
    offset_x, offset_y = (float(value) for value in calibration["local_photometric_refinement"]["raw_pixel_global_offset_xy"])

    stack = tifffile.memmap(args.tiff_b, series=0, mode="r")
    if stack.ndim != 4 or stack.shape[0] != 3 or stack.shape[1] < args.layer_z:
        raise ValueError(f"Expected read-only TZYX B hyperstack with 3 LEDs and layer {args.layer_z}, got {stack.shape}")
    raw_frame = np.asarray(stack[args.led - 1, args.layer_z - 1])
    display_image, display_stats = finite_percentile_display(raw_frame, args.max_display_pixels)
    machine_xy, machine_parts = read_xypt_laser_on(args.xypt)

    output_dir = args.output_dir
    prepare_output_directory(output_dir, args.overwrite)
    rows: list[dict[str, Any]] = []
    rank_summaries: list[dict[str, Any]] = []
    for rank in (rank_a, rank_b):
        hypothesis = hypotheses[rank - 1]
        rank_info = metadata(rank, hypothesis)
        projected, projectable = project_points(np.asarray(hypothesis["H"], dtype=np.float64), machine_xy, offset_x, offset_y)
        in_sensor = projectable & (projected[:, 0] >= 0) & (projected[:, 0] < raw_frame.shape[1]) & (projected[:, 1] >= 0) & (projected[:, 1] < raw_frame.shape[0])
        raw_by_part: dict[str, np.ndarray] = {}
        part_counts: dict[str, dict[str, int | float]] = {}
        for part in PARTS:
            part_mask = machine_parts == str(part)
            count_input = int(part_mask.sum())
            selected_points = projected[part_mask & in_sensor]
            selected_indices = downsample_indices(len(selected_points), args.max_points_per_part)
            raw_by_part[str(part)] = selected_points[selected_indices]
            part_counts[str(part)] = {
                "laser_on_command_count": count_input,
                "projectable_count": int((part_mask & projectable).sum()),
                "in_sensor_count": int((part_mask & in_sensor).sum()),
                "plotted_count": int(len(raw_by_part[str(part)])),
                "in_sensor_fraction": None if count_input == 0 else float((part_mask & in_sensor).sum() / count_input),
            }
        overlay_path = output_dir / f"layer125_B_led{args.led}_rank{rank}_{rank_info['orientation']}_overlay.png"
        plot_overlay(display_image, raw_frame.shape, rank_info, raw_by_part, overlay_path)
        rank_summaries.append({**rank_info, "projected_path_counts_by_part": part_counts, "overlay_png": str(overlay_path)})
        rows.append({
            "rank": rank,
            "orientation": rank_info["orientation"],
            "fit_rmse_px": rank_info["fit_rmse_px"],
            "loo_rmse_px": rank_info["loo_rmse_px"],
            "in_sensor_projected_fraction": float(in_sensor.mean()),
            "overlay_png": str(overlay_path),
        })

    comparison_path = output_dir / "rank_overlay_projection_summary.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "audit_type": "read-only deterministic layer-125 B-stage orientation overlay; not rank selection, metrology validation, or defect labeling",
        "narrative_anchor": {
            "source": "Lane and Yeung (2020), NIST J. Res. 125:125027, Fig. 2",
            "criterion": "At layer 125 the cylindrical cavity is on each part's left (-X) side and the overhang feature is on the right (+X) side; Fig. 2 gives Part 1-4 layout.",
            "source_url": "https://doi.org/10.6028/jres.125.027",
        },
        "inputs": {
            "tiff_b": str(args.tiff_b),
            "xypt": str(args.xypt),
            "calibration_config": str(args.calibration_config),
            "layer_z": args.layer_z,
            "led": args.led,
            "tiff_access": "tifffile.memmap(..., series=0, mode='r')",
            "raw_frame_shape": [int(raw_frame.shape[0]), int(raw_frame.shape[1])],
            "display": display_stats,
        },
        "calibration_status": calibration["status"],
        "selected_config_rank": selected_rank,
        "compared_existing_ranks": rank_summaries,
        "visual_review_required": {
            "automatic_rank_selection": False,
            "checklist": [
                "For each overlay, compare the projected laser-on path with visible raw B-stage part shapes.",
                "Use the authoritative layer-125 asymmetry: cavity must align with the -X/left side and overhang with the +X/right side.",
                "Check whether the projected Part 1-to-Part 4 diagonal ordering matches the visible build layout.",
                "Treat any apparent preference as visual evidence only; do not modify calibration_v1.yaml without separate review.",
            ],
        },
        "storage_policy": "writes two compact QC overlay PNGs and CSV/JSON summaries only; no raw TIFF, XYPT CSV, calibration config, target, model, checkpoint, or dense heatmap is modified",
        "outputs": {
            "projection_summary_csv": str(comparison_path),
            "summary_json": str(output_dir / "layer125_orientation_overlay_summary.json"),
        },
    }
    summary_path = output_dir / "layer125_orientation_overlay_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Layer-125 orientation overlay audit complete. No raw TIFF/XYPT CSV, calibration config, target, model, checkpoint, or dense heatmap was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
