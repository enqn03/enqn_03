#!/usr/bin/env python3
"""Audit DotGrid coverage evidence without changing a calibration or gate.

This script reads (1) the immutable layer-camera DotGrid TIFF using the existing
read-only memmap helper and (2) the compact feature CSV emitted by the completed
method-#2 V3 correspondence audit. It re-fits an *image-lattice-only* mapping in
memory solely to locate the nominal 50x50 image-lattice cells. It measures which
cells are assigned, predicted inside the camera sensor, and close to a fresh
ROI-restricted dot detector candidate.

The audit does not alter the nominal 50x50 assumption, the fixed 40-row/column
coverage gate, V3 correspondence, calibration_v1.yaml, transform rank,
orientation, machine origin, raw input, target, model, checkpoint, decoder, or
camera-primary candidate reporting. Its outcome is evidence for a later human
review of coverage definitions only.
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

from audit_independent_metrology_fiducials_refined import (
    density_roi,
    dot_grid_candidates,
    grayscale,
    read_tiff,
    refined_feature_candidates,
)
from audit_independent_method2_calibration_candidate import (
    GRID_SIZE,
    inlier_fit,
    nearest_camera_dot_pitch_px,
    project,
)

SENSOR_MIN_PX = 0.0
NOMINAL_GRID_SIZE = GRID_SIZE
V3_ASSIGNMENT_MAX_PITCH = 0.45
PROFILE_SUPPORT_MIN_CELLS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid TIFF; opened read-only via tifffile.memmap mode='r'.")
    parser.add_argument("--v3-features", required=True, type=Path, help="Compact method2_refined_2d_lattice_features.csv from completed V3 audit.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New ignored directory for compact CSV/JSON and at most two QC overlays.")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace only the requested output directory after review.")
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def load_v3_features(path: Path) -> list[dict[str, Any]]:
    required = {
        "source_candidate_index",
        "image_lattice_col_index_0_to_49",
        "image_lattice_row_index_0_to_49",
        "raw_x_px",
        "raw_y_px",
        "detector_response",
        "full_refined_fit_inlier",
        "heldout_block",
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"V3 feature CSV lacks required fields: {missing}")
        rows: list[dict[str, Any]] = []
        for record in reader:
            row = {
                "source_candidate_index": int(record["source_candidate_index"]),
                "col": int(record["image_lattice_col_index_0_to_49"]),
                "row": int(record["image_lattice_row_index_0_to_49"]),
                "raw_x_px": float(record["raw_x_px"]),
                "raw_y_px": float(record["raw_y_px"]),
                "detector_response": float(record["detector_response"]),
                "full_refined_fit_inlier": str(record["full_refined_fit_inlier"]).strip().lower() == "true",
                "heldout_block": str(record["heldout_block"]).strip().lower() == "true",
            }
            if not (0 <= row["col"] < NOMINAL_GRID_SIZE and 0 <= row["row"] < NOMINAL_GRID_SIZE):
                raise ValueError(f"V3 feature has non-nominal image-lattice index: {(row['col'], row['row'])}")
            rows.append(row)
    if len(rows) < 8:
        raise ValueError("Need at least eight V3 assigned cells for image-lattice coverage audit.")
    keys = {(int(row["col"]), int(row["row"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("V3 feature CSV contains duplicate image-lattice cells; coverage cannot be interpreted deterministically.")
    return rows


def arrays_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray([[float(row["col"]), float(row["row"])] for row in rows], dtype=np.float64)
    raw = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in rows], dtype=np.float64)
    return source, raw


def fit_image_lattice_only(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source, raw = arrays_from_rows(rows)
    h_matrix, inliers, residual, _ = inlier_fit(source, raw)
    pitch = nearest_camera_dot_pitch_px(raw[inliers])
    if pitch is None or not math.isfinite(pitch) or pitch <= 0.0:
        raise RuntimeError("Could not estimate positive camera-dot pitch from V3 inliers.")
    return h_matrix, inliers, residual, float(pitch)


def sensor_membership(predicted: np.ndarray, height: int, width: int) -> np.ndarray:
    return (
        (predicted[:, 0] >= SENSOR_MIN_PX)
        & (predicted[:, 0] <= float(width - 1))
        & (predicted[:, 1] >= SENSOR_MIN_PX)
        & (predicted[:, 1] <= float(height - 1))
    )


def nearest_detector_distances(predicted: np.ndarray, detector_points: np.ndarray) -> np.ndarray:
    if len(detector_points) == 0:
        raise RuntimeError("No fresh DotGrid detector candidates are available for coverage definition.")
    distances: list[np.ndarray] = []
    # Fixed 256-cell chunks avoid persisting or allocating a dense 2500x1616 tensor.
    for start in range(0, len(predicted), 256):
        chunk = predicted[start:start + 256]
        delta = chunk[:, None, :] - detector_points[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances, axis=0)


def local_darkness(gray: np.ndarray, predicted: np.ndarray, radius: int = 3) -> np.ndarray:
    values: list[float] = []
    height, width = gray.shape
    for x_float, y_float in predicted:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            values.append(float("nan"))
        else:
            patch = np.asarray(gray[y0:y1, x0:x1], dtype=np.float64)
            values.append(float(1.0 - np.mean(patch) / 255.0))
    return np.asarray(values, dtype=np.float64)


def contiguous_runs(indices: list[int]) -> list[dict[str, int]]:
    if not indices:
        return []
    values = sorted(set(int(value) for value in indices))
    start = previous = values[0]
    runs: list[dict[str, int]] = []
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append({"start_index": start, "end_index": previous, "length": previous - start + 1})
        start = previous = value
    runs.append({"start_index": start, "end_index": previous, "length": previous - start + 1})
    return runs


def profile_rows(feature_rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    other = "row" if key == "col" else "col"
    result: list[dict[str, Any]] = []
    for index in range(NOMINAL_GRID_SIZE):
        selected = [row for row in feature_rows if int(row[key]) == index]
        other_indices = sorted(int(row[other]) for row in selected)
        runs = contiguous_runs(other_indices)
        result.append({
            f"image_lattice_{key}_index_0_to_49": index,
            "assigned_cell_count": len(selected),
            f"unique_{other}_count": len(set(other_indices)),
            f"{other}_min_assigned": None if not other_indices else min(other_indices),
            f"{other}_max_assigned": None if not other_indices else max(other_indices),
            "contiguous_run_count": len(runs),
            "longest_contiguous_run": 0 if not runs else max(int(run["length"]) for run in runs),
            "has_any_assigned_cell": bool(len(selected) >= PROFILE_SUPPORT_MIN_CELLS),
            "runs_json": json.dumps(runs, separators=(",", ":")),
        })
    return result


def classify_coverage_evidence(cell_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [row for row in cell_rows if not bool(row["assigned_in_v3"])]
    if not missing:
        return {"classification": "no_missing_nominal_cells", "important_limit": "Not a calibration or gate-change recommendation."}
    in_sensor = [row for row in missing if bool(row["predicted_in_sensor"])]
    outside_sensor = [row for row in missing if not bool(row["predicted_in_sensor"])]
    nearby = [row for row in in_sensor if bool(row["fresh_detector_candidate_within_v3_assignment_bound"])]
    not_nearby = [row for row in in_sensor if not bool(row["fresh_detector_candidate_within_v3_assignment_bound"])]
    if len(missing) and len(outside_sensor) / len(missing) >= 0.80:
        classification = "evidence_consistent_with_nominal_cell_extrapolation_outside_sensor; requires human review"
    elif len(in_sensor) and len(nearby) / len(in_sensor) >= 0.50:
        classification = "evidence_consistent_with_in_sensor_detector_presence_but_unassigned_cells; indexing_or_reassignment boundary remains plausible"
    elif len(in_sensor) and len(not_nearby) / len(in_sensor) >= 0.50:
        classification = "evidence_consistent_with_in_sensor_nominal_cells_without fresh detector support; visible-target extent_or_detector boundary remains plausible"
    else:
        classification = "mixed coverage evidence; no single shortfall mechanism dominates"
    return {
        "classification": classification,
        "nominal_cell_count": NOMINAL_GRID_SIZE * NOMINAL_GRID_SIZE,
        "missing_nominal_cell_count": len(missing),
        "missing_predicted_inside_sensor_count": len(in_sensor),
        "missing_predicted_outside_sensor_count": len(outside_sensor),
        "missing_inside_sensor_with_fresh_detector_candidate_within_bound": len(nearby),
        "missing_inside_sensor_without_fresh_detector_candidate_within_bound": len(not_nearby),
        "important_limit": "This classifies image-space coverage evidence only. It neither changes 50x50/40-row assumptions nor chooses a transform, rank, orientation, machine origin, or physical part location.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for compact CSV: {path.name}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_nominal_coverage(gray: np.ndarray, feature_rows: list[dict[str, Any]], cell_rows: list[dict[str, Any]], output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(gray[::stride, ::stride], cmap="gray", origin="upper", interpolation="nearest")
    assigned = np.asarray([[float(row["predicted_x_px"]), float(row["predicted_y_px"])] for row in cell_rows if bool(row["assigned_in_v3"])], dtype=np.float64)
    if len(assigned):
        axis.scatter(assigned[:, 0] / stride, assigned[:, 1] / stride, s=3.0, c="cyan", linewidths=0, label="V3 assigned cell")
    missing_inside_nearby = np.asarray([[float(row["predicted_x_px"]), float(row["predicted_y_px"])] for row in cell_rows if (not bool(row["assigned_in_v3"]) and bool(row["predicted_in_sensor"]) and bool(row["fresh_detector_candidate_within_v3_assignment_bound"]))], dtype=np.float64)
    if len(missing_inside_nearby):
        axis.scatter(missing_inside_nearby[:, 0] / stride, missing_inside_nearby[:, 1] / stride, s=4.0, c="orange", marker="+", linewidths=0.45, label="missing / detector nearby")
    missing_inside_no_candidate = np.asarray([[float(row["predicted_x_px"]), float(row["predicted_y_px"])] for row in cell_rows if (not bool(row["assigned_in_v3"]) and bool(row["predicted_in_sensor"]) and not bool(row["fresh_detector_candidate_within_v3_assignment_bound"]))], dtype=np.float64)
    if len(missing_inside_no_candidate):
        axis.scatter(missing_inside_no_candidate[:, 0] / stride, missing_inside_no_candidate[:, 1] / stride, s=3.5, c="red", marker="x", linewidths=0.4, label="missing / no detector nearby")
    outside = np.asarray([[float(row["predicted_x_px"]), float(row["predicted_y_px"])] for row in cell_rows if not bool(row["predicted_in_sensor"])], dtype=np.float64)
    if len(outside):
        axis.scatter(outside[:, 0] / stride, outside[:, 1] / stride, s=2.0, c="magenta", marker=".", linewidths=0, label="nominal prediction outside sensor")
    axis.set_title("Method-#2 nominal 50x50 image-lattice coverage evidence\nImage coordinates only — no grid-size/gate/config change")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=7, framealpha=0.88)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_profiles(row_profile: list[dict[str, Any]], col_profile: list[dict[str, Any]], output_path: Path) -> None:
    rows = np.arange(NOMINAL_GRID_SIZE)
    row_counts = np.asarray([int(row["assigned_cell_count"]) for row in row_profile], dtype=np.int64)
    col_counts = np.asarray([int(row["assigned_cell_count"]) for row in col_profile], dtype=np.int64)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=160, sharex=True)
    axes[0].bar(rows, row_counts, color="#3478bf", width=0.85)
    axes[0].set_ylabel("assigned cells")
    axes[0].set_title("Per-image-lattice row coverage (nominal indices only)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(rows, col_counts, color="#2f9e6d", width=0.85)
    axes[1].set_ylabel("assigned cells")
    axes[1].set_xlabel("image-lattice index 0..49 (not machine D coordinate)")
    axes[1].set_title("Per-image-lattice column coverage (nominal indices only)")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.dot_grid.is_file():
        raise FileNotFoundError(f"Required immutable DotGrid TIFF not found: {args.dot_grid}")
    if not args.v3_features.is_file():
        raise FileNotFoundError(f"Required completed V3 feature CSV not found: {args.v3_features}")
    prepare_output_directory(args.output_dir, args.overwrite)

    feature_rows = load_v3_features(args.v3_features)
    h_matrix, inliers, residual, pitch = fit_image_lattice_only(feature_rows)
    channels, metadata = read_tiff(args.dot_grid)
    gray = grayscale(channels)
    coarse_points, _, _ = dot_grid_candidates(gray)
    roi, roi_metrics = density_roi(coarse_points, gray.shape)
    fresh_points, _, fresh_detector_metrics = refined_feature_candidates(gray, roi, "dot")
    fresh_points = np.asarray(fresh_points, dtype=np.float64)

    cells = np.asarray([[float(col), float(row)] for row in range(NOMINAL_GRID_SIZE) for col in range(NOMINAL_GRID_SIZE)], dtype=np.float64)
    predicted = project(h_matrix, cells)
    in_sensor = sensor_membership(predicted, gray.shape[0], gray.shape[1])
    nearest_distance = nearest_detector_distances(predicted, fresh_points)
    darkness = local_darkness(gray, predicted)
    assigned_map = {(int(row["col"]), int(row["row"])): row for row in feature_rows}
    assignment_bound = V3_ASSIGNMENT_MAX_PITCH * pitch
    cell_rows: list[dict[str, Any]] = []
    for flat, (col_float, row_float) in enumerate(cells):
        col, row = int(col_float), int(row_float)
        assigned = assigned_map.get((col, row))
        cell_rows.append({
            "image_lattice_col_index_0_to_49": col,
            "image_lattice_row_index_0_to_49": row,
            "predicted_x_px": float(predicted[flat, 0]),
            "predicted_y_px": float(predicted[flat, 1]),
            "predicted_in_sensor": bool(in_sensor[flat]),
            "assigned_in_v3": bool(assigned is not None),
            "assigned_source_candidate_index": None if assigned is None else int(assigned["source_candidate_index"]),
            "assigned_raw_x_px": None if assigned is None else float(assigned["raw_x_px"]),
            "assigned_raw_y_px": None if assigned is None else float(assigned["raw_y_px"]),
            "nearest_fresh_detector_candidate_distance_px": float(nearest_distance[flat]),
            "fresh_detector_candidate_within_v3_assignment_bound": bool(nearest_distance[flat] <= assignment_bound),
            "local_mean_darkness_0_to_1": None if not math.isfinite(float(darkness[flat])) else float(darkness[flat]),
        })

    row_profile = profile_rows(feature_rows, "row")
    col_profile = profile_rows(feature_rows, "col")
    visible_rows = [int(row["image_lattice_row_index_0_to_49"]) for row in row_profile if bool(row["has_any_assigned_cell"])]
    visible_cols = [int(row["image_lattice_col_index_0_to_49"]) for row in col_profile if bool(row["has_any_assigned_cell"])]
    evidence = classify_coverage_evidence(cell_rows)

    cells_csv = args.output_dir / "method2_v3_nominal_50x50_cell_coverage.csv"
    row_csv = args.output_dir / "method2_v3_row_coverage_profile.csv"
    col_csv = args.output_dir / "method2_v3_column_coverage_profile.csv"
    coverage_overlay = args.output_dir / "method2_v3_nominal_coverage_evidence_overlay.png"
    profile_plot = args.output_dir / "method2_v3_row_column_coverage_profiles.png"
    summary_path = args.output_dir / "independent_method2_dotgrid_coverage_definition_summary.json"
    write_csv(cells_csv, cell_rows)
    write_csv(row_csv, row_profile)
    write_csv(col_csv, col_profile)
    plot_nominal_coverage(gray, feature_rows, cell_rows, coverage_overlay)
    plot_profiles(row_profile, col_profile, profile_plot)

    summary = {
        "audit_type": "read-only DotGrid coverage-definition audit; no grid-size/coverage-gate/config change or calibration selection",
        "purpose": "Explain the V3 39-row fixed-coverage shortfall by separating nominal cell sensor visibility, fresh dot-detector proximity, and assignment occupancy.",
        "inputs": {
            "dot_grid": metadata,
            "v3_features_csv": str(args.v3_features),
            "v3_feature_row_count": int(len(feature_rows)),
        },
        "nominal_grid_assumption_under_audit": {
            "size": [NOMINAL_GRID_SIZE, NOMINAL_GRID_SIZE],
            "status": "documented nominal assumption is inspected, never modified by this script",
            "important_limit": "Image-lattice indices remain orientation-free and are not physical D coordinates or machine axes.",
        },
        "image_lattice_refit_for_coverage_only": {
            "method": "robust fit from compact V3 assigned image-lattice cells to raw camera pixels; used only to predict nominal-cell visibility/proximity",
            "inlier_count": int(inliers.sum()),
            "inlier_fraction": float(inliers.mean()),
            "inlier_rmse_px": float(math.sqrt(float(np.mean(residual[inliers] ** 2)))),
            "detected_inlier_camera_dot_pitch_px": pitch,
            "fresh_detector_assignment_proximity_bound_px": assignment_bound,
        },
        "fresh_detector_for_missing_cell_evidence": {
            "automatic_roi": roi_metrics,
            "metrics": fresh_detector_metrics,
            "important_limit": "Fresh detector points support only image-space coverage classification; they are not labels or a calibration target.",
        },
        "profile": {
            "assigned_unique_cell_count": int(len(feature_rows)),
            "assigned_image_lattice_row_count": int(len(visible_rows)),
            "assigned_image_lattice_column_count": int(len(visible_cols)),
            "assigned_row_runs": contiguous_runs(visible_rows),
            "assigned_column_runs": contiguous_runs(visible_cols),
            "predeclared_v3_coverage_rule": "unique cells>=1200 and unique image-lattice rows/columns>=40",
            "v3_row_gate_status_unchanged": bool(len(visible_rows) >= 40),
            "v3_column_gate_status_unchanged": bool(len(visible_cols) >= 40),
        },
        "coverage_shortfall_evidence": evidence,
        "recommendation": "hold_all_method2_transform_candidates; perform human review of coverage-definition evidence before any separately approved grid-size or coverage-threshold design decision",
        "prohibitions": [
            "Does not write raw TIFF/CSV.",
            "Does not modify the nominal 50x50 assumption or fixed 40-row/column coverage gate.",
            "Does not read or edit calibration_v1.yaml, choose a transform/rank/orientation, or assert machine origin/part location.",
            "Does not access A/B manufacturing TIFF, registered XCT, weak target/support, model, checkpoint, training, decoder, or candidate output.",
            "Does not change camera-primary XCT-derived continuous quality candidate reporting.",
        ],
        "storage_policy": "writes three compact CSVs, one JSON summary, and exactly two deterministic QC overlays; no dense crop, rectified image, mask, heatmap, target, model output, transform config, or raw data is persisted",
        "outputs": {
            "nominal_cell_coverage_csv": str(cells_csv),
            "row_profile_csv": str(row_csv),
            "column_profile_csv": str(col_csv),
            "summary_json": str(summary_path),
            "nominal_coverage_overlay_png": str(coverage_overlay),
            "row_column_profiles_png": str(profile_plot),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("DotGrid coverage-definition audit complete. No raw TIFF/CSV, coverage gate, calibration config, model, target, checkpoint, decoder, or candidate output was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
