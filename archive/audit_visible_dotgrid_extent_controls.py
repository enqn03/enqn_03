#!/usr/bin/env python3
"""Validate four human-reviewed visible DotGrid outer controls without calibrating.
Inputs are the immutable DotGrid TIFF, a compact four-point control JSON from
select_visible_dotgrid_extent_controls.py, and the completed V3 feature CSV.
The audit snaps clicks only after a frozen image-space tolerance check, verifies
convex human panel geometry and edge-dot support, and compares that panel to
V3 assigned/predicted image-lattice footprint.
It does not select a physical D origin, fit or deploy a machine calibration,
change grid size or the 40-row gate, edit config, choose rank/orientation, or
access production TIFF/XCT/target/model/candidate data.
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
EXPECTED_ORDER = ["top_left_outer_dot_center", "top_right_outer_dot_center", "bottom_right_outer_dot_center", "bottom_left_outer_dot_center"]
CLICK_SNAP_MAX_PITCH = 0.60
EDGE_SUPPORT_MAX_PITCH = 0.55
NOMINAL_GRID_SIZE = GRID_SIZE
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid TIFF, opened read-only via memmap(mode='r').")
    parser.add_argument("--controls-json", required=True, type=Path, help="Compact human outer-dot control JSON from the selector.")
    parser.add_argument("--v3-features", required=True, type=Path, help="Completed V3 compact feature CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New ignored directory for compact validation CSV/JSON and at most two overlays.")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace only an existing output directory after review.")
    return parser.parse_args()
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def load_controls(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    controls = payload.get("controls")
    if not isinstance(controls, list) or len(controls) != 4:
        raise ValueError("Controls JSON must contain exactly four controls.")
    ordered = sorted(controls, key=lambda row: int(row.get("selection_order", 0)))
    names = [str(row.get("semantic_name", "")) for row in ordered]
    if names != EXPECTED_ORDER:
        raise ValueError(f"Controls must be ordered exactly as {EXPECTED_ORDER}; received {names}")
    output: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        value = row.get("raw_camera_xy_px")
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Control {index} lacks raw_camera_xy_px=[x,y].")
        x, y = float(value[0]), float(value[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Control {index} has non-finite raw coordinate.")
        output.append({"selection_order": index, "semantic_name": EXPECTED_ORDER[index - 1], "clicked_x_px": x, "clicked_y_px": y})
    return output
def load_v3_features(path: Path) -> list[dict[str, Any]]:
    required = {"image_lattice_col_index_0_to_49", "image_lattice_row_index_0_to_49", "raw_x_px", "raw_y_px"}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"V3 feature CSV missing required fields: {sorted(required.difference(set(reader.fieldnames or [])))}")
        rows: list[dict[str, Any]] = []
        for record in reader:
            rows.append({
                "col": int(record["image_lattice_col_index_0_to_49"]),
                "row": int(record["image_lattice_row_index_0_to_49"]),
                "raw_x_px": float(record["raw_x_px"]),
                "raw_y_px": float(record["raw_y_px"]),
            })
    if len(rows) < 8:
        raise ValueError("V3 feature CSV requires at least eight rows.")
    return rows
def nearest_indices_and_distances(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for start in range(0, len(query), 256):
        chunk = query[start:start + 256]
        delta = chunk[:, None, :] - reference[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
        index = np.argmin(squared, axis=1)
        indices.append(index)
        distances.append(np.sqrt(squared[np.arange(len(chunk)), index]))
    return np.concatenate(indices), np.concatenate(distances)
def cross(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])
def is_strictly_convex_quad(points: np.ndarray) -> tuple[bool, list[float]]:
    values: list[float] = []
    for index in range(4):
        first = points[(index + 1) % 4] - points[index]
        second = points[(index + 2) % 4] - points[(index + 1) % 4]
        values.append(cross(first, second))
    positive = all(value > 0.0 for value in values)
    negative = all(value < 0.0 for value in values)
    return bool(positive or negative), values
def points_inside_convex_quad(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    signs = []
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        edge = end - start
        vector = points - start
        signs.append(edge[0] * vector[:, 1] - edge[1] * vector[:, 0])
    stacked = np.stack(signs, axis=1)
    return np.all(stacked >= -1.0e-9, axis=1) | np.all(stacked <= 1.0e-9, axis=1)
def point_segment_distance_and_fraction(points: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = second - first
    denom = float(np.dot(direction, direction))
    if denom <= 0.0:
        raise ValueError("Control edge has zero length.")
    fraction = ((points - first) @ direction) / denom
    projection = first + fraction[:, None] * direction[None, :]
    distance = np.linalg.norm(points - projection, axis=1)
    return distance, fraction
def contiguous_run_count(values: np.ndarray, gap_limit: float) -> int:
    if len(values) == 0:
        return 0
    ordered = np.sort(values)
    return int(1 + np.sum(np.diff(ordered) > gap_limit))
def edge_support_records(quad: np.ndarray, fresh_points: np.ndarray, pitch: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    max_distance = EDGE_SUPPORT_MAX_PITCH * pitch
    for index in range(4):
        first = quad[index]
        second = quad[(index + 1) % 4]
        distance, fraction = point_segment_distance_and_fraction(fresh_points, first, second)
        mask = (distance <= max_distance) & (fraction >= -0.05) & (fraction <= 1.05)
        selected_fraction = np.sort(fraction[mask])
        edge_length = float(np.linalg.norm(second - first))
        expected_intervals = max(1.0, edge_length / pitch)
        records.append({
            "edge_index": index + 1,
            "edge_name": f"{EXPECTED_ORDER[index]}_to_{EXPECTED_ORDER[(index + 1) % 4]}",
            "edge_length_px": edge_length,
            "fresh_candidate_count_within_edge_band": int(mask.sum()),
            "edge_band_max_distance_px": max_distance,
            "fraction_min": None if len(selected_fraction) == 0 else float(selected_fraction.min()),
            "fraction_max": None if len(selected_fraction) == 0 else float(selected_fraction.max()),
            "fraction_contiguous_run_count": contiguous_run_count(selected_fraction, 2.5 / expected_intervals),
            "approx_intervals_at_camera_pitch": float(expected_intervals),
            "important_limit": "Edge candidate count is an image-space visible-panel metric, not a physical target count or D-coordinate index.",
        })
    return records
def fit_v3_image_lattice(features: list[dict[str, Any]]) -> np.ndarray:
    source = np.asarray([[float(row["col"]), float(row["row"])] for row in features], dtype=np.float64)
    raw = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in features], dtype=np.float64)
    h_matrix, _, _, _ = inlier_fit(source, raw)
    return h_matrix
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty validation CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def plot_controls_and_edges(gray: np.ndarray, quad: np.ndarray, fresh_points: np.ndarray, edge_records: list[dict[str, Any]], output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    display = gray[::stride, ::stride]
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    inside = points_inside_convex_quad(fresh_points, quad)
    if inside.any():
        axis.scatter(fresh_points[inside, 0] / stride, fresh_points[inside, 1] / stride, s=3.0, c="cyan", linewidths=0, label="fresh candidates inside controls")
    outline = np.vstack([quad, quad[0]])
    axis.plot(outline[:, 0] / stride, outline[:, 1] / stride, color="yellow", linewidth=1.0, label="human outer-dot control quad")
    axis.scatter(quad[:, 0] / stride, quad[:, 1] / stride, s=35, c="magenta", marker="x", linewidths=1.2, label="snapped control dots")
    for index, record in enumerate(edge_records, start=1):
        midpoint = 0.5 * (quad[index - 1] + quad[index % 4])
        axis.text(midpoint[0] / stride, midpoint[1] / stride, f"E{index}: {record['fresh_candidate_count_within_edge_band']}", color="yellow", fontsize=7, ha="center", va="center", bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})
    axis.set_title("Human-reviewed visible DotGrid outer extent\nImage extent only — no physical grid/gate/config decision")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=7, framealpha=0.88)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def plot_v3_footprint(gray: np.ndarray, quad: np.ndarray, features: list[dict[str, Any]], nominal_predicted: np.ndarray, output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(gray[::stride, ::stride], cmap="gray", origin="upper", interpolation="nearest")
    assigned = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in features], dtype=np.float64)
    assigned_inside = points_inside_convex_quad(assigned, quad)
    predicted_inside = points_inside_convex_quad(nominal_predicted, quad)
    if predicted_inside.any():
        axis.scatter(nominal_predicted[predicted_inside, 0] / stride, nominal_predicted[predicted_inside, 1] / stride, s=2.0, c="orange", marker="+", linewidths=0.3, label="nominal V3 cell predicted inside controls")
    if assigned_inside.any():
        axis.scatter(assigned[assigned_inside, 0] / stride, assigned[assigned_inside, 1] / stride, s=3.5, c="cyan", linewidths=0, label="V3 assigned inside controls")
    if (~assigned_inside).any():
        axis.scatter(assigned[~assigned_inside, 0] / stride, assigned[~assigned_inside, 1] / stride, s=4.0, c="red", marker="x", linewidths=0.5, label="V3 assigned outside controls")
    outline = np.vstack([quad, quad[0]])
    axis.plot(outline[:, 0] / stride, outline[:, 1] / stride, color="yellow", linewidth=1.0, label="human outer-dot control quad")
    axis.set_title("V3 image-lattice footprint vs human visible DotGrid extent\nNo D-origin, rank, transform, or gate selection")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=7, framealpha=0.88)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    if not args.dot_grid.is_file():
        raise FileNotFoundError(f"Required immutable DotGrid TIFF not found: {args.dot_grid}")
    if not args.controls_json.is_file():
        raise FileNotFoundError(f"Required human controls JSON not found: {args.controls_json}")
    if not args.v3_features.is_file():
        raise FileNotFoundError(f"Required completed V3 feature CSV not found: {args.v3_features}")
    prepare_output_directory(args.output_dir, args.overwrite)
    controls = load_controls(args.controls_json)
    features = load_v3_features(args.v3_features)
    channels, metadata = read_tiff(args.dot_grid)
    gray = grayscale(channels)
    coarse_points, _, _ = dot_grid_candidates(gray)
    roi, roi_metrics = density_roi(coarse_points, gray.shape)
    fresh_points, _, detector_metrics = refined_feature_candidates(gray, roi, "dot")
    fresh_points = np.asarray(fresh_points, dtype=np.float64)
    v3_raw = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in features], dtype=np.float64)
    pitch = nearest_camera_dot_pitch_px(v3_raw)
    if pitch is None or not math.isfinite(pitch) or pitch <= 0.0:
        raise RuntimeError("Could not estimate positive DotGrid camera pitch from V3 assigned features.")
    snap_bound = CLICK_SNAP_MAX_PITCH * float(pitch)
    clicked = np.asarray([[float(row["clicked_x_px"]), float(row["clicked_y_px"])] for row in controls], dtype=np.float64)
    snap_indices, snap_distances = nearest_indices_and_distances(clicked, fresh_points)
    snapped = fresh_points[snap_indices]
    distinct = len(set(int(value) for value in snap_indices)) == 4
    convex, cross_values = is_strictly_convex_quad(snapped)
    all_snapped = bool(np.all(snap_distances <= snap_bound))
    all_inside_sensor = bool(np.all((snapped[:, 0] >= 0.0) & (snapped[:, 0] < gray.shape[1]) & (snapped[:, 1] >= 0.0) & (snapped[:, 1] < gray.shape[0])))
    control_rows: list[dict[str, Any]] = []
    for index, control in enumerate(controls):
        control_rows.append({
            **control,
            "snapped_fresh_candidate_index": int(snap_indices[index]),
            "snapped_x_px": float(snapped[index, 0]),
            "snapped_y_px": float(snapped[index, 1]),
            "click_to_fresh_candidate_distance_px": float(snap_distances[index]),
            "snap_bound_px": snap_bound,
            "snap_pass": bool(snap_distances[index] <= snap_bound),
        })
    edge_rows = edge_support_records(snapped, fresh_points, float(pitch))
    h_matrix = fit_v3_image_lattice(features)
    nominal_cells = np.asarray([[float(col), float(row)] for row in range(NOMINAL_GRID_SIZE) for col in range(NOMINAL_GRID_SIZE)], dtype=np.float64)
    nominal_predicted = project(h_matrix, nominal_cells)
    assigned_inside = points_inside_convex_quad(v3_raw, snapped)
    fresh_inside = points_inside_convex_quad(fresh_points, snapped)
    nominal_inside = points_inside_convex_quad(nominal_predicted, snapped)
    summary_row = {
        "v3_assigned_cell_count": int(len(features)),
        "v3_assigned_inside_human_quad_count": int(assigned_inside.sum()),
        "v3_assigned_outside_human_quad_count": int((~assigned_inside).sum()),
        "v3_assigned_inside_human_quad_fraction": float(assigned_inside.mean()),
        "fresh_detector_candidate_count": int(len(fresh_points)),
        "fresh_detector_candidate_inside_human_quad_count": int(fresh_inside.sum()),
        "fresh_detector_candidate_inside_human_quad_fraction": float(fresh_inside.mean()),
        "nominal_50x50_predicted_inside_human_quad_count": int(nominal_inside.sum()),
        "nominal_50x50_predicted_outside_human_quad_count": int((~nominal_inside).sum()),
        "important_limit": "Human visible extent is image-space evidence only; these counts do not alter grid size, coverage gate, physical target convention, D origin, machine coordinates, transform/rank, or config.",
    }
    controls_csv = args.output_dir / "visible_dotgrid_extent_control_validation.csv"
    edges_csv = args.output_dir / "visible_dotgrid_extent_edge_support.csv"
    footprint_csv = args.output_dir / "visible_dotgrid_extent_v3_footprint_summary.csv"
    controls_overlay = args.output_dir / "visible_dotgrid_extent_controls_overlay.png"
    footprint_overlay = args.output_dir / "visible_dotgrid_extent_v3_footprint_overlay.png"
    summary_path = args.output_dir / "visible_dotgrid_extent_validation_summary.json"
    write_csv(controls_csv, control_rows)
    write_csv(edges_csv, edge_rows)
    write_csv(footprint_csv, [summary_row])
    plot_controls_and_edges(gray, snapped, fresh_points, edge_rows, controls_overlay)
    plot_v3_footprint(gray, snapped, features, nominal_predicted, footprint_overlay)
    min_edge_candidates = min(int(row["fresh_candidate_count_within_edge_band"]) for row in edge_rows)
    validation_pass = bool(all_snapped and distinct and convex and all_inside_sensor and min_edge_candidates >= 3)
    summary = {
        "audit_type": "human-reviewed visible DotGrid outer-extent validation; no physical calibration or grid/gate decision",
        "purpose": "Validate four human-selected visible outer dot centres and compare the resulting image-space panel extent to V3 correspondence footprint before any separate policy design review.",
        "inputs": {
            "dot_grid": metadata,
            "controls_json": str(args.controls_json),
            "v3_features_csv": str(args.v3_features),
        },
        "human_control_validation": {
            "required_order": EXPECTED_ORDER,
            "click_snap_bound_px": snap_bound,
            "all_clicks_snap_to_distinct_fresh_candidates_within_bound": bool(all_snapped and distinct),
            "all_snapped_controls_inside_sensor": all_inside_sensor,
            "strictly_convex_ordered_quad": convex,
            "convex_cross_products": cross_values,
            "minimum_edge_fresh_candidate_count": min_edge_candidates,
            "edge_support_minimum_for_extent_evidence": 3,
            "all_control_validity_checks_pass": validation_pass,
            "important_limit": "Validation does not label D origin, physical units, machine orientation, or calibration control points.",
        },
        "image_space_panel_and_v3_footprint": summary_row,
        "fresh_detector": {"automatic_roi": roi_metrics, "metrics": detector_metrics},
        "edge_support": edge_rows,
        "recommendation": "eligible_for_human_review_of_visible_panel_extent_only" if validation_pass else "hold_extent_interpretation; recheck the four ordered visible outer dot clicks before any policy discussion",
        "prohibitions": [
            "Does not write raw TIFF/CSV or modify the controls JSON.",
            "Does not change nominal grid size or coverage gate.",
            "Does not read or edit calibration_v1.yaml, choose transform/rank/orientation, or assert machine origin/part location.",
            "Does not access A/B manufacturing TIFF, XCT, weak target/support, model, checkpoint, training, decoder, or candidate output.",
            "Does not change camera-primary XCT-derived continuous quality candidate reporting.",
        ],
        "storage_policy": "writes three compact CSVs, one JSON summary, and exactly two deterministic QC overlays; no dense crop, rectified image, mask, heatmap, target, model output, transform config, or raw data is persisted",
        "outputs": {
            "control_validation_csv": str(controls_csv),
            "edge_support_csv": str(edges_csv),
            "v3_footprint_summary_csv": str(footprint_csv),
            "summary_json": str(summary_path),
            "controls_overlay_png": str(controls_overlay),
            "v3_footprint_overlay_png": str(footprint_overlay),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Visible DotGrid extent controls validated. No raw TIFF/CSV, grid/coverage policy, calibration, model, target, checkpoint, decoder, or candidate output was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
