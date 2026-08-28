#!/usr/bin/env python3
"""Read-only calibration design review for extent, coverage, and orientation.

This audit compares frozen detector ROI and existing V2 human extent evidence,
measures V3 row/column occupancy without changing the fixed coverage gate, and
quantifies residual ties in the existing 192-hypothesis ranking. It never
selects a final extent/orientation, edits controls/configuration, refits a
calibration for deployment, or accesses manufacturing/XCT/model data.
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

from audit_independent_metrology_fiducials_refined import grayscale, read_tiff

NOMINAL_GRID_SIZE = 50
MIN_COVERAGE_ROWS = 40
MIN_COVERAGE_COLUMNS = 50
FROZEN_SNAP_BOUND_PX = 8.738353576116276
TOP_TIE_TOLERANCE_PX = 1.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path)
    parser.add_argument("--v2-validation-summary", required=True, type=Path)
    parser.add_argument("--v2-controls-csv", required=True, type=Path)
    parser.add_argument("--outer-boundary-csv", required=True, type=Path)
    parser.add_argument("--v3-features", required=True, type=Path)
    parser.add_argument("--orientation-ranking-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it before using --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in {"", "nan", "NaN"}:
            return float(row[name])
    raise KeyError(f"None of the required numeric fields are present: {names}")


def as_int(row: dict[str, str], *names: str) -> int:
    return int(round(as_float(row, *names)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def point_inside_rectangle(points: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)


def point_inside_convex_quad(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    signs: list[np.ndarray] = []
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        edge = end - start
        vector = points - start
        signs.append(edge[0] * vector[:, 1] - edge[1] * vector[:, 0])
    stacked = np.stack(signs, axis=1)
    return np.all(stacked >= -1.0e-9, axis=1) | np.all(stacked <= 1.0e-9, axis=1)


def polygon_area(points: np.ndarray) -> float:
    return float(0.5 * abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))))


def load_human_quad(path: Path) -> np.ndarray:
    rows = read_csv_rows(path)
    if len(rows) != 4:
        raise ValueError(f"Expected exactly four snapped V2 controls, found {len(rows)} in {path}")
    ordered = sorted(rows, key=lambda row: int(row["selection_order"]))
    return np.asarray([[as_float(row, "snapped_x_px"), as_float(row, "snapped_y_px")] for row in ordered], dtype=np.float64)


def load_v3(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"V3 feature CSV is empty: {path}")
    points: list[list[float]] = []
    labels: list[list[int]] = []
    for row in rows:
        try:
            col = as_int(row, "col", "image_lattice_col_index_0_to_49")
            lattice_row = as_int(row, "row", "image_lattice_row_index_0_to_49")
            x = as_float(row, "raw_x_px")
            y = as_float(row, "raw_y_px")
        except (KeyError, ValueError) as error:
            raise ValueError(f"V3 feature CSV lacks required compact fields: {error}") from error
        points.append([x, y])
        labels.append([col, lattice_row])
    return np.asarray(points, dtype=np.float64), np.asarray(labels, dtype=np.int64), np.asarray(rows, dtype=object)


def occupancy_metrics(points: np.ndarray, labels: np.ndarray, inside: np.ndarray, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_labels = labels[inside]
    selected_points = points[inside]
    rows_present = sorted(set(int(value) for value in selected_labels[:, 1])) if len(selected_labels) else []
    cols_present = sorted(set(int(value) for value in selected_labels[:, 0])) if len(selected_labels) else []
    row_counts = {row: int(np.sum(selected_labels[:, 1] == row)) for row in rows_present}
    col_counts = {col: int(np.sum(selected_labels[:, 0] == col)) for col in cols_present}
    missing_rows = [row for row in range(NOMINAL_GRID_SIZE) if row not in rows_present]
    missing_cols = [col for col in range(NOMINAL_GRID_SIZE) if col not in cols_present]
    summary = {
        "extent_candidate": name,
        "point_count_in_extent": int(inside.sum()),
        "point_fraction_in_extent": float(inside.mean()),
        "unique_assigned_rows": int(len(rows_present)),
        "unique_assigned_columns": int(len(cols_present)),
        "row_min": None if not rows_present else int(min(rows_present)),
        "row_max": None if not rows_present else int(max(rows_present)),
        "column_min": None if not cols_present else int(min(cols_present)),
        "column_max": None if not cols_present else int(max(cols_present)),
        "missing_nominal_rows": ";".join(str(value) for value in missing_rows),
        "missing_nominal_columns": ";".join(str(value) for value in missing_cols),
        "coverage_rows_gate_pass": bool(len(rows_present) >= MIN_COVERAGE_ROWS),
        "coverage_columns_gate_pass": bool(len(cols_present) >= MIN_COVERAGE_COLUMNS),
        "fixed_rows_ge_40_and_columns_ge_50_pass": bool(len(rows_present) >= MIN_COVERAGE_ROWS and len(cols_present) >= MIN_COVERAGE_COLUMNS),
        "important_limit": "Occupancy is descriptive image-space evidence; it cannot promote an extent or change the fixed coverage gate.",
    }
    detail_rows: list[dict[str, Any]] = []
    for row in range(NOMINAL_GRID_SIZE):
        detail_rows.append({
            "extent_candidate": name,
            "lattice_row": row,
            "assigned_point_count": row_counts.get(row, 0),
            "row_present": row in rows_present,
        })
    for col in range(NOMINAL_GRID_SIZE):
        detail_rows.append({
            "extent_candidate": name,
            "lattice_row": f"column_{col}",
            "assigned_point_count": col_counts.get(col, 0),
            "row_present": col in cols_present,
        })
    return summary, detail_rows


def orientation_review(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Orientation ranking CSV is empty: {path}")
    ranked: list[dict[str, Any]] = []
    for row in rows:
        ranked.append({
            "rank": int(row["rank"]),
            "loo_rmse_px": float(row["loo_rmse_px"]),
            "fit_rmse_px": float(row["fit_rmse_px"]),
            "orientation": row["orientation"],
            "assignment": row["screen_A_to_D_machine_parts"],
            "corner_index_order": row.get("corner_index_order", ""),
        })
    ranked.sort(key=lambda item: item["rank"])
    best = ranked[0]
    tie_rows = [row for row in ranked if abs(float(row["loo_rmse_px"]) - float(best["loo_rmse_px"])) <= TOP_TIE_TOLERANCE_PX]
    summary = {
        "hypothesis_count": int(len(ranked)),
        "top_rank": best,
        "top_residual_tie_count": int(len(tie_rows)),
        "top_residual_tie_ranks": ";".join(str(row["rank"]) for row in tie_rows),
        "top_residual_tie_delta_max_px": float(max(abs(float(row["loo_rmse_px"]) - float(best["loo_rmse_px"])) for row in tie_rows)),
        "independent_asymmetric_anchor_available": False,
        "orientation_ambiguity_reduced": bool(len(tie_rows) == 1),
        "decision": "hold_orientation; residual tie and no cross-camera/asymmetric anchor in this audit",
        "important_limit": "Residual ranking is candidate evidence only; it does not select rank/orientation or edit calibration_v1.yaml.",
    }
    return summary, ranked[: min(12, len(ranked))]


def plot_extent_overlay(gray: np.ndarray, frozen_roi: tuple[float, float, float, float], human_quad: np.ndarray, v3_points: np.ndarray, extent_rows: list[dict[str, Any]], output: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(gray[::stride, ::stride], cmap="gray", origin="upper", interpolation="nearest")
    human_inside = point_inside_convex_quad(v3_points, human_quad)
    if human_inside.any():
        axis.scatter(v3_points[human_inside, 0] / stride, v3_points[human_inside, 1] / stride, s=2.5, c="cyan", linewidths=0, label="V3 assigned inside human extent")
    if (~human_inside).any():
        axis.scatter(v3_points[~human_inside, 0] / stride, v3_points[~human_inside, 1] / stride, s=3.5, c="red", marker="x", linewidths=0.4, label="V3 assigned outside human extent")
    outline = np.vstack([human_quad, human_quad[0]])
    axis.plot(outline[:, 0] / stride, outline[:, 1] / stride, color="yellow", linewidth=1.2, label="V2 human extent (held)")
    x0, y0, x1, y1 = frozen_roi
    axis.plot(np.asarray([x0, x1, x1, x0, x0]) / stride, np.asarray([y0, y0, y1, y1, y0]) / stride, color="lime", linestyle="--", linewidth=1.0, label="frozen detector ROI")
    for row in extent_rows:
        if row["extent_candidate"] in {"frozen_detector_roi", "v2_human_quad"}:
            pass
    axis.set_title("Calibration design review: extent candidates\nNo extent/gate/calibration selection")
    axis.set_xlabel("display raw camera x [px]")
    axis.set_ylabel("display raw camera y [px]")
    axis.legend(loc="upper right", fontsize=7, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_orientation(ranked: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), dpi=160)
    labels = [f"r{row['rank']}\n{row['orientation']}" for row in ranked]
    values = [float(row["loo_rmse_px"]) for row in ranked]
    colors = ["tab:red" if abs(value - values[0]) <= TOP_TIE_TOLERANCE_PX else "tab:blue" for value in values]
    axis.bar(np.arange(len(values)), values, color=colors)
    axis.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("LOO RMSE [px]")
    axis.set_title("Top orientation hypotheses\nRed bars are residual ties; no rank selected")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    required = [
        (args.dot_grid, "immutable DotGrid TIFF"),
        (args.v2_validation_summary, "V2 validation summary"),
        (args.v2_controls_csv, "V2 control validation CSV"),
        (args.outer_boundary_csv, "outer-boundary diagnostic CSV"),
        (args.v3_features, "V3 feature CSV"),
        (args.orientation_ranking_csv, "orientation ranking CSV"),
    ]
    for path, label in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    prepare_output_directory(args.output_dir, args.overwrite)

    with args.v2_validation_summary.open(encoding="utf-8") as handle:
        v2_summary = json.load(handle)
    if v2_summary.get("human_control_validation", {}).get("all_control_validity_checks_pass") is not False:
        raise ValueError("Expected the preserved V2 human extent strict-snap hold; refusing to review a different state.")
    if int(v2_summary.get("image_space_panel_and_v3_footprint", {}).get("nominal_50x50_predicted_outside_human_quad_count", -1)) < 0:
        raise ValueError("V2 summary lacks the expected nominal 50x50 footprint evidence.")

    human_quad = load_human_quad(args.v2_controls_csv)
    outer_rows = read_csv_rows(args.outer_boundary_csv)
    if len(outer_rows) != 4:
        raise ValueError(f"Expected four outer-boundary diagnostic rows, found {len(outer_rows)}")
    outer_classes = {row.get("evidence_class", "") for row in outer_rows}
    frozen_roi_values = v2_summary["fresh_detector"]["automatic_roi"]["roi_xyxy_px"]
    frozen_roi = tuple(float(value) for value in frozen_roi_values)
    if len(frozen_roi) != 4:
        raise ValueError("V2 summary has an invalid frozen ROI.")
    v3_points, v3_labels, _ = load_v3(args.v3_features)
    frozen_inside = point_inside_rectangle(v3_points, frozen_roi)
    human_inside = point_inside_convex_quad(v3_points, human_quad)
    frozen_summary, frozen_detail = occupancy_metrics(v3_points, v3_labels, frozen_inside, "frozen_detector_roi")
    human_summary, human_detail = occupancy_metrics(v3_points, v3_labels, human_inside, "v2_human_quad")
    extent_summaries = [frozen_summary, human_summary]
    extent_detail = frozen_detail + human_detail
    orientation_summary, orientation_rows = orientation_review(args.orientation_ranking_csv)
    dot_channels, dot_metadata = read_tiff(args.dot_grid)
    gray = grayscale(dot_channels)

    by_control_rows = []
    for row in outer_rows:
        by_control_rows.append({
            "selection_order": row.get("selection_order", ""),
            "semantic_name": row.get("semantic_name", ""),
            "evidence_class": row.get("evidence_class", ""),
            "current_frozen_snap_pass": row.get("current_frozen_snap_pass", ""),
            "local_lattice_evidence_pass": row.get("local_lattice_evidence_pass", ""),
            "current_frozen_nearest_distance_px": row.get("current_frozen_nearest_distance_px", ""),
            "nearest_local_candidate_distance_px": row.get("nearest_local_candidate_distance_px", ""),
            "nearest_v3_assigned_distance_px": row.get("nearest_v3_assigned_distance_px", ""),
            "nearest_nominal_prediction_distance_px": row.get("nearest_nominal_prediction_distance_px", ""),
            "extent_design_use": "evidence_only; no automatic extent construction",
        })

    extent_eligible = bool(
        len(outer_classes) == 1 and
        "current_detector_supported" not in outer_classes and
        all(row.get("local_lattice_evidence_pass") == "true" for row in outer_rows)
    )
    summary_path = args.output_dir / "calibration_design_review_summary.json"
    extent_csv = args.output_dir / "calibration_design_review_extent_candidates.csv"
    detail_csv = args.output_dir / "calibration_design_review_occupancy_by_row_column.csv"
    controls_csv = args.output_dir / "calibration_design_review_outer_control_evidence.csv"
    extent_overlay = args.output_dir / "calibration_design_review_extent_overlay.png"
    orientation_overlay = args.output_dir / "calibration_design_review_orientation_overlay.png"
    write_csv(extent_csv, extent_summaries)
    write_csv(detail_csv, extent_detail)
    write_csv(controls_csv, by_control_rows)
    plot_extent_overlay(gray, frozen_roi, human_quad, v3_points, extent_summaries, extent_overlay)
    plot_orientation(orientation_rows, orientation_overlay)

    summary = {
        "audit_type": "read-only calibration design review; no extent/orientation/calibration selection",
        "purpose": "Compare frozen detector and held V2 human extent, test fixed rows>=40 coverage descriptively, and quantify orientation residual ties before any separate policy decision.",
        "inputs": {
            "dot_grid": dot_metadata,
            "v2_validation_summary": str(args.v2_validation_summary),
            "v2_controls_csv": str(args.v2_controls_csv),
            "outer_boundary_csv": str(args.outer_boundary_csv),
            "v3_features_csv": str(args.v3_features),
            "orientation_ranking_csv": str(args.orientation_ranking_csv),
        },
        "fixed_rules": {
            "nominal_grid_size": NOMINAL_GRID_SIZE,
            "coverage_rows_minimum": MIN_COVERAGE_ROWS,
            "coverage_columns_minimum": MIN_COVERAGE_COLUMNS,
            "frozen_snap_bound_px": FROZEN_SNAP_BOUND_PX,
            "human_extent_status": "held; V2 strict snap failed",
            "evidence_expanded_extent_status": "blocked unless all four controls have local lattice evidence; current classes are mixed",
        },
        "extent_candidates": extent_summaries,
        "outer_boundary_evidence_classes": sorted(outer_classes),
        "evidence_expanded_extent_eligible": extent_eligible,
        "orientation_review": orientation_summary,
        "design_review_decision": "hold_extent_and_orientation; no candidate satisfies all predeclared independent requirements",
        "prohibitions": [
            "Does not modify raw TIFF/CSV, V2 controls, V3 artifacts, or existing validation outputs.",
            "Does not change GRID_SIZE=50, rows>=40 coverage gate, detector threshold/ROI, snap tolerance, or nominal window.",
            "Does not fit/deploy a homography, choose transform/rank/orientation, assert D/machine origin, or edit calibration_v1.yaml.",
            "Does not access A/B manufacturing TIFF, XCT, weak target/support, model, checkpoint, training, decoder, or candidate output.",
            "Does not change raw-camera-primary XCT-derived continuous quality candidate reporting.",
        ],
        "storage_policy": "writes compact extent/occupancy/control CSVs, one JSON summary, and two deterministic QC overlays; no dense crop, mask, rectification, target, or model output",
        "outputs": {
            "extent_candidates_csv": str(extent_csv),
            "occupancy_csv": str(detail_csv),
            "outer_control_evidence_csv": str(controls_csv),
            "summary_json": str(summary_path),
            "extent_overlay_png": str(extent_overlay),
            "orientation_overlay_png": str(orientation_overlay),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Calibration design review complete. No extent, coverage gate, calibration, rank, orientation, target, model, or candidate policy was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
