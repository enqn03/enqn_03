utf-8
#!/usr/bin/env python3
"""Audit NIST method-#2 D-to-C calibration candidates without changing deployment calibration.
This script uses only the layer-camera DotGrid TIFF, the existing provisional
calibration config/control JSON, and published method-#2 constants. It detects
subpixel dot candidates, constructs an approximate 50x50 dot-grid lattice,
fits D-to-C homography *candidates*, and evaluates them on held-out lattice
blocks. It writes a candidate set for human review only.
It never changes calibration_v1.yaml, selects a rank/orientation, rewrites raw
TIFF/CSV, accesses A/B manufacturing image stacks, target/model/checkpoints, or
relabels any quality candidate. The primary candidate location policy remains
raw layer-camera pixels.
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
import yaml
from audit_independent_metrology_fiducials_refined import (
    density_roi,
    dot_grid_candidates,
    dot_response,
    grayscale,
    read_tiff,
    refined_feature_candidates,
)
from audit_machine_camera_calibration import Hfit, build_candidates, project
GRID_SIZE = 50
DOT_PITCH_MM = 1.0
A_ORIGIN_IN_D_MM = np.array([28.25, 24.25], dtype=np.float64)
D_TO_A_ABS_ANGLE_DEG = 2.5
HOLDOUT_BLOCK_SIZE = 5
HOLDOUT_MODULUS = 5
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid_2000x2000.tif read via memmap(mode='r').")
    parser.add_argument("--calibration-config", required=True, type=Path, help="Existing provisional config read for comparison only.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def hom(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(len(points), dtype=np.float64)])
def add_raw_offset(h_matrix: np.ndarray, offset_xy: tuple[float, float]) -> np.ndarray:
    """Compose a raw-camera pixel translation after a source-to-camera transform."""
    dx, dy = offset_xy
    translation = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return translation @ np.asarray(h_matrix, dtype=np.float64)
def subpixel_dark_blob_centers(gray: np.ndarray, points: np.ndarray, radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Use local positive dark-blob response weights to obtain conservative subpixel centroids."""
    response = dot_response(gray)
    height, width = response.shape
    centers: list[list[float]] = []
    shifts: list[float] = []
    for x_value, y_value in np.asarray(points, dtype=np.float64):
        x, y = int(round(float(x_value))), int(round(float(y_value)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = response[y0:y1, x0:x1]
        weights = np.clip(np.asarray(patch, dtype=np.float64), 0.0, None)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            centers.append([float(x), float(y)])
            shifts.append(0.0)
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        refined_x = float((xx * weights).sum() / total)
        refined_y = float((yy * weights).sum() / total)
        shift = float(math.hypot(refined_x - x, refined_y - y))
        if shift > radius:
            refined_x, refined_y, shift = float(x), float(y), 0.0
        centers.append([refined_x, refined_y])
        shifts.append(shift)
    return np.asarray(centers, dtype=np.float64), np.asarray(shifts, dtype=np.float64)
def kmeans_1d(values: np.ndarray, cluster_count: int, iterations: int = 100) -> np.ndarray:
    """Deterministic 1D k-means initialized at evenly spaced empirical quantiles."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < cluster_count:
        raise ValueError("Insufficient 1D values for requested lattice cluster count.")
    quantiles = np.linspace(0.0, 1.0, cluster_count)
    centers = np.quantile(values, quantiles)
    for _ in range(iterations):
        nearest = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for index in range(cluster_count):
            group = values[nearest == index]
            if len(group):
                updated[index] = float(group.mean())
        updated.sort()
        if float(np.max(np.abs(updated - centers))) < 1.0e-8:
            centers = updated
            break
        centers = updated
    return centers
def pca_coordinates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered PCA coordinates and orthonormal image-space basis vectors."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 4:
        raise ValueError("At least four dot candidates are required for PCA lattice coordinates.")
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    basis = vectors[:, order]
    return (points - center) @ basis, center, basis
def assign_lattice_cells(points: np.ndarray, responses: np.ndarray, grid_size: int = GRID_SIZE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign candidates to one approximate 50x50 lattice under image-space PCA axes.
    The output index is intentionally orientation-agnostic. Eight D-axis variants
    are composed later and all are retained; no image/machine orientation is selected.
    """
    coordinates, center, basis = pca_coordinates(points)
    axis0_centers = kmeans_1d(coordinates[:, 0], grid_size)
    axis1_centers = kmeans_1d(coordinates[:, 1], grid_size)
    col = np.argmin(np.abs(coordinates[:, 0, None] - axis0_centers[None, :]), axis=1)
    row = np.argmin(np.abs(coordinates[:, 1, None] - axis1_centers[None, :]), axis=1)
    retained: dict[tuple[int, int], int] = {}
    for index, key in enumerate(zip(col.tolist(), row.tolist(), strict=True)):
        previous = retained.get(key)
        if previous is None or float(responses[index]) > float(responses[previous]):
            retained[key] = index
    rows: list[dict[str, Any]] = []
    for (col_index, row_index), index in sorted(retained.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.append({
            "source_candidate_index": int(index),
            "pca_col_index_0_to_49": int(col_index),
            "pca_row_index_0_to_49": int(row_index),
            "raw_x_px": float(points[index, 0]),
            "raw_y_px": float(points[index, 1]),
            "detector_response": float(responses[index]),
        })
    unique_col_count = len({int(row["pca_col_index_0_to_49"]) for row in rows})
    unique_row_count = len({int(row["pca_row_index_0_to_49"]) for row in rows})
    return rows, {
        "method": "PCA image coordinates followed by deterministic 1D 50-cluster quantization per axis; duplicate cells retain higher-response point",
        "grid_size_per_axis": grid_size,
        "unique_indexed_cell_count": len(rows),
        "grid_cell_coverage_fraction": float(len(rows) / float(grid_size * grid_size)),
        "unique_pca_column_count": unique_col_count,
        "unique_pca_row_count": unique_row_count,
        "pca_center_raw_xy_px": [float(center[0]), float(center[1])],
        "pca_basis_columns_raw_xy": basis.tolist(),
        "important_limit": "PCA cell indices are image-lattice indices. Their lower-left D origin and axis directions are intentionally unresolved here and retained as alternatives.",
    }
def orientation_variants() -> list[dict[str, Any]]:
    """Enumerate eight D-axis assignments for an orientation-ambiguous PCA lattice."""
    variants: list[dict[str, Any]] = []
    for swapped in (False, True):
        for flip_x in (False, True):
            for flip_y in (False, True):
                variants.append({
                    "pca_axes_swapped": swapped,
                    "pca_col_reversed_for_d_x": flip_x,
                    "pca_row_reversed_for_d_y": flip_y,
                    "orientation_variant": f"{'swap_' if swapped else ''}pca_xy__dx_{'reverse' if flip_x else 'forward'}__dy_{'reverse' if flip_y else 'forward'}",
                })
    return variants
def d_coordinates_from_index_rows(index_rows: list[dict[str, Any]], variant: dict[str, Any]) -> np.ndarray:
    coordinates: list[list[float]] = []
    for row in index_rows:
        col = int(row["pca_col_index_0_to_49"])
        image_row = int(row["pca_row_index_0_to_49"])
        first, second = (image_row, col) if bool(variant["pca_axes_swapped"]) else (col, image_row)
        d_x = GRID_SIZE - 1 - first if bool(variant["pca_col_reversed_for_d_x"]) else first
        d_y = GRID_SIZE - 1 - second if bool(variant["pca_row_reversed_for_d_y"]) else second
        coordinates.append([float(d_x) * DOT_PITCH_MM, float(d_y) * DOT_PITCH_MM])
    return np.asarray(coordinates, dtype=np.float64)
def raw_coordinates_from_index_rows(index_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in index_rows], dtype=np.float64)
def inlier_fit(source_d: np.ndarray, raw_c: np.ndarray, initial_threshold_fraction: float = 0.30, iterations: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Iteratively refit a D-to-C candidate and reject large pixel residuals deterministically."""
    if len(source_d) != len(raw_c) or len(source_d) < 8:
        raise ValueError("At least eight paired lattice candidates are required for robust homography fit.")
    active = np.ones(len(source_d), dtype=bool)
    final_threshold = float("nan")
    for _ in range(iterations):
        if int(active.sum()) < 8:
            raise RuntimeError("Too few inlier lattice cells remain for candidate transform fit.")
        h_matrix = Hfit(source_d[active], raw_c[active])
        residual = np.linalg.norm(project(h_matrix, source_d) - raw_c, axis=1)
        active_residual = residual[active]
        robust_scale = float(np.median(active_residual))
        threshold = max(2.0, 3.0 * robust_scale, initial_threshold_fraction * float(np.percentile(active_residual, 90)))
        updated = residual <= threshold
        final_threshold = threshold
        if np.array_equal(updated, active):
            break
        active = updated
    h_matrix = Hfit(source_d[active], raw_c[active])
    residual = np.linalg.norm(project(h_matrix, source_d) - raw_c, axis=1)
    return h_matrix, active, residual, final_threshold
def heldout_block_mask(index_rows: list[dict[str, Any]]) -> np.ndarray:
    """Hold out deterministic 5x5 lattice blocks, not individual neighboring dots."""
    values = []
    for row in index_rows:
        col = int(row["pca_col_index_0_to_49"])
        image_row = int(row["pca_row_index_0_to_49"])
        block_id = (col // HOLDOUT_BLOCK_SIZE + 2 * (image_row // HOLDOUT_BLOCK_SIZE)) % HOLDOUT_MODULUS
        values.append(block_id == 0)
    return np.asarray(values, dtype=bool)
def a_to_d_matrix(angle_deg: float) -> np.ndarray:
    """Map machine A coordinates to D coordinates under one explicit angle-sign alternative."""
    theta = math.radians(angle_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float64)
    return np.array([
        [rotation[0, 0], rotation[0, 1], A_ORIGIN_IN_D_MM[0]],
        [rotation[1, 0], rotation[1, 1], A_ORIGIN_IN_D_MM[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=np.float64)]
def rmse(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    return float(math.sqrt(float(np.mean(values ** 2))))
def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    return None if len(values) == 0 else float(np.percentile(values, percentile))
def nearest_camera_dot_pitch_px(points: np.ndarray) -> float | None:
    """Return the median nearest-neighbor spacing in raw camera pixels for a lattice subset."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None
    differences = points[:, None, :] - points[None, :, :]
    squared_distance = np.einsum("ijk,ijk->ij", differences, differences, optimize=True)
    np.fill_diagonal(squared_distance, np.inf)
    nearest = np.sqrt(np.min(squared_distance, axis=1))
    value = float(np.median(nearest))
    return value if math.isfinite(value) and value > 0.0 else None
def canonical_a_anchors() -> np.ndarray:
    """Reference points within the method-#2 dot-grid span; they are not candidate/model coordinates."""
    return np.asarray([[0.0, 0.0], [-10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [10.0, 10.0]], dtype=np.float64)
def transform_displacement_summary(reference_h: np.ndarray, candidate_h: np.ndarray) -> dict[str, float | None]:
    anchors = canonical_a_anchors()
    displacement = np.linalg.norm(project(reference_h, anchors) - project(candidate_h, anchors), axis=1)
    return {
        "anchor_count": int(len(displacement)),
        "raw_pixel_shift_min": float(displacement.min()),
        "raw_pixel_shift_median": float(np.median(displacement)),
        "raw_pixel_shift_p95": float(np.percentile(displacement, 95)),
        "raw_pixel_shift_max": float(displacement.max()),
    }
def load_current_comparison_transforms(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("status") != "provisional":
        raise ValueError("Expected current provisional calibration_v1.yaml-like configuration.")
    controls_value = config["control_points"]["path"]
    controls_path = Path(str(controls_value))
    if not controls_path.is_absolute():
        controls_path = Path.cwd() / controls_path
    with controls_path.open(encoding="utf-8") as handle:
        control_payload = json.load(handle)
    ranked = build_candidates(control_payload["control_points"])
    selected_rank = int(config["geometry_candidate"]["rank"])
    if selected_rank < 1 or selected_rank > len(ranked):
        raise ValueError("Configured geometry rank is outside current screen-control candidate list.")
    offset = tuple(float(value) for value in config["local_photometric_refinement"]["raw_pixel_global_offset_xy"])
    comparisons = [
        {
            "reference_name": "screen_control_rank1_base_no_offset",
            "rank": 1,
            "orientation": str(ranked[0]["orientation"]),
            "machine_a_to_raw_c": np.asarray(ranked[0]["H"], dtype=np.float64),
            "note": "Residual-only screen-control rank 1. No local offset was fit for this rank.",
        },
        {
            "reference_name": "screen_control_rank2_base_no_offset",
            "rank": selected_rank,
            "orientation": str(ranked[selected_rank - 1]["orientation"]),
            "machine_a_to_raw_c": np.asarray(ranked[selected_rank - 1]["H"], dtype=np.float64),
            "note": "Configured rank-2 geometry before the selected local photometric raw offset.",
        },
        {
            "reference_name": "screen_control_rank2_with_configured_offset",
            "rank": selected_rank,
            "orientation": str(ranked[selected_rank - 1]["orientation"]),
            "machine_a_to_raw_c": add_raw_offset(np.asarray(ranked[selected_rank - 1]["H"], dtype=np.float64), offset),
            "note": "Current configured geometry composed with its raw pixel global offset; comparison only.",
        },
    ]
    return config, comparisons
def write_indexed_features_csv(path: Path, index_rows: list[dict[str, Any]], shifts: np.ndarray, inlier_mask: np.ndarray, holdout_mask: np.ndarray, residual: np.ndarray) -> None:
    fields = [
        "pca_col_index_0_to_49", "pca_row_index_0_to_49", "raw_x_px", "raw_y_px", "detector_response", "subpixel_shift_px",
        "robust_fit_inlier", "heldout_block", "full_fit_residual_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, shift, inlier, heldout, distance in zip(index_rows, shifts, inlier_mask, holdout_mask, residual, strict=True):
            writer.writerow({
                **row,
                "subpixel_shift_px": float(shift),
                "robust_fit_inlier": bool(inlier),
                "heldout_block": bool(heldout),
                "full_fit_residual_px": float(distance),
            })
def plot_indexing_overlay(gray: np.ndarray, index_rows: list[dict[str, Any]], inlier_mask: np.ndarray, holdout_mask: np.ndarray, output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    display = gray[::stride, ::stride]
    points = raw_coordinates_from_index_rows(index_rows)
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    non_holdout = inlier_mask & ~holdout_mask
    axis.scatter(points[non_holdout, 0] / stride, points[non_holdout, 1] / stride, s=2.0, c="cyan", linewidths=0, label="fit inlier")
    heldout = inlier_mask & holdout_mask
    axis.scatter(points[heldout, 0] / stride, points[heldout, 1] / stride, s=4.0, c="yellow", marker="x", linewidths=0.6, label="held-out block")
    rejected = ~inlier_mask
    if int(rejected.sum()):
        axis.scatter(points[rejected, 0] / stride, points[rejected, 1] / stride, s=3.0, c="red", marker="+", linewidths=0.5, label="robust rejection")
    axis.set_title("Method-#2 DotGrid indexed cells\nImage-pixel candidates only — D-axis orientation alternatives retained")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def plot_holdout_residual_overlay(gray: np.ndarray, raw_c: np.ndarray, holdout_mask: np.ndarray, predicted_c: np.ndarray, output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    display = gray[::stride, ::stride]
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    actual = raw_c[holdout_mask]
    predicted = predicted_c[holdout_mask]
    if len(actual):
        axis.scatter(actual[:, 0] / stride, actual[:, 1] / stride, s=5.0, c="yellow", marker="x", linewidths=0.6, label="held-out actual")
        vectors = (predicted - actual) / stride
        axis.quiver(actual[:, 0] / stride, actual[:, 1] / stride, vectors[:, 0], vectors[:, 1], color="magenta", angles="xy", scale_units="xy", scale=1.0, width=0.0015, label="prediction residual")
    axis.set_title("Method-#2 held-out 5x5-block residuals\nOne orientation representative; no rank/config selection")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    if not args.dot_grid.is_file():
        raise FileNotFoundError(f"Required dot-grid TIFF not found: {args.dot_grid}")
    if not args.calibration_config.is_file():
        raise FileNotFoundError(f"Required comparison calibration config not found: {args.calibration_config}")
    output_dir = args.output_dir
    prepare_output_directory(output_dir, args.overwrite)
    channels, dot_metadata = read_tiff(args.dot_grid)
    gray = grayscale(channels)
    coarse_points, _, _ = dot_grid_candidates(gray)
    roi, roi_metrics = density_roi(coarse_points, gray.shape)
    pixel_points, response_scores, detector_metrics = refined_feature_candidates(gray, roi, "dot")
    subpixel_points, shifts = subpixel_dark_blob_centers(gray, pixel_points)
    index_rows, lattice_metrics = assign_lattice_cells(subpixel_points, response_scores)
    raw_c = raw_coordinates_from_index_rows(index_rows)
    holdout_mask = heldout_block_mask(index_rows)
    representative = orientation_variants()[0]
    representative_d = d_coordinates_from_index_rows(index_rows, representative)
    full_h, inlier_mask, full_residual, full_threshold = inlier_fit(representative_d, raw_c)
    train_mask = inlier_mask & ~holdout_mask
    test_mask = inlier_mask & holdout_mask
    if int(train_mask.sum()) < 8 or int(test_mask.sum()) < 8:
        raise RuntimeError(f"Insufficient inlier train/test lattice cells for held-out validation: train={int(train_mask.sum())}, test={int(test_mask.sum())}.")
    holdout_h = Hfit(representative_d[train_mask], raw_c[train_mask])
    holdout_predicted_c = project(holdout_h, representative_d)
    holdout_residual = np.linalg.norm(holdout_predicted_c[test_mask] - raw_c[test_mask], axis=1)
    inlier_spacing = nearest_camera_dot_pitch_px(raw_c[inlier_mask])
    holdout_rmse = rmse(holdout_residual)
    holdout_rmse_as_pitch = None if holdout_rmse is None or inlier_spacing is None else float(holdout_rmse / inlier_spacing)
    config, comparison_refs = load_current_comparison_transforms(args.calibration_config)
    candidates: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for orientation_index, variant in enumerate(orientation_variants(), start=1):
        source_d = d_coordinates_from_index_rows(index_rows, variant)
        h_d_to_c, candidate_inliers, candidate_residual, candidate_threshold = inlier_fit(source_d, raw_c)
        candidate_train = candidate_inliers & ~holdout_mask
        candidate_test = candidate_inliers & holdout_mask
        if int(candidate_train.sum()) < 8 or int(candidate_test.sum()) < 8:
            raise RuntimeError("Orientation variant unexpectedly lacks enough held-out inliers.")
        h_block = Hfit(source_d[candidate_train], raw_c[candidate_train])
        block_residual = np.linalg.norm(project(h_block, source_d[candidate_test]) - raw_c[candidate_test], axis=1)
        full_candidate_rmse = rmse(candidate_residual[candidate_inliers])
        block_candidate_rmse = rmse(block_residual)
        candidate_camera_pitch = nearest_camera_dot_pitch_px(raw_c[candidate_inliers])
        d_to_c_holdout_relative_pitch = None if block_candidate_rmse is None or candidate_camera_pitch is None else float(block_candidate_rmse / candidate_camera_pitch)
        for angle_sign in (+1.0, -1.0):
            angle_deg = angle_sign * D_TO_A_ABS_ANGLE_DEG
            h_a_to_d = a_to_d_matrix(angle_deg)
            h_a_to_c = h_d_to_c @ h_a_to_d
            transform_id = f"candidate_{orientation_index:02d}_{variant['orientation_variant']}__A_to_D_{angle_deg:+.1f}deg"
            candidate = {
                "candidate_transform_id": transform_id,
                "candidate_transform_family": "published_method2_dot_grid_D_to_layer_camera_C_composed_with_A_to_D",
                "lattice_orientation_variant": variant["orientation_variant"],
                "published_relative_d_a_angle_deg_sign_alternative": float(angle_deg),
                "published_a_origin_in_d_mm": [float(A_ORIGIN_IN_D_MM[0]), float(A_ORIGIN_IN_D_MM[1])],
                "dot_pitch_mm": DOT_PITCH_MM,
                "d_to_c_homography": matrix_to_list(h_d_to_c),
                "a_to_d_affine": matrix_to_list(h_a_to_d),
                "machine_a_to_raw_camera_c_candidate_homography": matrix_to_list(h_a_to_c),
                "full_fit_inlier_count": int(candidate_inliers.sum()),
                "full_fit_inlier_fraction": float(candidate_inliers.mean()),
                "full_fit_rmse_px": full_candidate_rmse,
                "full_fit_p95_residual_px": percentile_or_none(candidate_residual[candidate_inliers], 95),
                "heldout_block_count": int(candidate_test.sum()),
                "heldout_block_rmse_px": block_candidate_rmse,
                "heldout_block_p95_residual_px": percentile_or_none(block_residual, 95),
                "detected_camera_dot_pitch_px": candidate_camera_pitch,
                "heldout_block_rmse_camera_dot_pitch_fraction": d_to_c_holdout_relative_pitch,
                "inlier_threshold_px": float(candidate_threshold),
                "status": "candidate_for_human_review_only_not_auto_selected",
            }
            candidates.append(candidate)
            for reference in comparison_refs:
                comparison_rows.append({
                    "candidate_transform_id": transform_id,
                    "comparison_reference": str(reference["reference_name"]),
                    "reference_rank": int(reference["rank"]),
                    "reference_orientation": str(reference["orientation"]),
                    **transform_displacement_summary(np.asarray(reference["machine_a_to_raw_c"], dtype=np.float64), h_a_to_c),
                    "reference_note": str(reference["note"]),
                })
    coverage_gate = bool(lattice_metrics["unique_indexed_cell_count"] >= 1200 and lattice_metrics["unique_pca_column_count"] >= 40 and lattice_metrics["unique_pca_row_count"] >= 40)
    holdout_p95 = percentile_or_none(holdout_residual, 95)
    holdout_gate = bool(holdout_rmse_as_pitch is not None and holdout_rmse_as_pitch <= 0.25 and holdout_p95 is not None and inlier_spacing is not None and float(holdout_p95) <= 0.50 * float(inlier_spacing))
    all_gates = bool(coverage_gate and holdout_gate)
    indexed_csv = output_dir / "dot_grid_indexed_subpixel_features.csv"
    transform_csv = output_dir / "method2_candidate_transforms.csv"
    comparison_csv = output_dir / "method2_candidate_vs_existing_comparison.csv"
    with indexed_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "source_candidate_index", "pca_col_index_0_to_49", "pca_row_index_0_to_49", "raw_x_px", "raw_y_px", "detector_response", "subpixel_shift_px",
            "robust_fit_inlier", "heldout_block", "full_fit_residual_px",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, inlier, heldout, residual_value in zip(index_rows, inlier_mask, holdout_mask, full_residual, strict=True):
            source_index = int(row["source_candidate_index"])
            writer.writerow({**row, "subpixel_shift_px": float(shifts[source_index]), "robust_fit_inlier": bool(inlier), "heldout_block": bool(heldout), "full_fit_residual_px": float(residual_value)})
    transform_fields = list(candidates[0].keys())
    with transform_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=transform_fields)
        writer.writeheader()
        for row in candidates:
            flattened = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            writer.writerow(flattened)
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    indexing_overlay = output_dir / "method2_dot_grid_indexing_overlay.png"
    holdout_overlay = output_dir / "method2_dot_grid_heldout_residual_overlay.png"
    plot_indexing_overlay(gray, index_rows, inlier_mask, holdout_mask, indexing_overlay)
    plot_holdout_residual_overlay(gray, raw_c, test_mask, holdout_predicted_c, holdout_overlay)
    summary = {
        "audit_type": "read-only independent NIST method-2 candidate D-to-C calibration audit; no calibration config update or rank selection",
        "narrative": {
            "published_method2_relation": "NIST reports a 50x50 dot grid with 1.00 mm pitch, D origin at lower-left dot, A(0,0)=D(28.25,24.25) mm, and a 2.5 degree D/A relative orientation. The image-axis sign is retained as two explicit alternatives.",
            "what_is_fit": "An image-pixel D-to-C homography candidate from automatically indexed DotGrid features, evaluated on deterministic held-out 5x5 lattice blocks.",
            "what_is_not_decided": "No candidate transform is selected; red cluster is not asserted as machine origin; existing rank1/rank2 are only comparison references; calibration_v1.yaml remains unchanged.",
        },
        "inputs": {"dot_grid": dot_metadata, "calibration_config_read_only": str(args.calibration_config)},
        "dot_grid_detection": {"automatic_roi": roi_metrics, "refined_detector": detector_metrics, "subpixel_center_method": "response-weighted centroid in radius=4 local dark-blob patch; shifts beyond radius rejected", "subpixel_shift_median_px": float(np.median(shifts)), "subpixel_shift_p95_px": float(np.percentile(shifts, 95))},
        "lattice_indexing": lattice_metrics,
        "representative_orientation_for_validation_only": representative,
        "representative_d_to_c_validation": {
            "full_fit_inlier_count": int(inlier_mask.sum()), "full_fit_inlier_fraction": float(inlier_mask.mean()), "full_fit_rmse_px": rmse(full_residual[inlier_mask]), "full_fit_p95_residual_px": percentile_or_none(full_residual[inlier_mask], 95), "inlier_threshold_px": float(full_threshold), "detected_camera_dot_pitch_px": inlier_spacing,
            "heldout_scheme": f"5x5 lattice blocks where (block_col + 2*block_row) mod {HOLDOUT_MODULUS} == 0", "heldout_block_count": int(test_mask.sum()), "heldout_block_rmse_px": holdout_rmse, "heldout_block_p95_residual_px": holdout_p95, "heldout_block_rmse_camera_dot_pitch_fraction": holdout_rmse_as_pitch,
        },
        "gates": {
            "coverage_pass": coverage_gate,
            "coverage_rule": "at least 1200 unique indexed cells and at least 40 represented PCA rows and columns",
            "heldout_residual_pass": holdout_gate,
            "heldout_residual_rule": "held-out block RMSE <= 0.25 of the detected inlier camera dot pitch and p95 residual <= 0.50 of that pitch; this is an image-pixel consistency gate, not a final calibration uncertainty claim",
            "all_independent_candidate_gates_pass": all_gates,
        },
        "candidate_transform_count": int(len(candidates)),
        "candidate_transform_status": "all retained for human review; no automatic selection, rank substitution, target projection change, or physical candidate-location claim",
        "provisional_existing_calibration_status": str(config.get("status")),
        "outputs": {"indexed_features_csv": str(indexed_csv), "candidate_transforms_csv": str(transform_csv), "candidate_vs_existing_csv": str(comparison_csv), "indexing_overlay_png": str(indexing_overlay), "heldout_overlay_png": str(holdout_overlay), "summary_json": str(output_dir / "independent_method2_calibration_candidate_summary.json")},
        "prohibitions": ["Does not write raw TIFF/CSV.", "Does not edit calibration_v1.yaml.", "Does not select rank 1/2 or a method-2 alternative.", "Does not access model, XCT target, support, checkpoint, or decoder.", "Does not change camera-primary candidate reporting."],
        "reference": "Lane and Yeung (2020), J. Res. NIST 125:125027, Sec. 5.2 and Figs. 14-15, DOI 10.6028/jres.125.027",
    }
    summary_path = output_dir / "independent_method2_calibration_candidate_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Independent method-#2 calibration candidate audit complete. No raw TIFF/CSV, calibration config, model, target, checkpoint, or deployment candidate output was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
