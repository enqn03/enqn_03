utf-8
#!/usr/bin/env python3
"""Diagnose V2 visible-DotGrid outer-control snap failures in image space only.
This read-only follow-up compares each existing human click with four independent
image-space references: the frozen refined detector, a deterministic local
same-response detector, V3 assigned cells, and the V3 nominal 50x50 prediction.
It classifies evidence only; it never edits clicks, thresholds, grid/gates,
calibration, model data, or candidate reporting.
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
from audit_independent_metrology_fiducials import greedy_nms, robust_normalize
from audit_independent_metrology_fiducials_refined import (
    density_roi,
    dot_grid_candidates,
    dot_response,
    grayscale,
    read_tiff,
    refined_feature_candidates,
)
from audit_independent_method2_calibration_candidate import GRID_SIZE, nearest_camera_dot_pitch_px, project
from audit_visible_dotgrid_extent_controls import (
    EXPECTED_ORDER,
    fit_v3_image_lattice,
    load_controls,
    load_v3_features,
    nearest_indices_and_distances,
)
CURRENT_SNAP_MAX_PITCH = 0.60
LOCAL_PATCH_RADIUS_PITCH = 4.0
LOCAL_RESPONSE_QUANTILE = 0.990
LOCAL_NMS_MIN_DISTANCE_PX = 8
LOCAL_NMS_MAXIMUM = 256
LOCAL_NEIGHBOR_MIN_PITCH = 0.55
LOCAL_NEIGHBOR_MAX_PITCH = 1.55
LOCAL_ORTHOGONAL_MAX_ABS_COSINE = 0.50
OUTSIDE_EVIDENCE_MIN_PITCH = 1.25
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid TIFF; read-only memmap only.")
    parser.add_argument("--controls-json", required=True, type=Path, help="Existing completed V2 four-control JSON; never modified.")
    parser.add_argument("--v3-features", required=True, type=Path, help="Existing completed V3 compact lattice-feature CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New ignored directory for compact metrics and five QC PNGs.")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace only this diagnostic output after review.")
    return parser.parse_args()
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty diagnostic CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def distance_to_roi(point: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[bool, float]:
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = roi
    inside = bool(x0 <= x < x1 and y0 <= y < y1)
    dx = max(float(x0) - x, 0.0, x - float(x1 - 1))
    dy = max(float(y0) - y, 0.0, y - float(y1 - 1))
    return inside, float(math.hypot(dx, dy))
def nearest_one(point: np.ndarray, reference: np.ndarray) -> tuple[int, float, np.ndarray]:
    if len(reference) == 0:
        raise ValueError("Nearest-reference set is empty.")
    indices, distances = nearest_indices_and_distances(point[None, :], reference)
    index = int(indices[0])
    return index, float(distances[0]), np.asarray(reference[index], dtype=np.float64)
def has_orthogonal_neighbor_pair(vectors: np.ndarray) -> tuple[bool, float | None]:
    if len(vectors) < 2:
        return False, None
    best_abs_cosine = float("inf")
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            denom = float(np.linalg.norm(vectors[left]) * np.linalg.norm(vectors[right]))
            if denom <= 1.0e-12:
                continue
            absolute_cosine = abs(float(np.dot(vectors[left], vectors[right]) / denom))
            best_abs_cosine = min(best_abs_cosine, absolute_cosine)
    if not math.isfinite(best_abs_cosine):
        return False, None
    return bool(best_abs_cosine <= LOCAL_ORTHOGONAL_MAX_ABS_COSINE), best_abs_cosine
def local_dot_evidence(gray: np.ndarray, click: np.ndarray, pitch: float) -> dict[str, Any]:
    height, width = gray.shape
    radius = max(24, int(math.ceil(LOCAL_PATCH_RADIUS_PITCH * pitch)))
    center_x, center_y = float(click[0]), float(click[1])
    x0 = max(0, int(math.floor(center_x)) - radius)
    y0 = max(0, int(math.floor(center_y)) - radius)
    x1 = min(width, int(math.floor(center_x)) + radius + 1)
    y1 = min(height, int(math.floor(center_y)) + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if min(patch.shape) < 16:
        raise RuntimeError(f"Local diagnostic patch is too small around click {click.tolist()}.")
    response = dot_response(patch)
    points, scores, nms = greedy_nms(
        response,
        quantile=LOCAL_RESPONSE_QUANTILE,
        min_distance_px=LOCAL_NMS_MIN_DISTANCE_PX,
        maximum=LOCAL_NMS_MAXIMUM,
    )
    points = np.asarray(points, dtype=np.float64)
    points[:, 0] += float(x0)
    points[:, 1] += float(y0)
    local_index, local_distance, local_center = nearest_one(click, points)
    vectors = points - local_center[None, :]
    distances = np.linalg.norm(vectors, axis=1)
    neighbor_mask = (distances >= LOCAL_NEIGHBOR_MIN_PITCH * pitch) & (distances <= LOCAL_NEIGHBOR_MAX_PITCH * pitch)
    neighbor_vectors = vectors[neighbor_mask]
    orthogonal_pair, best_abs_cosine = has_orthogonal_neighbor_pair(neighbor_vectors)
    center_pass = bool(local_distance <= CURRENT_SNAP_MAX_PITCH * pitch)
    lattice_support = bool(center_pass and len(neighbor_vectors) >= 2 and orthogonal_pair)
    return {
        "patch_xyxy_px": [x0, y0, x1, y1],
        "patch_radius_px": radius,
        "points": points,
        "scores": np.asarray(scores, dtype=np.float64),
        "response_metrics": nms,
        "nearest_local_candidate_index": local_index,
        "nearest_local_candidate": local_center,
        "nearest_local_candidate_distance_px": local_distance,
        "local_center_within_frozen_snap_bound": center_pass,
        "local_neighbor_count_in_pitch_band": int(len(neighbor_vectors)),
        "local_orthogonal_neighbor_pair_pass": orthogonal_pair,
        "local_best_neighbor_pair_abs_cosine": best_abs_cosine,
        "local_lattice_evidence_pass": lattice_support,
    }
def evidence_class(
    current_pass: bool,
    local_result: dict[str, Any],
    current_distance: float,
    v3_distance: float,
    nominal_distance: float,
    pitch: float,
) -> tuple[str, str]:
    if current_pass:
        return "current_detector_supported", "The existing frozen detector already supports this click within the predeclared snap bound."
    if bool(local_result["local_lattice_evidence_pass"]):
        return (
            "printed_dot_visible_but_current_detector_missed",
            "A near-click local same-response candidate has at least two pitch-band neighbors with an approximately orthogonal pair, while the frozen detector misses the click.",
        )
    outside_threshold = OUTSIDE_EVIDENCE_MIN_PITCH * pitch
    local_distance = float(local_result["nearest_local_candidate_distance_px"])
    if current_distance > outside_threshold and local_distance > outside_threshold and v3_distance > outside_threshold and nominal_distance > outside_threshold:
        return (
            "click_outside_printed_dot",
            "Frozen/local detector, V3 assigned cells, and nominal V3 predictions all lack near-click support beyond the conservative 1.25-pitch evidence bound.",
        )
    return (
        "ambiguous",
        "At least one image-space reference is near the click, but the independent local lattice-support gate is incomplete; no click or detector conclusion is promoted.",
    )
def plot_local_patch(
    gray: np.ndarray,
    click: np.ndarray,
    local_result: dict[str, Any],
    fresh_points: np.ndarray,
    current_nearest: np.ndarray,
    v3_raw: np.ndarray,
    nominal_predicted: np.ndarray,
    roi: tuple[int, int, int, int],
    classification: str,
    output_path: Path,
) -> None:
    x0, y0, x1, y1 = (int(value) for value in local_result["patch_xyxy_px"])
    patch = gray[y0:y1, x0:x1]
    display, _ = robust_normalize(patch)
    figure, axis = plt.subplots(figsize=(6.4, 6.4), dpi=160)
    axis.imshow(display, cmap="gray", origin="upper", extent=[x0, x1, y1, y0], interpolation="nearest")
    local_points = np.asarray(local_result["points"], dtype=np.float64)
    axis.scatter(local_points[:, 0], local_points[:, 1], s=18, facecolors="none", edgecolors="orange", linewidths=0.7, label="independent local candidates")
    fresh_mask = (
        (fresh_points[:, 0] >= x0) & (fresh_points[:, 0] < x1) &
        (fresh_points[:, 1] >= y0) & (fresh_points[:, 1] < y1)
    )
    if fresh_mask.any():
        axis.scatter(fresh_points[fresh_mask, 0], fresh_points[fresh_mask, 1], s=13, c="cyan", linewidths=0, label="frozen detector candidates")
    assigned_mask = (
        (v3_raw[:, 0] >= x0) & (v3_raw[:, 0] < x1) &
        (v3_raw[:, 1] >= y0) & (v3_raw[:, 1] < y1)
    )
    if assigned_mask.any():
        axis.scatter(v3_raw[assigned_mask, 0], v3_raw[assigned_mask, 1], s=16, facecolors="none", edgecolors="dodgerblue", linewidths=0.6, label="V3 assigned")
    nominal_mask = (
        (nominal_predicted[:, 0] >= x0) & (nominal_predicted[:, 0] < x1) &
        (nominal_predicted[:, 1] >= y0) & (nominal_predicted[:, 1] < y1)
    )
    if nominal_mask.any():
        axis.scatter(nominal_predicted[nominal_mask, 0], nominal_predicted[nominal_mask, 1], s=13, c="red", marker="+", linewidths=0.5, label="nominal V3 predictions")
    local_nearest = np.asarray(local_result["nearest_local_candidate"], dtype=np.float64)
    axis.scatter([click[0]], [click[1]], s=65, c="magenta", marker="x", linewidths=1.6, label="human click")
    axis.scatter([local_nearest[0]], [local_nearest[1]], s=42, c="yellow", marker="+", linewidths=1.2, label="nearest local candidate")
    if x0 <= current_nearest[0] < x1 and y0 <= current_nearest[1] < y1:
        axis.scatter([current_nearest[0]], [current_nearest[1]], s=42, c="lime", marker="x", linewidths=1.2, label="nearest frozen candidate")
    roi_x0, roi_y0, roi_x1, roi_y1 = roi
    if x0 <= roi_x0 <= x1:
        axis.axvline(roi_x0, color="lime", linestyle="--", linewidth=0.8, label="frozen ROI boundary")
    if x0 <= roi_x1 <= x1:
        axis.axvline(roi_x1, color="lime", linestyle="--", linewidth=0.8, label="frozen ROI boundary")
    if y0 <= roi_y0 <= y1:
        axis.axhline(roi_y0, color="lime", linestyle="--", linewidth=0.8, label="frozen ROI boundary")
    if y0 <= roi_y1 <= y1:
        axis.axhline(roi_y1, color="lime", linestyle="--", linewidth=0.8, label="frozen ROI boundary")
    axis.set_xlim(x0, x1)
    axis.set_ylim(y1, y0)
    axis.set_title(f"Outer-control local evidence: {classification}\nImage-space diagnostic only")
    axis.set_xlabel("raw camera x [px]")
    axis.set_ylabel("raw camera y [px]")
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axis.legend(unique.values(), unique.keys(), loc="best", fontsize=6.5, framealpha=0.88)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def plot_full_panel(
    gray: np.ndarray,
    clicks: np.ndarray,
    fresh_points: np.ndarray,
    local_nearest: np.ndarray,
    v3_raw: np.ndarray,
    nominal_predicted: np.ndarray,
    roi: tuple[int, int, int, int],
    classes: list[str],
    output_path: Path,
) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(gray[::stride, ::stride], cmap="gray", origin="upper", interpolation="nearest")
    axis.scatter(fresh_points[:, 0] / stride, fresh_points[:, 1] / stride, s=2.0, c="cyan", linewidths=0, label="frozen refined detector")
    axis.scatter(v3_raw[:, 0] / stride, v3_raw[:, 1] / stride, s=2.2, facecolors="none", edgecolors="dodgerblue", linewidths=0.25, label="V3 assigned")
    axis.scatter(nominal_predicted[:, 0] / stride, nominal_predicted[:, 1] / stride, s=1.7, c="orange", marker="+", linewidths=0.25, label="nominal V3 50x50 predictions")
    axis.scatter(clicks[:, 0] / stride, clicks[:, 1] / stride, s=45, c="magenta", marker="x", linewidths=1.1, label="human V2 clicks")
    axis.scatter(local_nearest[:, 0] / stride, local_nearest[:, 1] / stride, s=40, c="yellow", marker="+", linewidths=1.0, label="nearest independent local candidates")
    roi_x0, roi_y0, roi_x1, roi_y1 = roi
    rectangle_x = np.asarray([roi_x0, roi_x1, roi_x1, roi_x0, roi_x0], dtype=np.float64) / stride
    rectangle_y = np.asarray([roi_y0, roi_y0, roi_y1, roi_y1, roi_y0], dtype=np.float64) / stride
    axis.plot(rectangle_x, rectangle_y, color="lime", linestyle="--", linewidth=0.9, label="frozen detector ROI")
    for index, (click, classification) in enumerate(zip(clicks, classes), start=1):
        axis.text(click[0] / stride, click[1] / stride, f"{index}:{classification}", color="white", fontsize=6, ha="left", va="bottom", bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})
    axis.set_title("Visible DotGrid outer-boundary diagnostic\nNo reclick, detector/gate/config change, or calibration selection")
    axis.set_xlabel("display raw camera x [px]")
    axis.set_ylabel("display raw camera y [px]")
    axis.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    for path, label in ((args.dot_grid, "DotGrid TIFF"), (args.controls_json, "V2 controls JSON"), (args.v3_features, "V3 compact feature CSV")):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    prepare_output_directory(args.output_dir, args.overwrite)
    controls = load_controls(args.controls_json)
    features = load_v3_features(args.v3_features)
    channels, metadata = read_tiff(args.dot_grid)
    gray = grayscale(channels)
    coarse_points, _, _ = dot_grid_candidates(gray)
    roi, roi_metrics = density_roi(coarse_points, gray.shape)
    fresh_points, _, fresh_metrics = refined_feature_candidates(gray, roi, "dot")
    fresh_points = np.asarray(fresh_points, dtype=np.float64)
    v3_raw = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in features], dtype=np.float64)
    pitch = nearest_camera_dot_pitch_px(v3_raw)
    if pitch is None or not math.isfinite(pitch) or pitch <= 0.0:
        raise RuntimeError("Could not estimate a positive V3 camera-dot pitch.")
    snap_bound = CURRENT_SNAP_MAX_PITCH * float(pitch)
    h_matrix = fit_v3_image_lattice(features)
    nominal_cells = np.asarray([[float(col), float(row)] for row in range(GRID_SIZE) for col in range(GRID_SIZE)], dtype=np.float64)
    nominal_predicted = project(h_matrix, nominal_cells)
    clicks = np.asarray([[float(row["clicked_x_px"]), float(row["clicked_y_px"])] for row in controls], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    local_results: list[dict[str, Any]] = []
    current_nearest_points: list[np.ndarray] = []
    classes: list[str] = []
    for index, (control, click) in enumerate(zip(controls, clicks), start=1):
        current_index, current_distance, current_nearest = nearest_one(click, fresh_points)
        _, v3_distance, v3_nearest = nearest_one(click, v3_raw)
        _, nominal_distance, nominal_nearest = nearest_one(click, nominal_predicted)
        roi_inside, roi_distance = distance_to_roi(click, roi)
        local_result = local_dot_evidence(gray, click, float(pitch))
        classification, reason = evidence_class(
            current_pass=bool(current_distance <= snap_bound),
            local_result=local_result,
            current_distance=current_distance,
            v3_distance=v3_distance,
            nominal_distance=nominal_distance,
            pitch=float(pitch),
        )
        local_nearest = np.asarray(local_result["nearest_local_candidate"], dtype=np.float64)
        rows.append({
            "selection_order": index,
            "semantic_name": str(control["semantic_name"]),
            "clicked_x_px": float(click[0]),
            "clicked_y_px": float(click[1]),
            "inside_frozen_detector_roi": roi_inside,
            "distance_outside_frozen_detector_roi_px": roi_distance,
            "current_frozen_candidate_index": current_index,
            "current_frozen_nearest_x_px": float(current_nearest[0]),
            "current_frozen_nearest_y_px": float(current_nearest[1]),
            "current_frozen_nearest_distance_px": current_distance,
            "current_frozen_snap_bound_px": snap_bound,
            "current_frozen_snap_pass": bool(current_distance <= snap_bound),
            "local_patch_x0_px": int(local_result["patch_xyxy_px"][0]),
            "local_patch_y0_px": int(local_result["patch_xyxy_px"][1]),
            "local_patch_x1_px": int(local_result["patch_xyxy_px"][2]),
            "local_patch_y1_px": int(local_result["patch_xyxy_px"][3]),
            "local_candidate_count": int(len(local_result["points"])),
            "nearest_local_candidate_x_px": float(local_nearest[0]),
            "nearest_local_candidate_y_px": float(local_nearest[1]),
            "nearest_local_candidate_distance_px": float(local_result["nearest_local_candidate_distance_px"]),
            "local_center_within_frozen_snap_bound": bool(local_result["local_center_within_frozen_snap_bound"]),
            "local_neighbor_count_in_pitch_band": int(local_result["local_neighbor_count_in_pitch_band"]),
            "local_orthogonal_neighbor_pair_pass": bool(local_result["local_orthogonal_neighbor_pair_pass"]),
            "local_best_neighbor_pair_abs_cosine": local_result["local_best_neighbor_pair_abs_cosine"],
            "local_lattice_evidence_pass": bool(local_result["local_lattice_evidence_pass"]),
            "nearest_v3_assigned_x_px": float(v3_nearest[0]),
            "nearest_v3_assigned_y_px": float(v3_nearest[1]),
            "nearest_v3_assigned_distance_px": v3_distance,
            "nearest_nominal_prediction_x_px": float(nominal_nearest[0]),
            "nearest_nominal_prediction_y_px": float(nominal_nearest[1]),
            "nearest_nominal_prediction_distance_px": nominal_distance,
            "evidence_class": classification,
            "evidence_reason": reason,
            "important_limit": "Image-space evidence class only; does not edit clicks/detector/grid/gate/calibration.",
        })
        local_results.append(local_result)
        current_nearest_points.append(current_nearest)
        classes.append(classification)
    local_nearest_array = np.asarray([result["nearest_local_candidate"] for result in local_results], dtype=np.float64)
    current_nearest_array = np.asarray(current_nearest_points, dtype=np.float64)
    csv_path = args.output_dir / "visible_dotgrid_outer_boundary_diagnostic_by_control.csv"
    full_overlay = args.output_dir / "visible_dotgrid_outer_boundary_full_panel_overlay.png"
    local_overlays: list[Path] = []
    write_csv(csv_path, rows)
    for index, (click, local_result, current_nearest, classification) in enumerate(zip(clicks, local_results, current_nearest_array, classes), start=1):
        semantic = EXPECTED_ORDER[index - 1].replace("_outer_dot_center", "")
        output_path = args.output_dir / f"visible_dotgrid_outer_boundary_control_{index:02d}_{semantic}.png"
        plot_local_patch(gray, click, local_result, fresh_points, current_nearest, v3_raw, nominal_predicted, roi, classification, output_path)
        local_overlays.append(output_path)
    plot_full_panel(gray, clicks, fresh_points, local_nearest_array, v3_raw, nominal_predicted, roi, classes, full_overlay)
    class_counts = {name: int(classes.count(name)) for name in sorted(set(classes))}
    failing_classes = [classification for row, classification in zip(rows, classes) if not bool(row["current_frozen_snap_pass"])]
    if failing_classes and all(value == "printed_dot_visible_but_current_detector_missed" for value in failing_classes):
        recommendation = "detector_footprint_limitation_supported_for_separate_design_review_only"
    elif any(value == "click_outside_printed_dot" for value in failing_classes):
        recommendation = "human_click_placement_issue_supported_for_separate_reselection_design_review_only"
    else:
        recommendation = "hold_outer_boundary_diagnosis; mixed_or_ambiguous_image_space_evidence"
    summary_path = args.output_dir / "visible_dotgrid_outer_boundary_diagnostic_summary.json"
    summary = {
        "audit_type": "read-only visible DotGrid outer-boundary snap-failure diagnostic; no calibration or coverage-policy decision",
        "purpose": "Separate human-click placement evidence from frozen detector ROI/response limitations after the V2 strict snap hold.",
        "inputs": {
            "dot_grid": metadata,
            "controls_json": str(args.controls_json),
            "v3_features_csv": str(args.v3_features),
        },
        "frozen_reference": {
            "camera_dot_pitch_px": float(pitch),
            "current_snap_bound_pitch": CURRENT_SNAP_MAX_PITCH,
            "current_snap_bound_px": snap_bound,
            "detector_roi": roi_metrics,
            "detector_metrics": fresh_metrics,
            "nominal_grid_size_for_comparison_only": int(GRID_SIZE),
        },
        "independent_local_evidence_policy": {
            "patch_radius_pitch": LOCAL_PATCH_RADIUS_PITCH,
            "response": "same dark-dot response as frozen detector, recomputed within each local patch",
            "response_quantile": LOCAL_RESPONSE_QUANTILE,
            "nms_min_distance_px": LOCAL_NMS_MIN_DISTANCE_PX,
            "center_max_distance_pitch": CURRENT_SNAP_MAX_PITCH,
            "neighbor_pitch_band": [LOCAL_NEIGHBOR_MIN_PITCH, LOCAL_NEIGHBOR_MAX_PITCH],
            "orthogonal_pair_max_absolute_cosine": LOCAL_ORTHOGONAL_MAX_ABS_COSINE,
            "click_outside_all_references_min_distance_pitch": OUTSIDE_EVIDENCE_MIN_PITCH,
            "important_limit": "Local evidence is diagnostic only and cannot replace the frozen detector or validate a physical DotGrid cell.",
        },
        "evidence_class_counts": class_counts,
        "recommendation": recommendation,
        "prohibitions": [
            "Does not modify raw TIFF/CSV, V2 controls, V3 artifacts, detector thresholds, ROI, or validation tolerance.",
            "Does not change GRID_SIZE=50 or the rows/columns>=40 coverage gate.",
            "Does not fit/deploy a homography, choose transform/rank/orientation, assert D/machine origin, or edit calibration_v1.yaml.",
            "Does not access A/B manufacturing TIFF, XCT, weak target/support, model, checkpoint, training, decoder, or candidate output.",
            "Does not change raw-camera-primary XCT-derived continuous quality candidate reporting.",
        ],
        "storage_policy": "writes one compact CSV, one JSON, four local patch QC PNGs, and one full-panel QC PNG; no dense crop/mask/response/rectification is persisted",
        "outputs": {
            "by_control_csv": str(csv_path),
            "summary_json": str(summary_path),
            "local_patch_overlays": [str(path) for path in local_overlays],
            "full_panel_overlay": str(full_overlay),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Visible DotGrid outer-boundary diagnostic complete. No raw/control/V3 data, detector threshold, grid/gate, calibration, model, target, checkpoint, decoder, or candidate output was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
