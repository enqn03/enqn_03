utf-8
#!/usr/bin/env python3
"""Audit compact A-only candidate coordinates against the provisional calibration.
The script reads only the compact candidate JSON, model configuration,
normalization ROI, calibration configuration, and control-point JSON. It checks
coordinate-domain consistency and round-trip arithmetic; it does not read TIFF
or XCT CSV data, create targets/heatmaps, run inference, or modify a model.
A successful round trip proves only internal consistency under the provisional
transform. It is not independent metrology calibration or defect confirmation.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from audit_machine_camera_calibration import PARTS, RECT, build_candidates, project
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return loaded
def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded
def resolve_from_working_directory(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path
def inverse_project(h_inverse: np.ndarray, raw_unoffset_xy: np.ndarray) -> tuple[np.ndarray | None, float]:
    homogeneous = h_inverse @ np.array([raw_unoffset_xy[0], raw_unoffset_xy[1], 1.0], dtype=np.float64)
    denominator = float(homogeneous[2])
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return None, denominator
    machine_xy = homogeneous[:2] / denominator
    if not np.isfinite(machine_xy).all():
        return None, denominator
    return machine_xy, denominator
def containing_part(machine_xy: np.ndarray, atol: float = 1.0e-9) -> str | None:
    x_value, y_value = float(machine_xy[0]), float(machine_xy[1])
    for part in PARTS:
        x_min, x_max, y_min, y_max = RECT[part]
        if x_min - atol <= x_value <= x_max + atol and y_min - atol <= y_value <= y_max + atol:
            return str(part)
    return None
def percentile_summary(values: list[float]) -> dict[str, float | int | None]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {"count": 0, "nonfinite_count": len(values), "min": None, "median": None, "p95": None, "max": None}
    array = np.asarray(finite_values, dtype=np.float64)
    return {
        "count": len(finite_values),
        "nonfinite_count": len(values) - len(finite_values),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--roundtrip-atol-raw-px", type=float, default=1.0e-6)
    parser.add_argument("--roundtrip-atol-model-px", type=float, default=1.0e-6)
    parser.add_argument("--minimum-model-edge-margin-px", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def main() -> None:
    args = parse_args()
    if min(args.roundtrip_atol_raw_px, args.roundtrip_atol_model_px, args.minimum_model_edge_margin_px) < 0.0:
        raise ValueError("Round-trip tolerances and minimum edge margin must be non-negative.")
    for required in (args.candidate_json, args.model_config, args.normalization_config, args.calibration_config):
        if not required.is_file():
            raise FileNotFoundError(f"Missing input: {required}")
    candidates_payload = load_json(args.candidate_json)
    model_config = load_yaml(args.model_config)
    normalization_config = load_yaml(args.normalization_config)
    calibration_config = load_yaml(args.calibration_config)
    candidates = candidates_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate JSON contains no emitted compact candidates to audit.")
    declared_count = int(candidates_payload.get("candidate_count", -1))
    if declared_count != len(candidates):
        raise ValueError(f"candidate_count={declared_count} does not match candidates length={len(candidates)}.")
    if candidates_payload.get("stage") != "A":
        raise ValueError("Coordinate audit currently requires A-stage compact candidates.")
    if str(candidates_payload.get("response_direction")) != "unresolved":
        raise ValueError("Expected unresolved response direction contract in candidate JSON.")
    data_config = model_config["data"]
    model_resolution = tuple(int(value) for value in data_config["model_resolution"])
    if len(model_resolution) != 2 or min(model_resolution) < 1:
        raise ValueError("model_config data.model_resolution must be [height, width] with positive values.")
    model_height, model_width = model_resolution
    roi_values = normalization_config["working_roi"]["coordinates_raw_camera_pixels"]
    x0, y0, x1, y1 = (float(value) for value in roi_values)
    if not x1 > x0 or not y1 > y0:
        raise ValueError("Invalid normalization working ROI.")
    raw_width, raw_height = (float(value) for value in calibration_config["reference_frame"]["raw_dimensions_px"])
    if not raw_width > 0 or not raw_height > 0:
        raise ValueError("Invalid raw camera dimensions in calibration config.")
    x_scale = (x1 - x0) / model_width
    y_scale = (y1 - y0) / model_height
    control_path = resolve_from_working_directory(str(calibration_config["control_points"]["path"]))
    control_payload = load_json(control_path)
    control_points = control_payload.get("control_points")
    if not isinstance(control_points, list):
        raise ValueError("control-point JSON is missing control_points list.")
    rank = int(calibration_config["geometry_candidate"]["rank"])
    calibration_candidates = build_candidates(control_points)
    if not 1 <= rank <= len(calibration_candidates):
        raise ValueError(f"Configured calibration rank {rank} is outside available candidate range.")
    homography = np.asarray(calibration_candidates[rank - 1]["H"], dtype=np.float64)
    homography_inverse = np.linalg.inv(homography)
    offset_x, offset_y = (float(value) for value in calibration_config["local_photometric_refinement"]["raw_pixel_global_offset_xy"])
    rows: list[dict[str, Any]] = []
    raw_roundtrip_errors: list[float] = []
    model_roundtrip_errors: list[float] = []
    within_endpoint_keys: list[tuple[int, int, int]] = []
    across_layer_keys: list[tuple[int, int]] = []
    for candidate in candidates:
        required_fields = ("rank", "x_pixel", "y_pixel", "x_model_pixel", "y_model_pixel", "layer_z", "score", "sample_id", "stage")
        missing = [field for field in required_fields if field not in candidate]
        if missing:
            raise ValueError(f"Candidate missing fields {missing}: {candidate}")
        if candidate["stage"] != "A":
            raise ValueError(f"Candidate stage must be A, got {candidate['stage']!r}")
        x_model = int(candidate["x_model_pixel"])
        y_model = int(candidate["y_model_pixel"])
        layer_z = int(candidate["layer_z"])
        score = float(candidate["score"])
        raw_x = float(candidate["x_pixel"])
        raw_y = float(candidate["y_pixel"])
        expected_raw_x = x0 + (x_model + 0.5) * x_scale
        expected_raw_y = y0 + (y_model + 0.5) * y_scale
        raw_mapping_error = float(math.hypot(raw_x - expected_raw_x, raw_y - expected_raw_y))
        x_model_roundtrip = (raw_x - x0) / x_scale - 0.5
        y_model_roundtrip = (raw_y - y0) / y_scale - 0.5
        model_roundtrip_error = float(math.hypot(x_model_roundtrip - x_model, y_model_roundtrip - y_model))
        grid_in_bounds = 0 <= x_model < model_width and 0 <= y_model < model_height
        raw_in_roi = x0 <= raw_x < x1 and y0 <= raw_y < y1
        raw_in_sensor_fov = 0.0 <= raw_x < raw_width and 0.0 <= raw_y < raw_height
        score_finite = math.isfinite(score)
        score_in_unit_interval = score_finite and 0.0 <= score <= 1.0
        raw_finite = math.isfinite(raw_x) and math.isfinite(raw_y)
        edge_margin = float(min(x_model, model_width - 1 - x_model, y_model, model_height - 1 - y_model))
        edge_safe = edge_margin >= float(args.minimum_model_edge_margin_px)
        machine_xy: np.ndarray | None = None
        inverse_denominator: float | None = None
        reprojection_error = math.inf
        reprojected_raw_x: float | None = None
        reprojected_raw_y: float | None = None
        part_name: str | None = None
        if raw_finite:
            machine_xy, inverse_denominator_value = inverse_project(
                homography_inverse,
                np.array([raw_x - offset_x, raw_y - offset_y], dtype=np.float64),
            )
            inverse_denominator = float(inverse_denominator_value)
            if machine_xy is not None:
                reprojected = project(homography, machine_xy[None, :])[0] + np.array([offset_x, offset_y], dtype=np.float64)
                reprojected_raw_x = float(reprojected[0])
                reprojected_raw_y = float(reprojected[1])
                reprojection_error = float(math.hypot(reprojected_raw_x - raw_x, reprojected_raw_y - raw_y))
                part_name = containing_part(machine_xy)
        raw_roundtrip_errors.append(reprojection_error)
        model_roundtrip_errors.append(model_roundtrip_error)
        within_endpoint_keys.append((layer_z, x_model, y_model))
        across_layer_keys.append((x_model, y_model))
        rows.append(
            {
                "sample_id": str(candidate["sample_id"]),
                "layer_z": layer_z,
                "rank": int(candidate["rank"]),
                "score": score,
                "x_model_pixel": x_model,
                "y_model_pixel": y_model,
                "raw_camera_x_px": raw_x,
                "raw_camera_y_px": raw_y,
                "expected_raw_camera_x_px": expected_raw_x,
                "expected_raw_camera_y_px": expected_raw_y,
                "model_grid_roundtrip_x": x_model_roundtrip,
                "model_grid_roundtrip_y": y_model_roundtrip,
                "model_grid_roundtrip_error_px": model_roundtrip_error,
                "raw_center_mapping_error_px": raw_mapping_error,
                "grid_in_bounds": grid_in_bounds,
                "raw_in_working_roi": raw_in_roi,
                "raw_in_sensor_fov": raw_in_sensor_fov,
                "model_edge_margin_px": edge_margin,
                "model_edge_safe": edge_safe,
                "score_finite": score_finite,
                "score_in_unit_interval": score_in_unit_interval,
                "inverse_homography_denominator": inverse_denominator,
                "machine_x": None if machine_xy is None else float(machine_xy[0]),
                "machine_y": None if machine_xy is None else float(machine_xy[1]),
                "containing_machine_part": part_name,
                "inverse_transform_solvable": machine_xy is not None,
                "reprojected_raw_camera_x_px": reprojected_raw_x,
                "reprojected_raw_camera_y_px": reprojected_raw_y,
                "raw_camera_roundtrip_error_px": reprojection_error,
                "score_semantics": str(candidate.get("score_semantics", "")),
            }
        )
    within_endpoint_duplicates = {key: count for key, count in Counter(within_endpoint_keys).items() if count > 1}
    repeated_model_cells_across_layers = {key: count for key, count in Counter(across_layer_keys).items() if count > 1}
    raw_error_summary = percentile_summary(raw_roundtrip_errors)
    model_error_summary = percentile_summary(model_roundtrip_errors)
    raw_mapping_error_summary = percentile_summary([float(row["raw_center_mapping_error_px"]) for row in rows])
    edge_margin_summary = percentile_summary([float(row["model_edge_margin_px"]) for row in rows])
    checks = {
        "all_scores_finite": all(bool(row["score_finite"]) for row in rows),
        "all_scores_in_unit_interval": all(bool(row["score_in_unit_interval"]) for row in rows),
        "all_model_indices_in_bounds": all(bool(row["grid_in_bounds"]) for row in rows),
        "all_raw_coordinates_in_working_roi": all(bool(row["raw_in_working_roi"]) for row in rows),
        "all_raw_coordinates_in_sensor_fov": all(bool(row["raw_in_sensor_fov"]) for row in rows),
        "all_inverse_transforms_solvable": all(bool(row["inverse_transform_solvable"]) for row in rows),
        "all_machine_coordinates_inside_known_part_rectangles": all(row["containing_machine_part"] is not None for row in rows),
        "all_raw_center_mapping_errors_within_atol": all(float(row["raw_center_mapping_error_px"]) <= args.roundtrip_atol_raw_px for row in rows),
        "all_raw_roundtrip_errors_within_atol": all(float(row["raw_camera_roundtrip_error_px"]) <= args.roundtrip_atol_raw_px for row in rows),
        "all_model_roundtrip_errors_within_atol": all(float(row["model_grid_roundtrip_error_px"]) <= args.roundtrip_atol_model_px for row in rows),
        "no_duplicate_model_cells_within_same_endpoint": not within_endpoint_duplicates,
        "all_candidates_meet_minimum_model_edge_margin": all(bool(row["model_edge_safe"]) for row in rows),
    }
    arithmetic_checks = (
        "all_scores_finite",
        "all_scores_in_unit_interval",
        "all_model_indices_in_bounds",
        "all_raw_coordinates_in_working_roi",
        "all_raw_coordinates_in_sensor_fov",
        "all_inverse_transforms_solvable",
        "all_raw_center_mapping_errors_within_atol",
        "all_raw_roundtrip_errors_within_atol",
        "all_model_roundtrip_errors_within_atol",
        "no_duplicate_model_cells_within_same_endpoint",
    )
    coordinate_consistency_pass = all(bool(checks[name]) for name in arithmetic_checks)
    operational_geometry_pass = coordinate_consistency_pass and bool(checks["all_machine_coordinates_inside_known_part_rectangles"])
    edge_safety_pass = bool(checks["all_candidates_meet_minimum_model_edge_margin"])
    prepare_output_directory(args.output_dir, args.overwrite)
    candidate_csv = args.output_dir / "candidate_coordinate_audit.csv"
    duplicate_csv = args.output_dir / "repeated_model_cells_across_layers.csv"
    summary_json = args.output_dir / "candidate_coordinate_audit_summary.json"
    fieldnames = list(rows[0].keys())
    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with duplicate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x_model_pixel", "y_model_pixel", "repeat_count_across_layers"])
        writer.writeheader()
        for (x_model, y_model), count in sorted(repeated_model_cells_across_layers.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"x_model_pixel": x_model, "y_model_pixel": y_model, "repeat_count_across_layers": count})
    part_counts = Counter(str(row["containing_machine_part"]) if row["containing_machine_part"] is not None else "outside_known_part_rectangles" for row in rows)
    summary = {
        "audit_type": "calibration-aware compact A-only candidate coordinate audit; not metrology calibration or defect labeling",
        "candidate_json": str(args.candidate_json),
        "candidate_count": len(rows),
        "candidate_stage": candidates_payload["stage"],
        "response_direction": candidates_payload["response_direction"],
        "score_semantics": "XCT-derived continuous quality candidate; not confirmed defect or anomaly probability",
        "calibration_status": calibration_config["status"],
        "calibration_geometry_candidate": {
            "rank": rank,
            "orientation": calibration_config["geometry_candidate"]["orientation"],
            "fit_rmse_px": float(calibration_config["geometry_candidate"]["fit_rmse_px"]),
            "loo_rmse_px": float(calibration_config["geometry_candidate"]["loo_rmse_px"]),
            "raw_pixel_global_offset_xy": [offset_x, offset_y],
        },
        "coordinate_contract": {
            "model_resolution_height_width": [model_height, model_width],
            "working_roi_raw_camera_pixels_x0_y0_x1_y1_exclusive": [x0, y0, x1, y1],
            "raw_camera_dimensions_width_height": [raw_width, raw_height],
            "model_to_raw_center_equation": "raw = roi_start + (model_index + 0.5) * raw_pixels_per_model_pixel",
            "raw_to_machine_equation": "machine = inverse(H_machine_to_raw) * (raw - configured_global_offset)",
            "machine_to_raw_equation": "raw = H_machine_to_raw * machine + configured_global_offset",
        },
        "thresholds": {
            "roundtrip_atol_raw_px": float(args.roundtrip_atol_raw_px),
            "roundtrip_atol_model_px": float(args.roundtrip_atol_model_px),
            "minimum_model_edge_margin_px": float(args.minimum_model_edge_margin_px),
        },
        "counts": {
            "machine_part_containment": dict(sorted(part_counts.items())),
            "within_endpoint_duplicate_model_cell_count": len(within_endpoint_duplicates),
            "repeated_model_cell_across_layer_count": len(repeated_model_cells_across_layers),
            "candidates_below_minimum_model_edge_margin": int(sum(not bool(row["model_edge_safe"]) for row in rows)),
        },
        "error_summaries": {
            "raw_center_mapping_error_px": raw_mapping_error_summary,
            "raw_camera_roundtrip_error_px": raw_error_summary,
            "model_grid_roundtrip_error_px": model_error_summary,
            "model_edge_margin_px": edge_margin_summary,
        },
        "checks": checks,
        "coordinate_consistency_pass": coordinate_consistency_pass,
        "operational_geometry_pass_under_provisional_calibration": operational_geometry_pass,
        "edge_safety_pass": edge_safety_pass,
        "recommendation": (
            "internal_coordinate_contract_passed_but_absolute_machine_coordinates_remain_provisional"
            if operational_geometry_pass and edge_safety_pass
            else "hold_affected_candidates_for_coordinate_review_before_any_machine-coordinate interpretation"
        ),
        "important_limit": (
            "Round-trip arithmetic validates only the configured provisional homography and ROI-grid convention. "
            "It does not validate absolute metrology accuracy, response direction, physical defect presence, "
            "or unsampled-region score quality."
        ),
        "storage_policy": "writes compact CSV/JSON metrics only; no TIFF, XCT CSV, target, heatmap, model, or checkpoint is read or modified",
        "outputs": {
            "candidate_coordinate_audit_csv": str(candidate_csv),
            "repeated_model_cells_across_layers_csv": str(duplicate_csv),
            "summary_json": str(summary_json),
        },
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Candidate coordinate audit complete. No raw TIFF/XCT CSV, target, model, checkpoint, or dense heatmap was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
