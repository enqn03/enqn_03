utf-8
#!/usr/bin/env python3
"""Compare compact candidate coordinates under rank-1 and rank-2 calibration transforms.
The script performs a read-only sensitivity analysis on already emitted,
geometry-filtered compact candidates. It recomputes existing calibration ranks
from the existing 16 control points, but never fits new controls, changes the
calibration config, reruns model inference, or reads TIFF/XCT data.
A rank comparison quantifies coordinate sensitivity. It neither chooses a new
calibration rank nor validates absolute metrology accuracy or physical defects.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from audit_machine_camera_calibration import PARTS, RECT, build_candidates, project
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
def inverse_machine_xy(inverse_homography: np.ndarray, raw_x: float, raw_y: float, offset_x: float, offset_y: float) -> tuple[np.ndarray | None, float]:
    homogeneous = inverse_homography @ np.array([raw_x - offset_x, raw_y - offset_y, 1.0], dtype=np.float64)
    denominator = float(homogeneous[2])
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return None, denominator
    machine_xy = homogeneous[:2] / denominator
    if not np.isfinite(machine_xy).all():
        return None, denominator
    return machine_xy, denominator
def containing_part(machine_xy: np.ndarray | None, tolerance: float = 1.0e-9) -> str | None:
    if machine_xy is None:
        return None
    x_value, y_value = float(machine_xy[0]), float(machine_xy[1])
    for part in PARTS:
        x_min, x_max, y_min, y_max = RECT[part]
        if x_min - tolerance <= x_value <= x_max + tolerance and y_min - tolerance <= y_value <= y_max + tolerance:
            return str(part)
    return None
def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
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
def candidate_rank_metadata(rank: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "loo_rmse_px": float(item["loo_rmse"]),
        "fit_rmse_px": float(item["fit_rmse"]),
        "orientation": str(item["orientation"]),
        "screen_A_to_D_machine_parts": list(item["assignment"]),
        "corner_index_order": list(item["corner_index_order"]),
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--compare-ranks", nargs=2, type=int, default=[1, 2], metavar=("ALTERNATIVE_RANK", "SELECTED_RANK"))
    parser.add_argument("--output-dir", required=True, type=Path)
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
    alternative_rank, selected_rank = (int(value) for value in args.compare_ranks)
    if alternative_rank == selected_rank or min(alternative_rank, selected_rank) < 1:
        raise ValueError("--compare-ranks requires two distinct positive ranks.")
    if not args.candidate_json.is_file() or not args.calibration_config.is_file():
        raise FileNotFoundError("Candidate JSON and calibration config must both exist.")
    candidate_payload = load_json(args.candidate_json)
    candidates = candidate_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate JSON contains no compact candidates to compare.")
    if int(candidate_payload.get("candidate_count", -1)) != len(candidates):
        raise ValueError("candidate_count does not match candidates list length.")
    if candidate_payload.get("stage") != "A" or candidate_payload.get("response_direction") != "unresolved":
        raise ValueError("Expected A-stage candidate JSON with unresolved response direction.")
    gate_metadata = candidate_payload.get("provisional_part_geometry_gate", {})
    if not isinstance(gate_metadata, dict) or not bool(gate_metadata.get("enabled", False)):
        raise ValueError("This sensitivity audit requires candidates emitted by the enabled provisional part geometry gate.")
    calibration_config = load_yaml(args.calibration_config)
    configured_rank = int(calibration_config["geometry_candidate"]["rank"])
    if selected_rank != configured_rank:
        raise ValueError(f"selected compare rank {selected_rank} must equal calibration_config geometry_candidate.rank {configured_rank}.")
    controls_path = resolve_from_working_directory(str(calibration_config["control_points"]["path"]))
    controls_payload = load_json(controls_path)
    control_points = controls_payload.get("control_points")
    if not isinstance(control_points, list):
        raise ValueError("Control JSON missing control_points list.")
    ranked_candidates = build_candidates(control_points)
    if max(alternative_rank, selected_rank) > len(ranked_candidates):
        raise ValueError("Requested compare rank is outside existing calibration candidate range.")
    alternative = ranked_candidates[alternative_rank - 1]
    selected = ranked_candidates[selected_rank - 1]
    selected_homography = np.asarray(selected["H"], dtype=np.float64)
    alternative_homography = np.asarray(alternative["H"], dtype=np.float64)
    selected_inverse = np.linalg.inv(selected_homography)
    alternative_inverse = np.linalg.inv(alternative_homography)
    offset_x, offset_y = (float(value) for value in calibration_config["local_photometric_refinement"]["raw_pixel_global_offset_xy"])
    rows: list[dict[str, Any]] = []
    shift_distances: list[float] = []
    agreement_count = 0
    selected_inside_count = 0
    alternative_inside_count = 0
    both_inside_count = 0
    endpoint_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"candidate_count": 0, "selected_inside": 0, "alternative_inside": 0, "both_inside": 0, "same_part": 0})
    for candidate in candidates:
        fields = ("sample_id", "layer_z", "rank", "score", "x_model_pixel", "y_model_pixel", "x_pixel", "y_pixel")
        missing = [name for name in fields if name not in candidate]
        if missing:
            raise ValueError(f"Candidate missing fields {missing}: {candidate}")
        raw_x, raw_y = float(candidate["x_pixel"]), float(candidate["y_pixel"])
        if not math.isfinite(raw_x) or not math.isfinite(raw_y):
            raise ValueError("All raw candidate coordinates must be finite.")
        selected_machine, selected_denominator = inverse_machine_xy(selected_inverse, raw_x, raw_y, offset_x, offset_y)
        alternative_machine, alternative_denominator = inverse_machine_xy(alternative_inverse, raw_x, raw_y, offset_x, offset_y)
        selected_part = containing_part(selected_machine)
        alternative_part = containing_part(alternative_machine)
        selected_inside = selected_part is not None
        alternative_inside = alternative_part is not None
        both_inside = selected_inside and alternative_inside
        same_part = both_inside and selected_part == alternative_part
        if selected_machine is not None and alternative_machine is not None:
            shift = float(np.linalg.norm(selected_machine - alternative_machine))
        else:
            shift = math.inf
        layer_z = int(candidate["layer_z"])
        endpoint = endpoint_counts[layer_z]
        endpoint["candidate_count"] += 1
        endpoint["selected_inside"] += int(selected_inside)
        endpoint["alternative_inside"] += int(alternative_inside)
        endpoint["both_inside"] += int(both_inside)
        endpoint["same_part"] += int(same_part)
        selected_inside_count += int(selected_inside)
        alternative_inside_count += int(alternative_inside)
        both_inside_count += int(both_inside)
        agreement_count += int(same_part)
        shift_distances.append(shift)
        rows.append(
            {
                "sample_id": str(candidate["sample_id"]),
                "layer_z": layer_z,
                "rank_within_endpoint": int(candidate["rank"]),
                "score": float(candidate["score"]),
                "x_model_pixel": int(candidate["x_model_pixel"]),
                "y_model_pixel": int(candidate["y_model_pixel"]),
                "raw_camera_x_px": raw_x,
                "raw_camera_y_px": raw_y,
                f"rank_{selected_rank}_machine_x": None if selected_machine is None else float(selected_machine[0]),
                f"rank_{selected_rank}_machine_y": None if selected_machine is None else float(selected_machine[1]),
                f"rank_{selected_rank}_inverse_denominator": selected_denominator,
                f"rank_{selected_rank}_containing_part": selected_part,
                f"rank_{alternative_rank}_machine_x": None if alternative_machine is None else float(alternative_machine[0]),
                f"rank_{alternative_rank}_machine_y": None if alternative_machine is None else float(alternative_machine[1]),
                f"rank_{alternative_rank}_inverse_denominator": alternative_denominator,
                f"rank_{alternative_rank}_containing_part": alternative_part,
                "both_ranks_inside_known_part_rectangles": both_inside,
                "same_containing_part_under_both_ranks": same_part,
                "machine_coordinate_shift_rank_alternative_to_selected": shift,
            }
        )
    endpoint_rows: list[dict[str, Any]] = []
    for layer_z, count in sorted(endpoint_counts.items()):
        candidate_count = count["candidate_count"]
        endpoint_rows.append(
            {
                "layer_z": layer_z,
                "candidate_count": candidate_count,
                f"rank_{selected_rank}_inside_count": count["selected_inside"],
                f"rank_{alternative_rank}_inside_count": count["alternative_inside"],
                "both_ranks_inside_count": count["both_inside"],
                "same_part_count": count["same_part"],
                "same_part_fraction": count["same_part"] / candidate_count,
            }
        )
    selected_part_counts = Counter(str(row[f"rank_{selected_rank}_containing_part"]) if row[f"rank_{selected_rank}_containing_part"] is not None else "outside_known_part_rectangles" for row in rows)
    alternative_part_counts = Counter(str(row[f"rank_{alternative_rank}_containing_part"]) if row[f"rank_{alternative_rank}_containing_part"] is not None else "outside_known_part_rectangles" for row in rows)
    candidate_count = len(rows)
    output_dir = args.output_dir
    prepare_output_directory(output_dir, args.overwrite)
    by_candidate_path = output_dir / "candidate_rank_sensitivity.csv"
    by_endpoint_path = output_dir / "candidate_rank_sensitivity_by_endpoint.csv"
    summary_path = output_dir / "candidate_rank_sensitivity_summary.json"
    with by_candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with by_endpoint_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    summary = {
        "audit_type": "read-only compact candidate calibration-rank sensitivity; not calibration selection, metrology validation, or defect labeling",
        "candidate_json": str(args.candidate_json),
        "candidate_count": candidate_count,
        "candidate_stage": candidate_payload["stage"],
        "response_direction": candidate_payload["response_direction"],
        "score_semantics": "XCT-derived continuous quality candidate; not confirmed defect or anomaly probability",
        "calibration_status": calibration_config["status"],
        "configured_selected_rank": configured_rank,
        "compared_ranks": {
            "alternative": candidate_rank_metadata(alternative_rank, alternative),
            "selected": candidate_rank_metadata(selected_rank, selected),
            "raw_pixel_global_offset_xy": [offset_x, offset_y],
        },
        "containment": {
            f"rank_{selected_rank}_inside_count": selected_inside_count,
            f"rank_{selected_rank}_inside_fraction": selected_inside_count / candidate_count,
            f"rank_{alternative_rank}_inside_count": alternative_inside_count,
            f"rank_{alternative_rank}_inside_fraction": alternative_inside_count / candidate_count,
            "both_ranks_inside_count": both_inside_count,
            "both_ranks_inside_fraction": both_inside_count / candidate_count,
            "same_containing_part_count": agreement_count,
            "same_containing_part_fraction": agreement_count / candidate_count,
            f"rank_{selected_rank}_part_distribution": dict(sorted(selected_part_counts.items())),
            f"rank_{alternative_rank}_part_distribution": dict(sorted(alternative_part_counts.items())),
        },
        "machine_coordinate_shift_alternative_to_selected": numeric_summary(shift_distances),
        "per_endpoint_summary": {
            "endpoint_count": len(endpoint_rows),
            "candidate_count_per_endpoint": sorted(set(int(row["candidate_count"]) for row in endpoint_rows)),
            "minimum_same_part_fraction": min(float(row["same_part_fraction"]) for row in endpoint_rows),
            "maximum_same_part_fraction": max(float(row["same_part_fraction"]) for row in endpoint_rows),
        },
        "interpretation": {
            "rank_2_is_not_reselected": "Rank 2 remains the configured provisional transform; this audit only measures sensitivity to existing rank 1.",
            "part_agreement": "Same-part fraction measures whether an already geometry-gated candidate retains the same configured part under the alternative transform.",
            "machine_shift": "Machine-coordinate shift measures coordinate dependence on transform rank; it has no physical-error threshold without independent metrology.",
            "reporting_policy": "If alternative-rank containment or same-part agreement is unstable, report candidate camera coordinates and selected-rank provisional machine coordinates separately; do not make absolute coordinate claims.",
        },
        "storage_policy": "writes compact CSV/JSON metrics only; reads no TIFF, XCT CSV, target, checkpoint, or score map",
        "outputs": {
            "candidate_rank_sensitivity_csv": str(by_candidate_path),
            "candidate_rank_sensitivity_by_endpoint_csv": str(by_endpoint_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Calibration-rank sensitivity audit complete. No raw TIFF/XCT CSV, target, model, checkpoint, calibration config, or dense heatmap was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
