#!/usr/bin/env python3
"""Audit a provisional machine-XY to raw layer-camera pixel calibration.

Input is a JSON file created by select_machine_camera_control_points.py. This
script uses all manually selected part-corner correspondences to estimate a
projective homography from machine XY [mm] to raw camera pixels. It reports
in-sample and leave-one-out residuals and overlays the predicted part outlines
on the selected B LED3 reference frame.

It is a calibration *audit*, not permission to generate weak heatmaps. A
mapping remains provisional until residuals and the visual overlay are reviewed.

Example
-------
cd ~/ammt_project
/usr/local/bin/python3 src/audit_machine_camera_calibration.py \
  --control-points processed/calibration/camera_control_points.json \
  --output-dir processed/calibration/audit_v1
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

import matplotlib.pyplot as plt
import numpy as np
import tifffile


PART_COLORS = {"part01": "tab:blue", "part02": "tab:orange", "part03": "tab:green", "part04": "tab:red"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a provisional machine-XY to raw-camera-pixel homography")
    parser.add_argument("--control-points", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-fit-rmse-px", type=float, default=10.0)
    parser.add_argument("--max-loo-rmse-px", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def homogeneous(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(len(points), dtype=np.float64)])


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    mean_distance = float(distances.mean())
    if mean_distance <= 0:
        raise ValueError("Degenerate control points: mean distance is zero")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    normalized = (transform @ homogeneous(points).T).T[:, :2]
    return normalized, transform


def estimate_homography(machine_xy: np.ndarray, raw_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalized DLT estimate of H with raw_pixel ~= H @ [machine_x, machine_y, 1]."""
    if machine_xy.shape != raw_xy.shape or machine_xy.ndim != 2 or machine_xy.shape[1] != 2:
        raise ValueError("machine_xy and raw_xy must both be [N,2]")
    if len(machine_xy) < 4:
        raise ValueError("At least four control points are required")
    source, source_transform = normalize_points(machine_xy)
    target, target_transform = normalize_points(raw_xy)
    matrix_rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        matrix_rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        matrix_rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(matrix_rows, dtype=np.float64)
    _, singular_values, vt = np.linalg.svd(design)
    h_normalized = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(target_transform) @ h_normalized @ source_transform
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("Degenerate homography: scale term is near zero")
    homography /= homography[2, 2]
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    return homography, condition


def project_machine_to_raw(homography: np.ndarray, machine_xy: np.ndarray) -> np.ndarray:
    projected = (homography @ homogeneous(machine_xy).T).T
    if np.any(np.isclose(projected[:, 2], 0.0)):
        raise ValueError("Projection reached infinity for one or more machine points")
    return projected[:, :2] / projected[:, 2:3]


def read_reference_frame(reference: dict[str, Any]) -> np.ndarray:
    tiff_path = Path(str(reference["tiff_path"]))
    if not tiff_path.is_file():
        raise FileNotFoundError(f"Reference TIFF recorded in control JSON is missing: {tiff_path}")
    axes = str(reference["axes"])
    shape = tuple(int(value) for value in reference["shape"])
    layer_z = int(reference["layer_z"])
    led = int(reference["led"])
    data = tifffile.memmap(tiff_path, series=0, mode="r")
    index: list[Any] = []
    for axis in axes:
        if axis == "T":
            index.append(led - 1)
        elif axis == "Z":
            index.append(layer_z - 1)
        elif axis == "C":
            index.append(0)
        elif axis in {"Y", "X"}:
            index.append(slice(None))
        else:
            raise ValueError(f"Unsupported TIFF axis: {axis}")
    frame = np.asarray(data[tuple(index)])
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D reference frame, got {frame.shape}")
    return frame


def validate_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "provisional_manual_selection_requires_residual_and_overlay_audit":
        raise ValueError("Control JSON does not have the expected provisional selection status")
    points = payload.get("control_points")
    if not isinstance(points, list) or len(points) < 8:
        raise ValueError("At least eight user-selected control points are required for this audit")
    required = {"part", "corner", "machine_x_mm", "machine_y_mm", "raw_camera_x_px", "raw_camera_y_px"}
    for point in points:
        if not required.issubset(point):
            raise ValueError(f"Control point missing required keys: {point}")
    return points


def compute_loo_residuals(machine_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    if len(machine_xy) < 5:
        return np.full(len(machine_xy), np.nan, dtype=np.float64)
    errors = np.empty(len(machine_xy), dtype=np.float64)
    for holdout in range(len(machine_xy)):
        keep = np.ones(len(machine_xy), dtype=bool)
        keep[holdout] = False
        homography, _ = estimate_homography(machine_xy[keep], raw_xy[keep])
        predicted = project_machine_to_raw(homography, machine_xy[holdout : holdout + 1])[0]
        errors[holdout] = float(np.linalg.norm(predicted - raw_xy[holdout]))
    return errors


def write_residual_csv(path: Path, points: list[dict[str, Any]], predicted: np.ndarray, residual: np.ndarray, loo_residual: np.ndarray) -> None:
    fields = [
        "part", "corner", "machine_x_mm", "machine_y_mm", "observed_raw_x_px", "observed_raw_y_px",
        "predicted_raw_x_px", "predicted_raw_y_px", "fit_residual_px", "leave_one_out_residual_px",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point, projected, fit_error, loo_error in zip(points, predicted, residual, loo_residual, strict=True):
            writer.writerow(
                {
                    "part": point["part"],
                    "corner": point["corner"],
                    "machine_x_mm": float(point["machine_x_mm"]),
                    "machine_y_mm": float(point["machine_y_mm"]),
                    "observed_raw_x_px": float(point["raw_camera_x_px"]),
                    "observed_raw_y_px": float(point["raw_camera_y_px"]),
                    "predicted_raw_x_px": float(projected[0]),
                    "predicted_raw_y_px": float(projected[1]),
                    "fit_residual_px": float(fit_error),
                    "leave_one_out_residual_px": None if not math.isfinite(float(loo_error)) else float(loo_error),
                }
            )


def make_qc_figure(
    output_path: Path,
    frame: np.ndarray,
    reference: dict[str, Any],
    points: list[dict[str, Any]],
    homography: np.ndarray,
    predicted: np.ndarray,
    residual: np.ndarray,
    loo_residual: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    display_low, display_high = reference.get("display_percentiles", [1.0, 99.5])
    vmin, vmax = np.percentile(frame, [float(display_low), float(display_high)])

    overlay_axis = axes[0, 0]
    overlay_axis.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    part_to_machine: dict[str, list[tuple[float, float]]] = {}
    for point in points:
        part_to_machine.setdefault(str(point["part"]), []).append((float(point["machine_x_mm"]), float(point["machine_y_mm"])))
    for part, machine_points in sorted(part_to_machine.items()):
        machine = np.asarray(machine_points, dtype=np.float64)
        transformed = project_machine_to_raw(homography, machine)
        closed = np.vstack([transformed, transformed[0]])
        color = PART_COLORS.get(part, "cyan")
        overlay_axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5, label=f"{part} projected")
    observed = np.asarray([[float(p["raw_camera_x_px"]), float(p["raw_camera_y_px"])] for p in points])
    overlay_axis.scatter(observed[:, 0], observed[:, 1], c="cyan", marker="x", s=45, label="selected pixel")
    overlay_axis.scatter(predicted[:, 0], predicted[:, 1], facecolors="none", edgecolors="magenta", marker="o", s=45, label="H(machine XY)")
    for index, (obs, pred) in enumerate(zip(observed, predicted, strict=True), start=1):
        overlay_axis.plot([obs[0], pred[0]], [obs[1], pred[1]], color="yellow", alpha=0.7, linewidth=0.8)
        overlay_axis.text(obs[0] + 8, obs[1] + 8, str(index), color="cyan", fontsize=7)
    overlay_axis.set_title("Raw B LED3 reference: selected controls and projected part outlines")
    overlay_axis.set_xlabel("Raw camera pixel x")
    overlay_axis.set_ylabel("Raw camera pixel y")
    overlay_axis.legend(loc="lower right", fontsize=8)

    fit_axis = axes[0, 1]
    labels = [f"{p['part'][-2:]}:{p['corner']}" for p in points]
    positions = np.arange(1, len(points) + 1)
    fit_axis.bar(positions - 0.18, residual, width=0.36, label="fit residual")
    if np.any(np.isfinite(loo_residual)):
        fit_axis.bar(positions + 0.18, loo_residual, width=0.36, label="leave-one-out")
    fit_axis.set_title("Per-control-point residual")
    fit_axis.set_xlabel("Control point")
    fit_axis.set_ylabel("Residual [raw pixel]")
    fit_axis.set_xticks(positions, labels, rotation=90, fontsize=7)
    fit_axis.legend(loc="best")

    machine_axis = axes[1, 0]
    machine = np.asarray([[float(p["machine_x_mm"]), float(p["machine_y_mm"])] for p in points])
    machine_axis.scatter(machine[:, 0], machine[:, 1], c=residual, cmap="magma", s=55)
    for index, coordinate in enumerate(machine, start=1):
        machine_axis.text(coordinate[0] + 0.12, coordinate[1] + 0.12, str(index), fontsize=8)
    machine_axis.set_title("Control-point coverage in machine XY")
    machine_axis.set_xlabel("Machine command X [mm]")
    machine_axis.set_ylabel("Machine command Y [mm]")
    machine_axis.set_aspect("equal", adjustable="box")

    hist_axis = axes[1, 1]
    finite_loo = loo_residual[np.isfinite(loo_residual)]
    hist_axis.hist(residual, bins=min(12, len(residual)), alpha=0.7, label="fit")
    if len(finite_loo):
        hist_axis.hist(finite_loo, bins=min(12, len(finite_loo)), alpha=0.7, label="leave-one-out")
    hist_axis.set_title("Residual distribution")
    hist_axis.set_xlabel("Residual [raw pixel]")
    hist_axis.set_ylabel("Control-point count")
    hist_axis.legend(loc="best")

    fig.suptitle("Provisional machine XY → raw layer-camera pixel homography audit", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    control_path = args.control_points.resolve()
    if not control_path.is_file():
        raise FileNotFoundError(f"Missing control-point JSON: {control_path}")
    with control_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    points = validate_payload(payload)
    reference = dict(payload["reference_frame"])
    machine_xy = np.asarray([[float(point["machine_x_mm"]), float(point["machine_y_mm"])] for point in points], dtype=np.float64)
    raw_xy = np.asarray([[float(point["raw_camera_x_px"]), float(point["raw_camera_y_px"])] for point in points], dtype=np.float64)
    homography, condition_number = estimate_homography(machine_xy, raw_xy)
    predicted = project_machine_to_raw(homography, machine_xy)
    residual = np.linalg.norm(predicted - raw_xy, axis=1)
    loo_residual = compute_loo_residuals(machine_xy, raw_xy)
    frame = read_reference_frame(reference)

    prepare_output_directory(args.output_dir.resolve(), args.overwrite)
    output_dir = args.output_dir.resolve()
    write_residual_csv(output_dir / "calibration_residuals.csv", points, predicted, residual, loo_residual)
    make_qc_figure(output_dir / "calibration_qc.png", frame, reference, points, homography, predicted, residual, loo_residual)

    fit_rmse = float(np.sqrt(np.mean(residual ** 2)))
    finite_loo = loo_residual[np.isfinite(loo_residual)]
    loo_rmse = float(np.sqrt(np.mean(finite_loo ** 2))) if len(finite_loo) else None
    fit_pass = fit_rmse <= args.max_fit_rmse_px
    loo_pass = loo_rmse is not None and loo_rmse <= args.max_loo_rmse_px
    summary = {
        "audit_type": "provisional manual-control-point machine-XY to raw-camera-pixel homography audit",
        "control_points_json": str(control_path),
        "raw_input_policy": "Reference TIFF was read via tifffile.memmap(mode='r') and was not modified.",
        "reference_frame": reference,
        "control_point_count": len(points),
        "homography_machine_xy_to_raw_camera_pixel": homography.tolist(),
        "dlt_condition_number": condition_number,
        "fit_residual_px": {
            "rmse": fit_rmse,
            "mean": float(residual.mean()),
            "median": float(np.median(residual)),
            "maximum": float(residual.max()),
        },
        "leave_one_out_residual_px": None if loo_rmse is None else {
            "rmse": loo_rmse,
            "mean": float(finite_loo.mean()),
            "median": float(np.median(finite_loo)),
            "maximum": float(finite_loo.max()),
        },
        "acceptance_gates": {
            "max_fit_rmse_px": args.max_fit_rmse_px,
            "max_leave_one_out_rmse_px": args.max_loo_rmse_px,
            "fit_rmse_pass": fit_pass,
            "leave_one_out_rmse_pass": loo_pass,
            "visual_overlay_review_required": True,
        },
        "status": "provisional_acceptance_candidate" if fit_pass and loo_pass else "provisional_reselection_or_model_revision_required",
        "prohibitions": [
            "Do not generate camera-space XCT weak heatmaps merely because a homography exists.",
            "Do not interpret calibration residual pass as a defect-label validation.",
            "Do not label pixels outside calibrated sparse support as normal/negative.",
        ],
        "outputs": {"residual_csv": "calibration_residuals.csv", "qc_png": "calibration_qc.png"},
    }
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Machine-to-camera calibration audit completed. No raw TIFF, CSV, camera heatmap or label was modified/created.")
    print(f"- control points: {len(points)}")
    print(f"- fit RMSE [px]: {fit_rmse:.3f}")
    print("- leave-one-out RMSE [px]: " + (f"{loo_rmse:.3f}" if loo_rmse is not None else "not available"))
    print(f"- provisional status: {summary['status']}")
    print(f"- output directory: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
