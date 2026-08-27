#!/usr/bin/env python3
"""Read-only support-density audit for continuous XCT-derived weak targets.

This audit reconstructs the same command-XY projection, train-p01/p99 response
scaling, model ROI/grid mapping, Gaussian support, and weighted-average blend
used by AMMTWeakTargetDataset. It compares only Gaussian sigma values in memory.
It does not open TIFF files, train a model, write a dense target, or alter raw data.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from audit_machine_camera_calibration import build_candidates, project

PARTS = ("part01", "part02", "part03", "part04")
DEFAULT_LAYERS = (4, 128, 157, 161, 180, 199, 203, 227, 250)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return np.exp(-((xx * xx + yy * yy) / (2.0 * sigma * sigma))).astype(np.float64)


def connected_component_statistics(mask: np.ndarray) -> tuple[int, int, float]:
    """Return 8-connected component count, largest size, and largest share."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    component_count = 0
    largest_size = 0

    for start_y, start_x in np.argwhere(binary):
        y0, x0 = int(start_y), int(start_x)
        if visited[y0, x0]:
            continue
        component_count += 1
        visited[y0, x0] = True
        queue: deque[tuple[int, int]] = deque([(y0, x0)])
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        largest_size = max(largest_size, size)

    total = int(binary.sum())
    largest_fraction = 0.0 if total == 0 else float(largest_size / total)
    return component_count, largest_size, largest_fraction


def pearson_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def load_manifest_endpoint_splits(manifest_path: Path) -> dict[int, str]:
    split_by_layer: dict[int, str] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            endpoint = int(row["endpoint_layer_z"])
            split = row["split"]
            if endpoint in split_by_layer and split_by_layer[endpoint] != split:
                raise ValueError(f"Endpoint layer {endpoint} appears in multiple splits.")
            split_by_layer[endpoint] = split
    return split_by_layer


def collect_projected_points(
    *,
    layer_z: int,
    registered_root: Path,
    homography: np.ndarray,
    offset_xy: tuple[float, float],
    roi_xyxy: tuple[int, int, int, int],
    model_height: int,
    model_width: int,
    response_p01: float,
    response_p99: float,
    clip_response: bool,
) -> tuple[list[tuple[int, int, float]], int]:
    """Use the same rounding and in-ROI policy as AMMTWeakTargetDataset._target."""
    x0_raw, y0_raw, x1_raw, y1_raw = roi_xyxy
    sx = model_width / (x1_raw - x0_raw)
    sy = model_height / (y1_raw - y0_raw)
    dx, dy = offset_xy
    points: list[tuple[int, int, float]] = []
    finite_count = 0

    for part in PARTS:
        csv_path = registered_root / part / f"L{layer_z:04d}.csv"
        if not csv_path.is_file():
            continue
        table = np.genfromtxt(csv_path, delimiter=",")
        table = np.atleast_2d(table)
        good = np.isfinite(table[:, 39])
        if not good.any():
            continue
        finite_count += int(good.sum())
        uv = project(homography, table[good, 2:4])
        values = (table[good, 39] - response_p01) / (response_p99 - response_p01)
        if clip_response:
            values = np.clip(values, 0.0, 1.0)
        for (u, v), value in zip(uv, values):
            ix = int(round((float(u) + dx - x0_raw) * sx))
            iy = int(round((float(v) + dy - y0_raw) * sy))
            if 0 <= ix < model_width and 0 <= iy < model_height:
                points.append((ix, iy, float(value)))
    return points, finite_count


def rasterize(
    *,
    points: list[tuple[int, int, float]],
    sigma: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    kernel = gaussian_kernel(sigma)
    radius = kernel.shape[0] // 2

    for ix, iy, value in points:
        y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
        x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
        ky0, kx0 = y0 - (iy - radius), x0 - (ix - radius)
        patch = kernel[ky0 : ky0 + (y1 - y0), kx0 : kx0 + (x1 - x0)]
        numerator[y0:y1, x0:x1] += patch * value
        weight[y0:y1, x0:x1] += patch

    support = weight > 0.0
    response = np.zeros((height, width), dtype=np.float64)
    response[support] = numerator[support] / weight[support]
    return response, support


def base_metrics(
    *,
    layer_z: int,
    split: str,
    sigma: float,
    points: list[tuple[int, int, float]],
    finite_xct_point_count: int,
    response: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    support_pixels = int(support.sum())
    component_count, largest_component_pixels, largest_component_fraction = connected_component_statistics(support)
    unique_centers = len({(x, y) for x, y, _ in points})
    in_roi_count = len(points)
    return {
        "layer_z": layer_z,
        "split": split,
        "sigma_model_px": sigma,
        "finite_xct_point_count": finite_xct_point_count,
        "projected_in_roi_point_count": in_roi_count,
        "unique_center_pixel_count": unique_centers,
        "center_collision_fraction": None if in_roi_count == 0 else float(1.0 - unique_centers / in_roi_count),
        "weak_target_available": bool(support_pixels > 0),
        "support_pixel_count": support_pixels,
        "support_fraction": float(support.mean()),
        "response_nonzero_pixel_count": int((response > 0.0).sum()),
        "response_nonzero_fraction": float((response > 0.0).mean()),
        "support_component_count_8_connected": component_count,
        "largest_support_component_pixel_count": largest_component_pixels,
        "largest_support_component_fraction": largest_component_fraction,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else ["layer_z"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Gaussian sigma=2/3 weak-target support density without model training."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--sigmas", nargs=2, type=float, default=[2.0, 3.0])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sigma_base, sigma_candidate = (float(args.sigmas[0]), float(args.sigmas[1]))
    if not (sigma_base > 0.0 and sigma_candidate > 0.0 and sigma_candidate > sigma_base):
        raise ValueError("--sigmas must be two positive increasing values, for example: 2 3.")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Review it or use --overwrite deliberately.")
        shutil.rmtree(output_dir)

    endpoint_splits = load_manifest_endpoint_splits(args.manifest)
    requested_layers = [int(layer) for layer in args.layers]
    unexpected = [layer for layer in requested_layers if layer not in endpoint_splits]
    if unexpected:
        raise ValueError(f"Requested layers are not manifest endpoints: {unexpected}")

    normalization = load_yaml(args.normalization_config)
    weak = load_yaml(args.weak_target_config)
    calibration = load_yaml(args.calibration_config)
    roi = tuple(int(value) for value in normalization["working_roi"]["coordinates_raw_camera_pixels"])
    model_height, model_width = (int(value) for value in weak["rasterization"]["model_resolution"])
    scaling = weak["response"]["robust_scaling"]
    if scaling["method"] != "train_p01_p99":
        raise ValueError("This audit requires train_p01_p99 scaling to match the baseline Dataset.")
    response_p01, response_p99 = float(scaling["train_p01"]), float(scaling["train_p99"])
    if response_p99 <= response_p01:
        raise ValueError("weak target train_p99 must be greater than train_p01.")

    controls_path = Path(calibration["control_points"]["path"])
    if not controls_path.is_absolute():
        controls_path = Path.cwd() / controls_path
    controls = json.loads(controls_path.read_text(encoding="utf-8"))["control_points"]
    candidate_rank = int(calibration["geometry_candidate"]["rank"]) - 1
    homography = build_candidates(controls)[candidate_rank]["H"]
    offset_xy = tuple(float(value) for value in calibration["local_photometric_refinement"]["raw_pixel_global_offset_xy"])

    per_sigma_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    comparable_layers: list[dict[str, Any]] = []
    for layer_z in requested_layers:
        split = endpoint_splits[layer_z]
        points, finite_xct_point_count = collect_projected_points(
            layer_z=layer_z,
            registered_root=args.registered_root,
            homography=homography,
            offset_xy=offset_xy,
            roi_xyxy=roi,
            model_height=model_height,
            model_width=model_width,
            response_p01=response_p01,
            response_p99=response_p99,
            clip_response=bool(scaling["clip_to_unit_interval"]),
        )
        response_base, support_base = rasterize(
            points=points, sigma=sigma_base, height=model_height, width=model_width
        )
        response_candidate, support_candidate = rasterize(
            points=points, sigma=sigma_candidate, height=model_height, width=model_width
        )
        metrics_base = base_metrics(
            layer_z=layer_z,
            split=split,
            sigma=sigma_base,
            points=points,
            finite_xct_point_count=finite_xct_point_count,
            response=response_base,
            support=support_base,
        )
        metrics_candidate = base_metrics(
            layer_z=layer_z,
            split=split,
            sigma=sigma_candidate,
            points=points,
            finite_xct_point_count=finite_xct_point_count,
            response=response_candidate,
            support=support_candidate,
        )
        per_sigma_rows.extend((metrics_base, metrics_candidate))

        base_support_count = int(support_base.sum())
        candidate_support_count = int(support_candidate.sum())
        common_support = support_base & support_candidate
        added_support = support_candidate & ~support_base
        retained_fraction = 1.0 if base_support_count == 0 else float(common_support.sum() / base_support_count)
        support_gain_fraction = None if base_support_count == 0 else float((candidate_support_count - base_support_count) / base_support_count)
        common_response_mae = None if not common_support.any() else float(
            np.abs(response_candidate[common_support] - response_base[common_support]).mean()
        )
        common_response_pearson = None if not common_support.any() else pearson_or_none(
            response_base[common_support], response_candidate[common_support]
        )
        component_ratio = None if metrics_base["support_component_count_8_connected"] == 0 else float(
            metrics_candidate["support_component_count_8_connected"]
            / metrics_base["support_component_count_8_connected"]
        )
        largest_component_growth_ratio = None
        if metrics_base["largest_support_component_fraction"] > 0.0:
            largest_component_growth_ratio = float(
                metrics_candidate["largest_support_component_fraction"]
                / metrics_base["largest_support_component_fraction"]
            )
        comparison = {
            "layer_z": layer_z,
            "split": split,
            "weak_target_available_sigma_base": metrics_base["weak_target_available"],
            "support_pixels_sigma_base": base_support_count,
            "support_pixels_sigma_candidate": candidate_support_count,
            "support_fraction_sigma_base": metrics_base["support_fraction"],
            "support_fraction_sigma_candidate": metrics_candidate["support_fraction"],
            "support_gain_fraction": support_gain_fraction,
            "added_support_pixels": int(added_support.sum()),
            "base_support_retained_in_candidate_fraction": retained_fraction,
            "common_support_response_mae": common_response_mae,
            "common_support_response_pearson": common_response_pearson,
            "component_count_sigma_base": metrics_base["support_component_count_8_connected"],
            "component_count_sigma_candidate": metrics_candidate["support_component_count_8_connected"],
            "component_count_candidate_to_base_ratio": component_ratio,
            "largest_component_fraction_sigma_base": metrics_base["largest_support_component_fraction"],
            "largest_component_fraction_sigma_candidate": metrics_candidate["largest_support_component_fraction"],
            "largest_component_fraction_growth_ratio": largest_component_growth_ratio,
        }
        comparison_rows.append(comparison)
        if base_support_count > 0:
            comparable_layers.append(comparison)

    support_gains = [float(row["support_gain_fraction"]) for row in comparable_layers if row["support_gain_fraction"] is not None]
    retained = [float(row["base_support_retained_in_candidate_fraction"]) for row in comparable_layers]
    response_maes = [float(row["common_support_response_mae"]) for row in comparable_layers if row["common_support_response_mae"] is not None]
    component_ratios = [float(row["component_count_candidate_to_base_ratio"]) for row in comparable_layers if row["component_count_candidate_to_base_ratio"] is not None]
    largest_component_growth = [float(row["largest_component_fraction_growth_ratio"]) for row in comparable_layers if row["largest_component_fraction_growth_ratio"] is not None]

    gates = {
        "minimum_median_support_gain_fraction": 0.25,
        "minimum_base_support_retained_fraction": 1.0,
        "maximum_median_common_support_response_mae": 0.05,
        "minimum_median_component_count_ratio": 0.5,
        "maximum_median_largest_component_fraction_growth_ratio": 1.5,
    }
    observed = {
        "comparable_available_layer_count": len(comparable_layers),
        "median_support_gain_fraction": median_or_none(support_gains),
        "minimum_base_support_retained_in_candidate_fraction": None if not retained else float(min(retained)),
        "median_common_support_response_mae": median_or_none(response_maes),
        "median_component_count_candidate_to_base_ratio": median_or_none(component_ratios),
        "median_largest_component_fraction_growth_ratio": median_or_none(largest_component_growth),
    }
    automatic_gate_pass = bool(comparable_layers) and (
        observed["median_support_gain_fraction"] is not None
        and observed["median_support_gain_fraction"] >= gates["minimum_median_support_gain_fraction"]
        and observed["minimum_base_support_retained_in_candidate_fraction"] >= gates["minimum_base_support_retained_fraction"]
        and observed["median_common_support_response_mae"] is not None
        and observed["median_common_support_response_mae"] <= gates["maximum_median_common_support_response_mae"]
        and observed["median_component_count_candidate_to_base_ratio"] is not None
        and observed["median_component_count_candidate_to_base_ratio"] >= gates["minimum_median_component_count_ratio"]
        and observed["median_largest_component_fraction_growth_ratio"] is not None
        and observed["median_largest_component_fraction_growth_ratio"] <= gates["maximum_median_largest_component_fraction_growth_ratio"]
    )

    summary = {
        "audit_type": "read-only weak-target Gaussian support-density comparison; not training or defect labeling",
        "baseline_sigma_model_px": sigma_base,
        "candidate_sigma_model_px": sigma_candidate,
        "requested_endpoint_layers": requested_layers,
        "model_resolution": [model_height, model_width],
        "working_roi_raw_camera_pixels": list(roi),
        "calibration_status": calibration.get("status", "unspecified"),
        "response_policy": {
            "source": weak["source"]["registered_xct_name"],
            "scaling": "train_p01_p99",
            "train_p01": response_p01,
            "train_p99": response_p99,
            "direction": weak["response"]["direction"],
            "binary_defect_threshold": weak["response"]["binary_defect_threshold"],
        },
        "unknown_policy": weak["support"]["outside_support"],
        "storage_policy": "all response/support arrays are in-memory only; CSV/JSON metrics only",
        "comparison_gates_provisional": gates,
        "observed_summary": observed,
        "automatic_gate_pass": automatic_gate_pass,
        "recommendation": (
            "candidate_sigma_is_eligible_for_separate_training_review"
            if automatic_gate_pass
            else "hold_candidate_sigma; review support expansion and local-merge metrics before any training config change"
        ),
        "important_limit": (
            "This audit measures weak-supervision coverage and rasterization change, not physical defect direction, "
            "model localization quality, or deployment readiness."
        ),
    }

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "per_layer_sigma_metrics.csv", per_sigma_rows)
    write_csv(output_dir / "sigma_comparison_by_layer.csv", comparison_rows)
    (output_dir / "support_density_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("Weak-target support-density audit complete. No TIFF, raw CSV, dense target, checkpoint, or model was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
