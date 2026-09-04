#!/usr/bin/env python3
"""Measure independent metrology fiducial detectability without calibration fitting.
The audit reads NIST dot-grid, checkerboard, and secondary-camera TIFF metadata
through read-only memmap access. It extracts compact *pixel-space feature
candidates* and measures local regularity/coverage only. It never computes a
machine-to-camera homography, selects a calibration rank, changes a config,
uses XCT/model output, or identifies a red component as the machine origin.
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
MAX_FEATURE_ROWS = 5000
MAX_RED_COMPONENT_ROWS = 50
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path)
    parser.add_argument("--secondary-camera", required=True, type=Path)
    parser.add_argument("--checkerboard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-display-pixels", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def robust_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Image has no finite values.")
    p01, p50, p99 = (float(np.percentile(finite, p)) for p in (1, 50, 99))
    if not math.isfinite(p01) or not math.isfinite(p99) or p99 <= p01:
        raise ValueError(f"Invalid robust display range: p01={p01}, p99={p99}.")
    return np.clip((values - p01) / (p99 - p01), 0.0, 1.0), {"p01": p01, "p50": p50, "p99": p99}
def downsample(image: np.ndarray, max_display_pixels: int) -> tuple[np.ndarray, int]:
    if max_display_pixels < 128:
        raise ValueError("--max-display-pixels must be at least 128.")
    height, width = image.shape[-2:]
    stride = max(1, int(math.ceil(max(height, width) / max_display_pixels)))
    return image[..., ::stride, ::stride], stride
def to_channels_yx(data: np.ndarray, axes: str) -> np.ndarray:
    """Keep Y/X and one samples/channels axis; select first non-spatial page."""
    if data.ndim != len(axes):
        raise ValueError(f"Series axes {axes!r} do not match array shape {data.shape}.")
    selectors: list[int | slice] = []
    kept: list[str] = []
    channel_kept = False
    for axis, size in zip(axes, data.shape, strict=True):
        if axis in {"Y", "X"}:
            selectors.append(slice(None))
            kept.append(axis)
        elif axis in {"C", "S"} and not channel_kept:
            selectors.append(slice(None))
            kept.append(axis)
            channel_kept = True
        else:
            if size < 1:
                raise ValueError(f"Axis {axis!r} has unusable size {size}.")
            selectors.append(0)
    selected = np.asarray(data[tuple(selectors)])
    if "Y" not in kept or "X" not in kept:
        raise ValueError(f"Metadata TIFF has no usable Y/X axes: {axes!r}.")
    channel_axis = next((index for index, axis in enumerate(kept) if axis in {"C", "S"}), None)
    y_axis, x_axis = kept.index("Y"), kept.index("X")
    if channel_axis is None:
        if selected.ndim != 2:
            raise ValueError(f"Expected 2D grayscale array, got {selected.shape} from {axes!r}.")
        return selected[None, ...]
    if selected.ndim != 3:
        raise ValueError(f"Expected multi-channel 2D array, got {selected.shape} from {axes!r}.")
    return np.transpose(selected, (channel_axis, y_axis, x_axis))
def read_tiff(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata TIFF not found: {path}")
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        metadata = {
            "path": str(path),
            "series_axes": str(series.axes),
            "series_shape": [int(value) for value in series.shape],
            "dtype": str(series.dtype),
            "is_imagej": bool(handle.is_imagej),
        }
    data = tifffile.memmap(path, series=0, mode="r")
    channels = to_channels_yx(data, str(metadata["series_axes"]))
    metadata.update({
        "tiff_access": "tifffile.memmap(..., series=0, mode='r')",
        "channel_count": int(channels.shape[0]),
        "raw_height_px": int(channels.shape[1]),
        "raw_width_px": int(channels.shape[2]),
    })
    return channels, metadata
def grayscale(channels: np.ndarray) -> np.ndarray:
    if channels.shape[0] == 1:
        return np.asarray(channels[0], dtype=np.float32)
    if channels.shape[0] >= 3:
        return (0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]).astype(np.float32)
    return channels.mean(axis=0, dtype=np.float32)
def box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """Edge-padded box mean with summed-area table; no third-party CV dependency."""
    if radius < 1:
        raise ValueError("box_mean radius must be positive.")
    kernel = 2 * radius + 1
    padded = np.pad(np.asarray(image, dtype=np.float32), radius, mode="edge")
    integral = np.pad(padded.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64), ((1, 0), (1, 0)))
    summed = integral[kernel:, kernel:] - integral[:-kernel, kernel:] - integral[kernel:, :-kernel] + integral[:-kernel, :-kernel]
    return (summed / float(kernel * kernel)).astype(np.float32)
def greedy_nms(response: np.ndarray, quantile: float, min_distance_px: int, maximum: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract high-response pixel candidates using bounded greedy square NMS."""
    if min_distance_px < 1 or maximum < 1:
        raise ValueError("NMS limits must be positive.")
    finite = response[np.isfinite(response)]
    if finite.size == 0:
        raise ValueError("Feature response has no finite values.")
    threshold = float(np.quantile(finite, quantile))
    yy, xx = np.nonzero(np.isfinite(response) & (response >= threshold))
    scores = response[yy, xx]
    order = np.argsort(scores, kind="stable")[::-1]
    suppressed = np.zeros(response.shape, dtype=bool)
    accepted_y: list[int] = []
    accepted_x: list[int] = []
    accepted_score: list[float] = []
    height, width = response.shape
    for index in order:
        y, x = int(yy[index]), int(xx[index])
        if suppressed[y, x]:
            continue
        accepted_y.append(y)
        accepted_x.append(x)
        accepted_score.append(float(scores[index]))
        y0, y1 = max(0, y - min_distance_px), min(height, y + min_distance_px + 1)
        x0, x1 = max(0, x - min_distance_px), min(width, x + min_distance_px + 1)
        suppressed[y0:y1, x0:x1] = True
        if len(accepted_y) >= maximum:
            break
    points = np.column_stack([accepted_x, accepted_y]).astype(np.float64, copy=False)
    return points, np.asarray(accepted_score, dtype=np.float64), {
        "response_quantile": float(quantile),
        "response_threshold": threshold,
        "pre_nms_pixel_count": int(len(order)),
        "min_distance_px": int(min_distance_px),
        "maximum_retained": int(maximum),
        "retained_count": int(len(points)),
    }
def nearest_neighbor_metrics(points: np.ndarray, sample_limit: int = 1500) -> dict[str, float | int | None]:
    if len(points) < 2:
        return {"sample_count": int(len(points)), "nearest_neighbor_median_px": None, "nearest_neighbor_p05_px": None, "nearest_neighbor_p95_px": None, "nearest_neighbor_cv": None}
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    indices = np.linspace(0, len(ordered) - 1, num=min(len(ordered), sample_limit), dtype=np.int64)
    sampled = ordered[indices].astype(np.float32, copy=False)
    differences = sampled[:, None, :] - sampled[None, :, :]
    squared_distance = np.einsum("ijk,ijk->ij", differences, differences, optimize=True)
    np.fill_diagonal(squared_distance, np.inf)
    nearest = np.sqrt(np.min(squared_distance, axis=1))
    median = float(np.median(nearest))
    return {
        "sample_count": int(len(sampled)),
        "nearest_neighbor_median_px": median,
        "nearest_neighbor_p05_px": float(np.percentile(nearest, 5)),
        "nearest_neighbor_p95_px": float(np.percentile(nearest, 95)),
        "nearest_neighbor_cv": None if median <= 1.0e-12 else float(np.std(nearest) / median),
    }
def feature_geometry_metrics(points: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    height, width = image_shape
    if len(points) == 0:
        return {"candidate_count": 0, "bbox_xyxy_px": None, "bbox_area_fraction": 0.0}
    x0, y0 = np.percentile(points, 2.5, axis=0)
    x1, y1 = np.percentile(points, 97.5, axis=0)
    area = max(float(x1 - x0), 0.0) * max(float(y1 - y0), 0.0)
    return {
        "candidate_count": int(len(points)),
        "bbox_xyxy_px": [float(x0), float(y0), float(x1), float(y1)],
        "bbox_area_fraction": float(area / float(height * width)),
    }
def dot_grid_candidates(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized, _ = robust_normalize(gray)
    response = box_mean(normalized, radius=3) - normalized
    points, scores, nms = greedy_nms(response, quantile=0.995, min_distance_px=3, maximum=MAX_FEATURE_ROWS)
    nearest = nearest_neighbor_metrics(points)
    geometry = feature_geometry_metrics(points, gray.shape)
    eligible = bool(len(points) >= 200 and nearest["nearest_neighbor_cv"] is not None and float(nearest["nearest_neighbor_cv"]) <= 0.65)
    return points, scores, {
        "detector": "dark_blob_response=box_mean(normalized_gray,r=3)-normalized_gray; greedy_nms",
        "candidate_type": "dot_center_candidate",
        "fit_eligibility_local_regular_lattice": eligible,
        "eligibility_rule": "retained_count>=200 and nearest_neighbor_cv<=0.65; not a calibration fit",
        **nms,
        **nearest,
        **geometry,
    }
def checkerboard_candidates(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized, _ = robust_normalize(gray)
    gy, gx = np.gradient(normalized)
    a = box_mean(gx * gx, radius=4)
    b = box_mean(gx * gy, radius=4)
    c = box_mean(gy * gy, radius=4)
    response = (a * c - b * b) - 0.04 * (a + c) ** 2
    points, scores, nms = greedy_nms(response, quantile=0.998, min_distance_px=6, maximum=MAX_FEATURE_ROWS)
    nearest = nearest_neighbor_metrics(points)
    geometry = feature_geometry_metrics(points, gray.shape)
    eligible = bool(len(points) >= 100 and nearest["nearest_neighbor_cv"] is not None and float(nearest["nearest_neighbor_cv"]) <= 0.75)
    return points, scores, {
        "detector": "Harris-like response with box-smoothed gradient tensor (r=4); greedy_nms",
        "candidate_type": "checkerboard_corner_candidate",
        "fit_eligibility_local_regular_lattice": eligible,
        "eligibility_rule": "retained_count>=100 and nearest_neighbor_cv<=0.75; not a calibration fit",
        **nms,
        **nearest,
        **geometry,
    }
def red_dominance(channels: np.ndarray) -> np.ndarray:
    if channels.shape[0] < 3:
        raise ValueError("Secondary-camera TIFF has fewer than three channels; red-component audit is unavailable.")
    red, _ = robust_normalize(channels[0])
    green, _ = robust_normalize(channels[1])
    blue, _ = robust_normalize(channels[2])
    return np.clip(red - 0.5 * (green + blue), 0.0, None).astype(np.float32)
def connected_components(mask: np.ndarray, weights: np.ndarray, minimum_area: int = 3, maximum: int = MAX_RED_COMPONENT_ROWS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Eight-connected components on a sparse thresholded mask only."""
    if mask.shape != weights.shape:
        raise ValueError("Mask and weight image shapes differ.")
    positive = np.argwhere(mask)
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[dict[str, Any]] = []
    for start_y, start_x in positive:
        y0, x0 = int(start_y), int(start_x)
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        pixels_y: list[int] = []
        pixels_x: list[int] = []
        while stack:
            y, x = stack.pop()
            pixels_y.append(y)
            pixels_x.append(x)
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
        if len(pixels_x) < minimum_area:
            continue
        ys = np.asarray(pixels_y, dtype=np.float64)
        xs = np.asarray(pixels_x, dtype=np.float64)
        component_weights = weights[ys.astype(np.int64), xs.astype(np.int64)].astype(np.float64)
        integrated = float(component_weights.sum())
        if integrated <= 0.0 or not math.isfinite(integrated):
            continue
        centroid_x = float((xs * component_weights).sum() / integrated)
        centroid_y = float((ys * component_weights).sum() / integrated)
        variance = float((((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2) * component_weights).sum() / integrated)
        components.append({
            "area_px": int(len(xs)),
            "integrated_red_dominance": integrated,
            "peak_red_dominance": float(component_weights.max()),
            "centroid_x_px": centroid_x,
            "centroid_y_px": centroid_y,
            "weighted_spread_px": float(math.sqrt(max(variance, 0.0))),
            "bbox_x0_px": int(xs.min()),
            "bbox_y0_px": int(ys.min()),
            "bbox_x1_px": int(xs.max()),
            "bbox_y1_px": int(ys.max()),
            "touches_image_boundary": bool(xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1),
        })
    components.sort(key=lambda row: (-float(row["integrated_red_dominance"]), -float(row["peak_red_dominance"]), int(row["area_px"])))
    for rank, row in enumerate(components[:maximum], start=1):
        row["rank_by_integrated_red_dominance"] = rank
    return components[:maximum], {
        "thresholded_pixel_count": int(mask.sum()),
        "component_count_minimum_area": int(len(components)),
        "minimum_component_area_px": int(minimum_area),
        "maximum_recorded_components": int(maximum),
    }
def red_component_candidates(channels: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    dominance = red_dominance(channels)
    positive = dominance[dominance > 0.0]
    if positive.size == 0:
        raise ValueError("Secondary-camera RGB data have no positive red-dominance pixels.")
    threshold = float(np.quantile(positive, 0.999))
    components, metrics = connected_components(dominance >= threshold, dominance)
    top = components[0] if components else None
    locally_compact = bool(top is not None and not bool(top["touches_image_boundary"]) and float(top["weighted_spread_px"]) <= 50.0)
    return components, dominance, {
        "detector": "per-channel robust normalization; red - 0.5*(green+blue); q=0.999 threshold; 8-connected components",
        "candidate_type": "secondary_red_component_candidate",
        "red_dominance_quantile": 0.999,
        "red_dominance_threshold": threshold,
        "top_component_locally_compact": locally_compact,
        "fit_eligibility_local_reference_component": locally_compact,
        "eligibility_rule": "top component exists, does not touch image boundary, weighted_spread_px<=50; not machine-origin attribution",
        "top_component": top,
        **metrics,
    }
def write_feature_csv(path: Path, points: np.ndarray, scores: np.ndarray, point_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["rank_by_response", "feature_type", "raw_x_px", "raw_y_px", "response"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, (point, score) in enumerate(zip(points, scores, strict=True), start=1):
            writer.writerow({"rank_by_response": rank, "feature_type": point_name, "raw_x_px": float(point[0]), "raw_y_px": float(point[1]), "response": float(score)})
def write_component_csv(path: Path, components: list[dict[str, Any]]) -> None:
    fields = [
        "rank_by_integrated_red_dominance", "area_px", "integrated_red_dominance", "peak_red_dominance", "centroid_x_px", "centroid_y_px", "weighted_spread_px",
        "bbox_x0_px", "bbox_y0_px", "bbox_x1_px", "bbox_y1_px", "touches_image_boundary",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(components)
def plot_feature_overlay(gray: np.ndarray, points: np.ndarray, title: str, label: str, output_path: Path) -> None:
    display, stride = downsample(gray, 1000)
    figure, axis = plt.subplots(figsize=(9, 9), dpi=150)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    if len(points):
        plot_points = points[::max(1, int(math.ceil(len(points) / 2500)))] / float(stride)
        axis.scatter(plot_points[:, 0], plot_points[:, 1], s=2.0, c="cyan", alpha=0.70, linewidths=0, label=label)
        axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    axis.set_title(f"{title}\nPixel-space candidates only — no calibration fit")
    axis.set_xlabel("display x [pixel]")
    axis.set_ylabel("display y [pixel]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
def plot_red_overlay(channels: np.ndarray, components: list[dict[str, Any]], output_path: Path) -> None:
    rgb = np.stack([robust_normalize(channel)[0] for channel in channels[:3]], axis=-1)
    display, stride = downsample(np.moveaxis(rgb, -1, 0), 1000)
    display_rgb = np.moveaxis(display, 0, -1)
    figure, axis = plt.subplots(figsize=(11, 8), dpi=150)
    axis.imshow(display_rgb, origin="upper", interpolation="nearest")
    colors = ["cyan", "lime", "yellow", "magenta", "orange"]
    for row, color in zip(components[:5], colors, strict=False):
        x0, x1 = float(row["bbox_x0_px"]) / stride, float(row["bbox_x1_px"]) / stride
        y0, y1 = float(row["bbox_y0_px"]) / stride, float(row["bbox_y1_px"]) / stride
        rectangle = plt.Rectangle((x0, y0), max(x1 - x0, 1.0), max(y1 - y0, 1.0), fill=False, edgecolor=color, linewidth=1.5)
        axis.add_patch(rectangle)
        axis.scatter([float(row["centroid_x_px"]) / stride], [float(row["centroid_y_px"]) / stride], c=color, marker="+", s=100, linewidths=1.5)
        axis.text(x0, max(0.0, y0 - 4.0), f"#{row['rank_by_integrated_red_dominance']}", color=color, fontsize=9, fontweight="bold")
    axis.set_title("Secondary-camera red connected components\nCandidates only — no machine-origin attribution or calibration fit")
    axis.set_xlabel("display x [pixel]")
    axis.set_ylabel("display y [pixel]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    for required in (args.dot_grid, args.secondary_camera, args.checkerboard):
        if not required.is_file():
            raise FileNotFoundError(f"Required metadata TIFF not found: {required}")
    output_dir = args.output_dir
    prepare_output_directory(output_dir, args.overwrite)
    dot_channels, dot_metadata = read_tiff(args.dot_grid)
    checker_channels, checker_metadata = read_tiff(args.checkerboard)
    secondary_channels, secondary_metadata = read_tiff(args.secondary_camera)
    dot_gray, checker_gray = grayscale(dot_channels), grayscale(checker_channels)
    dot_points, dot_scores, dot_metrics = dot_grid_candidates(dot_gray)
    checker_points, checker_scores, checker_metrics = checkerboard_candidates(checker_gray)
    red_components, _, red_metrics = red_component_candidates(secondary_channels)
    dot_csv = output_dir / "dot_grid_feature_candidates.csv"
    checker_csv = output_dir / "checkerboard_feature_candidates.csv"
    red_csv = output_dir / "secondary_red_component_candidates.csv"
    write_feature_csv(dot_csv, dot_points, dot_scores, "dot_center_candidate")
    write_feature_csv(checker_csv, checker_points, checker_scores, "checkerboard_corner_candidate")
    write_component_csv(red_csv, red_components)
    dot_png = output_dir / "dot_grid_detection_overlay.png"
    checker_png = output_dir / "checkerboard_detection_overlay.png"
    red_png = output_dir / "secondary_red_component_overlay.png"
    plot_feature_overlay(dot_gray, dot_points, "Dot-grid response candidates", "dot candidates", dot_png)
    plot_feature_overlay(checker_gray, checker_points, "Checkerboard response candidates", "corner candidates", checker_png)
    plot_red_overlay(secondary_channels, red_components, red_png)
    all_eligible = bool(dot_metrics["fit_eligibility_local_regular_lattice"] and checker_metrics["fit_eligibility_local_regular_lattice"] and red_metrics["fit_eligibility_local_reference_component"])
    summary = {
        "audit_type": "read-only independent fiducial detector-and-fit feasibility audit; not a homography fit, calibration selection, or defect-labeling operation",
        "inputs": {"dot_grid": dot_metadata, "checkerboard": checker_metadata, "secondary_camera": secondary_metadata},
        "dot_grid": dot_metrics,
        "checkerboard": checker_metrics,
        "secondary_red_components": red_metrics,
        "fit_feasibility": {
            "all_local_feature_gates_pass": all_eligible,
            "recommendation": "eligible_for_separately_approved_calibration_fit_design_review" if all_eligible else "hold_calibration_fit; inspect feature overlays and component/local-regularity metrics",
            "scope_boundary": "Feature counts, local spacing, and connected components are not an image-to-machine transform. No homography, rank selection, calibration config edit, machine-origin assertion, or physical candidate-location claim is produced.",
        },
        "storage_policy": "writes three compact candidate CSVs, one JSON summary, and exactly three deterministic detection overlays; no raw TIFF, raw CSV, dense crop/mask/heatmap, target, model, checkpoint, or calibration config is changed",
        "outputs": {
            "dot_candidates_csv": str(dot_csv),
            "checkerboard_candidates_csv": str(checker_csv),
            "red_components_csv": str(red_csv),
            "dot_overlay_png": str(dot_png),
            "checkerboard_overlay_png": str(checker_png),
            "red_overlay_png": str(red_png),
            "summary_json": str(output_dir / "independent_metrology_fiducial_detection_summary.json"),
        },
    }
    summary_path = output_dir / "independent_metrology_fiducial_detection_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Independent fiducial detector-and-fit feasibility audit complete. No raw TIFF/CSV, calibration config, target, model, checkpoint, or dense output was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
