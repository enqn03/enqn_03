#!/usr/bin/env python3
"""Refine independent fiducial candidates in image pixels without calibration fitting.
This read-only follow-up uses the same NIST metadata TIFFs as the V1 detector.
It limits dot/checkerboard candidates to an automatically selected high-density
planar candidate ROI and groups nearby red connected components before ranking.
It measures feature regularity and compactness only. It never computes or writes
a homography, selects a rank, attributes a red cluster to machine origin, changes
calibration_v1.yaml, or accesses the manufacturing model/XCT data.
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
from audit_independent_metrology_fiducials import (
    MAX_FEATURE_ROWS,
    box_mean,
    checkerboard_candidates,
    connected_components,
    dot_grid_candidates,
    grayscale,
    greedy_nms,
    nearest_neighbor_metrics,
    read_tiff,
    red_dominance,
    robust_normalize,
)
MAX_RED_CLUSTER_ROWS = 20
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path)
    parser.add_argument("--secondary-camera", required=True, type=Path)
    parser.add_argument("--checkerboard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def grid_components(mask: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    """Return 8-connected components from a small candidate-density bin grid."""
    if mask.shape != values.shape:
        raise ValueError("Grid mask and values have different shapes.")
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[dict[str, Any]] = []
    for y0, x0 in np.argwhere(mask):
        y_start, x_start = int(y0), int(x0)
        if visited[y_start, x_start]:
            continue
        stack = [(y_start, x_start)]
        visited[y_start, x_start] = True
        members: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            members.append((y, x))
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
        ys, xs = zip(*members, strict=True)
        components.append({
            "y_min": int(min(ys)), "y_max": int(max(ys)), "x_min": int(min(xs)), "x_max": int(max(xs)),
            "cell_count": int(len(members)), "density_sum": float(sum(values[y, x] for y, x in members)),
        })
    return components
def density_roi(points: np.ndarray, image_shape: tuple[int, int], bin_size_px: int = 32, pad_px: int = 64) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Select one dense candidate component in a bin grid; result remains a candidate ROI."""
    if len(points) < 1:
        raise ValueError("Cannot determine candidate ROI from zero points.")
    height, width = image_shape
    x_bins = int(math.ceil(width / bin_size_px))
    y_bins = int(math.ceil(height / bin_size_px))
    clipped_x = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
    clipped_y = np.clip(points[:, 1].astype(np.int64), 0, height - 1)
    density = np.zeros((y_bins, x_bins), dtype=np.int32)
    np.add.at(density, (clipped_y // bin_size_px, clipped_x // bin_size_px), 1)
    maximum = int(density.max())
    threshold = max(2, int(math.ceil(maximum * 0.15)))
    components = grid_components(density >= threshold, density)
    if not components:
        raise RuntimeError("No dense candidate-grid component found for planar ROI.")
    best = max(components, key=lambda item: (float(item["density_sum"]), int(item["cell_count"])))
    x0 = max(0, int(best["x_min"]) * bin_size_px - pad_px)
    y0 = max(0, int(best["y_min"]) * bin_size_px - pad_px)
    x1 = min(width, (int(best["x_max"]) + 1) * bin_size_px + pad_px)
    y1 = min(height, (int(best["y_max"]) + 1) * bin_size_px + pad_px)
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise RuntimeError("Automatically selected planar candidate ROI is too small.")
    return (x0, y0, x1, y1), {
        "method": "candidate-density bin grid; 8-connected high-density component; padded rectangular candidate ROI",
        "bin_size_px": int(bin_size_px),
        "relative_density_threshold": 0.15,
        "absolute_density_threshold": threshold,
        "max_bin_density": maximum,
        "selected_component": best,
        "roi_xyxy_px": [x0, y0, x1, y1],
        "roi_area_fraction": float((x1 - x0) * (y1 - y0) / float(height * width)),
        "important_limit": "Automatic ROI is a detector restriction, not a metrology board pose or physical calibration region.",
    }
def add_offset(points: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    result[:, 0] += x_offset
    result[:, 1] += y_offset
    return result
def dot_response(gray: np.ndarray) -> np.ndarray:
    normalized, _ = robust_normalize(gray)
    return box_mean(normalized, radius=3) - normalized
def checker_response(gray: np.ndarray) -> np.ndarray:
    normalized, _ = robust_normalize(gray)
    gy, gx = np.gradient(normalized)
    a = box_mean(gx * gx, radius=4)
    b = box_mean(gx * gy, radius=4)
    c = box_mean(gy * gy, radius=4)
    return (a * c - b * b) - 0.04 * (a + c) ** 2
def refined_feature_candidates(gray: np.ndarray, roi: tuple[int, int, int, int], kind: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = roi
    crop = gray[y0:y1, x0:x1]
    if kind == "dot":
        response, quantile, min_distance = dot_response(crop), 0.990, 8
        candidate_type = "roi_restricted_dot_center_candidate"
        cv_limit = 0.40
        count_minimum = 200
        detector = "ROI-restricted dark local blob response; greedy NMS min distance=8 px"
    elif kind == "checkerboard":
        response, quantile, min_distance = checker_response(crop), 0.990, 11
        candidate_type = "roi_restricted_checkerboard_corner_candidate"
        cv_limit = 0.45
        count_minimum = 100
        detector = "ROI-restricted Harris-like corner response; greedy NMS min distance=11 px"
    else:
        raise ValueError(f"Unsupported feature kind: {kind}")
    crop_points, scores, nms = greedy_nms(response, quantile=quantile, min_distance_px=min_distance, maximum=MAX_FEATURE_ROWS)
    points = add_offset(crop_points, x0, y0)
    nearest = nearest_neighbor_metrics(points)
    nn_cv = nearest["nearest_neighbor_cv"]
    regularity_pass = bool(len(points) >= count_minimum and nn_cv is not None and float(nn_cv) <= cv_limit)
    return points, scores, {
        "detector": detector,
        "candidate_type": candidate_type,
        "roi_xyxy_px": [x0, y0, x1, y1],
        "fit_eligibility_roi_regular_lattice": regularity_pass,
        "eligibility_rule": f"retained_count>={count_minimum} and nearest_neighbor_cv<={cv_limit}; no calibration fit",
        **nms,
        **nearest,
    }
def union_find_clusters(components: list[dict[str, Any]], max_link_distance_px: float = 120.0) -> list[list[int]]:
    count = len(components)
    parent = list(range(count))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    centers = np.asarray([[float(row["centroid_x_px"]), float(row["centroid_y_px"])] for row in components], dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            if float(np.linalg.norm(centers[left] - centers[right])) <= max_link_distance_px:
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())
def red_component_clusters(channels: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dominance = red_dominance(channels)
    positive = dominance[dominance > 0.0]
    if positive.size == 0:
        raise ValueError("Secondary-camera RGB data have no positive red dominance.")
    threshold = float(np.quantile(positive, 0.999))
    components, component_metrics = connected_components(dominance >= threshold, dominance, maximum=200)
    if not components:
        raise RuntimeError("No red components passed the thresholded connected-component extraction.")
    groups = union_find_clusters(components, max_link_distance_px=120.0)
    clusters: list[dict[str, Any]] = []
    height, width = dominance.shape
    for member_indices in groups:
        members = [components[index] for index in member_indices]
        weights = np.asarray([float(row["integrated_red_dominance"]) for row in members], dtype=np.float64)
        xs = np.asarray([float(row["centroid_x_px"]) for row in members], dtype=np.float64)
        ys = np.asarray([float(row["centroid_y_px"]) for row in members], dtype=np.float64)
        total = float(weights.sum())
        centroid_x, centroid_y = float((xs * weights).sum() / total), float((ys * weights).sum() / total)
        between_variance = float((((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2) * weights).sum() / total)
        within_variance = float(sum(float(row["weighted_spread_px"]) ** 2 * float(row["integrated_red_dominance"]) for row in members) / total)
        clusters.append({
            "component_count": int(len(members)),
            "area_px_sum": int(sum(int(row["area_px"]) for row in members)),
            "integrated_red_dominance_sum": total,
            "peak_red_dominance_max": float(max(float(row["peak_red_dominance"]) for row in members)),
            "centroid_x_px": centroid_x,
            "centroid_y_px": centroid_y,
            "combined_weighted_spread_px": float(math.sqrt(max(between_variance + within_variance, 0.0))),
            "bbox_x0_px": int(min(int(row["bbox_x0_px"]) for row in members)),
            "bbox_y0_px": int(min(int(row["bbox_y0_px"]) for row in members)),
            "bbox_x1_px": int(max(int(row["bbox_x1_px"]) for row in members)),
            "bbox_y1_px": int(max(int(row["bbox_y1_px"]) for row in members)),
            "touches_image_boundary": bool(any(bool(row["touches_image_boundary"]) for row in members)),
            "source_component_ranks": ";".join(str(row["rank_by_integrated_red_dominance"]) for row in members),
        })
    clusters.sort(key=lambda row: (-float(row["integrated_red_dominance_sum"]), -int(row["component_count"]), -int(row["area_px_sum"])))
    for rank, row in enumerate(clusters[:MAX_RED_CLUSTER_ROWS], start=1):
        row["rank_by_clustered_integrated_red_dominance"] = rank
    recorded = clusters[:MAX_RED_CLUSTER_ROWS]
    top = recorded[0]
    compact = bool(not bool(top["touches_image_boundary"]) and float(top["combined_weighted_spread_px"]) <= 100.0 and int(top["component_count"]) >= 2)
    return recorded, {
        "detector": "same q=0.999 red components as V1; agglomerative spatial grouping by centroid link distance<=120 px",
        "red_dominance_quantile": 0.999,
        "red_dominance_threshold": threshold,
        "max_link_distance_px": 120.0,
        "source_component_count": int(component_metrics["component_count_minimum_area"]),
        "recorded_cluster_count": int(len(recorded)),
        "top_cluster": top,
        "fit_eligibility_compact_red_cluster": compact,
        "eligibility_rule": "top clustered component has >=2 source components, is non-boundary, and combined spread<=100 px; not origin attribution",
    }
def write_feature_csv(path: Path, points: np.ndarray, scores: np.ndarray, kind: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["rank_by_response", "feature_type", "raw_x_px", "raw_y_px", "response"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, (point, score) in enumerate(zip(points, scores, strict=True), start=1):
            writer.writerow({"rank_by_response": rank, "feature_type": kind, "raw_x_px": float(point[0]), "raw_y_px": float(point[1]), "response": float(score)})
def write_cluster_csv(path: Path, clusters: list[dict[str, Any]]) -> None:
    fields = [
        "rank_by_clustered_integrated_red_dominance", "component_count", "area_px_sum", "integrated_red_dominance_sum", "peak_red_dominance_max",
        "centroid_x_px", "centroid_y_px", "combined_weighted_spread_px", "bbox_x0_px", "bbox_y0_px", "bbox_x1_px", "bbox_y1_px",
        "touches_image_boundary", "source_component_ranks",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clusters)
def plot_planar_overlay(gray: np.ndarray, points: np.ndarray, roi: tuple[int, int, int, int], title: str, output_path: Path) -> None:
    display_stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    display = gray[::display_stride, ::display_stride]
    x0, y0, x1, y1 = roi
    figure, axis = plt.subplots(figsize=(9, 9), dpi=150)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    if len(points):
        selection = points[::max(1, int(math.ceil(len(points) / 2500)))] / float(display_stride)
        axis.scatter(selection[:, 0], selection[:, 1], c="cyan", s=2.0, alpha=0.75, linewidths=0, label="ROI-restricted candidates")
    rectangle = plt.Rectangle((x0 / display_stride, y0 / display_stride), (x1 - x0) / display_stride, (y1 - y0) / display_stride, fill=False, edgecolor="yellow", linewidth=1.8, label="automatic candidate ROI")
    axis.add_patch(rectangle)
    axis.set_title(f"{title}\nCandidate ROI and image-pixel features only — no calibration fit")
    axis.set_xlabel("display x [pixel]")
    axis.set_ylabel("display y [pixel]")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
def plot_red_cluster_overlay(channels: np.ndarray, clusters: list[dict[str, Any]], output_path: Path) -> None:
    rgb = np.stack([robust_normalize(channel)[0] for channel in channels[:3]], axis=-1)
    stride = max(1, int(math.ceil(max(rgb.shape[:2]) / 1000)))
    display = np.moveaxis(np.moveaxis(rgb, -1, 0)[..., ::stride, ::stride], 0, -1)
    figure, axis = plt.subplots(figsize=(11, 8), dpi=150)
    axis.imshow(display, origin="upper", interpolation="nearest")
    colors = ["cyan", "lime", "yellow", "magenta", "orange"]
    for row, color in zip(clusters[:5], colors, strict=False):
        x0, y0 = float(row["bbox_x0_px"]) / stride, float(row["bbox_y0_px"]) / stride
        width = max((float(row["bbox_x1_px"]) - float(row["bbox_x0_px"])) / stride, 1.0)
        height = max((float(row["bbox_y1_px"]) - float(row["bbox_y0_px"])) / stride, 1.0)
        axis.add_patch(plt.Rectangle((x0, y0), width, height, fill=False, edgecolor=color, linewidth=1.8))
        axis.scatter([float(row["centroid_x_px"]) / stride], [float(row["centroid_y_px"]) / stride], marker="+", c=color, s=110, linewidths=1.8)
        axis.text(x0, max(0.0, y0 - 5.0), f"cluster #{row['rank_by_clustered_integrated_red_dominance']}", color=color, fontsize=9, fontweight="bold")
    axis.set_title("Secondary-camera spatially grouped red clusters\nCandidates only — no machine-origin attribution or calibration fit")
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
    coarse_dot_points, _, _ = dot_grid_candidates(dot_gray)
    coarse_checker_points, _, _ = checkerboard_candidates(checker_gray)
    dot_roi, dot_roi_metrics = density_roi(coarse_dot_points, dot_gray.shape)
    checker_roi, checker_roi_metrics = density_roi(coarse_checker_points, checker_gray.shape)
    dot_points, dot_scores, dot_metrics = refined_feature_candidates(dot_gray, dot_roi, "dot")
    checker_points, checker_scores, checker_metrics = refined_feature_candidates(checker_gray, checker_roi, "checkerboard")
    red_clusters, red_metrics = red_component_clusters(secondary_channels)
    dot_csv = output_dir / "dot_grid_roi_feature_candidates.csv"
    checker_csv = output_dir / "checkerboard_roi_feature_candidates.csv"
    red_csv = output_dir / "secondary_red_component_clusters.csv"
    write_feature_csv(dot_csv, dot_points, dot_scores, "roi_restricted_dot_center_candidate")
    write_feature_csv(checker_csv, checker_points, checker_scores, "roi_restricted_checkerboard_corner_candidate")
    write_cluster_csv(red_csv, red_clusters)
    dot_overlay = output_dir / "dot_grid_roi_refinement_overlay.png"
    checker_overlay = output_dir / "checkerboard_roi_refinement_overlay.png"
    red_overlay = output_dir / "secondary_red_cluster_refinement_overlay.png"
    plot_planar_overlay(dot_gray, dot_points, dot_roi, "Dot-grid refined candidates", dot_overlay)
    plot_planar_overlay(checker_gray, checker_points, checker_roi, "Checkerboard refined candidates", checker_overlay)
    plot_red_cluster_overlay(secondary_channels, red_clusters, red_overlay)
    all_eligible = bool(dot_metrics["fit_eligibility_roi_regular_lattice"] and checker_metrics["fit_eligibility_roi_regular_lattice"] and red_metrics["fit_eligibility_compact_red_cluster"])
    summary = {
        "audit_type": "read-only independent fiducial ROI and red-cluster refinement; not calibration fit, rank selection, origin attribution, or defect labeling",
        "inputs": {"dot_grid": dot_metadata, "checkerboard": checker_metadata, "secondary_camera": secondary_metadata},
        "dot_grid": {"automatic_roi": dot_roi_metrics, "refined_features": dot_metrics},
        "checkerboard": {"automatic_roi": checker_roi_metrics, "refined_features": checker_metrics},
        "secondary_red_clusters": red_metrics,
        "fit_design_feasibility": {
            "all_refined_local_feature_gates_pass": all_eligible,
            "recommendation": "eligible_for_separately_approved_calibration_fit_design_review" if all_eligible else "hold_calibration_fit; inspect refined ROI/cluster overlays and feature regularity before any fit design",
            "scope_boundary": "Refined image-pixel candidates and clusters are not an image-to-machine mapping. No homography, camera intrinsic/extrinsic parameter, calibration rank selection, config update, machine-origin assertion, or physical candidate-coordinate claim is produced.",
        },
        "storage_policy": "writes three compact candidate/cluster CSVs, one JSON summary, and exactly three deterministic refinement overlays; no raw TIFF, raw CSV, target, model, checkpoint, calibration config, dense crop/mask, or heatmap is changed",
        "outputs": {
            "dot_features_csv": str(dot_csv), "checkerboard_features_csv": str(checker_csv), "red_clusters_csv": str(red_csv),
            "dot_overlay_png": str(dot_overlay), "checkerboard_overlay_png": str(checker_overlay), "red_overlay_png": str(red_overlay),
            "summary_json": str(output_dir / "independent_metrology_fiducial_refinement_summary.json"),
        },
    }
    summary_path = output_dir / "independent_metrology_fiducial_refinement_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Independent fiducial ROI/cluster refinement complete. No raw TIFF/CSV, calibration config, target, model, checkpoint, or dense output was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
