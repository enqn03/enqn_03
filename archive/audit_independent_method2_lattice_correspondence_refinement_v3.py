utf-8
#!/usr/bin/env python3
"""Refine DotGrid 2D lattice correspondence before any calibration selection.
This V3 audit reads one immutable layer-camera DotGrid TIFF via the existing
read-only memmap helper. It retains the V1 local-neighbor graph thresholds, but
first enumerates every edge-connected component and seeds 2D propagation from
the largest component, not from one maximum-response point. It then performs
provisional 2D cell propagation and projective reassignment before applying the
*same* 5x5-block held-out residual gate used by the V1 method-#2 audit.
It does not update calibration_v1.yaml, fit a deployed calibration, select an
orientation/rank, access A/B manufacturing images, XCT, targets, models, or
checkpoints, or change camera-primary candidate reporting. Any D-to-C mapping
reported here is a correspondence diagnostic for human review only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import sys
from collections import deque
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
    HOLDOUT_BLOCK_SIZE,
    HOLDOUT_MODULUS,
    Hfit,
    inlier_fit,
    nearest_camera_dot_pitch_px,
    percentile_or_none,
    project,
    rmse,
    subpixel_dark_blob_centers,
)
MAX_NEIGHBORS = 6
AXIS_DIRECTION_COSINE_MIN = 0.92
NEIGHBOR_DISTANCE_MIN_PITCH = 0.45
NEIGHBOR_DISTANCE_MAX_PITCH = 1.75
PROJECTIVE_ASSIGNMENT_MAX_PITCH = 0.45
MAX_REASSIGNMENT_ITERATIONS = 5
MIN_COVERAGE_CELLS = 1200
MIN_AXIS_COVERAGE = 40
HELDOUT_RMSE_MAX_PITCH = 0.25
HELDOUT_P95_MAX_PITCH = 0.50
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-grid", required=True, type=Path, help="Immutable DotGrid_2000x2000.tif read via memmap(mode='r').")
    parser.add_argument("--output-dir", required=True, type=Path, help="New ignored directory for compact CSV/JSON and at most three QC overlays.")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace only an existing output directory after review.")
    return parser.parse_args()
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Review it or use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def pca_coordinates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("Expected at least four raw XY dot candidates for PCA.")
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - center).T))
    basis = vectors[:, np.argsort(values)[::-1]]
    return (points - center) @ basis, center, basis
def nearest_neighbor_graph(points: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    """Build a deterministic local graph and keep only near-horizontal/vertical PCA edges.
    The graph is an image-space correspondence aid. It does not assign machine
    orientation, physical D origin, or a calibration transform.
    """
    points = np.asarray(points, dtype=np.float64)
    pca, center, basis = pca_coordinates(points)
    count = len(points)
    delta = points[:, None, :] - points[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    np.fill_diagonal(squared, np.inf)
    order = np.argsort(squared, axis=1)[:, :MAX_NEIGHBORS]
    nearest = np.sqrt(np.min(squared, axis=1))
    pitch = float(np.median(nearest))
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise RuntimeError("Unable to estimate positive local DotGrid pitch from candidate neighbors.")
    raw_edges: dict[tuple[int, int], dict[str, Any]] = {}
    min_distance = NEIGHBOR_DISTANCE_MIN_PITCH * pitch
    max_distance = NEIGHBOR_DISTANCE_MAX_PITCH * pitch
    for left in range(count):
        for right in order[left]:
            right = int(right)
            if right == left:
                continue
            first, second = (left, right) if left < right else (right, left)
            if (first, second) in raw_edges:
                continue
            vector = points[second] - points[first]
            distance = float(np.linalg.norm(vector))
            if not (min_distance <= distance <= max_distance):
                continue
            pca_vector = pca[second] - pca[first]
            normalized = pca_vector / max(float(np.linalg.norm(pca_vector)), 1.0e-12)
            magnitude0, magnitude1 = abs(float(normalized[0])), abs(float(normalized[1]))
            dominant = max(magnitude0, magnitude1)
            if dominant < AXIS_DIRECTION_COSINE_MIN:
                continue
            axis = 0 if magnitude0 >= magnitude1 else 1
            sign = 1 if float(pca_vector[axis]) > 0.0 else -1
            raw_edges[(first, second)] = {
                "left": first,
                "right": second,
                "axis": axis,
                "sign_left_to_right": sign,
                "distance_px": distance,
                "axis_alignment_cosine": dominant,
            }
    edges = list(raw_edges.values())
    if len(edges) < 8:
        raise RuntimeError("Too few locally consistent DotGrid neighbor edges survived graph filtering.")
    degree = np.zeros(count, dtype=np.int32)
    for edge in edges:
        degree[int(edge["left"])] += 1
        degree[int(edge["right"])] += 1
    metrics = {
        "method": "candidate nearest-neighbor graph filtered by local camera-pitch range and PCA-axis alignment; no machine-axis assignment",
        "candidate_count": int(count),
        "max_considered_neighbors_per_candidate": MAX_NEIGHBORS,
        "estimated_local_camera_dot_pitch_px": pitch,
        "accepted_neighbor_distance_range_px": [min_distance, max_distance],
        "axis_direction_cosine_min": AXIS_DIRECTION_COSINE_MIN,
        "accepted_2d_neighbor_edge_count": int(len(edges)),
        "candidate_degree_median": float(np.median(degree)),
        "candidate_degree_p05": float(np.percentile(degree, 5)),
        "candidate_degree_p95": float(np.percentile(degree, 95)),
        "pca_center_raw_xy_px": [float(center[0]), float(center[1])],
        "pca_basis_columns_raw_xy": basis.tolist(),
    }
    return edges, metrics, pca, degree
def adjacency_from_edges(edges: list[dict[str, Any]], point_count: int) -> list[list[tuple[int, int, int]]]:
    """Create adjacency records `(neighbor, axis, signed_label_step)` for each direction."""
    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in range(point_count)]
    for edge in edges:
        left, right = int(edge["left"]), int(edge["right"])
        axis, sign = int(edge["axis"]), int(edge["sign_left_to_right"])
        adjacency[left].append((right, axis, sign))
        adjacency[right].append((left, axis, -sign))
    for neighbor_list in adjacency:
        neighbor_list.sort(key=lambda item: (item[1], item[2], item[0]))
    return adjacency
def enumerate_graph_components(points: np.ndarray, responses: np.ndarray, edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[int]], list[list[tuple[int, int, int]]]]:
    """Enumerate all edge-connected components and choose one deterministic panel seed.
    The primary sort key is vertex count, because a panel-wide DotGrid component
    must not lose to one bright local blob. Aggregate detector response and the
    minimum raw `(y, x)` provide only deterministic tie-breaks.
    """
    count = len(points)
    adjacency = adjacency_from_edges(edges, count)
    visited = np.zeros(count, dtype=bool)
    records: list[dict[str, Any]] = []
    members_by_rank: list[list[int]] = []
    response_array = np.asarray(responses, dtype=np.float64)
    for start in range(count):
        if visited[start] or not adjacency[start]:
            continue
        members: list[int] = []
        queue: deque[int] = deque([start])
        visited[start] = True
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor, _, _ in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        member_array = np.asarray(members, dtype=np.int64)
        edge_count = int(sum(len(adjacency[index]) for index in members) // 2)
        root = int(member_array[np.argmax(response_array[member_array])])
        min_y, min_x = float(np.min(points[member_array, 1])), float(np.min(points[member_array, 0]))
        records.append({
            "component_rank_by_size": 0,
            "vertex_count": int(len(members)),
            "edge_count": edge_count,
            "aggregate_detector_response": float(response_array[member_array].sum()),
            "maximum_detector_response": float(response_array[member_array].max()),
            "seed_source_candidate_index": root,
            "seed_raw_x_px": float(points[root, 0]),
            "seed_raw_y_px": float(points[root, 1]),
            "bbox_x0_px": float(np.min(points[member_array, 0])),
            "bbox_y0_px": float(np.min(points[member_array, 1])),
            "bbox_x1_px": float(np.max(points[member_array, 0])),
            "bbox_y1_px": float(np.max(points[member_array, 1])),
            "_tie_min_y": min_y,
            "_tie_min_x": min_x,
            "_members": members,
        })
    if not records:
        raise RuntimeError("No edge-connected DotGrid candidate component exists after local graph filtering.")
    records.sort(key=lambda row: (-int(row["vertex_count"]), -float(row["aggregate_detector_response"]), float(row["_tie_min_y"]), float(row["_tie_min_x"])))
    public_records: list[dict[str, Any]] = []
    for rank, row in enumerate(records, start=1):
        row["component_rank_by_size"] = rank
        members_by_rank.append(list(row["_members"]))
        public_records.append({key: value for key, value in row.items() if not key.startswith("_")})
    return public_records, members_by_rank, adjacency
def propagate_provisional_2d_labels(points: np.ndarray, edges: list[dict[str, Any]], component_members: list[int], root: int, adjacency: list[list[tuple[int, int, int]]] | None = None) -> tuple[dict[int, tuple[int, int]], dict[str, Any]]:
    """Propagate image-lattice labels only inside the deterministic largest component."""
    count = len(points)
    if root not in set(component_members):
        raise ValueError("The requested BFS root is not a member of the selected graph component.")
    if adjacency is None:
        adjacency = adjacency_from_edges(edges, count)
    allowed = set(component_members)
    labels: dict[int, tuple[int, int]] = {root: (0, 0)}
    queue: deque[int] = deque([root])
    conflicts = 0
    while queue:
        current = queue.popleft()
        current_label = labels[current]
        for neighbor, axis, sign in adjacency[current]:
            if neighbor not in allowed:
                continue
            proposal = list(current_label)
            proposal[axis] += sign
            proposal_tuple = (int(proposal[0]), int(proposal[1]))
            existing = labels.get(neighbor)
            if existing is None:
                labels[neighbor] = proposal_tuple
                queue.append(neighbor)
            elif existing != proposal_tuple:
                conflicts += 1
    metrics = {
        "method": "BFS label propagation from the maximum-response point inside the largest edge-connected DotGrid graph component; inconsistent cycles are counted and do not overwrite the first deterministic label",
        "seed_selection": "largest vertex-count component, then aggregate detector response, then minimum raw y/x; maximum response used only inside selected component",
        "root_source_candidate_index": root,
        "root_raw_xy_px": [float(points[root, 0]), float(points[root, 1])],
        "selected_component_vertex_count": int(len(component_members)),
        "graph_reachable_candidate_count": int(len(labels)),
        "unlabelled_candidate_count": int(count - len(labels)),
        "cycle_label_conflict_count": int(conflicts),
        "important_limit": "Labels are provisional image-grid coordinates. Their lower-left D origin and machine-axis orientation are not inferred here.",
    }
    return labels, metrics
def best_dense_grid_window(labels: dict[int, tuple[int, int]], grid_size: int = GRID_SIZE) -> tuple[int, int, dict[str, Any]]:
    """Choose a deterministic dense square label window, retaining only a 50x50 image-lattice candidate region."""
    if not labels:
        raise RuntimeError("No graph-propagated labels are available for dense-grid window selection.")
    values = np.asarray(list(labels.values()), dtype=np.int64)
    min_col, max_col = int(values[:, 0].min()), int(values[:, 0].max())
    min_row, max_row = int(values[:, 1].min()), int(values[:, 1].max())
    candidate_cols = range(min_col, max(min_col, max_col - grid_size + 1) + 1)
    candidate_rows = range(min_row, max(min_row, max_row - grid_size + 1) + 1)
    best: tuple[int, int, int] | None = None
    for col_start in candidate_cols:
        for row_start in candidate_rows:
            count = sum(col_start <= col < col_start + grid_size and row_start <= row < row_start + grid_size for col, row in labels.values())
            candidate = (count, -row_start, -col_start)
            if best is None or candidate > best:
                best = candidate
    if best is None or best[0] < 8:
        raise RuntimeError("Dense 50x50 provisional lattice window contains too few graph labels.")
    col_start, row_start = -best[2], -best[1]
    return col_start, row_start, {
        "method": "max-count 50x50 window over graph-propagated provisional 2D image-lattice labels",
        "graph_label_col_range": [min_col, max_col],
        "graph_label_row_range": [min_row, max_row],
        "selected_window_start_col": col_start,
        "selected_window_start_row": row_start,
        "graph_labels_inside_selected_window": int(best[0]),
        "important_limit": "This is an image-lattice window only; it does not select D lower-left origin or machine-axis orientation.",
    }
def select_one_per_cell(points: np.ndarray, responses: np.ndarray, labels: dict[int, tuple[int, int]], col_start: int, row_start: int, source: str) -> list[dict[str, Any]]:
    """Retain one deterministic candidate per provisional 50x50 cell.
    For graph labels, higher response wins. For projective reassignment, the
    caller passes only one proposal per candidate and this function retains the
    candidate with the smallest projective residual, then higher response.
    """
    selected: dict[tuple[int, int], int] = {}
    for index, (col, row) in labels.items():
        local_col, local_row = int(col - col_start), int(row - row_start)
        if not (0 <= local_col < GRID_SIZE and 0 <= local_row < GRID_SIZE):
            continue
        key = (local_col, local_row)
        previous = selected.get(key)
        if previous is None or float(responses[index]) > float(responses[previous]):
            selected[key] = index
    rows: list[dict[str, Any]] = []
    for (col, row), index in sorted(selected.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.append({
            "source_candidate_index": int(index),
            "image_lattice_col_index_0_to_49": int(col),
            "image_lattice_row_index_0_to_49": int(row),
            "raw_x_px": float(points[index, 0]),
            "raw_y_px": float(points[index, 1]),
            "detector_response": float(responses[index]),
            "correspondence_source": source,
        })
    return rows
def arrays_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray([[float(row["image_lattice_col_index_0_to_49"]), float(row["image_lattice_row_index_0_to_49"])] for row in rows], dtype=np.float64)
    raw = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in rows], dtype=np.float64)
    return source, raw
def projective_reassignment(points: np.ndarray, responses: np.ndarray, h_matrix: np.ndarray, camera_pitch_px: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign every dot candidate to nearest projectively predicted 2D cell, with a fixed local-distance bound."""
    cells = np.asarray([[float(col), float(row)] for row in range(GRID_SIZE) for col in range(GRID_SIZE)], dtype=np.float64)
    predicted = project(h_matrix, cells)
    deltas = points[:, None, :] - predicted[None, :, :]
    squared = np.einsum("ijk,ijk->ij", deltas, deltas, optimize=True)
    nearest_cell = np.argmin(squared, axis=1)
    nearest_distance = np.sqrt(squared[np.arange(len(points)), nearest_cell])
    max_distance = PROJECTIVE_ASSIGNMENT_MAX_PITCH * camera_pitch_px
    retained: dict[tuple[int, int], tuple[int, float]] = {}
    accepted = 0
    for index, (flat_cell, distance) in enumerate(zip(nearest_cell.tolist(), nearest_distance.tolist(), strict=True)):
        if not math.isfinite(distance) or float(distance) > max_distance:
            continue
        row, col = divmod(int(flat_cell), GRID_SIZE)
        key = (col, row)
        previous = retained.get(key)
        if previous is None or float(distance) < previous[1] - 1.0e-9 or (abs(float(distance) - previous[1]) <= 1.0e-9 and float(responses[index]) > float(responses[previous[0]])):
            retained[key] = (index, float(distance))
        accepted += 1
    rows: list[dict[str, Any]] = []
    for (col, row), (index, distance) in sorted(retained.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.append({
            "source_candidate_index": int(index),
            "image_lattice_col_index_0_to_49": int(col),
            "image_lattice_row_index_0_to_49": int(row),
            "raw_x_px": float(points[index, 0]),
            "raw_y_px": float(points[index, 1]),
            "detector_response": float(responses[index]),
            "projective_cell_residual_px": float(distance),
            "correspondence_source": "projective_2d_nearest_cell_reassignment",
        })
    return rows, {
        "method": "nearest full 50x50 projective 2D cell prediction; one candidate per cell; closest residual then response tie-break",
        "candidate_assignment_distance_limit_px": max_distance,
        "candidate_assignment_distance_limit_pitch_fraction": PROJECTIVE_ASSIGNMENT_MAX_PITCH,
        "accepted_candidate_proposals_before_cell_deduplication": int(accepted),
        "unique_projectively_reassigned_cell_count": int(len(rows)),
        "projective_assignment_residual_median_px": None if not rows else float(np.median([float(row["projective_cell_residual_px"]) for row in rows])),
        "projective_assignment_residual_p95_px": None if not rows else float(np.percentile([float(row["projective_cell_residual_px"]) for row in rows], 95)),
    }
def refine_correspondence(points: np.ndarray, responses: np.ndarray, edges: list[dict[str, Any]], graph_metrics: dict[str, Any], component_records: list[dict[str, Any]], component_members_by_rank: list[list[int]], adjacency: list[list[tuple[int, int, int]]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Create graph seed cells, iteratively reassign under a projective 2D lattice, and return final rows.
    This function never examines manufacturing data or a machine-coordinate
    transform. It only builds an image-lattice `0..49 x 0..49` correspondence.
    """
    if not component_records or not component_members_by_rank:
        raise RuntimeError("No selected graph component was supplied for correspondence refinement.")
    selected_component = component_records[0]
    selected_members = component_members_by_rank[0]
    selected_root = int(selected_component["seed_source_candidate_index"])
    labels, propagation_metrics = propagate_provisional_2d_labels(points, edges, selected_members, selected_root, adjacency)
    col_start, row_start, window_metrics = best_dense_grid_window(labels)
    graph_rows = select_one_per_cell(points, responses, labels, col_start, row_start, "local_2d_neighbor_graph_seed")
    if len(graph_rows) < 8:
        raise RuntimeError("Too few unique graph-seed cells for projective correspondence initialization.")
    source, raw = arrays_from_rows(graph_rows)
    h_matrix, inliers, _, _ = inlier_fit(source, raw)
    pitch = nearest_camera_dot_pitch_px(raw[inliers])
    if pitch is None:
        raise RuntimeError("Unable to estimate camera-dot pitch from graph-seed inliers.")
    iterations: list[dict[str, Any]] = []
    final_rows = graph_rows
    previous_keys: set[tuple[int, int]] | None = None
    for iteration in range(1, MAX_REASSIGNMENT_ITERATIONS + 1):
        reassigned_rows, reassignment_metrics = projective_reassignment(points, responses, h_matrix, pitch)
        if len(reassigned_rows) < 8:
            raise RuntimeError("Projective 2D reassignment retained fewer than eight cells.")
        source, raw = arrays_from_rows(reassigned_rows)
        h_matrix, inliers, residual, threshold = inlier_fit(source, raw)
        pitch = nearest_camera_dot_pitch_px(raw[inliers])
        if pitch is None:
            raise RuntimeError("Unable to estimate camera-dot pitch after projective reassignment.")
        current_keys = {(int(row["image_lattice_col_index_0_to_49"]), int(row["image_lattice_row_index_0_to_49"])) for row in reassigned_rows}
        iterations.append({
            "iteration": iteration,
            **reassignment_metrics,
            "robust_fit_inlier_count": int(inliers.sum()),
            "robust_fit_inlier_fraction": float(inliers.mean()),
            "robust_fit_rmse_px": rmse(residual[inliers]),
            "robust_fit_p95_residual_px": percentile_or_none(residual[inliers], 95),
            "detected_inlier_camera_dot_pitch_px": pitch,
            "robust_inlier_threshold_px": float(threshold),
            "same_unique_cell_keys_as_prior_iteration": None if previous_keys is None else bool(current_keys == previous_keys),
        })
        final_rows = reassigned_rows
        if previous_keys == current_keys:
            break
        previous_keys = current_keys
    final_source, final_raw = arrays_from_rows(final_rows)
    final_h, final_inliers, final_residual, final_threshold = inlier_fit(final_source, final_raw)
    for row, inlier, residual in zip(final_rows, final_inliers, final_residual, strict=True):
        row["full_refined_fit_inlier"] = bool(inlier)
        row["full_refined_fit_residual_px"] = float(residual)
    unique_cols = len({int(row["image_lattice_col_index_0_to_49"]) for row in final_rows})
    unique_rows = len({int(row["image_lattice_row_index_0_to_49"]) for row in final_rows})
    correspondence_metrics = {
        "method": "local 2D neighbor graph seed followed by iterative full-grid projective nearest-cell reassignment; image-lattice correspondence only",
        "selected_graph_component_rank": int(selected_component["component_rank_by_size"]),
        "selected_graph_component_vertex_count": int(selected_component["vertex_count"]),
        "graph_seed_unique_cell_count": int(len(graph_rows)),
        "final_unique_cell_count": int(len(final_rows)),
        "final_grid_cell_coverage_fraction": float(len(final_rows) / float(GRID_SIZE * GRID_SIZE)),
        "final_unique_image_lattice_column_count": unique_cols,
        "final_unique_image_lattice_row_count": unique_rows,
        "final_full_fit_inlier_count": int(final_inliers.sum()),
        "final_full_fit_inlier_fraction": float(final_inliers.mean()),
        "final_full_fit_rmse_px": rmse(final_residual[final_inliers]),
        "final_full_fit_p95_residual_px": percentile_or_none(final_residual[final_inliers], 95),
        "final_full_fit_inlier_threshold_px": float(final_threshold),
        "reassignment_iterations": iterations,
        "important_limit": "Full-fit residual is diagnostic only. The gate below is based on a transform fit without the held-out blocks.",
    }
    return final_rows, graph_metrics, propagation_metrics, edges, final_h, final_inliers
def holdout_mask(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        col = int(row["image_lattice_col_index_0_to_49"])
        image_row = int(row["image_lattice_row_index_0_to_49"])
        block = (col // HOLDOUT_BLOCK_SIZE + 2 * (image_row // HOLDOUT_BLOCK_SIZE)) % HOLDOUT_MODULUS
        values.append(block == 0)
    return np.asarray(values, dtype=bool)
def strict_heldout_validation(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Fit only non-held-out cells and evaluate fixed held-out 5x5 blocks."""
    source, raw = arrays_from_rows(rows)
    block_test = holdout_mask(rows)
    train = ~block_test
    if int(train.sum()) < 8 or int(block_test.sum()) < 8:
        raise RuntimeError(f"Insufficient train/test cells for fixed block-held-out validation: train={int(train.sum())}, test={int(block_test.sum())}.")
    train_h, train_inliers, train_residual, train_threshold = inlier_fit(source[train], raw[train])
    train_pitch = nearest_camera_dot_pitch_px(raw[train][train_inliers])
    if train_pitch is None:
        raise RuntimeError("Unable to determine a positive camera-dot pitch from train-only correspondence inliers.")
    predicted = project(train_h, source)
    residual = np.linalg.norm(predicted - raw, axis=1)
    heldout_residual = residual[block_test]
    heldout_rmse = rmse(heldout_residual)
    heldout_p95 = percentile_or_none(heldout_residual, 95)
    relative_rmse = None if heldout_rmse is None else float(heldout_rmse / train_pitch)
    coverage = {
        "unique_cells": int(len(rows)),
        "unique_columns": len({int(row["image_lattice_col_index_0_to_49"]) for row in rows}),
        "unique_rows": len({int(row["image_lattice_row_index_0_to_49"]) for row in rows}),
    }
    coverage_pass = bool(coverage["unique_cells"] >= MIN_COVERAGE_CELLS and coverage["unique_columns"] >= MIN_AXIS_COVERAGE and coverage["unique_rows"] >= MIN_AXIS_COVERAGE)
    heldout_pass = bool(relative_rmse is not None and relative_rmse <= HELDOUT_RMSE_MAX_PITCH and heldout_p95 is not None and float(heldout_p95) <= HELDOUT_P95_MAX_PITCH * train_pitch)
    metrics = {
        "heldout_scheme": f"same fixed 5x5 blocks: (col//{HOLDOUT_BLOCK_SIZE} + 2*(row//{HOLDOUT_BLOCK_SIZE})) mod {HOLDOUT_MODULUS} == 0",
        "train_cell_count_before_train_only_robust_fit": int(train.sum()),
        "heldout_cell_count": int(block_test.sum()),
        "train_only_robust_inlier_count": int(train_inliers.sum()),
        "train_only_robust_inlier_fraction": float(train_inliers.mean()),
        "train_only_fit_rmse_px": rmse(train_residual[train_inliers]),
        "train_only_fit_p95_residual_px": percentile_or_none(train_residual[train_inliers], 95),
        "train_only_inlier_threshold_px": float(train_threshold),
        "detected_train_inlier_camera_dot_pitch_px": train_pitch,
        "heldout_block_rmse_px": heldout_rmse,
        "heldout_block_p95_residual_px": heldout_p95,
        "heldout_block_rmse_camera_dot_pitch_fraction": relative_rmse,
        "coverage": coverage,
        "coverage_pass_same_v1_rule": coverage_pass,
        "heldout_residual_pass_same_v1_rule": heldout_pass,
        "all_fixed_gates_pass": bool(coverage_pass and heldout_pass),
        "gate_rules": {
            "coverage": f"unique cells>={MIN_COVERAGE_CELLS}, unique image-lattice rows/columns>={MIN_AXIS_COVERAGE}",
            "heldout_rmse": f"RMSE<={HELDOUT_RMSE_MAX_PITCH} of detected train-inlier camera-dot pitch",
            "heldout_p95": f"p95<={HELDOUT_P95_MAX_PITCH} of detected train-inlier camera-dot pitch",
        },
        "important_limit": "This is an image-pixel correspondence consistency test, not a final physical calibration uncertainty or an automatic deployment decision.",
    }
    return metrics, block_test, predicted, residual
def write_components_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["component_rank_by_size", "vertex_count", "edge_count", "aggregate_detector_response", "maximum_detector_response", "seed_source_candidate_index", "seed_raw_x_px", "seed_raw_y_px", "bbox_x0_px", "bbox_y0_px", "bbox_x1_px", "bbox_y1_px"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
def write_features_csv(path: Path, rows: list[dict[str, Any]], block_test: np.ndarray, predicted: np.ndarray, residual: np.ndarray) -> None:
    fields = [
        "source_candidate_index", "image_lattice_col_index_0_to_49", "image_lattice_row_index_0_to_49", "raw_x_px", "raw_y_px", "detector_response",
        "projective_cell_residual_px", "correspondence_source", "full_refined_fit_inlier", "full_refined_fit_residual_px",
        "heldout_block", "train_only_prediction_x_px", "train_only_prediction_y_px", "train_only_residual_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, test, prediction, distance in zip(rows, block_test, predicted, residual, strict=True):
            result = {field: row.get(field) for field in fields}
            result.update({
                "heldout_block": bool(test),
                "train_only_prediction_x_px": float(prediction[0]),
                "train_only_prediction_y_px": float(prediction[1]),
                "train_only_residual_px": float(distance),
            })
            writer.writerow(result)
def write_edge_csv(path: Path, edges: list[dict[str, Any]]) -> None:
    fields = ["left", "right", "axis", "sign_left_to_right", "distance_px", "axis_alignment_cosine"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edges)
def plot_neighbor_graph(gray: np.ndarray, points: np.ndarray, edges: list[dict[str, Any]], output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    display = gray[::stride, ::stride]
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    selected = edges[::max(1, int(math.ceil(len(edges) / 1000)))]
    for edge in selected:
        left, right = int(edge["left"]), int(edge["right"])
        color = "cyan" if int(edge["axis"]) == 0 else "lime"
        axis.plot([points[left, 0] / stride, points[right, 0] / stride], [points[left, 1] / stride, points[right, 1] / stride], color=color, linewidth=0.25, alpha=0.45)
    axis.scatter(points[::max(1, int(math.ceil(len(points) / 2500))), 0] / stride, points[::max(1, int(math.ceil(len(points) / 2500))), 1] / stride, s=1.5, c="yellow", linewidths=0, label="dot candidates")
    axis.set_title("Method-#2 local 2D neighbor graph\nCyan/lime edges are image-lattice directions only")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def plot_correspondence(gray: np.ndarray, rows: list[dict[str, Any]], block_test: np.ndarray, output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    display = gray[::stride, ::stride]
    points = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in rows], dtype=np.float64)
    inlier = np.asarray([bool(row["full_refined_fit_inlier"]) for row in rows], dtype=bool)
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    train_inlier = inlier & ~block_test
    axis.scatter(points[train_inlier, 0] / stride, points[train_inlier, 1] / stride, s=2.0, c="cyan", linewidths=0, label="full-fit inlier / train block")
    heldout = inlier & block_test
    axis.scatter(points[heldout, 0] / stride, points[heldout, 1] / stride, s=5.0, c="yellow", marker="x", linewidths=0.6, label="full-fit inlier / held-out block")
    rejected = ~inlier
    if int(rejected.sum()):
        axis.scatter(points[rejected, 0] / stride, points[rejected, 1] / stride, s=3.0, c="red", marker="+", linewidths=0.5, label="full-fit robust rejection")
    axis.set_title("Method-#2 refined 2D correspondence cells\nImage lattice only — no rank/orientation selection")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def plot_heldout_residual(gray: np.ndarray, rows: list[dict[str, Any]], block_test: np.ndarray, predicted: np.ndarray, output_path: Path) -> None:
    stride = max(1, int(math.ceil(max(gray.shape) / 1000)))
    display = gray[::stride, ::stride]
    actual = np.asarray([[float(row["raw_x_px"]), float(row["raw_y_px"])] for row in rows], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(9, 9), dpi=160)
    axis.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    test_actual = actual[block_test]
    test_prediction = predicted[block_test]
    axis.scatter(test_actual[:, 0] / stride, test_actual[:, 1] / stride, s=5.0, c="yellow", marker="x", linewidths=0.6, label="held-out actual")
    vectors = (test_prediction - test_actual) / stride
    axis.quiver(test_actual[:, 0] / stride, test_actual[:, 1] / stride, vectors[:, 0], vectors[:, 1], color="magenta", angles="xy", scale_units="xy", scale=1.0, width=0.0015, label="train-only residual")
    axis.set_title("Method-#2 refined correspondence held-out residuals\nSame fixed 5x5 blocks; no calibration selection")
    axis.set_xlabel("display raw camera x [pixel]")
    axis.set_ylabel("display raw camera y [pixel]")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    if not args.dot_grid.is_file():
        raise FileNotFoundError(f"Required immutable DotGrid TIFF not found: {args.dot_grid}")
    prepare_output_directory(args.output_dir, args.overwrite)
    channels, metadata = read_tiff(args.dot_grid)
    gray = grayscale(channels)
    coarse_points, _, _ = dot_grid_candidates(gray)
    roi, roi_metrics = density_roi(coarse_points, gray.shape)
    points, responses, detector_metrics = refined_feature_candidates(gray, roi, "dot")
    subpixel_points, shifts = subpixel_dark_blob_centers(gray, points)
    edges, graph_metrics, _, _ = nearest_neighbor_graph(subpixel_points)
    component_records, component_members_by_rank, adjacency = enumerate_graph_components(subpixel_points, responses, edges)
    features_csv = args.output_dir / "method2_refined_2d_lattice_features.csv"
    edges_csv = args.output_dir / "method2_refined_2d_neighbor_edges.csv"
    components_csv = args.output_dir / "method2_refined_2d_graph_components.csv"
    summary_path = args.output_dir / "independent_method2_lattice_correspondence_refinement_v3_summary.json"
    graph_overlay = args.output_dir / "method2_refined_2d_neighbor_graph_overlay.png"
    correspondence_overlay = args.output_dir / "method2_refined_2d_correspondence_overlay.png"
    residual_overlay = args.output_dir / "method2_refined_2d_heldout_residual_overlay.png"
    write_edge_csv(edges_csv, edges)
    write_components_csv(components_csv, component_records)
    plot_neighbor_graph(gray, subpixel_points, edges, graph_overlay)
    common = {
        "audit_type": "read-only perspective-aware 2D DotGrid lattice-correspondence refinement V3; no deployed calibration fit, config update, rank/orientation selection, or candidate-location change",
        "purpose": "Retain V2 largest-component graph seeding and correct only its final held-out overlay call arity; detector thresholds and fixed V1 held-out gates remain unchanged.",
        "inputs": {"dot_grid": metadata},
        "detector": {
            "automatic_roi": roi_metrics,
            "refined_dot_candidates": detector_metrics,
            "subpixel_center_method": "response-weighted centroid in local dark-blob radius=4; shifts beyond radius are rejected",
            "subpixel_shift_median_px": float(np.median(shifts)),
            "subpixel_shift_p95_px": float(np.percentile(shifts, 95)),
        },
        "local_2d_neighbor_graph": graph_metrics,
        "graph_components": {
            "selection_rule": "largest vertex_count, then aggregate_detector_response, then minimum raw y/x; selected component's maximum-response point is the BFS root",
            "component_count_with_at_least_one_edge": int(len(component_records)),
            "largest_component": component_records[0],
            "second_component": None if len(component_records) < 2 else component_records[1],
        },
        "prohibitions": [
            "Does not write raw TIFF/CSV.",
            "Does not read or edit calibration_v1.yaml or select any calibration rank/orientation.",
            "Does not create, select, or deploy an A-to-C machine calibration transform.",
            "Does not access A/B manufacturing TIFF, registered XCT, weak target/support, model, checkpoint, training, or decoder.",
            "Does not change camera-primary XCT-derived continuous quality candidate reporting.",
        ],
        "outputs": {
            "features_csv": str(features_csv),
            "neighbor_edges_csv": str(edges_csv),
            "graph_components_csv": str(components_csv),
            "summary_json": str(summary_path),
            "neighbor_graph_overlay_png": str(graph_overlay),
            "correspondence_overlay_png": str(correspondence_overlay),
            "heldout_residual_overlay_png": str(residual_overlay),
        },
    }
    try:
        rows, graph_metrics, propagation_metrics, edges, _, _ = refine_correspondence(subpixel_points, responses, edges, graph_metrics, component_records, component_members_by_rank, adjacency)
        validation, block_test, predicted, residual = strict_heldout_validation(rows)
    except Exception as error:
        summary = {
            **common,
            "status": "fail_closed_before_heldout_validation",
            "failure_stage": "largest_component_2d_correspondence_seed_or_reassignment",
            "failure_message": str(error),
            "recommendation": "hold_all_method2_transform_candidates; inspect saved component/edge diagnostics before any detector or correspondence change",
            "storage_policy": "writes only compact graph-components CSV, neighbor-edges CSV, one JSON failure summary, and one neighbor-graph QC overlay; no raw/config/model/target or calibration output is changed",
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        print("Perspective-aware 2D lattice-correspondence refinement V3 stopped fail-closed before held-out validation. No raw TIFF/CSV, calibration config, model, target, checkpoint, or candidate output was modified.")
        return
    write_features_csv(features_csv, rows, block_test, predicted, residual)
    plot_correspondence(gray, rows, block_test, correspondence_overlay)
    plot_heldout_residual(gray, rows, block_test, predicted, residual_overlay)
    summary = {
        **common,
        "status": "completed",
        "provisional_2d_label_propagation": propagation_metrics,
        "refined_2d_correspondence": {
            "method": "largest-component local neighbor graph seed followed by iterative full-grid projective nearest-cell reassignment; image-lattice correspondence only",
            "final_unique_cell_count": int(len(rows)),
            "final_grid_cell_coverage_fraction": float(len(rows) / float(GRID_SIZE * GRID_SIZE)),
            "final_unique_image_lattice_column_count": len({int(row["image_lattice_col_index_0_to_49"]) for row in rows}),
            "final_unique_image_lattice_row_count": len({int(row["image_lattice_row_index_0_to_49"]) for row in rows}),
            "important_limit": "Image-lattice row/column labels are not D-origin, machine-axis, part-ID, or physical-coordinate assignments.",
        },
        "same_v1_fixed_block_heldout_validation": validation,
        "recommendation": "eligible_for_human_review_of_refined_correspondence_only" if validation["all_fixed_gates_pass"] else "hold_all_method2_transform_candidates; refine correspondence/outlier handling before any further transform audit",
        "storage_policy": "writes three compact CSVs, one JSON summary, and exactly three deterministic QC overlays; no dense crop, rectified image, mask, heatmap, target, or model output is persisted",
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Perspective-aware 2D lattice-correspondence refinement V3 complete. No raw TIFF/CSV, calibration config, model, target, checkpoint, or candidate output was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
