utf-8
#!/usr/bin/env python3
"""Audit rank/margin mechanisms behind temporal-difference candidate switches.
For selected held-out endpoints, compare normal causal K=4 history against the
same diagnostic counterfactuals used by the matched stability audit. The audit
uses the unchanged support-independent local-maximum decoder, stores only compact
metrics and display-only overlays, and never alters decoder thresholds or policy.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from ammt_causal_dataset import AMMTCausalStageDataset
from audit_a_only_temporal_difference_stability_v1 import counterfactual_histories, decode
from train_a_only_baseline import choose_device, load_yaml
from train_a_only_temporal_difference_v1 import AOnlyCausalTemporalDifferenceCandidateNet
TOP_K = 5
MATCH_RADIUS_MODEL_PX = 1.0
NEAR_TIE_MARGIN_FRACTION_MAX = 0.05
HIGH_MARGIN_FRACTION_MIN = 0.20
MAX_SELECTED_ENDPOINTS = 3
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 24, 47])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    return parser.parse_args()
def prepare_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output directory already exists: {path}. Review it and choose a new --output-dir; this audit never overwrites it.")
    path.mkdir(parents=True, exist_ok=False)
def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def distance_model(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(math.hypot(float(left["x_model_pixel"]) - float(right["x_model_pixel"]), float(left["y_model_pixel"]) - float(right["y_model_pixel"])))
def distance_raw(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(math.hypot(float(left["x_pixel"]) - float(right["x_pixel"]), float(left["y_pixel"]) - float(right["y_pixel"])))
def score_metrics(candidates: list[dict[str, Any]], spatial_range: float) -> dict[str, float | None]:
    if not candidates:
        return {"top1_score": None, "top2_score": None, "top1_top2_margin": None, "top1_top2_margin_fraction_of_spatial_range": None, "top1_top2_separation_model_px": None, "top1_top2_separation_raw_px": None}
    first = candidates[0]
    if len(candidates) < 2:
        return {"top1_score": float(first["score"]), "top2_score": None, "top1_top2_margin": None, "top1_top2_margin_fraction_of_spatial_range": None, "top1_top2_separation_model_px": None, "top1_top2_separation_raw_px": None}
    second = candidates[1]
    margin = float(first["score"]) - float(second["score"])
    return {
        "top1_score": float(first["score"]),
        "top2_score": float(second["score"]),
        "top1_top2_margin": margin,
        "top1_top2_margin_fraction_of_spatial_range": None if spatial_range <= 0.0 else margin / spatial_range,
        "top1_top2_separation_model_px": distance_model(first, second),
        "top1_top2_separation_raw_px": distance_raw(first, second),
    }
def matched_candidate_rank(reference: dict[str, Any], candidates: list[dict[str, Any]], radius_model_px: float) -> tuple[int | None, float | None]:
    matches = [(int(candidate["rank"]), distance_model(reference, candidate)) for candidate in candidates]
    nearby = [(rank, distance) for rank, distance in matches if distance <= radius_model_px]
    if not nearby:
        return None, min((distance for _, distance in matches), default=None)
    rank, distance = min(nearby, key=lambda item: (item[0], item[1]))
    return rank, distance
def top_k_overlap(reference: list[dict[str, Any]], variant: list[dict[str, Any]], radius_model_px: float) -> int:
    used: set[int] = set()
    count = 0
    for candidate in reference:
        possible = [index for index, other in enumerate(variant) if index not in used and distance_model(candidate, other) <= radius_model_px]
        if possible:
            chosen = min(possible, key=lambda index: distance_model(candidate, variant[index]))
            used.add(chosen)
            count += 1
    return count
def classify_switch(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    baseline_metrics: dict[str, float | None],
    variant_metrics: dict[str, float | None],
    causal_top1_rank_in_variant: int | None,
    top1_displacement_model_px: float | None,
) -> str:
    """Descriptive class only; it must never change the decoder."""
    if baseline["candidate_status"] != "emitted" or variant["candidate_status"] != "emitted":
        return "not_rank_comparable"
    if top1_displacement_model_px is None or top1_displacement_model_px <= MATCH_RADIUS_MODEL_PX:
        return "top1_stable"
    baseline_fraction = baseline_metrics["top1_top2_margin_fraction_of_spatial_range"]
    variant_fraction = variant_metrics["top1_top2_margin_fraction_of_spatial_range"]
    if causal_top1_rank_in_variant is not None and causal_top1_rank_in_variant > 1 and (
        (baseline_fraction is not None and baseline_fraction <= NEAR_TIE_MARGIN_FRACTION_MAX)
        or (variant_fraction is not None and variant_fraction <= NEAR_TIE_MARGIN_FRACTION_MAX)
    ):
        return "near_tie_rank_switch_consistent"
    if (
        causal_top1_rank_in_variant is None
        and baseline_fraction is not None
        and variant_fraction is not None
        and baseline_fraction >= HIGH_MARGIN_FRACTION_MIN
        and variant_fraction >= HIGH_MARGIN_FRACTION_MIN
    ):
        return "high_margin_peak_relocation_consistent"
    return "ambiguous_switch_mechanism"
def plot_overlays(
    maps: list[tuple[str, Tensor]],
    decoded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    endpoint_layer: int,
    output_path: Path,
) -> None:
    values = torch.cat([prediction[0, 0].detach().float().cpu().reshape(-1) for _, prediction in maps])
    low = float(torch.quantile(values, 0.01).item())
    high = float(torch.quantile(values, 0.99).item())
    if high <= low:
        high = low + 1.0e-6
    causal_candidates = decoded["causal_history"][1]
    causal_top1 = causal_candidates[0] if causal_candidates else None
    figure, axes = plt.subplots(1, len(maps), figsize=(4.0 * len(maps), 3.7), dpi=150)
    palette = ["#00e5ff", "#ffeb3b", "#00e676", "#ff80ab", "#ffffff"]
    for axis, (name, prediction) in zip(axes, maps, strict=True):
        image = prediction[0, 0].detach().float().cpu().numpy()
        axis.imshow(image, cmap="magma", origin="upper", interpolation="nearest", vmin=low, vmax=high)
        if causal_top1 is not None:
            axis.scatter([float(causal_top1["x_model_pixel"])], [float(causal_top1["y_model_pixel"])], marker="x", s=40, linewidths=1.4, color="white", label="causal top1")
        for candidate in decoded[name][1]:
            rank = int(candidate["rank"])
            axis.scatter([float(candidate["x_model_pixel"])], [float(candidate["y_model_pixel"])], s=30, edgecolors="black", linewidths=0.5, color=palette[(rank - 1) % len(palette)])
            axis.text(float(candidate["x_model_pixel"]) + 2.0, float(candidate["y_model_pixel"]) - 2.0, str(rank), color="white", fontsize=6, weight="bold")
        axis.set_title(name.replace("_", "\n"), fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Temporal-difference top-K margin/rank diagnostic: endpoint z={endpoint_layer}\n"
        "white x = causal top1; colored markers/numbers = variant top-K rank; display-only",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    if not 1 <= len(args.indices) <= MAX_SELECTED_ENDPOINTS or len(set(args.indices)) != len(args.indices):
        raise ValueError(f"--indices requires 1..{MAX_SELECTED_ENDPOINTS} unique values.")
    if list(args.indices) != sorted(args.indices):
        raise ValueError("--indices must be ascending for reproducible artifact order.")
    for path, label in ((args.config, "config"), (args.checkpoint, "checkpoint"), (args.tiff_a, "A-stage TIFF"), (args.manifest, "causal manifest"), (args.normalization_config, "normalization config")):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    config = load_yaml(args.config)
    data = config["data"]
    model_config = config["model"]
    evaluation = config["evaluation"]
    training = config["training"]
    if model_config["name"] != "a_only_causal_temporal_difference_candidate_net":
        raise ValueError("Audit requires the temporal-difference experiment config.")
    if int(data["sequence_length_k"]) != 4 or data["stage"] != "A" or int(data["input_channels"]) != 6:
        raise ValueError("Audit requires fixed A-only six-channel K=4 input.")
    if bool(evaluation.get("provisional_part_geometry_gate", {}).get("enabled", False)):
        raise ValueError("This raw-camera audit requires geometry gate disabled.")
    if int(evaluation["top_k_candidates_per_endpoint"]) != TOP_K:
        raise ValueError(f"Audit requires fixed top-K={TOP_K} comparison.")
    prepare_output_directory(args.output_dir)
    device = choose_device(args.device or str(training["device"]))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = AOnlyCausalTemporalDifferenceCandidateNet(input_channels=6, base_channels=int(model_config["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = AMMTCausalStageDataset(stage="A", tiff_path=args.tiff_a, manifest_path=args.manifest, normalization_config_path=args.normalization_config, split=args.split, resize_hw=tuple(int(value) for value in data["model_resolution"]))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    qc_paths: list[str] = []
    with torch.no_grad():
        for dataset_index in args.indices:
            if not 0 <= dataset_index < len(dataset):
                raise IndexError(f"--indices value {dataset_index} outside split={args.split} range 0..{len(dataset)-1}")
            sample = dataset[dataset_index]
            history = sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)
            maps: list[tuple[str, Tensor]] = []
            decoded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
            for name, variant_history in counterfactual_histories(history):
                prediction = model(variant_history).cpu()
                status, candidates = decode(prediction, sample, evaluation)
                maps.append((name, prediction))
                decoded[name] = (status, candidates)
            baseline_status, baseline_candidates = decoded["causal_history"]
            baseline_metrics = score_metrics(baseline_candidates, float(baseline_status["spatial_range"]))
            baseline_top1 = baseline_candidates[0] if baseline_candidates else None
            classification_counts: dict[str, int] = {}
            for name, _ in maps:
                status, candidates = decoded[name]
                metrics = score_metrics(candidates, float(status["spatial_range"]))
                variant_top1 = candidates[0] if candidates else None
                top1_model_distance = None if baseline_top1 is None or variant_top1 is None else distance_model(baseline_top1, variant_top1)
                top1_raw_distance = None if baseline_top1 is None or variant_top1 is None else distance_raw(baseline_top1, variant_top1)
                causal_rank, causal_min_distance = (None, None) if baseline_top1 is None else matched_candidate_rank(baseline_top1, candidates, MATCH_RADIUS_MODEL_PX)
                overlap = top_k_overlap(baseline_candidates, candidates, MATCH_RADIUS_MODEL_PX)
                mechanism = classify_switch(baseline_status, status, baseline_metrics, metrics, causal_rank, top1_model_distance)
                classification_counts[mechanism] = classification_counts.get(mechanism, 0) + 1
                rows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "sample_id": str(sample["metadata"]["sample_id"]),
                        "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                        "history_layer_z": ";".join(str(int(value)) for value in sample["history_layer_z"].tolist()),
                        "variant": name,
                        "candidate_status": str(status["candidate_status"]),
                        "candidate_count": len(candidates),
                        "top_k": TOP_K,
                        "spatial_range": float(status["spatial_range"]),
                        "top1_score": metrics["top1_score"],
                        "top2_score": metrics["top2_score"],
                        "top1_top2_margin": metrics["top1_top2_margin"],
                        "top1_top2_margin_fraction_of_spatial_range": metrics["top1_top2_margin_fraction_of_spatial_range"],
                        "top1_top2_separation_model_px": metrics["top1_top2_separation_model_px"],
                        "top1_top2_separation_raw_px": metrics["top1_top2_separation_raw_px"],
                        "causal_top1_rank_in_variant_top_k": causal_rank,
                        "causal_top1_min_distance_to_variant_top_k_model_px": causal_min_distance,
                        "top_k_overlap_count_with_causal": overlap,
                        "top1_displacement_model_px_vs_causal": top1_model_distance,
                        "top1_displacement_raw_px_vs_causal": top1_raw_distance,
                        "match_radius_model_px": MATCH_RADIUS_MODEL_PX,
                        "near_tie_margin_fraction_max": NEAR_TIE_MARGIN_FRACTION_MAX,
                        "high_margin_fraction_min": HIGH_MARGIN_FRACTION_MIN,
                        "switch_mechanism_class": mechanism,
                    }
                )
            summaries.append(
                {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(sample["metadata"]["sample_id"]),
                    "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                    "class_counts_including_causal": classification_counts,
                }
            )
            qc_path = args.output_dir / f"temporal_difference_candidate_margin_endpoint_z{int(sample['endpoint_layer_z']):03d}.png"
            plot_overlays(maps, decoded, int(sample["endpoint_layer_z"]), qc_path)
            qc_paths.append(str(qc_path))
    csv_path = args.output_dir / "a_only_temporal_difference_candidate_margin_by_variant.csv"
    summary_path = args.output_dir / "a_only_temporal_difference_candidate_margin_summary.json"
    write_csv(csv_path, rows)
    noncausal_rows = [row for row in rows if row["variant"] != "causal_history"]
    aggregate_counts: dict[str, int] = {}
    for row in noncausal_rows:
        name = str(row["switch_mechanism_class"])
        aggregate_counts[name] = aggregate_counts.get(name, 0) + 1
    summary = {
        "audit_type": "read-only temporal-difference candidate top-K margin/rank robustness audit; not training or decoder-policy change",
        "purpose": "Separate near-tie rank switching from high-margin candidate peak relocation under prescribed diagnostic causal-history counterfactuals.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "selected_dataset_indices": [int(value) for value in args.indices],
        "device": str(device),
        "fixed_descriptive_thresholds": {
            "top_k": TOP_K,
            "spatial_match_radius_model_px": MATCH_RADIUS_MODEL_PX,
            "near_tie_top1_top2_margin_fraction_max": NEAR_TIE_MARGIN_FRACTION_MAX,
            "high_margin_top1_top2_margin_fraction_min": HIGH_MARGIN_FRACTION_MIN,
            "rule": "These labels describe counterfactual rank/margin evidence only. They never change decoder thresholds, add withholding, choose a model, or authorize physical candidate interpretation.",
        },
        "aggregate_counterfactual_switch_mechanism_counts": aggregate_counts,
        "endpoint_summaries": summaries,
        "interpretation": "A near-tie class suggests rank competition; a high-margin relocation class suggests stronger competing peak movement; ambiguous retains both possibilities. Counterfactual substitution is not observed machine motion.",
        "score_semantics": "sigmoid-scaled XCT-derived continuous quality candidate; response direction unresolved; not a defect/anomaly probability",
        "prohibitions": [
            "Does not train, modify, or save a checkpoint, optimizer, config, manifest, raw TIFF, registered XCT CSV, target, support, calibration artifact, or decoder policy.",
            "Uses only A-stage TIFF through read-only memmap and does not open registered XCT/weak response/support.",
            "Does not apply machine/part geometry or select calibration rank/orientation.",
            "Persists compact metrics plus three display-only top-K overlay PNGs only; no dense prediction arrays are saved.",
        ],
        "outputs": {"by_variant_csv": str(csv_path), "summary_json": str(summary_path), "endpoint_qc_pngs": qc_paths},
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Temporal-difference candidate-margin audit complete. No raw data, XCT/weak target, checkpoint, config, calibration, or decoder policy was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
