utf-8
#!/usr/bin/env python3
"""Read-only counterfactual stability audit for a temporal-difference checkpoint.
This script evaluates a saved temporal-difference model at selected endpoints,
replacing prior causal frames with endpoint frames only for diagnostic variants.
It uses no XCT support or geometry during decoding and stores compact metrics plus
three display-only QC PNGs, never dense prediction arrays.
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
import torch
from torch import Tensor
from ammt_causal_dataset import AMMTCausalStageDataset
from train_a_only_baseline import choose_device, load_yaml, local_maximum_candidates
from train_a_only_temporal_difference_v1 import AOnlyCausalTemporalDifferenceCandidateNet
MAP_CHANGE_MAE_MIN = 1.0e-4
RAW_COORDINATE_STABILITY_MAX_MODEL_PIXEL = 1.0
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
        raise ValueError("Cannot write an empty per-variant CSV.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def map_pearson(left: Tensor, right: Tensor) -> float | None:
    x = left.detach().float().reshape(-1).cpu()
    y = right.detach().float().reshape(-1).cpu()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denominator.item()) <= 0.0:
        return None
    return float(torch.dot(x_centered, y_centered).item() / denominator.item())
def counterfactual_histories(history: Tensor) -> list[tuple[str, Tensor]]:
    if history.ndim != 5 or history.shape[0] != 1:
        raise ValueError(f"Expected one [1,K,C,H,W] history, got {tuple(history.shape)}")
    steps = int(history.shape[1])
    if steps != 4:
        raise ValueError("This matched audit requires the fixed K=4 temporal-difference experiment.")
    endpoint = history[:, -1:].clone()
    variants: list[tuple[str, Tensor]] = [("causal_history", history.clone())]
    variants.append(("endpoint_repeated_history", endpoint.repeat(1, steps, 1, 1, 1)))
    for time_index in range(steps - 1):
        variant = history.clone()
        variant[:, time_index] = endpoint[:, 0]
        variants.append((f"prior_t{time_index}_replaced_with_endpoint", variant))
    return variants
def decode(prediction: Tensor, sample: dict[str, Any], evaluation: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = local_maximum_candidates(
        prediction,
        sample,
        top_k=int(evaluation["top_k_candidates_per_endpoint"]),
        kernel_size=int(evaluation["local_maximum_kernel_size"]),
        plateau_range_atol=float(evaluation["spatial_plateau_range_atol"]),
        top_score_tie_atol=float(evaluation["top_score_tie_atol"]),
        top_score_tie_fraction_max=float(evaluation["top_score_tie_fraction_max"]),
        previous_prediction=None,
        temporal_map_mae_atol=float(evaluation["temporal_map_mae_atol"]),
        temporal_map_max_abs_atol=float(evaluation["temporal_map_max_abs_atol"]),
        geometry_gate=None,
    )
    return result, result.pop("candidates")
def first_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return candidates[0] if candidates else None
def raw_displacement(reference: dict[str, Any] | None, variant: dict[str, Any] | None) -> float | None:
    if reference is None or variant is None:
        return None
    return float(math.hypot(float(reference["x_pixel"]) - float(variant["x_pixel"]), float(reference["y_pixel"]) - float(variant["y_pixel"])))
def plot_qc(maps: list[tuple[str, Tensor]], endpoint_layer: int, output_path: Path) -> None:
    values = torch.cat([prediction[0, 0].detach().float().cpu().reshape(-1) for _, prediction in maps])
    low = float(torch.quantile(values, 0.01).item())
    high = float(torch.quantile(values, 0.99).item())
    if high <= low:
        high = low + 1.0e-6
    figure, axes = plt.subplots(1, len(maps), figsize=(3.3 * len(maps), 3.2), dpi=150)
    for axis, (name, prediction) in zip(axes, maps, strict=True):
        axis.imshow(prediction[0, 0].detach().float().cpu().numpy(), cmap="magma", origin="upper", interpolation="nearest", vmin=low, vmax=high)
        axis.set_title(name.replace("_", "\n"), fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Temporal-difference counterfactual candidate-map QC: endpoint z={endpoint_layer}\n"
        "Display-only compact artifact; no dense prediction array persisted",
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
    prepare_output_directory(args.output_dir)
    device = choose_device(args.device or str(training["device"]))
    model = AOnlyCausalTemporalDifferenceCandidateNet(input_channels=6, base_channels=int(model_config["base_channels"])).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = AMMTCausalStageDataset(
        stage="A", tiff_path=args.tiff_a, manifest_path=args.manifest,
        normalization_config_path=args.normalization_config, split=args.split,
        resize_hw=tuple(int(value) for value in data["model_resolution"]),
    )
    rows: list[dict[str, Any]] = []
    endpoint_summaries: list[dict[str, Any]] = []
    qc_paths: list[str] = []
    with torch.no_grad():
        for dataset_index in args.indices:
            if not 0 <= dataset_index < len(dataset):
                raise IndexError(f"--indices value {dataset_index} outside split={args.split} range 0..{len(dataset)-1}")
            sample = dataset[dataset_index]
            history = sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)
            variants = counterfactual_histories(history)
            predictions: list[tuple[str, Tensor]] = []
            decoded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
            for name, variant_history in variants:
                prediction = model(variant_history).cpu()
                status, candidates = decode(prediction, sample, evaluation)
                predictions.append((name, prediction))
                decoded[name] = (status, candidates)
            baseline_prediction = predictions[0][1]
            baseline_candidates = decoded["causal_history"][1]
            baseline_first = first_candidate(baseline_candidates)
            metadata = sample["metadata"]
            raw_scale = max(float(metadata["raw_pixels_per_output_pixel_x"]), float(metadata["raw_pixels_per_output_pixel_y"]))
            endpoint_rows: list[dict[str, Any]] = []
            for name, prediction in predictions:
                status, candidates = decoded[name]
                variant_first = first_candidate(candidates)
                map_difference = (prediction - baseline_prediction).abs()
                displacement = raw_displacement(baseline_first, variant_first)
                row = {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(metadata["sample_id"]),
                    "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                    "history_layer_z": ";".join(str(int(value)) for value in sample["history_layer_z"].tolist()),
                    "variant": name,
                    "map_mae_vs_causal": float(map_difference.mean().item()),
                    "map_max_abs_vs_causal": float(map_difference.max().item()),
                    "map_pearson_vs_causal": map_pearson(prediction, baseline_prediction),
                    "map_change_mae_min": MAP_CHANGE_MAE_MIN,
                    "material_map_change_vs_causal": bool(float(map_difference.mean().item()) >= MAP_CHANGE_MAE_MIN),
                    "raw_coordinate_stability_max_px": RAW_COORDINATE_STABILITY_MAX_MODEL_PIXEL * raw_scale,
                    "raw_top_candidate_displacement_px_vs_causal": displacement,
                    "raw_top_candidate_coordinate_stable_vs_causal": None if displacement is None else bool(displacement <= RAW_COORDINATE_STABILITY_MAX_MODEL_PIXEL * raw_scale),
                    "candidate_status": str(status["candidate_status"]),
                    "candidate_count": len(candidates),
                    "top_score_tie_pixel_count": int(status["top_score_tie_pixel_count"]),
                    "top_score_tie_fraction": float(status["top_score_tie_fraction"]),
                    "spatial_range": float(status["spatial_range"]),
                    "top_candidate_x_pixel": None if variant_first is None else float(variant_first["x_pixel"]),
                    "top_candidate_y_pixel": None if variant_first is None else float(variant_first["y_pixel"]),
                    "top_candidate_score": None if variant_first is None else float(variant_first["score"]),
                }
                rows.append(row)
                endpoint_rows.append(row)
            counterfactuals = [row for row in endpoint_rows if row["variant"] != "causal_history"]
            repeated = next(row for row in counterfactuals if row["variant"] == "endpoint_repeated_history")
            priors = [row for row in counterfactuals if row["variant"].startswith("prior_t")]
            endpoint_summaries.append(
                {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(metadata["sample_id"]),
                    "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                    "history_layer_z": [int(value) for value in sample["history_layer_z"].tolist()],
                    "endpoint_repeated_material_map_change": bool(repeated["material_map_change_vs_causal"]),
                    "any_individual_prior_material_map_change": bool(any(bool(row["material_map_change_vs_causal"]) for row in priors)),
                    "history_contribution_supported_by_map_change": bool(repeated["material_map_change_vs_causal"] and any(bool(row["material_map_change_vs_causal"]) for row in priors)),
                    "all_counterfactual_top_coordinates_stable_within_one_model_pixel": bool(all(row["raw_top_candidate_coordinate_stable_vs_causal"] is True for row in counterfactuals)),
                    "baseline_decoder_status": str(decoded["causal_history"][0]["candidate_status"]),
                    "baseline_candidate_count": len(baseline_candidates),
                }
            )
            qc_path = args.output_dir / f"temporal_difference_stability_endpoint_z{int(sample['endpoint_layer_z']):03d}.png"
            plot_qc(predictions, int(sample["endpoint_layer_z"]), qc_path)
            qc_paths.append(str(qc_path))
    csv_path = args.output_dir / "a_only_temporal_difference_stability_by_variant.csv"
    summary_path = args.output_dir / "a_only_temporal_difference_stability_summary.json"
    write_csv(csv_path, rows)
    supported_count = sum(bool(row["history_contribution_supported_by_map_change"]) for row in endpoint_summaries)
    stable_count = sum(bool(row["all_counterfactual_top_coordinates_stable_within_one_model_pixel"]) for row in endpoint_summaries)
    summary = {
        "audit_type": "read-only temporal-difference causal-history counterfactual candidate stability audit; not training or defect classification",
        "purpose": "Test whether explicit endpoint-minus-mean-prior fusion responds to causal preceding A frames in raw-camera/model space, without XCT support or calibration geometry.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "selected_dataset_indices": [int(value) for value in args.indices],
        "device": str(device),
        "counterfactual_policy": {
            "baseline": "normal causal K=4 history",
            "endpoint_repeated": "all K time positions replaced by the endpoint A frame; diagnostic-only",
            "prior_replacements": "one preceding frame at a time replaced by endpoint A frame; diagnostic-only",
            "decoder_policy": "same plateau/tie/local-maximum safety gates; no XCT support, no provisional geometry, no nonconsecutive temporal-map comparison",
            "map_change_mae_min": MAP_CHANGE_MAE_MIN,
            "raw_coordinate_stability_max_model_pixels": RAW_COORDINATE_STABILITY_MAX_MODEL_PIXEL,
        },
        "endpoint_summaries": endpoint_summaries,
        "aggregate": {
            "selected_endpoint_count": len(endpoint_summaries),
            "history_contribution_supported_endpoint_count": supported_count,
            "all_counterfactual_top_coordinates_stable_endpoint_count": stable_count,
            "interpretation": "Material map sensitivity and raw top-coordinate stability are separate. Neither confirms defect meaning, target direction, calibration, physical location, or deployment readiness.",
        },
        "score_semantics": "sigmoid-scaled XCT-derived continuous quality candidate; direction unresolved; not a defect/anomaly probability",
        "prohibitions": [
            "Does not train, modify, or save a checkpoint, optimizer, config, manifest, raw TIFF, registered XCT CSV, target, or calibration artifact.",
            "Uses only A-stage TIFF through read-only memmap and does not open weak response/support or registered XCT data.",
            "Does not use machine/part geometry, select calibration rank/orientation, or alter any calibration hold.",
            "Persists compact metrics plus three display-only deterministic QC PNGs only; no dense prediction arrays are saved.",
        ],
        "outputs": {"by_variant_csv": str(csv_path), "summary_json": str(summary_path), "endpoint_qc_pngs": qc_paths},
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Temporal-difference stability audit complete. No raw data, XCT/weak target, checkpoint, config, calibration, or candidate policy was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
