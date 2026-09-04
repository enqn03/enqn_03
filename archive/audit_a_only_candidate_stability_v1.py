utf-8
#!/usr/bin/env python3
"""Read-only causal-history contribution audit for a saved A-only checkpoint.
For selected held-out A-stage endpoints, this audit compares the normal causal
K=4 history with endpoint-repeated and one-prior-frame-replaced counterfactual
histories. It assesses map and raw-camera candidate stability only; it neither
trains nor modifies a checkpoint, target, calibration, or decoder policy.
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
import torch
from torch import Tensor
from ammt_causal_dataset import AMMTCausalStageDataset
from train_a_only_baseline import (
    AOnlyCausalCandidateNet,
    choose_device,
    local_maximum_candidates,
    load_yaml,
)
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
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def map_pearson(left: Tensor, right: Tensor) -> float | None:
    x = left.detach().float().flatten().cpu()
    y = right.detach().float().flatten().cpu()
    if x.numel() < 2 or y.numel() < 2:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denominator.item()) <= 0.0:
        return None
    return float(torch.dot(x_centered, y_centered).item() / denominator.item())
def counterfactual_histories(history: Tensor) -> list[tuple[str, Tensor]]:
    if tuple(history.shape[:2]) != (1, history.shape[1]):
        raise ValueError(f"Expected history [1,K,C,H,W], got {tuple(history.shape)}")
    steps = int(history.shape[1])
    if steps < 2:
        raise ValueError("Counterfactual audit requires K>=2.")
    endpoint = history[:, -1:].clone()
    variants: list[tuple[str, Tensor]] = [("causal_history", history.clone())]
    variants.append(("endpoint_repeated_history", endpoint.repeat(1, steps, 1, 1, 1)))
    for time_index in range(steps - 1):
        variant = history.clone()
        variant[:, time_index] = endpoint[:, 0]
        variants.append((f"prior_t{time_index}_replaced_with_endpoint", variant))
    return variants
def decoded_prediction(
    prediction: Tensor,
    sample: dict[str, Any],
    evaluation: dict[str, Any],
    previous_prediction: Tensor | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = local_maximum_candidates(
        prediction,
        sample,
        top_k=int(evaluation["top_k_candidates_per_endpoint"]),
        kernel_size=int(evaluation["local_maximum_kernel_size"]),
        plateau_range_atol=float(evaluation["spatial_plateau_range_atol"]),
        top_score_tie_atol=float(evaluation["top_score_tie_atol"]),
        top_score_tie_fraction_max=float(evaluation["top_score_tie_fraction_max"]),
        previous_prediction=previous_prediction,
        temporal_map_mae_atol=float(evaluation["temporal_map_mae_atol"]),
        temporal_map_max_abs_atol=float(evaluation["temporal_map_max_abs_atol"]),
        geometry_gate=None,
    )
    candidates = result.pop("candidates")
    return result, candidates
def top_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return None if not candidates else candidates[0]
def raw_displacement(reference: dict[str, Any] | None, candidate: dict[str, Any] | None) -> float | None:
    if reference is None or candidate is None:
        return None
    return float(math.hypot(float(reference["x_pixel"]) - float(candidate["x_pixel"]), float(reference["y_pixel"]) - float(candidate["y_pixel"])))
def model_pixel_raw_scale(sample: dict[str, Any]) -> float:
    metadata = sample["metadata"]
    return max(float(metadata["raw_pixels_per_output_pixel_x"]), float(metadata["raw_pixels_per_output_pixel_y"]))
def compact_decoder_fields(status: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    first = top_candidate(candidates)
    return {
        "candidate_status": str(status["candidate_status"]),
        "candidate_count": int(len(candidates)),
        "top_score_tie_pixel_count": int(status["top_score_tie_pixel_count"]),
        "top_score_tie_fraction": float(status["top_score_tie_fraction"]),
        "spatial_range": float(status["spatial_range"]),
        "top_candidate_x_pixel": None if first is None else float(first["x_pixel"]),
        "top_candidate_y_pixel": None if first is None else float(first["y_pixel"]),
        "top_candidate_score": None if first is None else float(first["score"]),
    }
def plot_endpoint_qc(
    maps: list[tuple[str, Tensor]],
    endpoint_layer: int,
    output_path: Path,
) -> None:
    values = torch.cat([prediction[0, 0].detach().float().cpu().reshape(-1) for _, prediction in maps])
    lower = float(torch.quantile(values, 0.01).item())
    upper = float(torch.quantile(values, 0.99).item())
    if upper <= lower:
        upper = lower + 1.0e-6
    figure, axes = plt.subplots(1, len(maps), figsize=(3.3 * len(maps), 3.2), dpi=150)
    if len(maps) == 1:
        axes = [axes]
    for axis, (name, prediction) in zip(axes, maps, strict=True):
        image = prediction[0, 0].detach().float().cpu().numpy()
        axis.imshow(image, cmap="magma", origin="upper", vmin=lower, vmax=upper, interpolation="nearest")
        axis.set_title(name.replace("_", "\n"), fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"A-only counterfactual candidate-map QC: endpoint z={endpoint_layer}\nDisplay-only compact artifact; no dense prediction array persisted", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    if len(args.indices) < 1 or len(args.indices) > MAX_SELECTED_ENDPOINTS:
        raise ValueError(f"--indices requires 1..{MAX_SELECTED_ENDPOINTS} unique dataset indices.")
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("--indices must not contain duplicates.")
    if list(args.indices) != sorted(args.indices):
        raise ValueError("--indices must be ascending so baseline temporal safety comparisons are well-defined.")
    for path, label in ((args.config, "config"), (args.checkpoint, "checkpoint"), (args.tiff_a, "A-stage TIFF"), (args.manifest, "causal manifest"), (args.normalization_config, "normalization config")):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    prepare_output_directory(args.output_dir)
    config = load_yaml(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    if data_config.get("stage") != "A" or int(data_config.get("input_channels", -1)) != 6:
        raise ValueError("Audit requires the six-channel A-only config.")
    if bool(evaluation_config.get("provisional_part_geometry_gate", {}).get("enabled", False)):
        raise ValueError("Use the non-geometry C32 residual config: this raw-camera audit must not decode through provisional geometry.")
    device = choose_device(args.device or str(training_config["device"]))
    model = AOnlyCausalCandidateNet(
        input_channels=int(data_config["input_channels"]),
        base_channels=int(model_config["base_channels"]),
        temporal_kernel_size=int(model_config["temporal_kernel_size"]),
        use_endpoint_feature_residual=bool(model_config.get("use_endpoint_feature_residual", False)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = AMMTCausalStageDataset(
        stage="A",
        tiff_path=args.tiff_a,
        manifest_path=args.manifest,
        normalization_config_path=args.normalization_config,
        split=args.split,
        resize_hw=tuple(int(value) for value in data_config["model_resolution"]),
    )
    variant_rows: list[dict[str, Any]] = []
    endpoint_summaries: list[dict[str, Any]] = []
    qc_paths: list[str] = []
    with torch.no_grad():
        for dataset_index in args.indices:
            if not 0 <= dataset_index < len(dataset):
                raise IndexError(f"--indices value {dataset_index} is outside split={args.split} range 0..{len(dataset)-1}")
            sample = dataset[dataset_index]
            history = sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)
            variants = counterfactual_histories(history)
            predictions: list[tuple[str, Tensor]] = []
            decoded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
            for name, variant_history in variants:
                prediction = model(variant_history).cpu()
                predictions.append((name, prediction))
                status, candidates = decoded_prediction(
                    prediction,
                    sample,
                    evaluation_config,
                    None,
                )
                decoded[name] = (status, candidates)
            baseline_name, baseline_prediction = predictions[0]
            baseline_status, baseline_candidates = decoded[baseline_name]
            baseline_top = top_candidate(baseline_candidates)
            raw_scale = model_pixel_raw_scale(sample)
            endpoint_rows: list[dict[str, Any]] = []
            for name, prediction in predictions:
                status, candidates = decoded[name]
                candidate = top_candidate(candidates)
                relative_mae = float((prediction - baseline_prediction).abs().mean().item())
                relative_max_abs = float((prediction - baseline_prediction).abs().max().item())
                displacement = raw_displacement(baseline_top, candidate)
                coordinate_stable = None if displacement is None else bool(displacement <= RAW_COORDINATE_STABILITY_MAX_MODEL_PIXEL * raw_scale)
                row = {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(sample["metadata"]["sample_id"]),
                    "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                    "history_layer_z": ";".join(str(int(value)) for value in sample["history_layer_z"].tolist()),
                    "variant": name,
                    "map_mae_vs_causal": relative_mae,
                    "map_max_abs_vs_causal": relative_max_abs,
                    "map_pearson_vs_causal": map_pearson(prediction, baseline_prediction),
                    "map_change_mae_min": MAP_CHANGE_MAE_MIN,
                    "material_map_change_vs_causal": bool(relative_mae >= MAP_CHANGE_MAE_MIN),
                    "raw_coordinate_stability_max_px": RAW_COORDINATE_STABILITY_MAX_MODEL_PIXEL * raw_scale,
                    "raw_top_candidate_displacement_px_vs_causal": displacement,
                    "raw_top_candidate_coordinate_stable_vs_causal": coordinate_stable,
                    **compact_decoder_fields(status, candidates),
                }
                variant_rows.append(row)
                endpoint_rows.append(row)
            counterfactual_rows = [row for row in endpoint_rows if row["variant"] != "causal_history"]
            repeated_row = next(row for row in counterfactual_rows if row["variant"] == "endpoint_repeated_history")
            prior_rows = [row for row in counterfactual_rows if row["variant"].startswith("prior_t")]
            history_contribution_supported = bool(repeated_row["material_map_change_vs_causal"] and any(bool(row["material_map_change_vs_causal"]) for row in prior_rows))
            all_counterfactual_coordinates_stable = bool(
                all(row["raw_top_candidate_coordinate_stable_vs_causal"] is True for row in counterfactual_rows)
            )
            endpoint_summaries.append({
                "dataset_index": int(dataset_index),
                "sample_id": str(sample["metadata"]["sample_id"]),
                "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                "history_layer_z": [int(value) for value in sample["history_layer_z"].tolist()],
                "history_contribution_supported_by_map_change": history_contribution_supported,
                "all_counterfactual_top_coordinates_stable_within_one_model_pixel": all_counterfactual_coordinates_stable,
                "baseline_decoder_status": str(baseline_status["candidate_status"]),
                "baseline_candidate_count": int(len(baseline_candidates)),
                "counterfactual_variant_count": int(len(counterfactual_rows)),
            })
            qc_path = args.output_dir / f"candidate_stability_endpoint_z{int(sample['endpoint_layer_z']):03d}.png"
            plot_endpoint_qc(predictions, int(sample["endpoint_layer_z"]), qc_path)
            qc_paths.append(str(qc_path))
    history_support_count = sum(bool(row["history_contribution_supported_by_map_change"]) for row in endpoint_summaries)
    coordinate_stability_count = sum(bool(row["all_counterfactual_top_coordinates_stable_within_one_model_pixel"]) for row in endpoint_summaries)
    summary_path = args.output_dir / "a_only_candidate_stability_summary.json"
    csv_path = args.output_dir / "a_only_candidate_stability_by_variant.csv"
    write_csv(csv_path, variant_rows)
    summary = {
        "audit_type": "read-only A-only causal-history counterfactual candidate stability audit; not training or defect classification",
        "purpose": "Test whether the saved residual checkpoint's raw-camera candidate map changes when its causal history is counterfactually replaced, without using XCT support in decoder or using machine-coordinate calibration.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "selected_dataset_indices": [int(value) for value in args.indices],
        "device": str(device),
        "counterfactual_policy": {
            "baseline": "normal causal K=4 history",
            "endpoint_repeated": "all K input time steps replaced by the endpoint A frame; diagnostic-only counterfactual",
            "prior_replacements": "one preceding history frame at a time replaced by endpoint A frame; diagnostic-only counterfactual",
            "decoder_policy": "same spatial/tie safety gates; no XCT support mask, no provisional geometry gate, and no cross-endpoint or variant-to-variant temporal-invariance comparison for this sparse selected-endpoint audit",
            "map_change_mae_min": MAP_CHANGE_MAE_MIN,
            "raw_coordinate_stability_max_model_pixels": RAW_COORDINATE_STABILITY_MAX_MODEL_PIXEL,
        },
        "endpoint_summaries": endpoint_summaries,
        "aggregate": {
            "selected_endpoint_count": int(len(endpoint_summaries)),
            "history_contribution_supported_endpoint_count": int(history_support_count),
            "all_counterfactual_top_coordinates_stable_endpoint_count": int(coordinate_stability_count),
            "interpretation": "Map sensitivity and top-coordinate stability are separate diagnostics; neither establishes physical defect location, XCT response direction, calibration validity, or deployment readiness.",
        },
        "score_semantics": "sigmoid-scaled XCT-derived continuous quality candidate; direction unresolved; not a defect/anomaly probability",
        "prohibitions": [
            "Does not train, write, or modify a checkpoint, optimizer, config, manifest, raw TIFF, or registered XCT CSV.",
            "Does not build/read weak response or XCT support; decoder input is camera history only.",
            "Does not use provisional machine/part geometry, choose calibration rank/orientation, or change calibration holds.",
            "Does not persist dense prediction arrays; three display-only deterministic QC PNGs are the only map-like artifacts.",
        ],
        "outputs": {
            "by_variant_csv": str(csv_path),
            "summary_json": str(summary_path),
            "endpoint_qc_pngs": qc_paths,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("A-only candidate stability audit complete. No raw data, target/support, checkpoint, config, calibration, model, or candidate policy was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
