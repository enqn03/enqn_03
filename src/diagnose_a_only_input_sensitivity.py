#!/usr/bin/env python3
"""Read-only stagewise input-sensitivity diagnostic for an A-only checkpoint.

The diagnostic measures whether selected causal A histories remain different at
five points: model input, final-history frame encoder feature, all-history frame
encoder feature, final causal temporal feature, decoder logit, and sigmoid score
map. It never creates weak targets, opens registered XCT CSV files, runs an
optimizer, or writes any file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ammt_causal_dataset import AMMTCausalStageDataset
from train_a_only_baseline import AOnlyCausalCandidateNet, choose_device, load_yaml


def tensor_statistics(values: Tensor) -> dict[str, float]:
    """Return compact numerical statistics without storing the source tensor."""
    values_cpu = values.detach().float().cpu()
    return {
        "min": float(values_cpu.min().item()),
        "max": float(values_cpu.max().item()),
        "mean": float(values_cpu.mean().item()),
        "std_population": float(values_cpu.std(unbiased=False).item()),
        "range": float((values_cpu.max() - values_cpu.min()).item()),
        "l2_norm": float(torch.linalg.vector_norm(values_cpu).item()),
    }


def pairwise_distance(left: Tensor, right: Tensor) -> dict[str, float]:
    """Summarize a pairwise stage difference without retaining image/features."""
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    difference = left_cpu - right_cpu
    abs_difference = difference.abs()
    reference_norm = max(
        float(torch.linalg.vector_norm(left_cpu).item()),
        float(torch.linalg.vector_norm(right_cpu).item()),
        1.0e-12,
    )
    return {
        "mae": float(abs_difference.mean().item()),
        "max_abs": float(abs_difference.max().item()),
        "rmse": float(torch.sqrt((difference * difference).mean()).item()),
        "relative_l2": float(torch.linalg.vector_norm(difference).item() / reference_norm),
    }


def stagewise_forward(model: AOnlyCausalCandidateNet, history: Tensor) -> dict[str, Tensor]:
    """Reproduce the model forward path while exposing only transient features."""
    if history.ndim != 5:
        raise ValueError(f"history must be [B,K,C,H,W], got {tuple(history.shape)}")
    batch, steps, channels, height, width = history.shape
    if channels != 6:
        raise ValueError(f"A-only model requires six channels, got {channels}")

    encoded_flat = model.frame_encoder(history.reshape(batch * steps, channels, height, width))
    encoded_history = encoded_flat.reshape(batch, steps, encoded_flat.shape[1], height, width)
    encoded_for_temporal = encoded_history.permute(0, 2, 1, 3, 4)
    padded = F.pad(encoded_for_temporal, (0, 0, 0, 0, model.temporal_kernel_size - 1, 0))
    temporal_history = F.silu(model.temporal_norm(model.temporal(padded)))
    temporal_update = temporal_history[:, :, -1]
    temporal_final = encoded_history[:, -1] + temporal_update if model.use_endpoint_feature_residual else temporal_update
    logits = model.decoder(temporal_final)
    score = torch.sigmoid(logits)
    return {
        "input_history": history,
        "encoded_history": encoded_history,
        "encoded_final_history_frame": encoded_history[:, -1],
        "temporal_final": temporal_final,
        "logits": logits,
        "score": score,
    }


def distinct_for_all_pairs(pair_rows: list[dict[str, Any]], stage_name: str, atol: float) -> bool:
    return bool(pair_rows) and all(float(row["stage_distances"][stage_name]["max_abs"]) > atol for row in pair_rows)


def derive_collapse_interpretation(pair_rows: list[dict[str, Any]], atol: float) -> dict[str, Any]:
    ordered_stages = (
        "input_history",
        "encoded_final_history_frame",
        "encoded_history",
        "temporal_final",
        "logits",
        "score",
    )
    all_pairs_distinct = {stage: distinct_for_all_pairs(pair_rows, stage, atol) for stage in ordered_stages}
    earliest_not_distinct = next((stage for stage in ordered_stages if not all_pairs_distinct[stage]), None)

    if not pair_rows:
        conclusion = "need_at_least_two_indices_for_pairwise_sensitivity"
    elif not all_pairs_distinct["input_history"]:
        conclusion = "selected_inputs_are_numerically_indistinguishable; check Dataset selection before interpreting model"
    elif not all_pairs_distinct["encoded_final_history_frame"]:
        conclusion = "frame_encoder_insensitivity_candidate; inputs differ but final-frame encoder features are not distinct"
    elif not all_pairs_distinct["encoded_history"]:
        conclusion = "frame_encoder_history_insensitivity_candidate; input histories differ but all encoded histories are not distinct"
    elif not all_pairs_distinct["temporal_final"]:
        conclusion = "temporal_aggregation_insensitivity_candidate; encoded histories differ but final causal temporal features are not distinct"
    elif not all_pairs_distinct["logits"]:
        conclusion = "decoder_insensitivity_candidate; temporal features differ but decoder logits are not distinct"
    elif not all_pairs_distinct["score"]:
        conclusion = "sigmoid_saturation_or_quantization_candidate; logits differ but score maps are not distinct"
    else:
        conclusion = "selected_inputs_remain_distinct_through_score; inspect magnitude and spatial diagnostics before changing training"

    return {
        "stage_difference_atol": atol,
        "all_selected_pairs_distinct_by_stage": all_pairs_distinct,
        "earliest_stage_not_distinct_for_all_pairs": earliest_not_distinct,
        "conclusion": conclusion,
        "limit": (
            "A max_abs value above atol means numerical distinction, not sufficient localization quality. "
            "This diagnostic does not use XCT support or target values."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--stage-difference-atol", type=float, default=1.0e-6)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage_difference_atol < 0.0:
        raise ValueError("--stage-difference-atol must be non-negative.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    config = load_yaml(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    if data_config["stage"] != "A" or int(data_config["input_channels"]) != 6:
        raise ValueError("Diagnostic requires the six-channel A-only baseline config.")

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

    sample_rows: list[dict[str, Any]] = []
    features: list[dict[str, Tensor]] = []
    with torch.no_grad():
        for index in args.indices:
            if not 0 <= index < len(dataset):
                raise IndexError(f"--indices value {index} is outside split={args.split} range 0..{len(dataset) - 1}")
            sample = dataset[index]
            history = sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)
            stage_features = stagewise_forward(model, history)
            model_forward_score = model(history)
            reconstruction_error = pairwise_distance(stage_features["score"], model_forward_score)
            if reconstruction_error["max_abs"] > 1.0e-6:
                raise RuntimeError("Stagewise reconstruction does not match model.forward output.")
            features.append({name: value.cpu() for name, value in stage_features.items()})
            sample_rows.append(
                {
                    "dataset_index": int(index),
                    "sample_id": str(sample["metadata"]["sample_id"]),
                    "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                    "history_layer_z": [int(value) for value in sample["history_layer_z"].tolist()],
                    "per_stage_statistics": {
                        name: tensor_statistics(value)
                        for name, value in stage_features.items()
                        if name != "encoded_history"
                    },
                    "forward_reconstruction_max_abs": reconstruction_error["max_abs"],
                }
            )

    pair_rows: list[dict[str, Any]] = []
    pair_stage_names = (
        "input_history",
        "encoded_final_history_frame",
        "encoded_history",
        "temporal_final",
        "logits",
        "score",
    )
    for left_idx in range(len(features)):
        for right_idx in range(left_idx + 1, len(features)):
            pair_rows.append(
                {
                    "left_dataset_index": sample_rows[left_idx]["dataset_index"],
                    "left_endpoint_layer_z": sample_rows[left_idx]["endpoint_layer_z"],
                    "right_dataset_index": sample_rows[right_idx]["dataset_index"],
                    "right_endpoint_layer_z": sample_rows[right_idx]["endpoint_layer_z"],
                    "stage_distances": {
                        name: pairwise_distance(features[left_idx][name], features[right_idx][name])
                        for name in pair_stage_names
                    },
                }
            )

    output = {
        "audit_type": "A-only checkpoint stagewise input-sensitivity diagnostic; read-only, no XCT target/support",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": str(args.config),
        "split": args.split,
        "device": str(device),
        "model": {
            "base_channels": int(model_config["base_channels"]),
            "temporal_kernel_size": int(model_config["temporal_kernel_size"]),
            "use_endpoint_feature_residual": bool(model_config.get("use_endpoint_feature_residual", False)),
            "input_channels": int(data_config["input_channels"]),
        },
        "samples": sample_rows,
        "pairwise_stage_distances": pair_rows,
        "stagewise_interpretation": derive_collapse_interpretation(pair_rows, float(args.stage_difference_atol)),
        "storage_policy": "terminal JSON only; no TIFF crop, dense target, checkpoint, optimizer, or output file is created",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    print("Stagewise input-sensitivity diagnostic complete. No raw TIFF, XCT CSV, target, checkpoint, or output file was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
