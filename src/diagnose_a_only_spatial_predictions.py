#!/usr/bin/env python3
"""Read-only spatial diagnostic for a trained A-only AMMT checkpoint.

This script loads an existing checkpoint and selected causal samples, then
prints compact JSON statistics for prediction spatial variation, XCT-supported
target variation, and their supported-pixel correlation. It performs no
training and writes no checkpoint, dense heatmap, crop, target, or result file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ammt_weak_target_dataset import AMMTWeakTargetDataset
from train_a_only_baseline import AOnlyCausalCandidateNet, choose_device, load_yaml, local_maximum_candidates


def tensor_stats(values: Tensor) -> dict[str, float] | None:
    if values.numel() == 0:
        return None
    values = values.detach().float().cpu()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std_population": float(values.std(unbiased=False).item()),
        "spatial_range": float((values.max() - values.min()).item()),
    }


def pearson_correlation(x: Tensor, y: Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x.detach().float().flatten().cpu()
    y = y.detach().float().flatten().cpu()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denominator.item()) == 0.0:
        return None
    return float((torch.dot(x_centered, y_centered) / denominator).item())


def make_dataset(args: argparse.Namespace) -> AMMTWeakTargetDataset:
    return AMMTWeakTargetDataset(
        stage="A",
        tiff_path=args.tiff_a,
        manifest_path=args.manifest,
        normalization_config_path=args.normalization_config,
        split=args.split,
        registered_root=args.registered_root,
        calibration_config=args.calibration_config,
        weak_target_config=args.weak_target_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = config["data"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]
    training_config = config["training"]
    if data_config["stage"] != "A" or int(data_config["input_channels"]) != 6:
        raise ValueError("Diagnostic requires the six-channel A-only baseline config.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = choose_device(args.device or str(training_config["device"]))
    model = AOnlyCausalCandidateNet(
        input_channels=int(data_config["input_channels"]),
        base_channels=int(model_config["base_channels"]),
        temporal_kernel_size=int(model_config["temporal_kernel_size"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = make_dataset(args)
    plateau_atol = float(evaluation_config["spatial_plateau_range_atol"])

    summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in args.indices:
            if not 0 <= index < len(dataset):
                raise IndexError(f"--indices value {index} is outside split={args.split} range 0..{len(dataset)-1}")
            sample = dataset[index]
            prediction = model(sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)).cpu()[0, 0]
            target = sample["weak_response"][0].cpu()
            support = sample["weak_support_mask"][0].cpu() > 0
            predicted_supported = prediction[support]
            target_supported = target[support]
            prediction_stats = tensor_stats(prediction)
            supported_target_stats = tensor_stats(target_supported)
            supported_prediction_stats = tensor_stats(predicted_supported)
            decoder_result = local_maximum_candidates(
                prediction.unsqueeze(0).unsqueeze(0),
                sample,
                top_k=int(evaluation_config["top_k_candidates_per_endpoint"]),
                kernel_size=int(evaluation_config["local_maximum_kernel_size"]),
                plateau_range_atol=plateau_atol,
            )
            decoded_candidates = decoder_result.pop("candidates")
            summaries.append(
                {
                    "dataset_index": index,
                    "sample_id": str(sample["metadata"]["sample_id"]),
                    "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                    "weak_target_available": bool(sample["weak_target_available"]),
                    "supervised_pixel_count": int(support.sum().item()),
                    "prediction_statistics": prediction_stats,
                    "supported_prediction_statistics": supported_prediction_stats,
                    "supported_target_statistics": supported_target_stats,
                    "prediction_exact_unique_value_count": int(torch.unique(prediction).numel()),
                    "prediction_target_pearson_on_support": pearson_correlation(predicted_supported, target_supported),
                    "prediction_target_mae_on_support": None if target_supported.numel() == 0 else float((predicted_supported - target_supported).abs().mean().item()),
                    "flat_by_spatial_range_atol": bool(prediction_stats is not None and prediction_stats["spatial_range"] <= plateau_atol),
                    "spatial_plateau_range_atol": plateau_atol,
                    "candidate_decoder_status": str(decoder_result["candidate_status"]),
                    "candidate_decoder_reason": decoder_result["reason"],
                    "candidate_decoder_count": len(decoded_candidates),
                }
            )

    print(
        json.dumps(
            {
                "audit_type": "A-only checkpoint spatial prediction diagnostic; read-only, not training",
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "split": args.split,
                "device": str(device),
                "spatial_plateau_range_atol": plateau_atol,
                "samples": summaries,
                "interpretation": "Flat prediction maps must not produce physical coordinate candidates. Correlation is descriptive only while response direction remains unresolved.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Spatial diagnostic complete. No raw file, dense heatmap, checkpoint, or output file was written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
