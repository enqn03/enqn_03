#!/usr/bin/env python3
"""Evaluate an existing A-only AMMT checkpoint without training or overwriting it.

The script reads a previously saved checkpoint and evaluates the held-out causal
A-stage test split with SupportMaskedSmoothL1Loss. It writes only the missing
compact test metric and candidate JSON files to an existing run output directory.
It never opens raw TIFF except through read-only Dataset memmap access and never
modifies source TIFF, registered XCT CSV, checkpoint, history, or dense targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ammt_masked_regression_loss import SupportMaskedSmoothL1Loss
from train_a_only_baseline import (
    AOnlyCausalCandidateNet,
    build_provisional_part_geometry_gate,
    choose_device,
    evaluate_test_candidates,
    load_yaml,
    make_dataset,
    make_loader,
    run_epoch,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--loss-config", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def require_missing_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation output: {path}. Review it or choose a new output directory."
        )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    loss_config = load_yaml(args.loss_config)
    data_config: dict[str, Any] = config["data"]
    model_config: dict[str, Any] = config["model"]
    training_config: dict[str, Any] = config["training"]
    evaluation_config: dict[str, Any] = config["evaluation"]
    storage_config: dict[str, Any] = config["storage"]

    if data_config["stage"] != "A" or int(data_config["input_channels"]) != 6:
        raise ValueError("Checkpoint evaluation requires the six-channel A-only configuration.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.output_dir.is_dir():
        raise NotADirectoryError(f"Expected existing run output directory: {args.output_dir}")

    test_metrics_path = args.output_dir / str(storage_config["test_metrics_name"])
    test_candidates_path = args.output_dir / str(storage_config["test_candidates_name"])
    require_missing_output(test_metrics_path)
    require_missing_output(test_candidates_path)

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

    loss_fn = SupportMaskedSmoothL1Loss(beta=float(loss_config["objective"]["beta"]))
    test_dataset = make_dataset(args, split="test")
    test_loader = make_loader(
        test_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
    )
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        epoch=checkpoint_epoch,
        split="test",
        optimizer=None,
        gradient_clip_norm=0.0,
        max_batches=None,
    )
    geometry_gate = build_provisional_part_geometry_gate(evaluation_config, args.calibration_config)
    candidates, endpoint_statuses = evaluate_test_candidates(
        model=model,
        dataset=test_dataset,
        device=device,
        top_k=int(evaluation_config["top_k_candidates_per_endpoint"]),
        kernel_size=int(evaluation_config["local_maximum_kernel_size"]),
        plateau_range_atol=float(evaluation_config["spatial_plateau_range_atol"]),
        top_score_tie_atol=float(evaluation_config["top_score_tie_atol"]),
        top_score_tie_fraction_max=float(evaluation_config["top_score_tie_fraction_max"]),
        temporal_map_mae_atol=float(evaluation_config["temporal_map_mae_atol"]),
        temporal_map_max_abs_atol=float(evaluation_config["temporal_map_max_abs_atol"]),
        max_samples=args.max_test_samples,
        geometry_gate=geometry_gate,
    )
    status_counts: dict[str, int] = {}
    for endpoint_status in endpoint_statuses:
        status = str(endpoint_status["candidate_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    write_json(
        test_metrics_path,
        {
            "purpose": "Checkpoint-only held-out evaluation; no retraining or checkpoint overwrite",
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "test": asdict(test_metrics),
            "response_direction": "unresolved",
            "score_semantics": "continuous XCT-derived quality candidate, not defect probability",
        },
    )
    write_json(
        test_candidates_path,
        {
            "purpose": "Checkpoint-only compact coordinate candidate output with plateau-safe withholding",
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "stage": "A",
            "split": "test",
            "response_direction": "unresolved",
            "candidate_count": len(candidates),
            "top_k_per_endpoint": int(evaluation_config["top_k_candidates_per_endpoint"]),
            "spatial_plateau_range_atol": float(evaluation_config["spatial_plateau_range_atol"]),
            "top_score_tie_atol": float(evaluation_config["top_score_tie_atol"]),
            "top_score_tie_fraction_max": float(evaluation_config["top_score_tie_fraction_max"]),
            "temporal_map_mae_atol": float(evaluation_config["temporal_map_mae_atol"]),
            "temporal_map_max_abs_atol": float(evaluation_config["temporal_map_max_abs_atol"]),
            "provisional_part_geometry_gate": {"enabled": False} if geometry_gate is None else geometry_gate.metadata(),
            "endpoint_status_counts": status_counts,
            "endpoint_statuses": endpoint_statuses,
            "candidates": candidates,
        },
    )
    print(
        json.dumps(
            {
                "checkpoint_only_evaluation_complete": True,
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint_epoch,
                "device": str(device),
                "test": asdict(test_metrics),
                "candidate_count": len(candidates),
                "provisional_part_geometry_gate": {"enabled": False} if geometry_gate is None else geometry_gate.metadata(),
                "endpoint_status_counts": status_counts,
                "test_metrics_path": str(test_metrics_path),
                "test_candidates_path": str(test_candidates_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Checkpoint-only evaluation complete. No training, raw-file mutation, checkpoint overwrite, or dense output was performed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
