#!/usr/bin/env python3
"""Train a controlled A-only causal temporal-difference candidate model.

This experiment preserves the C32 residual baseline's Dataset, weak target,
loss, split, and decoder safety rules. Its only model-level change is explicit
fusion of encoded endpoint-minus-mean-prior features, which gives all three
preceding K=4 frames a direct causal path to the fused model representation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ammt_masked_regression_loss import SupportMaskedSmoothL1Loss
from train_a_only_baseline import (
    ConvNormAct,
    build_provisional_part_geometry_gate,
    choose_device,
    evaluate_test_candidates,
    load_yaml,
    make_dataset,
    make_loader,
    prepare_output_directory,
    run_epoch,
    set_seed,
    write_json,
)


class AOnlyCausalTemporalDifferenceCandidateNet(nn.Module):
    """A-only model with explicit causal endpoint-minus-prior feature fusion.

    The input remains [B,K=4,C=6,H,W]. Each frame is encoded independently.
    The three prior encoded frames are averaged, subtracted from endpoint
    encoding, transformed, and fused with endpoint encoding. This aggregation
    has no future-frame path and uses every preceding history position directly.
    """

    def __init__(self, input_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        if input_channels != 6:
            raise ValueError("Temporal-difference model requires 6 channels: 3 intensity + 3 validity masks.")
        self.frame_encoder = nn.Sequential(
            ConvNormAct(input_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
        )
        self.difference_encoder = nn.Sequential(
            ConvNormAct(base_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
        )
        self.fusion = nn.Sequential(
            ConvNormAct(2 * base_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
        )
        self.decoder = nn.Sequential(
            ConvNormAct(base_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

    def forward_components(self, history: Tensor) -> dict[str, Tensor]:
        if history.ndim != 5:
            raise ValueError(f"history must be [B,K,C,H,W], got {tuple(history.shape)}")
        batch, steps, channels, height, width = history.shape
        if channels != 6:
            raise ValueError(f"Temporal-difference model expects 6 channels, got {channels}.")
        if steps < 2:
            raise ValueError("Temporal-difference fusion requires K>=2 so at least one prior frame exists.")
        encoded = self.frame_encoder(history.reshape(batch * steps, channels, height, width))
        encoded_history = encoded.reshape(batch, steps, encoded.shape[1], height, width)
        encoded_endpoint = encoded_history[:, -1]
        encoded_prior_mean = encoded_history[:, :-1].mean(dim=1)
        difference_feature = encoded_endpoint - encoded_prior_mean
        encoded_difference = self.difference_encoder(difference_feature)
        fused_feature = self.fusion(torch.cat([encoded_endpoint, encoded_difference], dim=1))
        return {
            "encoded_endpoint": encoded_endpoint,
            "encoded_prior_mean": encoded_prior_mean,
            "difference_feature": difference_feature,
            "encoded_difference": encoded_difference,
            "fused_feature": fused_feature,
        }

    def forward(self, history: Tensor) -> Tensor:
        components = self.forward_components(history)
        return torch.sigmoid(self.decoder(components["fused_feature"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--loss-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-index", type=int, default=124)
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    evaluation = config["evaluation"]
    if data["stage"] != "A" or int(data["input_channels"]) != 6:
        raise ValueError("Temporal-difference training requires A-only six-channel input.")
    if int(data["sequence_length_k"]) != 4:
        raise ValueError("This controlled experiment requires the frozen K=4 causal comparison contract.")
    if model["name"] != "a_only_causal_temporal_difference_candidate_net":
        raise ValueError("Unexpected model.name for temporal-difference experiment.")
    if model["temporal_fusion"] != "endpoint_minus_mean_of_all_preceding_encoded_frames":
        raise ValueError("Unexpected temporal_fusion contract.")
    if model["output_activation"] != "sigmoid":
        raise ValueError("Continuous target contract requires sigmoid output.")
    if bool(evaluation.get("provisional_part_geometry_gate", {}).get("enabled", False)):
        raise ValueError("Controlled raw-camera comparison requires provisional geometry gate disabled.")


def make_candidate_evaluator(model: nn.Module, dataset: Any, device: torch.device, evaluation: dict[str, Any], max_samples: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse only decoder policy, passing this new model without geometry metadata."""
    return evaluate_test_candidates(
        model=model,
        dataset=dataset,
        device=device,
        top_k=int(evaluation["top_k_candidates_per_endpoint"]),
        kernel_size=int(evaluation["local_maximum_kernel_size"]),
        plateau_range_atol=float(evaluation["spatial_plateau_range_atol"]),
        top_score_tie_atol=float(evaluation["top_score_tie_atol"]),
        top_score_tie_fraction_max=float(evaluation["top_score_tie_fraction_max"]),
        temporal_map_mae_atol=float(evaluation["temporal_map_mae_atol"]),
        temporal_map_max_abs_atol=float(evaluation["temporal_map_max_abs_atol"]),
        max_samples=max_samples,
        geometry_gate=None,
    )


def dry_run(model: nn.Module, dataset: Any, loss_fn: SupportMaskedSmoothL1Loss, device: torch.device, sample_index: int) -> None:
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"--dry-run-index must be 0..{len(dataset)-1}")
    sample = dataset[sample_index]
    model.eval()
    with torch.no_grad():
        prediction = model(sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32))
        loss_result = loss_fn(
            prediction,
            sample["weak_response"].unsqueeze(0).to(device=device, dtype=torch.float32),
            sample["weak_support_mask"].unsqueeze(0).to(device=device, dtype=torch.float32),
        )
    print(
        json.dumps(
            {
                "audit_type": "A-only temporal-difference controlled experiment dry run; not training",
                "architecture": "endpoint_minus_mean_prior_encoded_feature_fusion",
                "input_shape": list(sample["model_input_history"].unsqueeze(0).shape),
                "prediction_shape": list(prediction.shape),
                "prediction_min": float(prediction.min().item()),
                "prediction_max": float(prediction.max().item()),
                "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                "weak_target_available": bool(sample["weak_target_available"]),
                "supervised_pixel_count": loss_result.supervised_pixel_count,
                "masked_loss": float(loss_result.loss.item()),
                "device": str(device),
                "limits": "Dry run does not train or modify checkpoint/config/TIFF/XCT/calibration; prediction is continuous XCT-derived quality candidate, not defect probability.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Temporal-difference dry run complete. No training, checkpoint, dense heatmap, raw-file, target, or calibration mutation occurred.")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)
    data_config = config["data"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    storage_config = config["storage"]
    loss_config = load_yaml(args.loss_config)
    objective = loss_config["objective"]
    if objective["name"] != "support_masked_smooth_l1":
        raise ValueError("loss config objective.name must be support_masked_smooth_l1.")

    set_seed(int(training_config["seed"]))
    device = choose_device(args.device or str(training_config["device"]))
    model = AOnlyCausalTemporalDifferenceCandidateNet(
        input_channels=int(data_config["input_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    loss_fn = SupportMaskedSmoothL1Loss(beta=float(objective["beta"]))
    train_dataset = make_dataset(args, split="train")
    if args.dry_run:
        dry_run(model, train_dataset, loss_fn, device, args.dry_run_index)
        return

    output_dir = args.output_dir or Path(storage_config["output_directory"])
    prepare_output_directory(output_dir)
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config["num_workers"])
    pin_memory = bool(data_config["pin_memory"]) and device.type == "cuda"
    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    validation_dataset = make_dataset(args, split="validation")
    validation_loader = make_loader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dataset = make_dataset(args, split="test")
    test_loader = make_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_config["learning_rate"]), weight_decay=float(optimizer_config["weight_decay"]))
    epochs = int(args.epochs or training_config["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch: int | None = None
    checkpoint_path = output_dir / str(storage_config["checkpoint_name"])
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_fn, device, epoch, "train", optimizer,
            gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]), max_batches=args.max_train_batches,
        )
        validation_metrics = run_epoch(
            model, validation_loader, loss_fn, device, epoch, "validation", None,
            gradient_clip_norm=0.0, max_batches=args.max_eval_batches,
        )
        history.extend([asdict(train_metrics), asdict(validation_metrics)])
        print(json.dumps({"train": asdict(train_metrics), "validation": asdict(validation_metrics)}, ensure_ascii=False))
        if validation_metrics.mean_loss is not None and validation_metrics.mean_loss < best_validation_loss:
            best_validation_loss = validation_metrics.mean_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_supported_pixel_mean_smooth_l1": best_validation_loss,
                    "config": config,
                    "loss_config": loss_config,
                },
                checkpoint_path,
            )
    if best_epoch is None:
        raise RuntimeError("No validation sparse support was found; no checkpoint was saved.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, loss_fn, device, best_epoch, "test", None, gradient_clip_norm=0.0, max_batches=args.max_eval_batches)
    candidates, endpoint_statuses = make_candidate_evaluator(model, test_dataset, device, evaluation_config, args.max_test_samples)
    status_counts: dict[str, int] = {}
    for status in endpoint_statuses:
        name = str(status["candidate_status"])
        status_counts[name] = status_counts.get(name, 0) + 1

    write_json(
        output_dir / str(storage_config["history_name"]),
        {
            "purpose": "Controlled temporal-difference training history; continuous XCT-derived response, not defect classification",
            "architecture": str(config["model"]["temporal_equation"]),
            "reference_checkpoint": str(config["comparison"]["reference_checkpoint"]),
            "best_epoch": best_epoch,
            "best_validation_supported_pixel_mean_smooth_l1": best_validation_loss,
            "history": history,
        },
    )
    write_json(
        output_dir / str(storage_config["test_metrics_name"]),
        {
            "purpose": "Held-out causal test metric on sparse XCT-supported pixels only; controlled temporal-difference comparison",
            "reference_test_loss": float(config["comparison"]["reference_test_loss"]),
            "best_checkpoint_epoch": best_epoch,
            "test": asdict(test_metrics),
            "response_direction": "unresolved",
            "interpretation": "XCT-derived continuous quality candidate; not anomaly probability or confirmed defect.",
        },
    )
    write_json(
        output_dir / str(storage_config["test_candidates_name"]),
        {
            "purpose": "Compact raw-camera coordinate candidates using unchanged support-independent safety decoder",
            "stage": "A",
            "response_direction": "unresolved",
            "candidate_count": len(candidates),
            "top_k_per_endpoint": int(evaluation_config["top_k_candidates_per_endpoint"]),
            "spatial_plateau_range_atol": float(evaluation_config["spatial_plateau_range_atol"]),
            "top_score_tie_atol": float(evaluation_config["top_score_tie_atol"]),
            "top_score_tie_fraction_max": float(evaluation_config["top_score_tie_fraction_max"]),
            "temporal_map_mae_atol": float(evaluation_config["temporal_map_mae_atol"]),
            "temporal_map_max_abs_atol": float(evaluation_config["temporal_map_max_abs_atol"]),
            "provisional_part_geometry_gate": {"enabled": False},
            "endpoint_status_counts": status_counts,
            "endpoint_statuses": endpoint_statuses,
            "candidates": candidates,
        },
    )
    print(
        json.dumps(
            {
                "training_complete": True,
                "architecture": "endpoint_minus_mean_prior_encoded_feature_fusion",
                "device": str(device),
                "best_epoch": best_epoch,
                "best_validation_supported_pixel_mean_smooth_l1": best_validation_loss,
                "reference_test_loss": float(config["comparison"]["reference_test_loss"]),
                "test": asdict(test_metrics),
                "output_directory": str(output_dir),
                "checkpoint": str(checkpoint_path),
                "candidate_count": len(candidates),
                "candidate_endpoint_status_counts": status_counts,
                "response_direction": "unresolved",
                "limit": "No calibration rank/orientation or physical candidate location is selected by this controlled model experiment.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Temporal-difference controlled experiment complete. Raw files were read-only; dense heatmaps were not persisted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
