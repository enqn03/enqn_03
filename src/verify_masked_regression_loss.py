#!/usr/bin/env python3
"""Read-only runtime checks for AMMT support-masked continuous regression loss.

This script does not train a model, write a checkpoint, or create a dense target.
It loads two existing AMMTWeakTargetDataset samples and verifies that:
1. an early endpoint without XCT support has exactly zero target loss/gradient;
2. unsupported pixels cannot change the loss or receive target-loss gradients;
3. a supported endpoint has a positive loss and nonzero supported gradients.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from ammt_masked_regression_loss import SupportMaskedSmoothL1Loss
from ammt_weak_target_dataset import AMMTWeakTargetDataset


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_dataset(args: argparse.Namespace) -> AMMTWeakTargetDataset:
    return AMMTWeakTargetDataset(
        stage=args.stage,
        tiff_path=args.tiff,
        manifest_path=args.manifest,
        normalization_config_path=args.normalization_config,
        split=args.split,
        registered_root=args.registered_root,
        calibration_config=args.calibration_config,
        weak_target_config=args.weak_target_config,
    )


def gradient_abs_sums(prediction: torch.Tensor, support: torch.Tensor) -> tuple[float, float]:
    gradient = prediction.grad.detach().abs()
    supported = float(gradient[support.bool()].sum().item())
    unsupported = float(gradient[~support.bool()].sum().item())
    return supported, unsupported


def zero_support_check(loss_fn: SupportMaskedSmoothL1Loss, sample: dict[str, Any]) -> dict[str, Any]:
    target = sample["weak_response"].unsqueeze(0)
    support = sample["weak_support_mask"].unsqueeze(0)
    prediction = torch.full_like(target, 0.37, requires_grad=True)
    result = loss_fn(prediction, target, support)
    result.loss.backward()
    supported_grad, unsupported_grad = gradient_abs_sums(prediction, support)

    passed = (
        not bool(sample["weak_target_available"])
        and result.supervised_pixel_count == 0
        and float(result.loss.detach().item()) == 0.0
        and supported_grad == 0.0
        and unsupported_grad == 0.0
    )
    return {
        "endpoint_layer_z": int(sample["endpoint_layer_z"]),
        "weak_target_available": bool(sample["weak_target_available"]),
        "supervised_pixel_count": result.supervised_pixel_count,
        "loss": float(result.loss.detach().item()),
        "supported_gradient_abs_sum": supported_grad,
        "unsupported_gradient_abs_sum": unsupported_grad,
        "pass": passed,
    }


def supported_target_check(loss_fn: SupportMaskedSmoothL1Loss, sample: dict[str, Any]) -> dict[str, Any]:
    target = sample["weak_response"].unsqueeze(0)
    support = sample["weak_support_mask"].unsqueeze(0)

    prediction = torch.zeros_like(target, requires_grad=True)
    baseline = loss_fn(prediction, target, support)
    baseline.loss.backward()
    supported_grad, unsupported_grad = gradient_abs_sums(prediction, support)

    altered_prediction = torch.zeros_like(target)
    altered_prediction[~support.bool()] = 1000.0
    altered = loss_fn(altered_prediction, target, support)
    loss_difference = float(torch.abs(baseline.loss.detach() - altered.loss.detach()).item())

    passed = (
        bool(sample["weak_target_available"])
        and baseline.supervised_pixel_count > 0
        and float(baseline.loss.detach().item()) > 0.0
        and supported_grad > 0.0
        and unsupported_grad == 0.0
        and loss_difference <= 1e-8
    )
    return {
        "endpoint_layer_z": int(sample["endpoint_layer_z"]),
        "weak_target_available": bool(sample["weak_target_available"]),
        "supervised_pixel_count": baseline.supervised_pixel_count,
        "baseline_loss": float(baseline.loss.detach().item()),
        "loss_when_only_unknown_predictions_are_set_to_1000": float(altered.loss.detach().item()),
        "absolute_loss_difference": loss_difference,
        "supported_gradient_abs_sum": supported_grad,
        "unsupported_gradient_abs_sum": unsupported_grad,
        "pass": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["A", "B"])
    parser.add_argument("--tiff", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normalization-config", required=True)
    parser.add_argument("--registered-root", required=True)
    parser.add_argument("--calibration-config", required=True)
    parser.add_argument("--weak-target-config", required=True)
    parser.add_argument("--loss-config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--unavailable-index", type=int, default=0)
    parser.add_argument("--available-index", type=int, default=124)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.loss_config))
    objective = config["objective"]
    if objective["name"] != "support_masked_smooth_l1":
        raise ValueError("loss config objective.name must be support_masked_smooth_l1.")
    loss_fn = SupportMaskedSmoothL1Loss(beta=float(objective["beta"]))
    dataset = make_dataset(args)

    unavailable_sample = dataset[args.unavailable_index]
    available_sample = dataset[args.available_index]
    unavailable = zero_support_check(loss_fn, unavailable_sample)
    available = supported_target_check(loss_fn, available_sample)

    summary = {
        "audit_type": "support-masked continuous regression loss runtime check; not model training",
        "stage": args.stage,
        "split": args.split,
        "loss_name": objective["name"],
        "beta": float(objective["beta"]),
        "zero_support_unknown_check": unavailable,
        "supported_target_masking_check": available,
        "overall_pass": bool(unavailable["pass"] and available["pass"]),
    }
    print(json.dumps(summary, indent=2))
    print("Masked regression loss verification complete. No raw file, dense target, checkpoint, or output file was written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
