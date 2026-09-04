utf-8
#!/usr/bin/env python3
"""Run a multi-seed matched comparison of the temporal-difference model vs residual baseline.
This script trains both architectures across a list of random seeds, strictly
controlling for seed pairing, identical data ordering, and disabled decoder rules.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any
import torch
from ammt_masked_regression_loss import SupportMaskedSmoothL1Loss
from train_a_only_baseline import (
    AOnlyCausalCandidateNet,
    choose_device,
    load_yaml,
    make_dataset,
    make_loader,
    prepare_output_directory,
    run_epoch,
    set_seed,
    write_json,
)
from train_a_only_temporal_difference_v1 import (
    AOnlyCausalTemporalDifferenceCandidateNet,
)
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: SupportMaskedSmoothL1Loss,
    optimizer_config: dict,
    epochs: int,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[int, float | None, float | None, float]:
    """Train the model and return (best_epoch, val_loss, test_loss, duration_seconds)."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    best_val_loss = math.inf
    best_epoch = -1
    best_state = None
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            epoch,
            "train",
            optimizer,
            gradient_clip_norm=gradient_clip_norm,
            max_batches=None,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            loss_fn,
            device,
            epoch,
            "validation",
            None,
            gradient_clip_norm=0.0,
            max_batches=None,
        )
        print(f"Epoch {epoch}: Train Loss = {train_metrics.mean_loss}, Val Loss = {validation_metrics.mean_loss}")
        if train_metrics.mean_loss is None or validation_metrics.mean_loss is None:
            raise RuntimeError(f"Fail-fast triggered: Epoch {epoch} encountered a None loss.")
        if validation_metrics.mean_loss is not None and validation_metrics.mean_loss < best_val_loss:
            best_val_loss = validation_metrics.mean_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No validation sparse support found.")
    model.load_state_dict(best_state)
    test_metrics = run_epoch(
        model,
        test_loader,
        loss_fn,
        device,
        best_epoch,
        "test",
        None,
        gradient_clip_norm=0.0,
        max_batches=None,
    )
    duration = time.time() - start_time
    return best_epoch, best_val_loss, test_metrics.mean_loss, duration
def check_dataset_support(dataset: Any, split_name: str) -> None:
    """Pre-flight check: ensure the dataset has at least one sample with a weak target available."""
    has_support = any(dataset[i]["weak_target_available"] for i in range(len(dataset)))
    if not has_support:
        raise RuntimeError(f"Pre-flight check failed: No weak target support found in the '{split_name}' dataset split.")
def setup_dataloaders(args: argparse.Namespace, config: dict, device: torch.device) -> tuple[Any, Any, Any]:
    """Creates fresh dataloaders to guarantee identical shuffling if seed is set right before."""
    data_config = config["data"]
    train_dataset = make_dataset(args, split="train")
    validation_dataset = make_dataset(args, split="validation")
    test_dataset = make_dataset(args, split="test")
    check_dataset_support(train_dataset, "train")
    check_dataset_support(validation_dataset, "validation")
    check_dataset_support(test_dataset, "test")
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config["num_workers"]) if device.type != "mps" else 0
    pin_memory = bool(data_config["pin_memory"]) and device.type == "cuda"
    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = make_loader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = make_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, validation_loader, test_loader
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--loss-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit_multi_seed_temporal_comparison_v1"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003, 1004, 1005])
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    if not args.registered_root.exists() or not args.registered_root.is_dir():
        raise FileNotFoundError(f"Registered XCT root directory not found: {args.registered_root}")
    ref_config = load_yaml(args.reference_config)
    cand_config = load_yaml(args.candidate_config)
    loss_config = load_yaml(args.loss_config)
    device = choose_device(args.device)
    loss_fn = SupportMaskedSmoothL1Loss(beta=float(loss_config["objective"]["beta"]))
    output_dir = args.output_dir
    per_seed_dir = output_dir / "per_seed_metrics"
    prepare_output_directory(output_dir)
    prepare_output_directory(per_seed_dir)
    base_channels = int(ref_config["model"]["base_channels"])
    input_channels = int(ref_config["data"]["input_channels"])
    epochs = int(ref_config["training"]["epochs"])
    gradient_clip_norm = float(ref_config["optimizer"]["gradient_clip_norm"])
    optimizer_config = ref_config["optimizer"]
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"\n{'='*50}\n--- Running comparison for seed {seed} ---\n{'='*50}")
        print(f"Training Reference model (Residual)...")
        set_seed(seed)
        train_loader, val_loader, test_loader = setup_dataloaders(args, ref_config, device)
        model_ref = AOnlyCausalCandidateNet(
            input_channels=input_channels,
            base_channels=base_channels,
            temporal_kernel_size=3,
            use_endpoint_feature_residual=True,
        ).to(device)
        ref_params = count_parameters(model_ref)
        ref_epoch, ref_val_loss, ref_test_loss, ref_time = train_model(
            model_ref, train_loader, val_loader, test_loader, loss_fn, optimizer_config, epochs, device, gradient_clip_norm
        )
        print(f"Training Candidate model (Temporal Difference)...")
        set_seed(seed)
        train_loader, val_loader, test_loader = setup_dataloaders(args, cand_config, device)
        model_cand = AOnlyCausalTemporalDifferenceCandidateNet(
            input_channels=input_channels,
            base_channels=base_channels,
        ).to(device)
        cand_params = count_parameters(model_cand)
        cand_epoch, cand_val_loss, cand_test_loss, cand_time = train_model(
            model_cand, train_loader, val_loader, test_loader, loss_fn, optimizer_config, epochs, device, gradient_clip_norm
        )
        paired_diff = (cand_test_loss - ref_test_loss) if cand_test_loss is not None and ref_test_loss is not None else None
        candidate_won = paired_diff < 0 if paired_diff is not None else False
        candidate_tie = paired_diff == 0 if paired_diff is not None else False
        record = {
            "seed": seed,
            "device": str(device),
            "deterministic_status": torch.are_deterministic_algorithms_enabled(),
            "reference_best_epoch": ref_epoch,
            "reference_validation_loss": ref_val_loss,
            "reference_test_loss": ref_test_loss,
            "reference_training_time_seconds": ref_time,
            "reference_parameter_count": ref_params,
            "candidate_best_epoch": cand_epoch,
            "candidate_validation_loss": cand_val_loss,
            "candidate_test_loss": cand_test_loss,
            "candidate_training_time_seconds": cand_time,
            "candidate_parameter_count": cand_params,
            "paired_difference": paired_diff,
            "candidate_won": candidate_won,
            "candidate_tie": candidate_tie,
        }
        results.append(record)
        write_json(per_seed_dir / f"seed_{seed}.json", record)
        print(json.dumps(record, indent=2))
    valid_diffs = [r["paired_difference"] for r in results if r["paired_difference"] is not None]
    win_count = sum(1 for r in results if r["candidate_won"])
    tie_count = sum(1 for r in results if r["candidate_tie"])
    loss_count = len(valid_diffs) - win_count - tie_count
    if valid_diffs:
        mean_diff = statistics.mean(valid_diffs)
        median_diff = statistics.median(valid_diffs)
        std_diff = statistics.stdev(valid_diffs) if len(valid_diffs) > 1 else 0.0
        min_diff = min(valid_diffs)
        max_diff = max(valid_diffs)
        sign_consistency = max(win_count, loss_count) / len(valid_diffs)
    else:
        mean_diff = median_diff = std_diff = min_diff = max_diff = sign_consistency = None
    summary = {
        "purpose": "Multi-seed matched comparison (reference vs candidate) with isolated evaluation environments",
        "seeds_tested": args.seeds,
        "total_valid_seeds": len(valid_diffs),
        "candidate_win_count": win_count,
        "candidate_tie_count": tie_count,
        "candidate_loss_count": loss_count,
        "paired_difference_mean": mean_diff,
        "paired_difference_median": median_diff,
        "paired_difference_std": std_diff,
        "paired_difference_min": min_diff,
        "paired_difference_max": max_diff,
        "sign_consistency": sign_consistency,
        "reference_avg_training_time": statistics.mean([r["reference_training_time_seconds"] for r in results]),
        "candidate_avg_training_time": statistics.mean([r["candidate_training_time_seconds"] for r in results]),
        "reference_parameter_count": results[0]["reference_parameter_count"] if results else None,
        "candidate_parameter_count": results[0]["candidate_parameter_count"] if results else None,
    }
    write_json(output_dir / "multi_seed_comparison.json", summary)
    csv_path = output_dir / "multi_seed_comparison.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nMulti-seed comparison complete.")
    print(f"Results saved to {output_dir}")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
