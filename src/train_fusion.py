utf-8
#!/usr/bin/env python3
"""Train an A+B Fusion causal temporal-difference candidate model.
This experiment uses paired encoders to process A (AfterSpreading) and B (Burned)
stage images simultaneously. It concatenates their temporal-difference features
and predicts the continuous XCT-derived quality candidate through a single decoder.
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
from ammt_masked_regression_loss import MaskedRegressionLossResult
import torch.nn.functional as F
from train_a_only_baseline import (
    ConvNormAct,
    build_provisional_part_geometry_gate,
    choose_device,
    evaluate_test_candidates,
    load_yaml,
    make_loader,
    prepare_output_directory,
    run_epoch,
    set_seed,
    write_json,
)
from ammt_weak_target_dataset import AMMTFusionWeakTargetDataset
class SupportMaskedBCELoss(nn.Module):
    """Computes BCE loss only on pixels where the support mask is valid.
    The continuous target is binarized using binary_defect_threshold.
    """
    def __init__(self, binary_defect_threshold: float, pos_weight: float = 1.0):
        super().__init__()
        self.binary_defect_threshold = binary_defect_threshold
        self.pos_weight = pos_weight
    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> MaskedRegressionLossResult:
        bool_mask = (mask > 0)
        supervised_pixel_count = int(bool_mask.sum().item())
        supervised_sample_count = int((bool_mask.flatten(start_dim=1).sum(dim=1) > 0).sum().item())
        if supervised_pixel_count == 0:
            loss = pred.sum() * 0.0
            return MaskedRegressionLossResult(
                loss=loss,
                supervised_pixel_count=0,
                supervised_sample_count=0,
            )
        p = pred[bool_mask]
        t = target[bool_mask].to(dtype=pred.dtype)
        binary_t = (t >= self.binary_defect_threshold).to(dtype=pred.dtype)
        weight = torch.ones_like(binary_t)
        weight[binary_t == 1.0] = self.pos_weight
        loss = F.binary_cross_entropy(p, binary_t, weight=weight)
        return MaskedRegressionLossResult(
            loss=loss,
            supervised_pixel_count=supervised_pixel_count,
            supervised_sample_count=supervised_sample_count,
        )
class TemporalDifferenceBranch(nn.Module):
    def __init__(self, input_channels: int, base_channels: int):
        super().__init__()
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
    def forward(self, history: Tensor) -> Tensor:
        batch, steps, channels, height, width = history.shape
        encoded = self.frame_encoder(history.reshape(batch * steps, channels, height, width))
        encoded_history = encoded.reshape(batch, steps, encoded.shape[1], height, width)
        encoded_endpoint = encoded_history[:, -1]
        encoded_prior_mean = encoded_history[:, :-1].mean(dim=1)
        difference_feature = encoded_endpoint - encoded_prior_mean
        encoded_difference = self.difference_encoder(difference_feature)
        return self.fusion(torch.cat([encoded_endpoint, encoded_difference], dim=1))
class CBAMFusion(nn.Module):
    """Convolutional Block Attention Module for selective feature fusion."""
    def __init__(self, channels: int):
        super().__init__()
        reduced_channels = max(1, (2 * channels) // 4)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, 2 * channels)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, feat_a: Tensor, feat_b: Tensor) -> Tensor:
        concat = torch.cat([feat_a, feat_b], dim=1)
        avg_pool = F.avg_pool2d(concat, concat.shape[2:]).flatten(1)
        max_pool = F.max_pool2d(concat, concat.shape[2:]).flatten(1)
        channel_gate = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        concat = concat * channel_gate.unsqueeze(-1).unsqueeze(-1)
        avg_spatial = torch.mean(concat, dim=1, keepdim=True)
        max_spatial, _ = torch.max(concat, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_gate = torch.sigmoid(self.spatial_conv(spatial_concat))
        return concat * spatial_gate
class ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet(nn.Module):
    """A+B fusion model with paired temporal-difference branches and CBAM.
    Processes A and B independently through paired encoders, then selectively
    weights their concatenated features using CBAM (Channel + Spatial Attention),
    and decodes them to a single score map.
    """
    def __init__(self, input_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        if input_channels != 6:
            raise ValueError("Temporal-difference model requires 6 channels: 3 intensity + 3 validity masks.")
        self.branch_a = TemporalDifferenceBranch(input_channels, base_channels)
        self.branch_b = TemporalDifferenceBranch(input_channels, base_channels)
        self.cbam_fusion = CBAMFusion(base_channels)
        self.decoder = nn.Sequential(
            ConvNormAct(2 * base_channels, base_channels),
            nn.Dropout2d(p=0.3),
            ConvNormAct(base_channels, base_channels),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
    def forward(self, history_a: Tensor, history_b: Tensor) -> Tensor:
        if history_a.ndim != 5 or history_b.ndim != 5:
            raise ValueError(f"history must be [B,K,C,H,W]")
        fused_a = self.branch_a(history_a)
        fused_b = self.branch_b(history_b)
        fused_features = self.cbam_fusion(fused_a, fused_b)
        return torch.sigmoid(self.decoder(fused_features))
def make_fusion_dataset(args: argparse.Namespace, split: str) -> AMMTFusionWeakTargetDataset:
    return AMMTFusionWeakTargetDataset(
        tiff_a_path=args.tiff_a,
        tiff_b_path=args.tiff_b,
        manifest_path=args.manifest,
        normalization_config_path=args.normalization_config,
        split=split,
        registered_root=args.registered_root,
        calibration_config=args.calibration_config,
        weak_target_config=args.weak_target_config,
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--tiff-b", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--loss-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    if data["stage"] != "fusion" or int(data["input_channels"]) != 6:
        raise ValueError("Fusion training requires fusion stage and six-channel input.")
    if int(data["sequence_length_k"]) != 4:
        raise ValueError("This controlled experiment requires the frozen K=4 causal comparison contract.")
    if model["name"] != "a_b_gated_fusion_regularized_causal_temporal_difference_candidate_net":
        raise ValueError("Unexpected model.name for fusion experiment.")
    if model["temporal_fusion"] != "paired_encoders_with_gated_cnn_fusion":
        raise ValueError("Unexpected temporal_fusion contract.")
    if model["output_activation"] != "sigmoid":
        raise ValueError("Continuous target contract requires sigmoid output.")
    if bool(evaluation.get("provisional_part_geometry_gate", {}).get("enabled", False)):
        raise ValueError("Controlled raw-camera comparison requires provisional geometry gate disabled.")
def dry_run(model: nn.Module, dataset: Any, loss_fn: SupportMaskedSmoothL1Loss, device: torch.device, sample_index: int) -> None:
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"--dry-run-index must be 0..{len(dataset)-1}")
    sample = dataset[sample_index]
    model.eval()
    with torch.no_grad():
        history_a = sample["model_input_history_a"].unsqueeze(0).to(device=device, dtype=torch.float32)
        history_b = sample["model_input_history_b"].unsqueeze(0).to(device=device, dtype=torch.float32)
        prediction = model(history_a, history_b)
        loss_result = loss_fn(
            prediction,
            sample["weak_response"].unsqueeze(0).to(device=device, dtype=torch.float32),
            sample["weak_support_mask"].unsqueeze(0).to(device=device, dtype=torch.float32),
        )
    print(
        json.dumps(
            {
                "audit_type": "A+B Gated Fusion temporal-difference controlled experiment dry run",
                "architecture": "paired_encoders_with_gated_cnn_fusion",
                "input_a_shape": list(history_a.shape),
                "input_b_shape": list(history_b.shape),
                "prediction_shape": list(prediction.shape),
                "prediction_min": float(prediction.min().item()),
                "prediction_max": float(prediction.max().item()),
                "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                "weak_target_available": bool(sample["weak_target_available"]),
                "supervised_pixel_count": loss_result.supervised_pixel_count,
                "masked_loss": float(loss_result.loss.item()),
                "device": str(device),
                "limits": "Dry run does not train or modify checkpoint/config/TIFF/XCT/calibration.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("A+B Gated Fusion dry run complete.")
def run_fusion_epoch(
    model: nn.Module,
    loader: Any,
    loss_fn: SupportMaskedBCELoss,
    device: torch.device,
    epoch: int,
    split_name: str,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 0.0,
    max_batches: int | None = None,
) -> Any:
    from dataclasses import dataclass
    @dataclass
    class EpochMetrics:
        split: str
        epoch: int
        mean_loss: float | None
        supported_pixel_count: int
        supported_sample_count: int
        total_sample_count: int
        optimizer_steps: int
    model.train() if optimizer is not None else model.eval()
    total_loss = 0.0
    supported_pixel_count = 0
    supported_sample_count = 0
    total_sample_count = 0
    optimizer_steps = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        history_a = batch["model_input_history_a"].to(device=device, dtype=torch.float32)
        history_b = batch["model_input_history_b"].to(device=device, dtype=torch.float32)
        response = batch["weak_response"].to(device=device, dtype=torch.float32)
        support = batch["weak_support_mask"].to(device=device, dtype=torch.float32)
        B = history_a.size(0)
        total_sample_count += B
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(history_a, history_b)
            result = loss_fn(prediction, response, support)
        batch_supported_pixels = result.supervised_pixel_count
        if batch_supported_pixels > 0:
            supported_pixel_count += batch_supported_pixels
            total_loss += result.loss.item() * batch_supported_pixels
            supported_sample_count += int((support.view(B, -1).sum(dim=1) > 0).sum().item())
            if optimizer is not None:
                result.loss.backward()
                if gradient_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                optimizer_steps += 1
        else:
            pass
    mean_loss = total_loss / supported_pixel_count if supported_pixel_count > 0 else None
    return EpochMetrics(
        split=split_name,
        epoch=epoch,
        mean_loss=mean_loss,
        supported_pixel_count=supported_pixel_count,
        supported_sample_count=supported_sample_count,
        total_sample_count=total_sample_count,
        optimizer_steps=optimizer_steps,
    )
import torch
import torch.nn.functional as F
from train_a_only_baseline import raw_coordinate_from_model_index
from typing import Any
def make_fusion_candidate_evaluator(model: torch.nn.Module, dataset: Any, device: torch.device, evaluation: dict[str, Any], max_samples: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    candidates: list[dict[str, Any]] = []
    endpoint_statuses: list[dict[str, Any]] = []
    top_k = int(evaluation["top_k_candidates_per_endpoint"])
    kernel_size = int(evaluation["local_maximum_kernel_size"])
    plateau_range_atol = float(evaluation["spatial_plateau_range_atol"])
    top_score_tie_atol = float(evaluation["top_score_tie_atol"])
    top_score_tie_fraction_max = float(evaluation["top_score_tie_fraction_max"])
    with torch.no_grad():
        sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for index in range(sample_count):
            sample = dataset[index]
            history_a = sample["model_input_history_a"].unsqueeze(0).to(device=device, dtype=torch.float32)
            history_b = sample["model_input_history_b"].unsqueeze(0).to(device=device, dtype=torch.float32)
            prediction = model(history_a, history_b).cpu()
            candidate_map = prediction[0, 0]
            prediction_min, prediction_max = float(candidate_map.min().item()), float(candidate_map.max().item())
            spatial_range = prediction_max - prediction_min
            z = int(sample["endpoint_layer_z"])
            top_score_tie_pixel_count = int(torch.isclose(candidate_map, candidate_map.max(), rtol=0.0, atol=top_score_tie_atol).sum().item())
            top_score_tie_fraction = top_score_tie_pixel_count / int(candidate_map.numel())
            if spatial_range <= plateau_range_atol:
                endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "withheld_spatial_plateau"})
                continue
            if top_score_tie_fraction > top_score_tie_fraction_max:
                endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "withheld_top_score_plateau"})
                continue
            pooled = F.max_pool2d(candidate_map[None, None], kernel_size=kernel_size, stride=1, padding=kernel_size // 2)[0, 0]
            maxima = candidate_map == pooled
            finite_maxima = maxima & torch.isfinite(candidate_map)
            scores = candidate_map.masked_fill(~finite_maxima, -torch.inf).flatten()
            count = min(top_k, int(torch.isfinite(scores).sum().item()))
            if count == 0:
                endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "withheld_no_local_maximum"})
                continue
            values, flat_indices = torch.topk(scores, k=count)
            width = candidate_map.shape[1]
            metadata = sample["metadata"]
            emitted = 0
            for rank, (score, flat_index) in enumerate(zip(values.tolist(), flat_indices.tolist()), start=1):
                y_model, x_model = divmod(int(flat_index), int(width))
                x_raw, y_raw = raw_coordinate_from_model_index(x_model, y_model, metadata)
                candidates.append({
                    "x_pixel": x_raw, "y_pixel": y_raw, "layer_z": z,
                    "score": float(score), "status": "candidate", "stage": "fusion"
                })
                emitted += 1
            endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "emitted", "count": emitted})
    return candidates, endpoint_statuses
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
    if objective["name"] != "support_masked_bce":
        raise ValueError("loss config objective.name must be support_masked_bce.")
    weak_target_config = load_yaml(Path(loss_config["provenance"]["weak_target_config"]))
    seed_value = args.seed if args.seed is not None else int(training_config["seed"])
    set_seed(seed_value)
    device = choose_device(args.device or str(training_config["device"]))
    threshold = float(weak_target_config["response"]["binary_defect_threshold"])
    pos_weight = float(objective.get("pos_weight", 1.0))
    model = ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet(
        input_channels=int(config["data"]["input_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    loss_fn = SupportMaskedBCELoss(binary_defect_threshold=threshold, pos_weight=pos_weight)
    train_dataset = make_fusion_dataset(args, split="train")
    if args.dry_run:
        dry_run(model, train_dataset, loss_fn, device, args.dry_run_index)
        return
    output_dir = args.output_dir or Path(storage_config["output_directory"])
    prepare_output_directory(output_dir)
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config["num_workers"])
    pin_memory = bool(data_config["pin_memory"]) and device.type == "cuda"
    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    validation_dataset = make_fusion_dataset(args, split="validation")
    validation_loader = make_loader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dataset = make_fusion_dataset(args, split="test")
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
        train_metrics = run_fusion_epoch(
            model, train_loader, loss_fn, device, epoch, "train", optimizer,
            gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]), max_batches=args.max_train_batches,
        )
        validation_metrics = run_fusion_epoch(
            model, validation_loader, loss_fn, device, epoch, "validation", None,
            gradient_clip_norm=0.0, max_batches=args.max_eval_batches,
        )
        history.extend([asdict(train_metrics), asdict(validation_metrics)])
        print(json.dumps({"train": asdict(train_metrics), "validation": asdict(validation_metrics)}, ensure_ascii=False))
        if validation_metrics.mean_loss is None:
            raise RuntimeError(f"Epoch {epoch} returned a None validation loss. Halting to prevent overwriting best_state with None.")
        if validation_metrics.mean_loss < best_validation_loss:
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
    test_metrics = run_fusion_epoch(model, test_loader, loss_fn, device, best_epoch, "test", None, gradient_clip_norm=0.0, max_batches=args.max_eval_batches)
    candidates, endpoint_statuses = make_fusion_candidate_evaluator(model, test_dataset, device, evaluation_config, args.max_test_samples)
    status_counts: dict[str, int] = {}
    for status in endpoint_statuses:
        name = str(status["candidate_status"])
        status_counts[name] = status_counts.get(name, 0) + 1
    write_json(
        output_dir / str(storage_config["history_name"]),
        {
            "purpose": "A+B Gated Fusion Temporal-difference training history",
            "architecture": str(config["model"]["temporal_equation"]),
            "best_epoch": best_epoch,
            "best_validation_supported_pixel_mean_smooth_l1": best_validation_loss,
            "history": history,
        },
    )
    write_json(
        output_dir / str(storage_config["test_metrics_name"]),
        {
            "purpose": "A+B Gated Fusion causal test metric on sparse XCT-supported pixels",
            "best_checkpoint_epoch": best_epoch,
            "test": asdict(test_metrics),
            "response_direction": "unresolved",
        },
    )
    write_json(
        output_dir / str(storage_config["test_candidates_name"]),
        {
            "purpose": "A+B Gated Fusion compact raw-camera coordinate candidates",
            "stage": "fusion",
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
                "architecture": "a_b_gated_fusion_temporal_difference",
                "device": str(device),
                "best_epoch": best_epoch,
                "best_validation_supported_pixel_mean_smooth_l1": best_validation_loss,
                "test": asdict(test_metrics),
                "output_directory": str(output_dir),
                "checkpoint": str(checkpoint_path),
                "candidate_count": len(candidates),
                "candidate_endpoint_status_counts": status_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("A+B Gated Fusion Temporal-difference controlled experiment complete.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
