utf-8
#!/usr/bin/env python3
"""Train and evaluate the first causal A-only AMMT quality-candidate baseline.
The script consumes only AfterSpreading (A-stage) histories shaped
[B, K=4, C=6, H=256, W=256]. It predicts a sigmoid-scaled continuous map and
uses SupportMaskedSmoothL1Loss only where the on-the-fly XCT support mask is 1.
Important semantics
-------------------
The prediction is an ``XCT-derived quality candidate`` response. It is NOT a
binary defect map, anomaly probability, defect class, or automatic pass/fail
decision while the high/low XCT response direction remains unresolved.
The script never modifies raw TIFF or registered XCT CSV files. A real training
run writes only the requested output directory under outputs/. No dense target
heatmaps are stored; coordinate candidates are compact JSON records.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml
from ammt_masked_regression_loss import SupportMaskedSmoothL1Loss
from ammt_weak_target_dataset import AMMTWeakTargetDataset
from audit_machine_camera_calibration import PARTS, RECT, build_candidates
@dataclass
class EpochMetrics:
    split: str
    epoch: int
    mean_loss: float | None
    supported_pixel_count: int
    supported_sample_count: int
    total_sample_count: int
    optimizer_steps: int
class ConvNormAct(nn.Module):
    """Small convolutional building block suitable for the first baseline."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
class AOnlyCausalCandidateNet(nn.Module):
    """Causal 6-channel baseline with temporal Conv3D aggregation.
    Input shape is [B, K, C=6, H, W]. The K axis is processed causally by a
    3D convolution whose time padding is only on the past side. The final map
    is a sigmoid-scaled [B,1,H,W] continuous response prediction.
    """
    def __init__(
        self,
        input_channels: int = 6,
        base_channels: int = 8,
        temporal_kernel_size: int = 3,
        use_endpoint_feature_residual: bool = False,
    ) -> None:
        super().__init__()
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer.")
        self.temporal_kernel_size = temporal_kernel_size
        self.use_endpoint_feature_residual = bool(use_endpoint_feature_residual)
        self.frame_encoder = nn.Sequential(
            ConvNormAct(input_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
        )
        self.temporal = nn.Conv3d(
            base_channels,
            base_channels,
            kernel_size=(temporal_kernel_size, 3, 3),
            padding=(0, 1, 1),
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.decoder = nn.Sequential(
            ConvNormAct(base_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 5:
            raise ValueError(f"history must be [B,K,C,H,W], got {tuple(history.shape)}")
        batch, steps, channels, height, width = history.shape
        if channels != 6:
            raise ValueError(f"A-only baseline expects six channels, got {channels}.")
        encoded_history = self.frame_encoder(history.reshape(batch * steps, channels, height, width))
        encoded_history = encoded_history.reshape(batch, steps, encoded_history.shape[1], height, width)
        encoded_endpoint = encoded_history[:, -1]
        encoded_for_temporal = encoded_history.permute(0, 2, 1, 3, 4)
        padded_history = F.pad(encoded_for_temporal, (0, 0, 0, 0, self.temporal_kernel_size - 1, 0))
        temporal_update = F.silu(self.temporal_norm(self.temporal(padded_history)))[:, :, -1]
        temporal_final = encoded_endpoint + temporal_update if self.use_endpoint_feature_residual else temporal_update
        logits = self.decoder(temporal_final)
        return torch.sigmoid(logits)
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return config
def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def make_dataset(args: argparse.Namespace, split: str) -> AMMTWeakTargetDataset:
    return AMMTWeakTargetDataset(
        stage="A",
        tiff_path=args.tiff_a,
        manifest_path=args.manifest,
        normalization_config_path=args.normalization_config,
        split=split,
        registered_root=args.registered_root,
        calibration_config=args.calibration_config,
        weak_target_config=args.weak_target_config,
    )
def make_loader(dataset: AMMTWeakTargetDataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
def limited_batches(loader: DataLoader, max_batches: int | None) -> Iterator[dict[str, Any]]:
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch
def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    history = batch["model_input_history"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch["weak_response"].to(device=device, dtype=torch.float32, non_blocking=True)
    support = batch["weak_support_mask"].to(device=device, dtype=torch.float32, non_blocking=True)
    return history, target, support
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SupportMaskedSmoothL1Loss,
    device: torch.device,
    epoch: int,
    split: str,
    optimizer: AdamW | None,
    gradient_clip_norm: float,
    max_batches: int | None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    weighted_loss_sum = 0.0
    supported_pixels = 0
    supported_samples = 0
    total_samples = 0
    optimizer_steps = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in limited_batches(loader, max_batches):
            history, target, support = batch_to_device(batch, device)
            total_samples += int(history.shape[0])
            prediction = model(history)
            result = loss_fn(prediction, target, support)
            supported_pixels += result.supervised_pixel_count
            supported_samples += result.supervised_sample_count
            if result.supervised_pixel_count > 0:
                weighted_loss_sum += float(result.loss.detach().item()) * result.supervised_pixel_count
            if training and result.supervised_pixel_count > 0:
                optimizer.zero_grad(set_to_none=True)
                result.loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
                optimizer_steps += 1
    mean_loss = None if supported_pixels == 0 else weighted_loss_sum / supported_pixels
    return EpochMetrics(
        split=split,
        epoch=epoch,
        mean_loss=mean_loss,
        supported_pixel_count=supported_pixels,
        supported_sample_count=supported_samples,
        total_sample_count=total_samples,
        optimizer_steps=optimizer_steps,
    )
def raw_coordinate_from_model_index(x_model: int, y_model: int, metadata: dict[str, Any]) -> tuple[float, float]:
    roi = metadata["working_roi_raw_camera_pixels"]
    x_scale = float(metadata["raw_pixels_per_output_pixel_x"])
    y_scale = float(metadata["raw_pixels_per_output_pixel_y"])
    x_raw = float(roi["x0"]) + (float(x_model) + 0.5) * x_scale
    y_raw = float(roi["y0"]) + (float(y_model) + 0.5) * y_scale
    return x_raw, y_raw
@dataclass(frozen=True)
class ProvisionalPartGeometryGate:
    """Optional candidate-only geometry filter; it never changes score-map inference."""
    inverse_homography: np.ndarray
    raw_offset_xy: tuple[float, float]
    calibration_status: str
    geometry_rank: int
    orientation: str
    def containing_part(self, raw_x: float, raw_y: float) -> str | None:
        offset_x, offset_y = self.raw_offset_xy
        homogeneous = self.inverse_homography @ np.array([raw_x - offset_x, raw_y - offset_y, 1.0], dtype=np.float64)
        denominator = float(homogeneous[2])
        if not math.isfinite(denominator) or abs(denominator) <= 1.0e-12:
            return None
        machine_xy = homogeneous[:2] / denominator
        if not np.isfinite(machine_xy).all():
            return None
        x_value, y_value = float(machine_xy[0]), float(machine_xy[1])
        tolerance = 1.0e-9
        for part in PARTS:
            x_min, x_max, y_min, y_max = RECT[part]
            if x_min - tolerance <= x_value <= x_max + tolerance and y_min - tolerance <= y_value <= y_max + tolerance:
                return str(part)
        return None
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "name": "provisional_known_part_rectangle_filter",
            "calibration_status": self.calibration_status,
            "geometry_rank": self.geometry_rank,
            "orientation": self.orientation,
            "raw_pixel_global_offset_xy": list(self.raw_offset_xy),
            "policy": "Filter local maxima to configured known part rectangles; withhold if no in-geometry maximum remains.",
            "limit": "This uses provisional calibration only and does not confirm physical defect location or calibration accuracy.",
        }
def build_provisional_part_geometry_gate(evaluation_config: dict[str, Any], calibration_config_path: Path) -> ProvisionalPartGeometryGate | None:
    """Return None by default so historical decoder behavior remains unchanged."""
    gate_config = evaluation_config.get("provisional_part_geometry_gate", {})
    if gate_config is None:
        gate_config = {}
    if not isinstance(gate_config, dict):
        raise ValueError("evaluation.provisional_part_geometry_gate must be a mapping when provided.")
    if not bool(gate_config.get("enabled", False)):
        return None
    if gate_config.get("policy") != "filter_local_maxima_to_known_part_rectangles":
        raise ValueError("Enabled provisional part geometry gate requires the supported local-maxima filter policy.")
    calibration_config = load_yaml(calibration_config_path)
    controls_value = calibration_config["control_points"]["path"]
    controls_path = Path(controls_value)
    if not controls_path.is_absolute():
        controls_path = Path.cwd() / controls_path
    with controls_path.open("r", encoding="utf-8") as handle:
        controls_payload = json.load(handle)
    control_points = controls_payload.get("control_points")
    if not isinstance(control_points, list):
        raise ValueError("Calibration control JSON is missing control_points.")
    rank = int(calibration_config["geometry_candidate"]["rank"])
    candidates = build_candidates(control_points)
    if not 1 <= rank <= len(candidates):
        raise ValueError(f"Calibration rank {rank} is outside available candidate range.")
    homography = np.asarray(candidates[rank - 1]["H"], dtype=np.float64)
    return ProvisionalPartGeometryGate(
        inverse_homography=np.linalg.inv(homography),
        raw_offset_xy=tuple(float(value) for value in calibration_config["local_photometric_refinement"]["raw_pixel_global_offset_xy"]),
        calibration_status=str(calibration_config["status"]),
        geometry_rank=rank,
        orientation=str(calibration_config["geometry_candidate"]["orientation"]),
    )
def local_maximum_candidates(
    prediction: Tensor,
    sample: dict[str, Any],
    top_k: int,
    kernel_size: int,
    plateau_range_atol: float,
    top_score_tie_atol: float,
    top_score_tie_fraction_max: float,
    previous_prediction: Tensor | None,
    temporal_map_mae_atol: float,
    temporal_map_max_abs_atol: float,
    geometry_gate: ProvisionalPartGeometryGate | None = None,
) -> dict[str, Any]:
    """Decode local maxima, withholding arbitrary coordinates on a flat map."""
    if prediction.shape != (1, 1, prediction.shape[-2], prediction.shape[-1]):
        raise ValueError(f"Expected one prediction [1,1,H,W], got {tuple(prediction.shape)}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("local maximum kernel_size must be a positive odd integer.")
    if min(plateau_range_atol, top_score_tie_atol, temporal_map_mae_atol, temporal_map_max_abs_atol) < 0:
        raise ValueError("All decoder tolerances must be non-negative.")
    if not 0.0 <= top_score_tie_fraction_max <= 1.0:
        raise ValueError("top_score_tie_fraction_max must be in [0, 1].")
    candidate_map = prediction[0, 0]
    prediction_min = float(candidate_map.min().item())
    prediction_max = float(candidate_map.max().item())
    spatial_range = prediction_max - prediction_min
    top_score_tie_pixel_count = int(torch.isclose(candidate_map, candidate_map.max(), rtol=0.0, atol=top_score_tie_atol).sum().item())
    top_score_tie_fraction = top_score_tie_pixel_count / int(candidate_map.numel())
    temporal_map_mae: float | None = None
    temporal_map_max_abs: float | None = None
    temporally_invariant = False
    if previous_prediction is not None:
        if tuple(previous_prediction.shape) != tuple(prediction.shape):
            raise ValueError("previous_prediction must have the same [1,1,H,W] shape as prediction.")
        temporal_difference = (prediction - previous_prediction).abs()
        temporal_map_mae = float(temporal_difference.mean().item())
        temporal_map_max_abs = float(temporal_difference.max().item())
        temporally_invariant = temporal_map_mae <= temporal_map_mae_atol and temporal_map_max_abs <= temporal_map_max_abs_atol
    metadata = sample["metadata"]
    status: dict[str, Any] = {
        "sample_id": str(metadata["sample_id"]),
        "layer_z": int(sample["endpoint_layer_z"]),
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "spatial_range": spatial_range,
        "plateau_range_atol": float(plateau_range_atol),
        "top_score_tie_atol": float(top_score_tie_atol),
        "top_score_tie_pixel_count": top_score_tie_pixel_count,
        "top_score_tie_fraction": top_score_tie_fraction,
        "top_score_tie_fraction_max": float(top_score_tie_fraction_max),
        "temporal_map_mae_to_previous": temporal_map_mae,
        "temporal_map_max_abs_to_previous": temporal_map_max_abs,
        "temporal_map_mae_atol": float(temporal_map_mae_atol),
        "temporal_map_max_abs_atol": float(temporal_map_max_abs_atol),
    }
    if spatial_range <= plateau_range_atol:
        status.update(
            {
                "candidate_status": "withheld_spatial_plateau",
                "reason": "prediction spatial range is at or below plateau tolerance; no meaningful location can be ranked",
                "candidate_count": 0,
                "candidates": [],
            }
        )
        return status
    if top_score_tie_fraction > top_score_tie_fraction_max:
        status.update(
            {
                "candidate_status": "withheld_top_score_plateau",
                "reason": "top-score tie occupies too much of the model map; no unique location can be ranked",
                "candidate_count": 0,
                "candidates": [],
            }
        )
        return status
    if temporally_invariant:
        status.update(
            {
                "candidate_status": "withheld_temporally_invariant_map",
                "reason": "consecutive endpoint prediction maps are numerically invariant",
                "candidate_count": 0,
                "candidates": [],
            }
        )
        return status
    pooled = F.max_pool2d(candidate_map[None, None], kernel_size=kernel_size, stride=1, padding=kernel_size // 2)[0, 0]
    maxima = candidate_map == pooled
    finite_maxima = maxima & torch.isfinite(candidate_map)
    width = candidate_map.shape[1]
    geometry_rejected_local_maximum_count = 0
    geometry_allowed_local_maximum_count: int | None = None
    geometry_part_by_flat_index: dict[int, str] = {}
    if geometry_gate is not None:
        for flat_index in torch.nonzero(finite_maxima.flatten(), as_tuple=False).flatten().tolist():
            y_model, x_model = divmod(int(flat_index), int(width))
            x_raw, y_raw = raw_coordinate_from_model_index(x_model, y_model, metadata)
            part_name = geometry_gate.containing_part(x_raw, y_raw)
            if part_name is None:
                finite_maxima.flatten()[flat_index] = False
                geometry_rejected_local_maximum_count += 1
            else:
                geometry_part_by_flat_index[int(flat_index)] = part_name
        geometry_allowed_local_maximum_count = int(finite_maxima.sum().item())
        status.update(
            {
                "provisional_part_geometry_gate": geometry_gate.metadata(),
                "geometry_rejected_local_maximum_count": geometry_rejected_local_maximum_count,
                "geometry_allowed_local_maximum_count": geometry_allowed_local_maximum_count,
            }
        )
    scores = candidate_map.masked_fill(~finite_maxima, -torch.inf).flatten()
    count = min(top_k, int(torch.isfinite(scores).sum().item()))
    if count == 0:
        status.update(
            {
                "candidate_status": "withheld_outside_provisional_part_geometry" if geometry_gate is not None else "withheld_no_local_maximum",
                "reason": (
                    "no finite local maximum remained inside the configured provisional part rectangles"
                    if geometry_gate is not None
                    else "no finite local maximum was found"
                ),
                "candidate_count": 0,
                "candidates": [],
            }
        )
        return status
    values, flat_indices = torch.topk(scores, k=count)
    candidates: list[dict[str, Any]] = []
    for rank, (score, flat_index) in enumerate(zip(values.tolist(), flat_indices.tolist()), start=1):
        y_model, x_model = divmod(int(flat_index), int(width))
        x_raw, y_raw = raw_coordinate_from_model_index(x_model, y_model, metadata)
        candidate_record: dict[str, Any] = {
            "rank": rank,
            "x_pixel": x_raw,
            "y_pixel": y_raw,
            "x_model_pixel": x_model,
            "y_model_pixel": y_model,
            "layer_z": int(sample["endpoint_layer_z"]),
            "score": float(score),
            "score_semantics": "sigmoid-scaled XCT-derived continuous quality candidate; direction unresolved",
            "stage": "A",
            "sample_id": str(metadata["sample_id"]),
        }
        if geometry_gate is not None:
            candidate_record["provisional_geometry_containing_part"] = geometry_part_by_flat_index.get(int(flat_index))
        candidates.append(candidate_record)
    status.update({"candidate_status": "emitted", "reason": None, "candidate_count": len(candidates), "candidates": candidates})
    return status
def evaluate_test_candidates(
    model: nn.Module,
    dataset: AMMTWeakTargetDataset,
    device: torch.device,
    top_k: int,
    kernel_size: int,
    plateau_range_atol: float,
    top_score_tie_atol: float,
    top_score_tie_fraction_max: float,
    temporal_map_mae_atol: float,
    temporal_map_max_abs_atol: float,
    max_samples: int | None,
    geometry_gate: ProvisionalPartGeometryGate | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    candidates: list[dict[str, Any]] = []
    endpoint_statuses: list[dict[str, Any]] = []
    previous_prediction: Tensor | None = None
    with torch.no_grad():
        sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for index in range(sample_count):
            sample = dataset[index]
            history = sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)
            prediction = model(history).cpu()
            result = local_maximum_candidates(
                prediction,
                sample,
                top_k=top_k,
                kernel_size=kernel_size,
                plateau_range_atol=plateau_range_atol,
                top_score_tie_atol=top_score_tie_atol,
                top_score_tie_fraction_max=top_score_tie_fraction_max,
                previous_prediction=previous_prediction,
                temporal_map_mae_atol=temporal_map_mae_atol,
                temporal_map_max_abs_atol=temporal_map_max_abs_atol,
                geometry_gate=geometry_gate,
            )
            candidates.extend(result.pop("candidates"))
            endpoint_statuses.append(result)
            previous_prediction = prediction.clone()
    return candidates, endpoint_statuses
def prepare_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Output directory already exists: {path}. Review it and choose a new --output-dir; this script never overwrites it."
        )
    path.mkdir(parents=True, exist_ok=False)
def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
def dry_run(model: nn.Module, dataset: AMMTWeakTargetDataset, loss_fn: SupportMaskedSmoothL1Loss, device: torch.device, sample_index: int) -> None:
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"--dry-run-index must be 0..{len(dataset) - 1}")
    sample = dataset[sample_index]
    model.eval()
    with torch.no_grad():
        prediction = model(sample["model_input_history"].unsqueeze(0).to(device))
        result = loss_fn(
            prediction,
            sample["weak_response"].unsqueeze(0).to(device),
            sample["weak_support_mask"].unsqueeze(0).to(device),
        )
    print(
        json.dumps(
            {
                "audit_type": "A-only baseline dry run; not model training",
                "input_shape": list(sample["model_input_history"].unsqueeze(0).shape),
                "prediction_shape": list(prediction.shape),
                "prediction_min": float(prediction.min().item()),
                "prediction_max": float(prediction.max().item()),
                "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                "weak_target_available": bool(sample["weak_target_available"]),
                "supervised_pixel_count": result.supervised_pixel_count,
                "masked_loss": float(result.loss.item()),
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("A-only baseline dry run complete. No training, checkpoint, dense heatmap, or output file was written.")
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
def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = config["data"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    storage_config = config["storage"]
    if data_config["stage"] != "A":
        raise ValueError("This script is A-only; config data.stage must be 'A'.")
    if int(data_config["input_channels"]) != 6:
        raise ValueError("A-only baseline requires six input channels: 3 intensity + 3 validity masks.")
    if model_config["output_activation"] != "sigmoid":
        raise ValueError("This baseline requires sigmoid output for the [0,1] continuous response contract.")
    loss_config = load_yaml(args.loss_config)
    objective = loss_config["objective"]
    if objective["name"] != "support_masked_smooth_l1":
        raise ValueError("loss config objective.name must be support_masked_smooth_l1.")
    set_seed(int(training_config["seed"]))
    device = choose_device(args.device or str(training_config["device"]))
    model = AOnlyCausalCandidateNet(
        input_channels=int(data_config["input_channels"]),
        base_channels=int(model_config["base_channels"]),
        temporal_kernel_size=int(model_config["temporal_kernel_size"]),
        use_endpoint_feature_residual=bool(model_config.get("use_endpoint_feature_residual", False)),
    ).to(device)
    loss_fn = SupportMaskedSmoothL1Loss(beta=float(objective["beta"]))
    train_dataset = make_dataset(args, split="train")
    if args.dry_run:
        dry_run(model, train_dataset, loss_fn, device=device, sample_index=args.dry_run_index)
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
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
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
            gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
            max_batches=args.max_train_batches,
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
    test_metrics = run_epoch(
        model, test_loader, loss_fn, device, best_epoch, "test", None,
        gradient_clip_norm=0.0, max_batches=args.max_eval_batches,
    )
    geometry_gate = build_provisional_part_geometry_gate(evaluation_config, args.calibration_config)
    candidates, candidate_endpoint_statuses = evaluate_test_candidates(
        model,
        test_dataset,
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
    candidate_status_counts: dict[str, int] = {}
    for status in candidate_endpoint_statuses:
        name = str(status["candidate_status"])
        candidate_status_counts[name] = candidate_status_counts.get(name, 0) + 1
    write_json(
        output_dir / str(storage_config["history_name"]),
        {
            "purpose": "A-only baseline training history; continuous XCT-derived response, not defect classification",
            "best_epoch": best_epoch,
            "best_validation_supported_pixel_mean_smooth_l1": best_validation_loss,
            "history": history,
        },
    )
    write_json(
        output_dir / str(storage_config["test_metrics_name"]),
        {
            "purpose": "Held-out causal test metric on sparse XCT-supported pixels only",
            "best_checkpoint_epoch": best_epoch,
            "test": asdict(test_metrics),
            "response_direction": "unresolved",
            "interpretation": "XCT-derived continuous quality candidate; not anomaly probability or confirmed defect",
        },
    )
    write_json(
        output_dir / str(storage_config["test_candidates_name"]),
        {
            "purpose": "Compact coordinate candidates, not dense heatmaps or confirmed defects",
            "stage": "A",
            "response_direction": "unresolved",
            "candidate_count": len(candidates),
            "top_k_per_endpoint": int(evaluation_config["top_k_candidates_per_endpoint"]),
            "spatial_plateau_range_atol": float(evaluation_config["spatial_plateau_range_atol"]),
            "top_score_tie_atol": float(evaluation_config["top_score_tie_atol"]),
            "top_score_tie_fraction_max": float(evaluation_config["top_score_tie_fraction_max"]),
            "temporal_map_mae_atol": float(evaluation_config["temporal_map_mae_atol"]),
            "temporal_map_max_abs_atol": float(evaluation_config["temporal_map_max_abs_atol"]),
            "provisional_part_geometry_gate": {"enabled": False} if geometry_gate is None else geometry_gate.metadata(),
            "endpoint_status_counts": candidate_status_counts,
            "endpoint_statuses": candidate_endpoint_statuses,
            "candidates": candidates,
        },
    )
    print(
        json.dumps(
            {
                "training_complete": True,
                "device": str(device),
                "best_epoch": best_epoch,
                "best_validation_supported_pixel_mean_smooth_l1": best_validation_loss,
                "test": asdict(test_metrics),
                "output_directory": str(output_dir),
                "checkpoint": str(checkpoint_path),
                "candidate_count": len(candidates),
                "candidate_endpoint_status_counts": candidate_status_counts,
                "response_direction": "unresolved",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("A-only baseline complete. Raw files were read-only; dense heatmaps were not persisted.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
