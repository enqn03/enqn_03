utf-8
#!/usr/bin/env python3
"""Read-only mechanism audit for temporal-path use in an A-only residual checkpoint.
The audit quantifies Conv3D lag-wise kernel energy and compares the full residual
prediction with endpoint-only and temporal-update-only branch inputs. It uses
only read-only A-stage causal histories and an existing checkpoint. It neither
trains nor reads XCT/weak targets, changes calibration, or saves dense arrays.
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
from torch.nn import functional as F
from ammt_causal_dataset import AMMTCausalStageDataset
from train_a_only_baseline import AOnlyCausalCandidateNet, choose_device, load_yaml
NUMERICAL_ZERO_ATOL = 1.0e-7
MATERIAL_BRANCH_MAP_MAE_MIN = 1.0e-4
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
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def tensor_stats(values: Tensor) -> dict[str, float]:
    tensor = values.detach().float().cpu()
    return {
        "mean": float(tensor.mean().item()),
        "std_population": float(tensor.std(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(tensor).item()),
        "mean_abs": float(tensor.abs().mean().item()),
        "max_abs": float(tensor.abs().max().item()),
    }
def map_difference(left: Tensor, right: Tensor) -> dict[str, float]:
    difference = (left - right).detach().float().cpu()
    return {
        "mae": float(difference.abs().mean().item()),
        "max_abs": float(difference.abs().max().item()),
        "l2_norm": float(torch.linalg.vector_norm(difference).item()),
    }
def map_pearson(left: Tensor, right: Tensor) -> float | None:
    x = left.detach().float().reshape(-1).cpu()
    y = right.detach().float().reshape(-1).cpu()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denominator.item()) <= 0.0:
        return None
    return float(torch.dot(x_centered, y_centered).item() / denominator.item())
def compute_branches(model: AOnlyCausalCandidateNet, history: Tensor) -> dict[str, Tensor]:
    """Replicate the frozen model forward pass while retaining its two residual inputs."""
    if history.ndim != 5 or history.shape[0] != 1:
        raise ValueError(f"Expected one history [1,K,C,H,W], got {tuple(history.shape)}")
    batch, steps, channels, height, width = history.shape
    encoded = model.frame_encoder(history.reshape(batch * steps, channels, height, width))
    encoded_history = encoded.reshape(batch, steps, encoded.shape[1], height, width)
    encoded_endpoint = encoded_history[:, -1]
    encoded_for_temporal = encoded_history.permute(0, 2, 1, 3, 4)
    padded = F.pad(encoded_for_temporal, (0, 0, 0, 0, model.temporal_kernel_size - 1, 0))
    temporal_update = F.silu(model.temporal_norm(model.temporal(padded)))[:, :, -1]
    full_decoder_input = encoded_endpoint + temporal_update
    return {
        "encoded_endpoint": encoded_endpoint,
        "temporal_update": temporal_update,
        "full_decoder_input": full_decoder_input,
        "full_prediction": torch.sigmoid(model.decoder(full_decoder_input)),
        "endpoint_only_prediction": torch.sigmoid(model.decoder(encoded_endpoint)),
        "temporal_update_only_prediction": torch.sigmoid(model.decoder(temporal_update)),
    }
def endpoint_repeated(history: Tensor) -> Tensor:
    return history[:, -1:].repeat(1, int(history.shape[1]), 1, 1, 1)
def temporal_kernel_rows(model: AOnlyCausalCandidateNet, sequence_length: int) -> list[dict[str, Any]]:
    weight = model.temporal.weight.detach().float().cpu()
    if weight.ndim != 5:
        raise ValueError(f"Expected Conv3D weight [out,in,time,H,W], got {tuple(weight.shape)}")
    kernel_steps = int(weight.shape[2])
    if kernel_steps != int(model.temporal_kernel_size):
        raise ValueError("Conv3D time dimension does not match declared temporal kernel size.")
    total_energy = float(weight.square().sum().item())
    rows: list[dict[str, Any]] = []
    for time_index in range(kernel_steps):
        lag_from_endpoint = kernel_steps - 1 - time_index
        energy = float(weight[:, :, time_index, :, :].square().sum().item())
        rows.append(
            {
                "kernel_time_index": time_index,
                "lag_from_endpoint_layers": lag_from_endpoint,
                "history_position_from_oldest_within_K": sequence_length - 1 - lag_from_endpoint,
                "is_endpoint_lag": lag_from_endpoint == 0,
                "frobenius_energy": energy,
                "energy_fraction": 0.0 if total_energy <= 0.0 else energy / total_energy,
                "effective_at_final_output": True,
                "note": "The K=4 earliest history frame at lag 3 is outside a final Conv3D kernel of temporal size 3; it has no direct path to the selected final temporal update.",
            }
        )
    return rows
def plot_qc(
    branches: dict[str, Tensor],
    repeated_branches: dict[str, Tensor],
    endpoint_layer: int,
    output_path: Path,
) -> None:
    full = branches["full_prediction"][0, 0].detach().float().cpu()
    endpoint_only = branches["endpoint_only_prediction"][0, 0].detach().float().cpu()
    temporal_only = branches["temporal_update_only_prediction"][0, 0].detach().float().cpu()
    full_minus_endpoint = (full - endpoint_only).abs()
    update_change = (branches["temporal_update"] - repeated_branches["temporal_update"]).abs().mean(dim=1)[0].detach().float().cpu()
    signal_values = torch.cat([full.reshape(-1), endpoint_only.reshape(-1), temporal_only.reshape(-1)])
    signal_low = float(torch.quantile(signal_values, 0.01).item())
    signal_high = float(torch.quantile(signal_values, 0.99).item())
    if signal_high <= signal_low:
        signal_high = signal_low + NUMERICAL_ZERO_ATOL
    diff_high = max(float(full_minus_endpoint.max().item()), float(update_change.max().item()), NUMERICAL_ZERO_ATOL)
    panels: list[tuple[str, Tensor, str, float, float]] = [
        ("full residual prediction", full, "magma", signal_low, signal_high),
        ("endpoint-only branch prediction", endpoint_only, "magma", signal_low, signal_high),
        ("temporal-update-only branch prediction", temporal_only, "magma", signal_low, signal_high),
        ("abs(full - endpoint-only)", full_minus_endpoint, "viridis", 0.0, diff_high),
        ("mean abs temporal-update\nchange under endpoint repeat", update_change, "viridis", 0.0, diff_high),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(3.15 * len(panels), 3.3), dpi=150)
    for axis, (title, values, cmap, lower, upper) in zip(axes, panels, strict=True):
        axis.imshow(values.numpy(), cmap=cmap, origin="upper", interpolation="nearest", vmin=lower, vmax=upper)
        axis.set_title(title, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"A-only temporal-path mechanism QC: endpoint z={endpoint_layer}\n"
        "Display-only deterministic branch diagnostic; no dense prediction array persisted",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
def main() -> None:
    args = parse_args()
    if not 1 <= len(args.indices) <= MAX_SELECTED_ENDPOINTS:
        raise ValueError(f"--indices requires 1..{MAX_SELECTED_ENDPOINTS} unique values.")
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("--indices must not contain duplicates.")
    for path, label in ((args.config, "config"), (args.checkpoint, "checkpoint"), (args.tiff_a, "A-stage TIFF"), (args.manifest, "causal manifest"), (args.normalization_config, "normalization config")):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    config = load_yaml(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    if data_config.get("stage") != "A" or int(data_config.get("input_channels", -1)) != 6:
        raise ValueError("Audit requires the six-channel A-only residual config.")
    if not bool(model_config.get("use_endpoint_feature_residual", False)):
        raise ValueError("Audit requires an endpoint-feature-residual checkpoint/config.")
    if int(data_config["sequence_length_k"]) < int(model_config["temporal_kernel_size"]):
        raise ValueError("Sequence length must be at least the temporal kernel size.")
    prepare_output_directory(args.output_dir)
    device = choose_device(args.device or str(training_config["device"]))
    model = AOnlyCausalCandidateNet(
        input_channels=int(data_config["input_channels"]),
        base_channels=int(model_config["base_channels"]),
        temporal_kernel_size=int(model_config["temporal_kernel_size"]),
        use_endpoint_feature_residual=True,
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
    kernel_rows = temporal_kernel_rows(model, int(data_config["sequence_length_k"]))
    endpoint_rows: list[dict[str, Any]] = []
    qc_paths: list[str] = []
    with torch.no_grad():
        for dataset_index in args.indices:
            if not 0 <= dataset_index < len(dataset):
                raise IndexError(f"--indices value {dataset_index} is outside split={args.split} range 0..{len(dataset) - 1}")
            sample = dataset[dataset_index]
            history = sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)
            branches = compute_branches(model, history)
            repeated_branches = compute_branches(model, endpoint_repeated(history))
            full_vs_endpoint = map_difference(branches["full_prediction"], branches["endpoint_only_prediction"])
            full_vs_temporal = map_difference(branches["full_prediction"], branches["temporal_update_only_prediction"])
            temporal_update_vs_repeated = map_difference(branches["temporal_update"], repeated_branches["temporal_update"])
            full_vs_repeated = map_difference(branches["full_prediction"], repeated_branches["full_prediction"])
            encoded_endpoint_vs_repeated = map_difference(branches["encoded_endpoint"], repeated_branches["encoded_endpoint"])
            endpoint_stats = tensor_stats(branches["encoded_endpoint"])
            update_stats = tensor_stats(branches["temporal_update"])
            final_stats = tensor_stats(branches["full_decoder_input"])
            update_l2_ratio = 0.0 if endpoint_stats["l2_norm"] <= 0.0 else update_stats["l2_norm"] / endpoint_stats["l2_norm"]
            endpoint_rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(sample["metadata"]["sample_id"]),
                    "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                    "history_layer_z": ";".join(str(int(value)) for value in sample["history_layer_z"].tolist()),
                    "effective_temporal_history_layers": ";".join(str(int(value)) for value in sample["history_layer_z"].tolist()[-int(model.temporal_kernel_size):]),
                    "earliest_K_frame_has_direct_final_temporal_path": False,
                    "encoded_endpoint_l2_norm": endpoint_stats["l2_norm"],
                    "encoded_endpoint_std_population": endpoint_stats["std_population"],
                    "temporal_update_l2_norm": update_stats["l2_norm"],
                    "temporal_update_std_population": update_stats["std_population"],
                    "temporal_update_to_endpoint_l2_ratio": update_l2_ratio,
                    "full_decoder_input_l2_norm": final_stats["l2_norm"],
                    "full_vs_endpoint_only_map_mae": full_vs_endpoint["mae"],
                    "full_vs_endpoint_only_map_max_abs": full_vs_endpoint["max_abs"],
                    "full_vs_temporal_only_map_mae": full_vs_temporal["mae"],
                    "full_vs_temporal_only_map_max_abs": full_vs_temporal["max_abs"],
                    "full_vs_endpoint_only_map_pearson": map_pearson(branches["full_prediction"], branches["endpoint_only_prediction"]),
                    "temporal_update_vs_endpoint_repeated_mae": temporal_update_vs_repeated["mae"],
                    "temporal_update_vs_endpoint_repeated_max_abs": temporal_update_vs_repeated["max_abs"],
                    "full_prediction_vs_endpoint_repeated_mae": full_vs_repeated["mae"],
                    "full_prediction_vs_endpoint_repeated_max_abs": full_vs_repeated["max_abs"],
                    "encoded_endpoint_vs_endpoint_repeated_mae": encoded_endpoint_vs_repeated["mae"],
                    "temporal_update_changes_when_prior_frames_replaced": bool(temporal_update_vs_repeated["max_abs"] > NUMERICAL_ZERO_ATOL),
                    "temporal_branch_materially_changes_prediction_vs_endpoint_only": bool(full_vs_endpoint["mae"] >= MATERIAL_BRANCH_MAP_MAE_MIN),
                    "full_prediction_changes_when_prior_frames_replaced": bool(full_vs_repeated["max_abs"] > NUMERICAL_ZERO_ATOL),
                    "numerical_zero_atol": NUMERICAL_ZERO_ATOL,
                    "material_branch_map_mae_min": MATERIAL_BRANCH_MAP_MAE_MIN,
                }
            )
            qc_path = args.output_dir / f"temporal_path_mechanism_endpoint_z{int(sample['endpoint_layer_z']):03d}.png"
            plot_qc(branches, repeated_branches, int(sample["endpoint_layer_z"]), qc_path)
            qc_paths.append(str(qc_path))
    kernel_csv = args.output_dir / "a_only_temporal_path_kernel_energy.csv"
    endpoints_csv = args.output_dir / "a_only_temporal_path_mechanism_by_endpoint.csv"
    summary_json = args.output_dir / "a_only_temporal_path_mechanism_summary.json"
    write_csv(kernel_csv, kernel_rows)
    write_csv(endpoints_csv, endpoint_rows)
    past_energy_fraction = sum(float(row["energy_fraction"]) for row in kernel_rows if int(row["lag_from_endpoint_layers"]) > 0)
    update_changed_count = sum(bool(row["temporal_update_changes_when_prior_frames_replaced"]) for row in endpoint_rows)
    branch_effect_count = sum(bool(row["temporal_branch_materially_changes_prediction_vs_endpoint_only"]) for row in endpoint_rows)
    full_changed_count = sum(bool(row["full_prediction_changes_when_prior_frames_replaced"]) for row in endpoint_rows)
    summary = {
        "audit_type": "read-only A-only temporal-path mechanism audit; not training or defect classification",
        "purpose": "Separate structural receptive-field limits, Conv3D lag-wise parameter energy, temporal-update activation sensitivity, and residual branch effect before any architecture change.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "selected_dataset_indices": [int(value) for value in args.indices],
        "device": str(device),
        "architecture_contract": {
            "sequence_length_k": int(data_config["sequence_length_k"]),
            "temporal_kernel_size": int(model.temporal_kernel_size),
            "residual_equation": "temporal_final = encoded_endpoint + temporal_update",
            "effective_final_temporal_receptive_field": f"endpoint and the {int(model.temporal_kernel_size) - 1} immediately preceding encoded frames",
            "structural_limit": "For K=4 and a final causal Conv3D kernel of size 3, oldest history position t0/layer z-3 has no direct final temporal-update path. Only t1/t2 and endpoint can affect the selected temporal Conv3D output.",
        },
        "fixed_interpretation_thresholds": {
            "numerical_zero_atol": NUMERICAL_ZERO_ATOL,
            "material_branch_map_mae_min": MATERIAL_BRANCH_MAP_MAE_MIN,
            "rule": "These thresholds classify numerical branch sensitivity only; they never authorize retraining, architecture selection, calibration changes, or physical candidate interpretation.",
        },
        "aggregate": {
            "past_lag_kernel_energy_fraction": past_energy_fraction,
            "selected_endpoint_count": len(endpoint_rows),
            "temporal_update_changes_under_endpoint_repeat_count": update_changed_count,
            "full_prediction_changes_under_endpoint_repeat_count": full_changed_count,
            "temporal_branch_materially_changes_prediction_vs_endpoint_only_count": branch_effect_count,
            "interpretation": "Kernel energy, activation sensitivity, and decoder-input branch effect are distinct. Inspect all three before deciding whether the non-use result arises from structural receptive field, endpoint-only temporal convolution, weak temporal update, or downstream residual fusion.",
        },
        "score_semantics": "sigmoid-scaled XCT-derived continuous quality candidate; direction unresolved; not a defect/anomaly probability",
        "image_artifacts": {
            "purpose": "Each QC PNG is a display-only deterministic visualization of in-memory branch maps for one selected endpoint.",
            "panels_left_to_right": [
                "full residual prediction used by the frozen checkpoint",
                "counterfactual decoder output when only encoded_endpoint is passed to the same decoder",
                "counterfactual decoder output when only temporal_update is passed to the same decoder",
                "absolute difference between full and endpoint-only predictions, showing temporal branch effect at decoder output",
                "per-pixel channel-mean absolute change in temporal_update after all prior frames are replaced by endpoint",
            ],
            "limits": "PNG colors are visualization aids only; they are not saved model tensors, targets, defect maps, anomaly probabilities, or physical location evidence.",
        },
        "prohibitions": [
            "Does not train, modify, or save a checkpoint, optimizer, config, manifest, raw TIFF, CSV, or calibration artifact.",
            "Uses A-stage TIFF only through read-only memmap via AMMTCausalStageDataset.",
            "Does not open registered XCT data or create/read weak response/support; no XCT support enters any decoder calculation.",
            "Does not apply provisional machine/part geometry, choose a calibration rank/orientation, or change calibration holds.",
            "Persists only compact metrics and three display-only deterministic QC PNGs; no dense feature/map/target arrays are saved.",
        ],
        "outputs": {
            "kernel_energy_csv": str(kernel_csv),
            "endpoint_mechanism_csv": str(endpoints_csv),
            "summary_json": str(summary_json),
            "endpoint_qc_pngs": qc_paths,
        },
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Temporal-path mechanism audit complete. No raw data, XCT/weak target, checkpoint, config, calibration, or candidate policy was modified.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
