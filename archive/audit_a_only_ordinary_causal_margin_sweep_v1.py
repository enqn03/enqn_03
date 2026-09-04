utf-8
#!/usr/bin/env python3
"""Read-only top1-top2 margin sensitivity sweep for ordinary causal held-out test maps.
This script evaluates all 48 endpoints in the test split using the temporal-difference 
model's normal causal K=4 history. It calculates the top1-top2 score margins, 
spatial separations, and hypothetical withholding counts at 1%, 2%, and 5% margin 
thresholds to inform future candidate safety design, without altering decoder policy.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
import torch
from ammt_causal_dataset import AMMTCausalStageDataset
from audit_a_only_temporal_difference_stability_v1 import decode
from audit_a_only_temporal_difference_candidate_margin_v1 import score_metrics
from train_a_only_baseline import choose_device, load_yaml
from train_a_only_temporal_difference_v1 import AOnlyCausalTemporalDifferenceCandidateNet
TOP_K = 5
MARGIN_THRESHOLDS = [0.01, 0.02, 0.05]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--split", choices=["test"], default="test")
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
        raise ValueError("Cannot write an empty CSV.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def main() -> None:
    args = parse_args()
    for path, label in ((args.config, "config"), (args.checkpoint, "checkpoint"), (args.tiff_a, "A-stage TIFF"), (args.manifest, "causal manifest"), (args.normalization_config, "normalization config")):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    config = load_yaml(args.config)
    data = config["data"]
    model_config = config["model"]
    evaluation = config["evaluation"]
    training = config["training"]
    if model_config["name"] != "a_only_causal_temporal_difference_candidate_net":
        raise ValueError("Audit requires the temporal-difference experiment config.")
    if int(data["sequence_length_k"]) != 4 or data["stage"] != "A" or int(data["input_channels"]) != 6:
        raise ValueError("Audit requires fixed A-only six-channel K=4 input.")
    if bool(evaluation.get("provisional_part_geometry_gate", {}).get("enabled", False)):
        raise ValueError("This raw-camera audit requires geometry gate disabled.")
    evaluation["top_k_candidates_per_endpoint"] = max(TOP_K, int(evaluation.get("top_k_candidates_per_endpoint", TOP_K)))
    prepare_output_directory(args.output_dir)
    device = choose_device(args.device or str(training["device"]))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = AOnlyCausalTemporalDifferenceCandidateNet(input_channels=6, base_channels=int(model_config["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = AMMTCausalStageDataset(
        stage="A", tiff_path=args.tiff_a, manifest_path=args.manifest, 
        normalization_config_path=args.normalization_config, split=args.split, 
        resize_hw=tuple(int(value) for value in data["model_resolution"])
    )
    rows: list[dict[str, Any]] = []
    withheld_counts = {f"margin_{int(t*100)}pct": 0 for t in MARGIN_THRESHOLDS}
    total_emitted = 0
    with torch.no_grad():
        for dataset_index in range(len(dataset)):
            sample = dataset[dataset_index]
            history = sample["model_input_history"].unsqueeze(0).to(device=device, dtype=torch.float32)
            prediction = model(history).cpu()
            status, candidates = decode(prediction, sample, evaluation)
            spatial_range = float(status["spatial_range"])
            metrics = score_metrics(candidates, spatial_range)
            is_emitted = status["candidate_status"] == "emitted"
            if is_emitted:
                total_emitted += 1
                margin_frac = metrics["top1_top2_margin_fraction_of_spatial_range"]
                if margin_frac is not None:
                    for t in MARGIN_THRESHOLDS:
                        if margin_frac < t:
                            withheld_counts[f"margin_{int(t*100)}pct"] += 1
            rows.append({
                "dataset_index": dataset_index,
                "sample_id": str(sample["metadata"]["sample_id"]),
                "endpoint_layer_z": int(sample["endpoint_layer_z"]),
                "candidate_status": str(status["candidate_status"]),
                "candidate_count": len(candidates),
                "spatial_range": spatial_range,
                "top1_score": metrics["top1_score"],
                "top2_score": metrics["top2_score"],
                "top1_top2_margin": metrics["top1_top2_margin"],
                "top1_top2_margin_fraction_of_spatial_range": metrics["top1_top2_margin_fraction_of_spatial_range"],
                "top1_top2_separation_model_px": metrics["top1_top2_separation_model_px"],
                "top1_top2_separation_raw_px": metrics["top1_top2_separation_raw_px"],
            })
    csv_path = args.output_dir / "a_only_ordinary_causal_margin_sweep_by_endpoint.csv"
    summary_path = args.output_dir / "a_only_ordinary_causal_margin_sweep_summary.json"
    write_csv(csv_path, rows)
    summary = {
        "audit_type": "read-only ordinary causal map margin sensitivity sweep",
        "purpose": "Evaluate distribution of top1-top2 margin on real causal maps to inform future safe withholding thresholds, without changing decoder.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "device": str(device),
        "total_endpoints": len(dataset),
        "total_emitted_endpoints_before_margin_gate": total_emitted,
        "hypothetical_withholding_counts": withheld_counts,
        "hypothetical_withholding_fractions": {
            k: (v / total_emitted if total_emitted > 0 else 0.0)
            for k, v in withheld_counts.items()
        },
        "score_semantics": "sigmoid-scaled XCT-derived continuous quality candidate; not a defect/anomaly probability",
        "prohibitions": [
            "Does not train, modify, or save a checkpoint, config, or decoder policy.",
            "Does not open registered XCT/weak response/support.",
            "Does not apply machine/part geometry or calibration."
        ],
        "outputs": {
            "by_endpoint_csv": str(csv_path),
            "summary_json": str(summary_path)
        }
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("Ordinary causal margin sweep complete. Decoder policy, configs, and raw data are unchanged.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
