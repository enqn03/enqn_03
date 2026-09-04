#!/usr/bin/env python3
"""Audit the physical semantics of the continuous XCT response.
This script extracts small visual patches from A-stage and B-stage images
at locations with extreme XCT scores (high > 0.9, low < 0.1) to determine
whether a high score indicates a defect (e.g., spatter/pore) or normal.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ammt_weak_target_dataset import AMMTFusionWeakTargetDataset
from train_a_only_baseline import load_yaml, set_seed
def extract_patches() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/a_b_fusion_temporal_difference_v1.yaml"))
    parser.add_argument("--tiff-a", type=Path, default=Path("raw_original/layer_camera/LayerCameraAfterSpreading.tif"))
    parser.add_argument("--tiff-b", type=Path, default=Path("raw_original/layer_camera/LayerCameraBurned.tif"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/causal_sequence_manifest.csv"))
    parser.add_argument("--normalization-config", type=Path, default=Path("configs/normalization_v1.yaml"))
    parser.add_argument("--registered-root", type=Path, default=Path("raw_original/registered_xct"))
    parser.add_argument("--calibration-config", type=Path, default=Path("configs/calibration_v1.yaml"))
    parser.add_argument("--weak-target-config", type=Path, default=Path("configs/weak_target_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/target_semantics_v1"))
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    dataset = AMMTFusionWeakTargetDataset(
        tiff_a_path=args.tiff_a,
        tiff_b_path=args.tiff_b,
        manifest_path=args.manifest,
        normalization_config_path=args.normalization_config,
        split="train",
        registered_root=args.registered_root,
        calibration_config=args.calibration_config,
        weak_target_config=args.weak_target_config,
    )
    high_score_candidates = []
    low_score_candidates = []
    print("Scanning dataset for high and low score pixels...")
    for i in range(len(dataset)):
        sample = dataset[i]
        support = sample["weak_support_mask"].bool()
        response = sample["weak_response"]
        valid_response = response[support]
        high_mask = (response > 0.9) & support
        low_mask = (response < 0.1) & support
        high_indices = high_mask.nonzero(as_tuple=False)
        low_indices = low_mask.nonzero(as_tuple=False)
        for idx in high_indices:
            if len(idx) == 3:
                c, y, x = idx.tolist()
            else:
                y, x = idx.tolist()
            high_score_candidates.append({
                "sample_idx": i,
                "y": y,
                "x": x,
                "score": float(response.squeeze()[y, x].item()),
                "z": int(sample["endpoint_layer_z"])
            })
        for idx in low_indices:
            if len(idx) == 3:
                c, y, x = idx.tolist()
            else:
                y, x = idx.tolist()
            low_score_candidates.append({
                "sample_idx": i,
                "y": y,
                "x": x,
                "score": float(response.squeeze()[y, x].item()),
                "z": int(sample["endpoint_layer_z"])
            })
    print(f"Found {len(high_score_candidates)} high-score pixels and {len(low_score_candidates)} low-score pixels.")
    random.shuffle(high_score_candidates)
    random.shuffle(low_score_candidates)
    selected_high = high_score_candidates[:args.num_samples]
    selected_low = low_score_candidates[:args.num_samples]
    fig, axes = plt.subplots(args.num_samples, 4, figsize=(16, 4 * args.num_samples))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    half_p = args.patch_size // 2
    def extract_patch(image_tensor: torch.Tensor, y: int, x: int) -> np.ndarray:
        img = image_tensor[-1, 2, :, :].numpy()
        H, W = img.shape
        y_min = max(0, y - half_p)
        y_max = min(H, y + half_p)
        x_min = max(0, x - half_p)
        x_max = min(W, x + half_p)
        patch = np.zeros((args.patch_size, args.patch_size))
        p_y_min = half_p - (y - y_min)
        p_y_max = half_p + (y_max - y)
        p_x_min = half_p - (x - x_min)
        p_x_max = half_p + (x_max - x)
        patch[p_y_min:p_y_max, p_x_min:p_x_max] = img[y_min:y_max, x_min:x_max]
        return patch
    for row, (h_cand, l_cand) in enumerate(zip(selected_high, selected_low)):
        h_sample = dataset[h_cand["sample_idx"]]
        l_sample = dataset[l_cand["sample_idx"]]
        h_patch_a = extract_patch(h_sample["model_input_history_a"], h_cand["y"], h_cand["x"])
        h_patch_b = extract_patch(h_sample["model_input_history_b"], h_cand["y"], h_cand["x"])
        l_patch_a = extract_patch(l_sample["model_input_history_a"], l_cand["y"], l_cand["x"])
        l_patch_b = extract_patch(l_sample["model_input_history_b"], l_cand["y"], l_cand["x"])
        ax = axes[row, 0]
        ax.imshow(h_patch_a, cmap="gray", vmin=-2, vmax=2)
        ax.set_title(f"High Score A: {h_cand['score']:.2f}")
        ax.axis("off")
        ax = axes[row, 1]
        ax.imshow(h_patch_b, cmap="gray", vmin=-2, vmax=2)
        ax.set_title(f"High Score B: {h_cand['score']:.2f}")
        ax.axis("off")
        ax = axes[row, 2]
        ax.imshow(l_patch_a, cmap="gray", vmin=-2, vmax=2)
        ax.set_title(f"Low Score A: {l_cand['score']:.2f}")
        ax.axis("off")
        ax = axes[row, 3]
        ax.imshow(l_patch_b, cmap="gray", vmin=-2, vmax=2)
        ax.set_title(f"Low Score B: {l_cand['score']:.2f}")
        ax.axis("off")
    output_path = args.output_dir / "target_semantics_patches.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {output_path}")
    summary = {
        "audit_type": "target_semantics_audit",
        "patch_size": args.patch_size,
        "high_score_count": len(high_score_candidates),
        "low_score_count": len(low_score_candidates),
        "selected_high": selected_high,
        "selected_low": selected_low,
    }
    with open(args.output_dir / "target_semantics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
if __name__ == "__main__":
    extract_patches()
