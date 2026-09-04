utf-8
#!/usr/bin/env python3
"""
CBAM XAI (Explainable AI) Audit Script
This script extracts the Spatial Attention Map from the CBAM module
to visualize where the model focuses its attention when predicting anomalies.
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_fusion import (
    ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet,
    make_fusion_dataset,
    load_yaml
)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--layer-idx", type=int, default=242, help="Layer endpoint_z to audit")
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--tiff-b", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[시스템] 장치 초기화: {device}")
    config = load_yaml(args.config)
    model = ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet(
        input_channels=int(config["data"]["input_channels"]),
        base_channels=int(config["model"]["base_channels"])
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    spatial_attention_maps = []
    def spatial_hook(module, inp, out):
        spatial_attention_maps.append(torch.sigmoid(out).detach().cpu())
    model.cbam_fusion.spatial_conv.register_forward_hook(spatial_hook)
    print("[시스템] 데이터셋 로드 중...")
    dataset = make_fusion_dataset(args, split="test")
    sample = None
    for i in range(len(dataset)):
        if dataset[i]["endpoint_layer_z"] == args.layer_idx:
            sample = dataset[i]
            break
    if sample is None:
        print(f"[오류] Layer {args.layer_idx} 를 찾을 수 없습니다.")
        return
    print(f"[시스템] Layer {args.layer_idx} 탐색 완료. XAI 추론 시작...")
    hist_a = sample["model_input_history_a"].unsqueeze(0).to(device)
    hist_b = sample["model_input_history_b"].unsqueeze(0).to(device)
    mask = sample["weak_support_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(hist_a, hist_b)
        probs = probs * mask                   
    prob_map = probs[0, 0].cpu().numpy()
    img_a = hist_a[0, -1, 0].cpu().numpy()
    img_b = hist_b[0, -1, 0].cpu().numpy()
    raw_spatial = spatial_attention_maps[0]
    spatial_up = F.interpolate(raw_spatial, size=(256, 256), mode='bilinear', align_corners=False)
    spatial_map = spatial_up[0, 0].numpy()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax = axes[0]
    ax.imshow(img_a, cmap='gray')
    ax.set_title("Input A (Spreading)")
    ax.axis('off')
    ax = axes[1]
    ax.imshow(img_b, cmap='gray')
    ax.set_title("Input B (Laser)")
    ax.axis('off')
    ax = axes[2]
    im2 = ax.imshow(spatial_map, cmap='jet')
    ax.set_title("CBAM Spatial Attention")
    fig.colorbar(im2, ax=ax, shrink=0.7, label='Attention Weight')
    ax.axis('off')
    ax = axes[3]
    im3 = ax.imshow(prob_map, cmap='YlOrRd', vmin=0.0, vmax=1.0)
    ax.set_title(f"Final Anomaly Probability (Max: {prob_map.max()*100:.1f}%)")
    fig.colorbar(im3, ax=ax, shrink=0.7, label='Probability')
    ax.axis('off')
    plt.tight_layout()
    out_path = Path(f"outputs/cbam_attention_layer{args.layer_idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[시스템] XAI 시각화 완료! 저장 위치: {out_path}")
if __name__ == "__main__":
    main()
