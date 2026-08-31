"""
Evaluate single modality (A-only or B-only) model performance.
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from tqdm import tqdm

from train_a_only_bce_v7 import AOnlyCausalTemporalDifferenceCandidateNet, make_dataset as make_a_dataset, load_yaml
from train_b_only_bce_v7 import BOnlyCausalTemporalDifferenceCandidateNet, make_b_dataset

def calculate_blob_metrics(binary_target, binary_pred, tolerance=2):
    gt_labeled, gt_num = label(binary_target)
    pred_labeled, pred_num = label(binary_pred)
    
    struct = np.ones((tolerance*2+1, tolerance*2+1), dtype=int)
    
    hits = 0
    for i in range(1, gt_num + 1):
        gt_blob = (gt_labeled == i)
        gt_blob_dilated = binary_dilation(gt_blob, structure=struct)
        if np.any(gt_blob_dilated & binary_pred):
            hits += 1
            
    valid_alarms = 0
    for i in range(1, pred_num + 1):
        pred_blob = (pred_labeled == i)
        pred_blob_dilated = binary_dilation(pred_blob, structure=struct)
        if np.any(pred_blob_dilated & binary_target):
            valid_alarms += 1
            
    return gt_num, pred_num, hits, valid_alarms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=["A", "B"])
    parser.add_argument("--tiff", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[시스템] 장치 초기화: {device}")

    config = load_yaml(args.config)
    weak_target_config = load_yaml(args.weak_target_config)
    gt_threshold = float(weak_target_config["response"]["binary_defect_threshold"])
    
    if args.modality == "A":
        args.tiff_a = args.tiff  # Alias for train_a_only_baseline
        model = AOnlyCausalTemporalDifferenceCandidateNet(
            input_channels=int(config["data"]["input_channels"]),
            base_channels=int(config["model"]["base_channels"])
        ).to(device)
        dataset = make_a_dataset(args, split="test")
    else:
        args.tiff_b = args.tiff  # Alias for train_b_only_baseline
        model = BOnlyCausalTemporalDifferenceCandidateNet(
            input_channels=int(config["data"]["input_channels"]),
            base_channels=int(config["model"]["base_channels"])
        ).to(device)
        dataset = make_b_dataset(args, split="test")
        
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    all_targets = []
    all_preds = []
    total_gt_blobs = 0
    total_pred_blobs = 0
    total_hits = 0
    total_valid_alarms = 0
    
    print(f"[시스템] 모달리티 {args.modality} 검증 시작...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            hist = sample["model_input_history"].unsqueeze(0).to(device)
            mask = sample["weak_support_mask"].squeeze().cpu().numpy()
            target = sample["weak_response"].squeeze().cpu().numpy()
            
            probs = model(hist)[0, 0].cpu().numpy()
            valid_idx = (mask > 0)
            
            binary_target = (target > gt_threshold).astype(int)
            binary_pred = (probs >= 0.85).astype(int)
            
            b_gt_num, b_pred_num, b_hits, b_valid_alarms = calculate_blob_metrics(binary_target * valid_idx, binary_pred * valid_idx, tolerance=2)
            total_gt_blobs += b_gt_num
            total_pred_blobs += b_pred_num
            total_hits += b_hits
            total_valid_alarms += b_valid_alarms
            
            all_targets.append(binary_target[valid_idx])
            all_preds.append(probs[valid_idx])
            
    print("\n==================================================")
    print(f" 🎯 객체 단위 성능 검증 (Blob-level @ 0.85) - {args.modality} Only")
    print("==================================================")
    
    blob_recall = total_hits / total_gt_blobs if total_gt_blobs > 0 else 0.0
    blob_precision = total_valid_alarms / total_pred_blobs if total_pred_blobs > 0 else 0.0
    
    print(f"총 정답 결함 덩어리 수 (GT Blobs): {total_gt_blobs}")
    print(f"모델이 찾아낸 결함 수 (Hits): {total_hits}")
    print(f"  -> Blob Recall (재현율): {blob_recall*100:.2f}%")
    print()
    print(f"모델이 울린 알람 덩어리 수 (Pred Blobs): {total_pred_blobs}")
    print(f"그 중 진짜 결함 근처였던 알람 수 (Valid Alarms): {total_valid_alarms}")
    print(f"  -> Blob Precision (정밀도): {blob_precision*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
