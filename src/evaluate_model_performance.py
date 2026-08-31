#!/usr/bin/env python3
"""
Quantitative Model Performance Validation Script

This script evaluates the CBAM Fusion V6 model on the test dataset.
It calculates pixel-level AUROC, AUPRC, Precision, Recall, and F1-score
using the weak_target tensor as pseudo ground truth, considering only
the pixels within the weak_support_mask.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from tqdm import tqdm

from train_a_b_cbam_fusion_bce_v6 import (
    ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet,
    make_fusion_dataset,
    load_yaml
)

def calculate_blob_metrics(binary_target, binary_pred, tolerance=2):
    """
    Blob-level (Object-level) Evaluation with spatial tolerance.
    """
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
    # Dataset args
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
    weak_target_config = load_yaml(args.weak_target_config)
    gt_threshold = float(weak_target_config["response"]["binary_defect_threshold"])
    print(f"[시스템] Ground Truth 결함 판정 임계값: {gt_threshold}")
    
    # 모델 로드
    model = ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet(
        input_channels=int(config["data"]["input_channels"]),
        base_channels=int(config["model"]["base_channels"])
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # 데이터셋 로드
    print("[시스템] Test 데이터셋 로드 중...")
    dataset = make_fusion_dataset(args, split="test")
    
    all_targets = []
    all_preds = []
    
    total_gt_blobs = 0
    total_pred_blobs = 0
    total_hits = 0
    total_valid_alarms = 0
    
    print("[시스템] 추론 및 예측 확률 수집 중...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            hist_a = sample["model_input_history_a"].unsqueeze(0).to(device)
            hist_b = sample["model_input_history_b"].unsqueeze(0).to(device)
            
            mask = sample["weak_support_mask"].squeeze().cpu().numpy()
            target = sample["weak_response"].squeeze().cpu().numpy()
            
            probs = model(hist_a, hist_b)
            probs = probs[0, 0].cpu().numpy()
            
            # mask가 1인 (부품 내부) 픽셀만 1D 배열로 추출
            valid_idx = (mask > 0)
            
            # 정답지 이진화
            binary_target = (target > gt_threshold).astype(int)
            binary_pred = (probs >= 0.85).astype(int)
            
            # Blob-level Metrics accumulation
            b_gt_num, b_pred_num, b_hits, b_valid_alarms = calculate_blob_metrics(binary_target * valid_idx, binary_pred * valid_idx, tolerance=2)
            total_gt_blobs += b_gt_num
            total_pred_blobs += b_pred_num
            total_hits += b_hits
            total_valid_alarms += b_valid_alarms
            
            # Pixel-level array accumulation (Without artificial dilation to show raw pixel performance vs blob performance)
            valid_probs = probs[valid_idx]
            
            all_targets.append(binary_target[valid_idx])
            all_preds.append(valid_probs)
            
    # 전체 데이터를 하나의 1D 배열로 병합
    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_preds)
    
    print("\n[시스템] 지표(Metrics) 계산 중...")
    
    # 1. AUROC 계산
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # 2. AUPRC 계산
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    prc_auc = average_precision_score(y_true, y_score)
    
    # 3. 임계값 별 Precision, Recall, F1
    thresholds = [0.5, 0.85, 0.95]
    
    print("\n==================================================")
    print(" 📊 정량적 성능 검증 결과 (Quantitative Evaluation)")
    print("==================================================")
    print(f"Total Valid Pixels: {len(y_true):,}")
    print(f"Defect Pixels (True): {y_true.sum():,} ({(y_true.sum()/len(y_true)*100) if len(y_true)>0 else 0:.3f}%)")
    print("--------------------------------------------------")
    print(f"Pixel-level AUROC: {roc_auc:.4f}")
    print(f"Pixel-level AUPRC: {prc_auc:.4f}")
    print("--------------------------------------------------")
    
    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"[Pixel-level] Threshold: {th:.2f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - (TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
        print("--------------------------------------------------")

    print("\n==================================================")
    print(" 🎯 객체 단위 성능 검증 (Blob-level Hit Rate @ 0.85)")
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
        
    # 시각화 저장
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC)')
    axes[0].legend(loc="lower right")
    
    # Precision-Recall Curve
    axes[1].plot(recall_curve, precision_curve, color='green', lw=2, label=f'PRC curve (area = {prc_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve (PRC)')
    axes[1].legend(loc="lower left")
    
    out_path = Path("outputs/roc_prc_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[시스템] ROC 및 PRC 곡선 이미지 저장 완료: {out_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
