#!/usr/bin/env python3
"""Simulate a real-time live inference stream using the v4 16-channel fusion model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from train_a_b_cbam_fusion_v5 import (
    ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet,
    choose_device,
    load_yaml,
    make_fusion_dataset,
)

def hom(p: np.ndarray) -> np.ndarray:
    return np.c_[p, np.ones(len(p))]

def norm(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = p.mean(0)
    d = np.sqrt(((p - c) ** 2).sum(1)).mean()
    if d <= 0:
        raise ValueError("Degenerate controls")
    s = math.sqrt(2) / d
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.]])
    return (T @ hom(p).T).T[:, :2], T

def Hfit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a, Ts = norm(src)
    b, Td = norm(dst)
    A = []
    for (x, y), (u, v) in zip(a, b):
        A.extend([[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]])
    _, _, V = np.linalg.svd(np.asarray(A))
    H = np.linalg.inv(Td) @ V[-1].reshape(3, 3) @ Ts
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Degenerate homography")
    return H / H[2, 2]

def project(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = (H @ hom(p).T).T
    return q[:, :2] / q[:, 2:3]

def get_machine_corners(part: str) -> np.ndarray:
    RECT = {"part01": (-6., 3., 11., 16.), "part02": (-2., 7., 2., 7.), "part03": (2., 11., -7., -2.), "part04": (6., 15., -16., -11.)}
    xmin, xmax, ymin, ymax = RECT[part]
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], float)

def load_calibration(controls_path: Path, parts_order: list[str]) -> np.ndarray:
    with open(controls_path) as f:
        data = json.load(f)
    
    src_points = []
    dst_points = []
    
    screen_keys = ["screen_part_A_topmost", "screen_part_B", "screen_part_C", "screen_part_D_bottommost"]
    corner_keys = ["screen_top_left", "screen_top_right", "screen_bottom_right", "screen_bottom_left"]
    
    # We apply mirror_rotate_270 directly to the corner indices
    # mirror_rotate_270 -> rev=(0,3,2,1) -> shifted by 3 -> (1,0,3,2) -> actually let's just use the known mapping
    # Actually, we can just use the mapping logic from audit script, but for simplicity, 
    # we know rank 2 with mirror_rotate_270 was selected. 
    # rev=(0,3,2,1), r=3 -> (1,0,3,2)
    corner_mapping = (1, 0, 3, 2)
    
    # Re-organize the control points by screen_part and screen_corner
    part_data = {}
    for pt in data["control_points"]:
        sp = pt["screen_part"]
        sc = pt["screen_corner"]
        if sp not in part_data:
            part_data[sp] = {}
        part_data[sp][sc] = [pt["raw_camera_x_px"], pt["raw_camera_y_px"]]
    
    for screen_key, machine_part in zip(screen_keys, parts_order):
        machine_corners = get_machine_corners(machine_part)
        for i, raw_corner_idx in enumerate(corner_mapping):
            corner_name = corner_keys[raw_corner_idx]
            src_points.append(part_data[screen_key][corner_name])
            dst_points.append(machine_corners[i])
            
    src = np.array(src_points, float)
    dst = np.array(dst_points, float)
    return Hfit(src, dst)

def extract_peaks(probs: torch.Tensor, threshold: float = 0.5, kernel_size: int = 7) -> list[tuple[int, int, float]]:
    # probs shape: [1, 1, 256, 256]
    
    # Find local maxima
    pad = kernel_size // 2
    pooled = F.max_pool2d(probs, kernel_size, stride=1, padding=pad)
    is_max = torch.isclose(probs, pooled, atol=1e-5) & (probs >= threshold)
    
    peaks = []
    if is_max.any():
        indices = torch.nonzero(is_max[0, 0])
        for idx in indices:
            y, x = int(idx[0]), int(idx[1])
            score = float(probs[0, 0, y, x])
            peaks.append((x, y, score))
    return peaks

def pixel_to_machine(x_256: int, y_256: int, H: np.ndarray) -> tuple[float, float, float, float]:
    # 1. 256x256 to 1500x1500 ROI
    x_1500 = x_256 * (1500 / 256.0)
    y_1500 = y_256 * (1500 / 256.0)
    
    # 2. 1500x1500 ROI to 2000x2000 raw camera (Shift by 250)
    x_raw = x_1500 + 250.0
    y_raw = y_1500 + 250.0
    
    # 3. Apply Homography
    pt = np.array([[x_raw, y_raw]])
    machine_pt = project(H, pt)[0]
    return float(machine_pt[0]), float(machine_pt[1]), float(x_raw), float(y_raw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tiff-a", required=True, type=Path)
    parser.add_argument("--tiff-b", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--weak-target-config", required=True, type=Path)
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", type=Path, default=None, help="Path to save the detected coordinates as CSV")
    args = parser.parse_args()
    
    print("[시스템] 파이프라인 초기화 중...")
    
    config = load_yaml(args.config)
    calib_config = load_yaml(args.calibration_config)
    
    device = choose_device("auto")
    
    model = ABCBAMFusionRegularizedCausalTemporalDifferenceCandidateNet(
        input_channels=int(config["data"]["input_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[시스템] 모델 가중치 로드 완료: {args.checkpoint.name}")
    
    # Load calibration
    controls_path = Path("processed/calibration/screen_corner_controls_v2.json")
    parts_order = calib_config["geometry_candidate"]["screen_A_to_D_machine_parts"]
    H = load_calibration(controls_path, parts_order)
    print("[시스템] 캘리브레이션 매트릭스(Homography) 초기화 완료.")
    
    # Simulate stream using the dataset
    print("[시스템] 카메라 스트림 연결 중...")
    dataset = make_fusion_dataset(args, split="test")
    
    print("\n=======================================================")
    print(f" [ AMMT 실시간 이상 탐지 스트림 가동 (임계값: {args.threshold}) ]")
    print("=======================================================\n")
    
    csv_file = None
    csv_writer = None
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["layer_z", "score_percent", "machine_x_mm", "machine_y_mm", "raw_image_x_px", "raw_image_y_px"])
        print(f"[시스템] 감지 결과가 CSV 파일로 저장됩니다: {args.output_csv}\n")
    
    try:
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                layer_z = sample["endpoint_layer_z"]
                
                # Simulate processing time delay
                time.sleep(0.5)
                
                history_a = sample["model_input_history_a"].unsqueeze(0).to(device)
                history_b = sample["model_input_history_b"].unsqueeze(0).to(device)
                
                # Inference
                start_time = time.time()
                probs = model(history_a, history_b)
                probs = probs * sample["weak_support_mask"].unsqueeze(0).to(device)
                inf_time = (time.time() - start_time) * 1000
                
                peaks = extract_peaks(probs, threshold=args.threshold)
                
                if len(peaks) > 0:
                    print(f"[Layer {layer_z:03d}] 🚨 결함 의심 구역 발견! (추론: {inf_time:.1f}ms)")
                    for (px, py, score) in peaks:
                        mx, my, rx, ry = pixel_to_machine(px, py, H)
                        print(f"   ➔ 확률: {score*100:.1f}% | 장비 위치: X={mx:+.2f}mm, Y={my:+.2f}mm | 이미지 좌표: X={rx:.1f}px, Y={ry:.1f}px")
                        if csv_writer:
                            csv_writer.writerow([layer_z, score*100, mx, my, rx, ry])
                    if csv_file:
                        csv_file.flush()
                else:
                    print(f"[Layer {layer_z:03d}] ✅ 정상 (추론: {inf_time:.1f}ms)")
    finally:
        if csv_file:
            csv_file.close()
            
    # 시각화 로직 추가
    if args.output_csv and args.output_csv.exists():
        df = pd.read_csv(args.output_csv)
        if len(df) > 0:
            print("\n[시스템] 감지된 결함 데이터를 바탕으로 분포도를 생성합니다...")
            # Tensor 문자열 제거 및 정수 변환
            df['layer_z'] = df['layer_z'].astype(str).str.extract(r'(\d+)').astype(int)
            
            # 2D 시각화
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                df['machine_x_mm'], df['machine_y_mm'], 
                c=df['score_percent'], cmap='YlOrRd', 
                s=50, alpha=0.8, edgecolors='k', linewidth=0.5
            )
            plt.colorbar(scatter, label='Confidence Score (%)')
            plt.title('AMMT Defect Distribution (Top-Down View)')
            plt.xlabel('Machine X (mm)')
            plt.ylabel('Machine Y (mm)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axis('equal')
            
            out_2d = args.output_csv.parent / f"{args.output_csv.stem}_2d.png"
            plt.savefig(out_2d, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3D 시각화
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter3d = ax.scatter(
                df['machine_x_mm'], df['machine_y_mm'], df['layer_z'], 
                c=df['score_percent'], cmap='YlOrRd', 
                s=50, alpha=0.8, edgecolors='k', linewidth=0.5
            )
            fig.colorbar(scatter3d, label='Confidence Score (%)', shrink=0.7, pad=0.1)
            ax.set_title('AMMT Defect Distribution (3D View)')
            ax.set_xlabel('Machine X (mm)')
            ax.set_ylabel('Machine Y (mm)')
            ax.set_zlabel('Layer (Z)')
            
            out_3d = args.output_csv.parent / f"{args.output_csv.stem}_3d.png"
            plt.savefig(out_3d, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[시스템] 시각화 완료! 저장 위치:")
            print(f"  - 2D 분포도: {out_2d}")
            print(f"  - 3D 분포도: {out_3d}")

if __name__ == "__main__":
    main()
