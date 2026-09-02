#!/bin/bash
set -e

echo "[Ablation] Starting A-only BCE V7 training..."
/usr/local/bin/python3 src/train_a_only.py \
  --config configs/a_only_bce_c32_v7_tuning.yaml \
  --loss-config configs/masked_bce_loss_v2.yaml \
  --tiff raw_original/layer_camera/LayerCameraAfterSpreading.tif \
  --manifest manifests/causal_sequence_manifest.csv \
  --normalization-config configs/normalization_v1.yaml \
  --calibration-config configs/calibration_v1.yaml \
  --weak-target-config configs/weak_target_v1.yaml \
  --registered-root raw_original/registered_xct \
  --device mps

echo "[Ablation] Starting B-only BCE V7 training..."
/usr/local/bin/python3 src/train_b_only.py \
  --config configs/b_only_bce_c32_v7_tuning.yaml \
  --loss-config configs/masked_bce_loss_v2.yaml \
  --tiff raw_original/layer_camera/LayerCameraBurned.tif \
  --manifest manifests/causal_sequence_manifest.csv \
  --normalization-config configs/normalization_v1.yaml \
  --calibration-config configs/calibration_v1.yaml \
  --weak-target-config configs/weak_target_v1.yaml \
  --registered-root raw_original/registered_xct \
  --device mps

echo "[Ablation] All ablation trainings completed!"
