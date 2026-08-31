import argparse
from pathlib import Path
from src.train_a_only_baseline import load_yaml, make_dataset, make_loader
import torch

class DummyArgs:
    def __init__(self):
        self.config = Path("configs/a_only_temporal_difference_v1.yaml")
        self.tiff_a = Path("raw_original/layer_camera/LayerCameraAfterSpreading.tif")
        self.manifest = Path("manifests/causal_sequence_manifest.csv")
        self.normalization_config = Path("configs/normalization_v1.yaml")
        self.registered_root = Path("registered_xct_v1")
        self.calibration_config = Path("configs/calibration_v1.yaml")
        self.weak_target_config = Path("configs/weak_target_v1.yaml")
        self.loss_config = Path("configs/masked_regression_loss_v1.yaml")

args = DummyArgs()
val_ds = make_dataset(args, "validation")

supp_cnt = 0
for i in range(len(val_ds)):
    item = val_ds[i]
    if item["weak_target_available"]:
        supp_cnt += 1

print(f"Val samples with target: {supp_cnt} out of {len(val_ds)}")
