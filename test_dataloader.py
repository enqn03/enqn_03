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
print(f"Val dataset size: {len(val_ds)}")
val_loader = make_loader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

total_support = 0
for i, batch in enumerate(val_loader):
    supp = batch["weak_support_mask"].sum().item()
    print(f"Batch {i}: supp {supp}")
    total_support += supp
print(f"Total val support: {total_support}")
