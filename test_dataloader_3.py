import argparse
from pathlib import Path
from src.train_a_only_baseline import load_yaml, make_dataset
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

for split in ["train", "validation", "test"]:
    ds = make_dataset(args, split)
    supp_cnt = sum(1 for i in range(len(ds)) if ds[i]["weak_target_available"])
    print(f"{split} samples with target: {supp_cnt} out of {len(ds)}")
