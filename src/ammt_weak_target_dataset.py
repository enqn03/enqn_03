#!/usr/bin/env python3
"""On-the-fly sparse XCT weak target wrapper for AMMTCausalStageDataset.

The response is continuous XCT-derived supervision, not a defect label. Pixels
outside sparse Gaussian support remain unknown and are excluded by the support mask.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from ammt_causal_dataset import AMMTCausalStageDataset
from audit_machine_camera_calibration import build_candidates, project

PARTS = ["part01", "part02", "part03", "part04"]

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f: return yaml.safe_load(f)

def gaussian_kernel(sigma: float) -> torch.Tensor:
    r=max(1, int(np.ceil(3*sigma))); y,x=torch.meshgrid(torch.arange(-r,r+1),torch.arange(-r,r+1),indexing="ij")
    return torch.exp(-(x*x+y*y)/(2*sigma*sigma)).float()

class AMMTWeakTargetDataset(Dataset):
    def __init__(self, *, registered_root: str|Path, calibration_config: str|Path, weak_target_config: str|Path, **base_kwargs: Any):
        self.base=AMMTCausalStageDataset(**base_kwargs); self.root=Path(registered_root); self.cal=load_yaml(Path(calibration_config)); self.weak=load_yaml(Path(weak_target_config))
        controls=Path(self.cal["control_points"]["path"]); controls = controls if controls.is_absolute() else Path.cwd()/controls
        points=json.loads(controls.read_text(encoding="utf-8"))["control_points"]
        rank=int(self.cal["geometry_candidate"]["rank"])-1; self.H=build_candidates(points)[rank]["H"]
        self.dx,self.dy=self.cal["local_photometric_refinement"]["raw_pixel_global_offset_xy"]; self.sigma=float(self.weak["rasterization"]["gaussian_sigma_model_px"]); self.kernel=gaussian_kernel(self.sigma)
        scaling=self.weak["response"]["robust_scaling"]
        if scaling["method"] != "train_p01_p99": raise ValueError("Only train_p01_p99 robust scaling is supported.")
        self.response_p01=float(scaling["train_p01"]); self.response_p99=float(scaling["train_p99"])
        if not self.response_p99 > self.response_p01: raise ValueError("weak response train_p99 must exceed train_p01.")
        self.clip_response=bool(scaling["clip_to_unit_interval"])
    def __len__(self): return len(self.base)
    def _target(self, z:int, h:int, w:int, roi:dict[str,int]):
        response=torch.zeros((1,h,w)); support=torch.zeros((1,h,w)); weight=torch.zeros((1,h,w)); valid=False
        sy=h/(roi["y1"]-roi["y0"]); sx=w/(roi["x1"]-roi["x0"]); k=self.kernel; r=k.shape[0]//2
        for part in PARTS:
            p=self.root/part/f"L{z:04d}.csv"
            if not p.is_file(): continue
            a=np.genfromtxt(p,delimiter=","); a=np.atleast_2d(a); good=np.isfinite(a[:,39])
            if not good.any(): continue
            xy=a[good,2:4]; uv=project(self.H,xy); vals=(a[good,39]-self.response_p01)/(self.response_p99-self.response_p01)
            if self.clip_response: vals=np.clip(vals,0.0,1.0)
            for (u,v),value in zip(uv,vals):
                ix=int(round((u+self.dx-roi["x0"])*sx)); iy=int(round((v+self.dy-roi["y0"])*sy))
                if not (0<=ix<w and 0<=iy<h): continue
                y0,y1=max(0,iy-r),min(h,iy+r+1); x0,x1=max(0,ix-r),min(w,ix+r+1); ky0,kx0=y0-(iy-r),x0-(ix-r); patch=k[ky0:ky0+y1-y0,kx0:kx0+x1-x0]
                response[0,y0:y1,x0:x1]+=patch*float(value); weight[0,y0:y1,x0:x1]+=patch; valid=True
        nz=weight>0; response[nz]=response[nz]/weight[nz]; support[nz]=1.0
        return response,support,valid
    def __getitem__(self,index:int):
        sample=self.base[index]; h,w=sample["intensity_history"].shape[-2:]; z=int(sample["endpoint_layer_z"]); roi=sample["metadata"]["working_roi_raw_camera_pixels"]
        response,support,available=self._target(z,h,w,roi); sample.update({"weak_response":response,"weak_support_mask":support,"weak_target_available":torch.tensor(available,dtype=torch.bool),"weak_supervised_pixel_count":torch.tensor(int(support.sum().item()),dtype=torch.int64)})
        return sample

class AMMTFusionWeakTargetDataset(Dataset):
    def __init__(self, *, registered_root: str|Path, calibration_config: str|Path, weak_target_config: str|Path, tiff_a_path: str|Path, tiff_b_path: str|Path, **base_kwargs: Any):
        self.base_a = AMMTCausalStageDataset(stage="A", tiff_path=tiff_a_path, **base_kwargs)
        self.base_b = AMMTCausalStageDataset(stage="B", tiff_path=tiff_b_path, **base_kwargs)
        if len(self.base_a) != len(self.base_b):
            raise ValueError("A and B datasets must have the same length")
        
        self.root=Path(registered_root); self.cal=load_yaml(Path(calibration_config)); self.weak=load_yaml(Path(weak_target_config))
        controls=Path(self.cal["control_points"]["path"]); controls = controls if controls.is_absolute() else Path.cwd()/controls
        points=json.loads(controls.read_text(encoding="utf-8"))["control_points"]
        rank=int(self.cal["geometry_candidate"]["rank"])-1; self.H=build_candidates(points)[rank]["H"]
        self.dx,self.dy=self.cal["local_photometric_refinement"]["raw_pixel_global_offset_xy"]; self.sigma=float(self.weak["rasterization"]["gaussian_sigma_model_px"]); self.kernel=gaussian_kernel(self.sigma)
        scaling=self.weak["response"]["robust_scaling"]
        if scaling["method"] != "train_p01_p99": raise ValueError("Only train_p01_p99 robust scaling is supported.")
        self.response_p01=float(scaling["train_p01"]); self.response_p99=float(scaling["train_p99"])
        if not self.response_p99 > self.response_p01: raise ValueError("weak response train_p99 must exceed train_p01.")
        self.clip_response=bool(scaling["clip_to_unit_interval"])

    def __len__(self): return len(self.base_a)
    
    def _target(self, z:int, h:int, w:int, roi:dict[str,int]):
        response=torch.zeros((1,h,w)); support=torch.zeros((1,h,w)); weight=torch.zeros((1,h,w)); valid=False
        sy=h/(roi["y1"]-roi["y0"]); sx=w/(roi["x1"]-roi["x0"]); k=self.kernel; r=k.shape[0]//2
        for part in PARTS:
            p=self.root/part/f"L{z:04d}.csv"
            if not p.is_file(): continue
            a=np.genfromtxt(p,delimiter=","); a=np.atleast_2d(a); good=np.isfinite(a[:,39])
            if not good.any(): continue
            xy=a[good,2:4]; uv=project(self.H,xy); vals=(a[good,39]-self.response_p01)/(self.response_p99-self.response_p01)
            if self.clip_response: vals=np.clip(vals,0.0,1.0)
            for (u,v),value in zip(uv,vals):
                ix=int(round((u+self.dx-roi["x0"])*sx)); iy=int(round((v+self.dy-roi["y0"])*sy))
                if not (0<=ix<w and 0<=iy<h): continue
                y0,y1=max(0,iy-r),min(h,iy+r+1); x0,x1=max(0,ix-r),min(w,ix+r+1); ky0,kx0=y0-(iy-r),x0-(ix-r); patch=k[ky0:ky0+y1-y0,kx0:kx0+x1-x0]
                response[0,y0:y1,x0:x1]+=patch*float(value); weight[0,y0:y1,x0:x1]+=patch; valid=True
        nz=weight>0; response[nz]=response[nz]/weight[nz]; support[nz]=1.0
        return response,support,valid

    def __getitem__(self,index:int):
        sample_a = self.base_a[index]
        sample_b = self.base_b[index]
        if sample_a["endpoint_layer_z"] != sample_b["endpoint_layer_z"]:
            raise ValueError(f"Mismatch in layer z: A={sample_a['endpoint_layer_z']}, B={sample_b['endpoint_layer_z']}")
        
        h,w=sample_a["intensity_history"].shape[-2:]; z=int(sample_a["endpoint_layer_z"]); roi=sample_a["metadata"]["working_roi_raw_camera_pixels"]
        response,support,available=self._target(z,h,w,roi)
        
        return {
            "model_input_history_a": sample_a["model_input_history"],
            "model_input_history_b": sample_b["model_input_history"],
            "weak_response": response,
            "weak_support_mask": support,
            "weak_target_available": torch.tensor(available,dtype=torch.bool),
            "weak_supervised_pixel_count": torch.tensor(int(support.sum().item()),dtype=torch.int64),
            "endpoint_layer_z": sample_a["endpoint_layer_z"],
            "metadata": sample_a["metadata"]
        }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",required=True,choices=["A","B"]); p.add_argument("--tiff",required=True); p.add_argument("--manifest",required=True); p.add_argument("--normalization-config",required=True); p.add_argument("--registered-root",required=True); p.add_argument("--calibration-config",required=True); p.add_argument("--weak-target-config",required=True); p.add_argument("--split",required=True,choices=["train","validation","test"]); p.add_argument("--index",type=int,default=0); args=p.parse_args()
    ds=AMMTWeakTargetDataset(stage=args.stage,tiff_path=args.tiff,manifest_path=args.manifest,normalization_config_path=args.normalization_config,split=args.split,registered_root=args.registered_root,calibration_config=args.calibration_config,weak_target_config=args.weak_target_config)
    s=ds[args.index]; support=s["weak_support_mask"].bool(); supervised=s["weak_response"][support]
    print(json.dumps({"input_shape":list(s["model_input_history"].shape),"weak_response_shape":list(s["weak_response"].shape),"support_fraction":float(s["weak_support_mask"].mean()),"weak_supervised_pixel_count":int(s["weak_supervised_pixel_count"]),"weak_response_supported_min":None if supervised.numel()==0 else float(supervised.min()),"weak_response_supported_max":None if supervised.numel()==0 else float(supervised.max()),"weak_target_available":bool(s["weak_target_available"]),"endpoint_layer_z":int(s["endpoint_layer_z"])},indent=2)); print("Weak target Dataset inspection complete. No raw file or dense target file was written.")
if __name__=="__main__":
    try: main()
    except Exception as e: print(f"ERROR: {e}",file=sys.stderr); raise
