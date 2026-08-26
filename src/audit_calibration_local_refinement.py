#!/usr/bin/env python3
"""Local offset refinement for the top two screen-corner calibration candidates."""
from __future__ import annotations
import argparse, csv, json, shutil, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from audit_machine_camera_calibration import build_candidates, project

# mean5 LWI columns only: A LED1/2/3 then B LED1/2/3.
FEATURES=[("A",1,22),("A",2,25),("A",3,28),("B",1,31),("B",2,34),("B",3,37)]
def args():
 p=argparse.ArgumentParser();p.add_argument("--control-points",required=True,type=Path);p.add_argument("--registered-root",required=True,type=Path);p.add_argument("--after-tiff",required=True,type=Path);p.add_argument("--burned-tiff",required=True,type=Path);p.add_argument("--output-dir",required=True,type=Path);p.add_argument("--layers",nargs="+",type=int,default=[25,75,125,150]);p.add_argument("--max-offset",type=int,default=12);p.add_argument("--offset-step",type=int,default=2);p.add_argument("--patch-radius",type=int,default=2);p.add_argument("--max-rows",type=int,default=600);p.add_argument("--overwrite",action="store_true");return p.parse_args()
def info(p):
 with tifffile.TiffFile(p) as t:return str(t.series[0].axes)
def frame(p,axes,z,led):
 d=tifffile.memmap(p,series=0,mode="r");i=[]
 for a in axes:
  if a=="T":i.append(led-1)
  elif a=="Z":i.append(z-1)
  elif a=="C":i.append(0)
  else:i.append(slice(None))
 return np.asarray(d[tuple(i)])
def rank(x):
 o=np.argsort(x,kind="mergesort");r=np.empty(len(x));r[o]=np.arange(len(x));return r
def spear(x,y):
 if len(x)<3 or np.std(x)==0 or np.std(y)==0:return np.nan
 return float(np.corrcoef(rank(x),rank(y))[0,1])
def patch_mean(img,xy,r):
 out=np.full(len(xy),np.nan);h,w=img.shape
 for n,(x,y) in enumerate(np.rint(xy).astype(int)):
  if x-r<0 or x+r>=w or y-r<0 or y+r>=h:continue
  v=img[y-r:y+r+1,x-r:x+r+1];v=v[v<65535]
  if len(v):out[n]=np.mean(v)
 return out
def main():
 a=args();out=a.output_dir.resolve()
 if out.exists():
  if not a.overwrite:raise FileExistsError(f"Output directory already exists: {out}. Use --overwrite only after review.")
  shutil.rmtree(out)
 with a.control_points.open() as f:p=json.load(f)
 cands=build_candidates(p["control_points"]);chosen=[(1,cands[0]),(2,cands[1])]
 aa,bb=a.after_tiff.resolve(),a.burned_tiff.resolve();axes={"A":info(aa),"B":info(bb)};paths={"A":aa,"B":bb}
 shifts=range(-a.max_offset,a.max_offset+1,a.offset_step);rows=[]
 for cr,c in chosen:
  for dx in shifts:
   for dy in shifts:
    values=[]
    for z in a.layers:
     for part in c["assignment"]:
      f=a.registered_root/part/f"L{z:04d}.csv"
      if not f.exists():continue
      t=np.genfromtxt(f,delimiter=",");t=t[::max(1,int(np.ceil(len(t)/a.max_rows)))]
      xy=project(c["H"],t[:,2:4])+np.array([dx,dy])
      for st,led,col in FEATURES:
       raw=patch_mean(frame(paths[st],axes[st],z,led),xy,a.patch_radius);valid=np.isfinite(raw)&np.isfinite(t[:,col-1]);rho=spear(raw[valid],t[valid,col-1]);
       if np.isfinite(rho):values.append(abs(rho))
    rows.append({"candidate_rank":cr,"dx_px":dx,"dy_px":dy,"median_abs_spearman":float(np.median(values)) if values else np.nan,"comparison_count":len(values)})
 out.mkdir(parents=True)
 with (out/"local_offset_scores.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 best=[]
 for cr in (1,2):
  r=max((x for x in rows if x["candidate_rank"]==cr and np.isfinite(x["median_abs_spearman"])),key=lambda x:x["median_abs_spearman"]);best.append(r)
 fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
 for ax,cr in zip(axs,(1,2)):
  r=[x for x in rows if x["candidate_rank"]==cr];xs=sorted(set(x["dx_px"] for x in r));ys=sorted(set(x["dy_px"] for x in r));m=np.array([[next(x["median_abs_spearman"] for x in r if x["dx_px"]==xv and x["dy_px"]==yv) for xv in xs] for yv in ys]);im=ax.imshow(m,origin="lower",extent=(min(xs),max(xs),min(ys),max(ys)),aspect="auto");ax.set_title(f"candidate {cr}");ax.set_xlabel("global dx [px]");ax.set_ylabel("global dy [px]");fig.colorbar(im,ax=ax,label="median |Spearman|")
 fig.savefig(out/"local_refinement_qc.png",dpi=180);plt.close(fig)
 summary={"audit_type":"local 5x5 valid-patch photometric offset refinement; not a heatmap/label generator","layers":a.layers,"patch_radius_px":a.patch_radius,"offset_grid":{"max_px":a.max_offset,"step_px":a.offset_step},"best_by_candidate":best,"required_check":"One candidate must show a clearly stronger and stable local correlation peak; otherwise orientation remains unresolved."}
 with (out/"local_refinement_summary.json").open("w") as f:json.dump(summary,f,indent=2);f.write("\n")
 print("Local photometric refinement completed. No raw TIFF, CSV, heatmap or label was modified/created.")
 for r in best:print(f"- rank {r['candidate_rank']}: best |rho|={r['median_abs_spearman']:.4f} at dx={r['dx_px']}, dy={r['dy_px']}")
if __name__=="__main__":
 try:main()
 except Exception as e:print(f"ERROR: {e}",file=sys.stderr);raise
