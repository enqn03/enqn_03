#!/usr/bin/env python3
"""Tie-break mirror-equivalent calibration candidates with registered LWI features.
For selected build layers, this audit projects sparse registered XY points to raw
A/B camera pixels under candidate homographies. It compares raw valid pixel
intensities with registered layerwise-image (LWI) values using Pearson and
Spearman correlations. It ranks candidates by the median absolute correlation
across stage/LED/filter combinations. This is a calibration consistency audit,
not a defect-label or heatmap generator.
"""
from __future__ import annotations
import argparse, csv, json, shutil, sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from audit_machine_camera_calibration import build_candidates, project
FEATURES = []
for stage, first_col in (("A", 20), ("B", 29)):
    for led in (1, 2, 3):
        for filt, offset in (("original", 0), ("mean3", 1), ("mean5", 2)):
            FEATURES.append((stage, led, filt, first_col + (led - 1) * 3 + offset))
def cli() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Photometric tie-break for top machine-camera calibration candidates")
    p.add_argument("--control-points",required=True,type=Path)
    p.add_argument("--registered-root",required=True,type=Path)
    p.add_argument("--after-tiff",required=True,type=Path)
    p.add_argument("--burned-tiff",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--candidate-ranks",nargs="+",type=int,default=[1,2])
    p.add_argument("--layers",nargs="+",type=int,default=[25,50,75,100,125,150])
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()
def frame_info(path:Path)->tuple[str,tuple[int,...]]:
    with tifffile.TiffFile(path) as tif: return str(tif.series[0].axes),tuple(int(x) for x in tif.series[0].shape)
def raw_frame(path:Path,axes:str,z:int,led:int)->np.ndarray:
    data=tifffile.memmap(path,series=0,mode="r"); ix=[]
    for a in axes:
        if a=="T": ix.append(led-1)
        elif a=="Z": ix.append(z-1)
        elif a=="C": ix.append(0)
        elif a in "YX": ix.append(slice(None))
        else: raise ValueError(f"Unsupported TIFF axis {a}")
    return np.asarray(data[tuple(ix)])
def ranks(x:np.ndarray)->np.ndarray:
    order=np.argsort(x,kind="mergesort"); out=np.empty(len(x),float); out[order]=np.arange(len(x),dtype=float)
    vals=x[order]; start=0
    for i in range(1,len(x)+1):
        if i==len(x) or vals[i]!=vals[start]:
            out[order[start:i]]=0.5*(start+i-1); start=i
    return out
def corr(x:np.ndarray,y:np.ndarray)->tuple[float,float]:
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return float("nan"),float("nan")
    return float(np.corrcoef(x,y)[0,1]),float(np.corrcoef(ranks(x),ranks(y))[0,1])
def read_csv(path:Path)->np.ndarray:
    return np.genfromtxt(path,delimiter=",",dtype=float)
def sample(frame:np.ndarray,xy:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    x=np.rint(xy[:,0]).astype(int); y=np.rint(xy[:,1]).astype(int); h,w=frame.shape
    inside=(x>=0)&(x<w)&(y>=0)&(y<h); values=np.full(len(x),np.nan); values[inside]=frame[y[inside],x[inside]]
    return values,inside
def main()->None:
    a=cli(); cp=a.control_points.resolve(); root=a.registered_root.resolve(); out=a.output_dir.resolve()
    if out.exists():
        if not a.overwrite: raise FileExistsError(f"Output directory already exists: {out}. Use --overwrite only after review.")
        shutil.rmtree(out)
    with cp.open(encoding="utf-8") as f: payload=json.load(f)
    if payload.get("schema")!="screen_corners_v2_orientation_agnostic": raise ValueError("Expected v2 screen-corner JSON")
    all_cands=build_candidates(payload["control_points"])
    selected=[]
    for rank in a.candidate_ranks:
        if not 1<=rank<=len(all_cands): raise ValueError(f"Candidate rank must be 1..{len(all_cands)}")
        selected.append((rank,all_cands[rank-1]))
    axes_a,_=frame_info(a.after_tiff.resolve()); axes_b,_=frame_info(a.burned_tiff.resolve())
    out.mkdir(parents=True); rows=[]
    for rank,cand in selected:
        for z in a.layers:
            for part_i,part in enumerate(cand["assignment"]):
                csv_path=root/part/f"L{z:04d}.csv"
                if not csv_path.is_file(): continue
                table=read_csv(csv_path)
                if table.ndim!=2 or table.shape[1]<40: continue
                machine=table[:,2:4]; pixels=project(cand["H"],machine)
                for stage,led,filt,column in FEATURES:
                    frame=raw_frame(a.after_tiff.resolve() if stage=="A" else a.burned_tiff.resolve(),axes_a if stage=="A" else axes_b,z,led)
                    raw,inside=sample(frame,pixels); registered=table[:,column-1]
                    valid=inside & np.isfinite(registered) & np.isfinite(raw) & (raw<65535)
                    p,s=corr(raw[valid],registered[valid])
                    rows.append({"candidate_rank":rank,"orientation":cand["orientation"],"screen_A_to_D_machine_parts":";".join(cand["assignment"]),"part":part,"layer_z":z,"stage":stage,"led":led,"filter":filt,"registered_column_1_based":column,"sample_count":int(valid.sum()),"in_frame_fraction":float(inside.mean()),"valid_fraction":float(valid.mean()),"pearson_r":p,"spearman_r":s,"abs_spearman_r":abs(s) if np.isfinite(s) else float("nan")})
    if not rows: raise RuntimeError("No comparable registered/raw samples were found")
    fields=list(rows[0]);
    with (out/"photometric_candidate_scores.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary_rows=[]
    for rank,_ in selected:
        vals=np.asarray([r["abs_spearman_r"] for r in rows if r["candidate_rank"]==rank and np.isfinite(r["abs_spearman_r"])])
        summary_rows.append({"candidate_rank":rank,"median_abs_spearman":float(np.median(vals)),"mean_abs_spearman":float(np.mean(vals)),"comparison_count":int(len(vals))})
    summary_rows.sort(key=lambda r:r["median_abs_spearman"],reverse=True)
    fig,ax=plt.subplots(figsize=(10,5)); labels=[f"rank {r['candidate_rank']}" for r in summary_rows]; ax.bar(labels,[r["median_abs_spearman"] for r in summary_rows]); ax.set_ylim(0,1);ax.set_ylabel("median |Spearman r|");ax.set_title("Photometric consistency by calibration candidate");
    for i,r in enumerate(summary_rows):ax.text(i,r["median_abs_spearman"]+0.02,f"n={r['comparison_count']}",ha="center")
    fig.savefig(out/"photometric_qc.png",dpi=180,bbox_inches="tight");plt.close(fig)
    summary={"audit_type":"registered LWI/raw camera photometric calibration tie-break; not a weak heatmap or defect label","raw_input_policy":"Registered CSV and TIFF files read only; no raw data modified.","layers":a.layers,"candidate_summary":summary_rows,"feature_schema":{"A_columns_1_based":"20-28","B_columns_1_based":"29-37","filters":["original","mean3","mean5"]},"validity_rule":"Projected points must be in frame, have finite registered value, and raw intensity < 65535.","recommended_by_median_abs_spearman":summary_rows[0],"required_check":"A clear advantage over the mirror candidate is required; a marginal tie keeps absolute orientation unresolved.","outputs":{"scores_csv":"photometric_candidate_scores.csv","qc_png":"photometric_qc.png"}}
    with (out/"photometric_summary.json").open("w",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write("\n")
    print("Photometric calibration tie-break completed. No raw TIFF, CSV, heatmap or label was modified/created.")
    for r in summary_rows: print(f"- candidate rank {r['candidate_rank']}: median |Spearman r|={r['median_abs_spearman']:.4f} (n={r['comparison_count']})")
    print(f"- output directory: {out}")
if __name__=="__main__":
    try: main()
    except Exception as e: print(f"ERROR: {e}",file=sys.stderr);raise
