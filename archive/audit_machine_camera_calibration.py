#!/usr/bin/env python3
"""Compare machine-part and machine-axis hypotheses from screen-corner controls.

Input JSON contains only visible screen TL/TR/BR/BL corners for four parts.
This tool evaluates 24 assignments of screen parts to part01..04 and eight
cyclic/mirrored mappings of screen corners to machine rectangle corners. It
ranks 192 provisional homography hypotheses by leave-one-out residual, renders
the top candidates, and requires visual overlay review before any weak target
projection is permitted.
"""
from __future__ import annotations

import argparse, csv, itertools, json, math, shutil, sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile

RECT = {"part01": (-6., 3., 11., 16.), "part02": (-2., 7., 2., 7.), "part03": (2., 11., -7., -2.), "part04": (6., 15., -16., -11.)}
PARTS = tuple(RECT)
COLORS = {"part01":"tab:blue","part02":"tab:orange","part03":"tab:green","part04":"tab:red"}


def cli() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Rank machine XY→raw camera hypotheses from screen controls")
    p.add_argument("--control-points",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--top-k",type=int,default=4); p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def hom(p:np.ndarray)->np.ndarray: return np.c_[p,np.ones(len(p))]
def norm(p:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    c=p.mean(0); d=np.sqrt(((p-c)**2).sum(1)).mean()
    if d<=0: raise ValueError("Degenerate controls")
    s=math.sqrt(2)/d; T=np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1.]])
    return (T@hom(p).T).T[:,:2],T

def Hfit(src:np.ndarray,dst:np.ndarray)->np.ndarray:
    a,Ts=norm(src); b,Td=norm(dst); A=[]
    for (x,y),(u,v) in zip(a,b,strict=True): A.extend([[-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]])
    _,_,V=np.linalg.svd(np.asarray(A)); H=np.linalg.inv(Td)@V[-1].reshape(3,3)@Ts
    if abs(H[2,2])<1e-12: raise ValueError("Degenerate homography")
    return H/H[2,2]
def project(H:np.ndarray,p:np.ndarray)->np.ndarray:
    q=(H@hom(p).T).T
    return q[:,:2]/q[:,2:3]
def loo(src:np.ndarray,dst:np.ndarray)->np.ndarray:
    out=[]
    for i in range(len(src)):
        k=np.ones(len(src),bool); k[i]=False; out.append(np.linalg.norm(project(Hfit(src[k],dst[k]),src[i:i+1])[0]-dst[i]))
    return np.asarray(out)
def corners(part:str)->np.ndarray:
    xmin,xmax,ymin,ymax=RECT[part]; return np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]],float)
def orientations()->list[tuple[str,tuple[int,...]]]:
    base=(0,1,2,3); out=[]
    for r in range(4): out.append((f"rotate_{r*90}",base[r:]+base[:r]))
    rev=(0,3,2,1)
    for r in range(4): out.append((f"mirror_rotate_{r*90}",rev[r:]+rev[:r]))
    return out

def read_frame(ref:dict[str,Any])->np.ndarray:
    path=Path(ref["tiff_path"]); axes=str(ref["axes"]); data=tifffile.memmap(path,series=0,mode="r")
    ix=[]
    for a in axes:
        if a=="T": ix.append(int(ref["led"])-1)
        elif a=="Z": ix.append(int(ref["layer_z"])-1)
        elif a=="C": ix.append(0)
        elif a in "YX": ix.append(slice(None))
        else: raise ValueError(f"Unsupported axis {a}")
    return np.asarray(data[tuple(ix)])

def candidate(src:np.ndarray,dst:np.ndarray,assignment:tuple[str,...],name:str,order:tuple[int,...])->dict[str,Any]:
    H=Hfit(src,dst); fit=np.linalg.norm(project(H,src)-dst,axis=1); lv=loo(src,dst)
    return {"assignment":assignment,"orientation":name,"corner_index_order":order,"H":H,"fit":fit,"loo":lv,"fit_rmse":float(np.sqrt(np.mean(fit**2))),"loo_rmse":float(np.sqrt(np.mean(lv**2))),"score":float(np.sqrt(np.mean(lv**2)))}

def build_candidates(points:list[dict[str,Any]])->list[dict[str,Any]]:
    screen_parts=[]; grouped={}
    for p in points: grouped.setdefault(p["screen_part"],[]).append(p)
    for name, rows in grouped.items():
        if len(rows)!=4: raise ValueError(f"{name} requires exactly four screen corners")
        screen_parts.append(name)
    screen_parts=sorted(screen_parts,key=lambda s: points.index(grouped[s][0]))
    dst=np.asarray([[float(p["raw_camera_x_px"]),float(p["raw_camera_y_px"])] for s in screen_parts for p in grouped[s]],float)
    result=[]
    for assignment in itertools.permutations(PARTS):
        for oname,order in orientations():
            src=np.vstack([corners(part)[list(order)] for part in assignment])
            result.append(candidate(src,dst,assignment,oname,order))
    return sorted(result,key=lambda c:c["score"])

def qc(path:Path,frame:np.ndarray,ref:dict[str,Any],cands:list[dict[str,Any]],points:list[dict[str,Any]])->None:
    k=min(4,len(cands)); fig,axs=plt.subplots(2,2,figsize=(16,14),constrained_layout=True); lo,hi=np.percentile(frame,ref.get("display_percentiles",[1,99.5]))
    dst=np.asarray([[float(p["raw_camera_x_px"]),float(p["raw_camera_y_px"])] for p in points],float)
    for rank,(ax,c) in enumerate(zip(axs.ravel(),cands[:k]),1):
        ax.imshow(frame,cmap="gray",vmin=lo,vmax=hi,origin="upper"); ax.scatter(dst[:,0],dst[:,1],c="cyan",marker="x",s=35,label="screen controls")
        for j,part in enumerate(c["assignment"]):
            poly=project(c["H"],corners(part)); poly=np.vstack([poly,poly[0]]); ax.plot(poly[:,0],poly[:,1],color=COLORS[part],lw=2,label=part if j==0 else None)
            ax.text(poly[:,0].mean(),poly[:,1].mean(),f"screen {chr(65+j)}→{part}",color=COLORS[part],fontsize=8)
        ax.set_title(f"rank {rank}: LOO RMSE={c['loo_rmse']:.2f}px\n{c['orientation']} | {list(c['assignment'])}"); ax.set_xlabel("raw x"); ax.set_ylabel("raw y")
    fig.suptitle("Top provisional machine-part / orientation hypotheses — visual review required",fontsize=15,fontweight="bold")
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)

def main()->None:
    a=cli(); cp=a.control_points.resolve(); out=a.output_dir.resolve()
    if out.exists():
        if not a.overwrite: raise FileExistsError(f"Output directory already exists: {out}. Use --overwrite only after review.")
        shutil.rmtree(out)
    with cp.open(encoding="utf-8") as f: payload=json.load(f)
    if payload.get("schema")!="screen_corners_v2_orientation_agnostic": raise ValueError("Use the v2 screen-corner selection JSON, not the older machine-corner JSON")
    points=payload.get("control_points",[])
    if len(points)!=16: raise ValueError("Exactly 16 screen control points are required")
    out.mkdir(parents=True); cands=build_candidates(points); ref=payload["reference_frame"]; frame=read_frame(ref)
    rows=[]
    for i,c in enumerate(cands,1): rows.append({"rank":i,"loo_rmse_px":c["loo_rmse"],"fit_rmse_px":c["fit_rmse"],"orientation":c["orientation"],"screen_A_to_D_machine_parts":";".join(c["assignment"]),"corner_index_order":";".join(map(str,c["corner_index_order"]))})
    with (out/"calibration_candidate_ranking.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    qc(out/"calibration_candidate_qc.png",frame,ref,cands[:a.top_k],points)
    best=cands[0]; summary={"audit_type":"screen-corner machine-part/orientation hypothesis ranking; not a final calibration","input_control_json":str(cp),"reference_frame":ref,"candidate_count":len(cands),"recommended_by_residual_only":{"rank":1,"loo_rmse_px":best["loo_rmse"],"fit_rmse_px":best["fit_rmse"],"orientation":best["orientation"],"screen_A_to_D_machine_parts":list(best["assignment"]),"homography_machine_xy_to_raw_camera_pixel":best["H"].tolist()},"required_human_check":"Inspect calibration_candidate_qc.png; choose no candidate if outlines do not follow all four visible part boundaries.","status":"candidate_ranking_ready_visual_selection_required","outputs":{"ranking_csv":"calibration_candidate_ranking.csv","qc_png":"calibration_candidate_qc.png"},"prohibitions":["No weak heatmap is generated by this audit.","Pixels outside sparse support remain unknown, never negative labels."]}
    with (out/"calibration_hypothesis_summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2);f.write("\n")
    print("Orientation-hypothesis calibration audit completed. No raw TIFF, CSV, heatmap or label was modified/created.")
    print(f"- hypotheses ranked: {len(cands)}")
    print(f"- best residual-only candidate: LOO RMSE={best['loo_rmse']:.3f}px, {best['orientation']}, {list(best['assignment'])}")
    print(f"- output directory: {out}")

if __name__=="__main__":
    try: main()
    except Exception as e: print(f"ERROR: {e}",file=sys.stderr);raise
