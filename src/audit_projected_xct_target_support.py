#!/usr/bin/env python3
"""Audit camera-space support of provisional sparse XCT weak targets."""
from __future__ import annotations
import argparse,csv,json,shutil,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,yaml
from audit_machine_camera_calibration import build_candidates,project

def args():
 p=argparse.ArgumentParser();p.add_argument('--calibration-config',required=True,type=Path);p.add_argument('--registered-root',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--qc-layer',type=int,default=125);p.add_argument('--overwrite',action='store_true');return p.parse_args()
def train_layers(manifest):
 s=set()
 with manifest.open() as f:
  for r in csv.DictReader(f):
   if r['split']=='train':s.update(map(int,r['history_layer_z'].split(';')))
 return sorted(s)
def main():
 a=args();out=a.output_dir.resolve()
 if out.exists():
  if not a.overwrite:raise FileExistsError(f'Output directory already exists: {out}. Use --overwrite only after review.')
  shutil.rmtree(out)
 cfg=yaml.safe_load(a.calibration_config.read_text());cp=Path(cfg['control_points']['path'])
 if not cp.is_absolute():cp=Path.cwd()/cp
 controls=json.loads(cp.read_text());rank=int(cfg['geometry_candidate']['rank']);cand=build_candidates(controls['control_points'])[rank-1];off=np.asarray(cfg['local_photometric_refinement']['raw_pixel_global_offset_xy'])
 layers=train_layers(a.manifest.resolve());rows=[];qc=[]
 for part in ('part01','part02','part03','part04'):
  for z in layers:
   f=a.registered_root/part/f'L{z:04d}.csv'
   if not f.exists():rows.append({'part':part,'layer_z':z,'finite_xct5':0,'in_fov':0,'unique_pixels':0,'in_fov_fraction':np.nan,'collision_fraction':np.nan});continue
   t=np.genfromtxt(f,delimiter=',');v=np.isfinite(t[:,39]);xy=project(cand['H'],t[v,2:4])+off;inside=(xy[:,0]>=0)&(xy[:,0]<2000)&(xy[:,1]>=0)&(xy[:,1]<2000);pix=np.rint(xy[inside]).astype(int);unique=len(np.unique(pix,axis=0)) if len(pix) else 0
   rows.append({'part':part,'layer_z':z,'finite_xct5':int(v.sum()),'in_fov':int(inside.sum()),'unique_pixels':unique,'in_fov_fraction':float(inside.mean()) if len(inside) else np.nan,'collision_fraction':float(1-unique/len(pix)) if len(pix) else np.nan})
   if z==a.qc_layer and len(pix):qc.append((part,pix))
 out.mkdir(parents=True)
 with (out/'projected_support_by_layer.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 finite=sum(r['finite_xct5'] for r in rows);inside=sum(r['in_fov'] for r in rows);unique=sum(r['unique_pixels'] for r in rows)
 summary={'audit_type':'provisional sparse XCT 5x5x5 camera-space support audit; not a heatmap or binary label','train_layers':layers,'calibration_config':str(a.calibration_config),'finite_xct5_total':finite,'in_fov_total':inside,'in_fov_fraction':inside/finite if finite else None,'unique_projected_pixels_total':unique,'pixel_collision_fraction':1-unique/inside if inside else None,'unknown_policy':'outside FOV and unsupported locations remain unknown, never negative labels.'}
 (out/'projected_support_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(8,8));colors=['r','g','b','m']
 for (part,pix),c in zip(qc,colors):ax.scatter(pix[:,0],pix[:,1],s=.4,c=c,label=part)
 ax.set(xlim=(0,2000),ylim=(2000,0),aspect='equal',title=f'Projected finite XCT support, layer {a.qc_layer}',xlabel='raw x',ylabel='raw y');ax.legend(markerscale=8);fig.savefig(out/'projected_support_qc.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print('Projected sparse-XCT support audit completed. No raw TIFF, CSV, heatmap or label was modified/created.');print(f'- finite/in-FOV support: {finite}/{inside} ({inside/finite:.4%})');print(f'- output directory: {out}')
if __name__=='__main__':
 try:main()
 except Exception as e:print(f'ERROR: {e}',file=sys.stderr);raise
