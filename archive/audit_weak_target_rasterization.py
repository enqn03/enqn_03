utf-8
#!/usr/bin/env python3
"""Rasterization audit: sparse continuous XCT response + support mask, not defect labels."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,yaml
from audit_machine_camera_calibration import build_candidates,project
def main():
 p=argparse.ArgumentParser();p.add_argument('--calibration-config',required=True,type=Path);p.add_argument('--registered-root',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--layer-z',type=int,default=125);p.add_argument('--resolution',nargs=2,type=int,default=[256,256]);p.add_argument('--sigmas',nargs='+',type=float,default=[1,2,3,4]);p.add_argument('--overwrite',action='store_true');a=p.parse_args();out=a.output_dir.resolve()
 if out.exists():
  if not a.overwrite:raise FileExistsError(f'Output directory already exists: {out}. Use --overwrite only after review.')
  shutil.rmtree(out)
 cfg=yaml.safe_load(a.calibration_config.read_text());cp=Path(cfg['control_points']['path']);cp=cp if cp.is_absolute() else Path.cwd()/cp
 c=build_candidates(json.loads(cp.read_text())['control_points'])[int(cfg['geometry_candidate']['rank'])-1];off=np.asarray(cfg['local_photometric_refinement']['raw_pixel_global_offset_xy']);H,W=a.resolution;pts=[]
 for part in ('part01','part02','part03','part04'):
  f=a.registered_root/part/f'L{a.layer_z:04d}.csv';t=np.genfromtxt(f,delimiter=',');v=np.isfinite(t[:,39]);xy=project(c['H'],t[v,2:4])+off;inside=(xy[:,0]>=0)&(xy[:,0]<2000)&(xy[:,1]>=0)&(xy[:,1]<2000);pts.append((xy[inside]*np.array([W/2000,H/2000]),t[v,39][inside]))
 xy=np.concatenate([x for x,_ in pts]);val=np.concatenate([v for _,v in pts]);q1,q99=np.percentile(val,[1,99]);value=np.clip((val-q1)/(q99-q1),0,1);out.mkdir(parents=True);summary={'audit_type':'continuous XCT-derived response/support rasterization; not anomaly/defect label','layer_z':a.layer_z,'resolution':[H,W],'finite_points':int(len(value)),'response_direction':'unresolved','sigmas_model_px':a.sigmas,'unknown_policy':'support=0 is unknown, not negative label'};fig,axs=plt.subplots(2,len(a.sigmas),figsize=(4*len(a.sigmas),7),constrained_layout=True)
 yy,xx=np.mgrid[0:H,0:W]
 for j,s in enumerate(a.sigmas):
  num=np.zeros((H,W));den=np.zeros((H,W))
  for (x,y),v in zip(xy,value):
   r=int(np.ceil(3*s));x0=max(0,int(x)-r);x1=min(W,int(x)+r+1);y0=max(0,int(y)-r);y1=min(H,int(y)+r+1);g=np.exp(-((xx[y0:y1,x0:x1]-x)**2+(yy[y0:y1,x0:x1]-y)**2)/(2*s*s));num[y0:y1,x0:x1]+=g*v;den[y0:y1,x0:x1]+=g
  response=np.divide(num,den,out=np.zeros_like(num),where=den>0);support=(den>1e-6).astype(float);axs[0,j].imshow(response,cmap='viridis',vmin=0,vmax=1);axs[0,j].set_title(f'response sigma={s}');axs[1,j].imshow(support,cmap='gray',vmin=0,vmax=1);axs[1,j].set_title(f'support={support.mean():.3f}')
  summary[str(s)]={'support_fraction':float(support.mean()),'response_nonzero_fraction':float((response>0).mean())}
 fig.savefig(out/'weak_target_rasterization_qc.png',dpi=180);plt.close(fig);(out/'weak_target_rasterization_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Weak target rasterization audit completed. No heatmap label file or raw data was modified.')
if __name__=='__main__':
 try:main()
 except Exception as e:print(f'ERROR: {e}',file=sys.stderr);raise
