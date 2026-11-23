"""Validate new ViPE head support and express legacy candidate poses in native ViPE units."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from check_overlap_geometry import load_k,unpack

def prepare(geometry,meta_path,anchors_path,scale,output):
 if not np.isfinite(scale) or scale <= 0: raise ValueError('Scale must be positive and finite')
 with np.load(geometry/'pose/exo.npz',allow_pickle=False) as p:
  if not np.array_equal(p['inds'],np.arange(49)) or not np.allclose(p['data'],np.broadcast_to(np.eye(4),(49,4,4)),atol=1e-6): raise ValueError('EXO pose is not a fixed identity frame')
 meta=json.loads(meta_path.read_text()); row,=meta['test_datasets']; E=np.asarray(row['ego_extrinsics'],float)
 exo=np.asarray(row['camera_extrinsics'],float)
 if exo.shape!=(3,4) or not np.allclose(exo,np.eye(4)[:3],atol=1e-8): raise ValueError('Candidate EXO W2C is not identity')
 if E.shape!=(49,3,4) or not np.isfinite(E).all(): raise ValueError('Invalid camera poses')
 r=E[:,:,:3];
 if not np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(r),1,atol=1e-5): raise ValueError('Invalid rotation matrices')
 oldk=np.asarray(row['camera_intrinsics'],float); new=load_k(geometry)
 if np.max(abs(new[:2]/[oldk[0,0],oldk[1,1]]-1))>.01 or np.max(abs(new[2:]-[oldk[0,2],oldk[1,2]]))>.5: raise ValueError('New camera K differs from candidate K')
 anchors=json.loads(anchors_path.read_text())['frames'];
 if [a['frame'] for a in anchors]!=list(range(49)): raise ValueError('Head anchors not indexed')
 person={int(line.split(':',1)[0]) for line in (geometry/'mask/exo.txt').read_text().splitlines() if line.split(':',1)[1].strip().lower()=='person'}
 if not person: raise ValueError('Person mask labels absent')
 centers=-np.einsum('nji,nj->ni',r,E[:,:,3]); rows=[]
 if np.any(centers[:,2]<=0): raise ValueError('Head center behind EXO camera')
 pixels=np.column_stack((new[0]*centers[:,0]/centers[:,2]+new[2],new[1]*centers[:,1]/centers[:,2]+new[3]))
 marks=np.array([[v['x'],v['y']] for v in anchors]); reprojection=np.linalg.norm(pixels-marks,axis=1)
 if np.max(reprojection)>5: raise ValueError(f'Head center misses EXO anchor: {np.max(reprojection):.2f}px')
 with zipfile.ZipFile(geometry/'depth/exo.zip') as depths, zipfile.ZipFile(geometry/'mask/exo.zip') as masks:
  for i,a in enumerate(anchors):
   x,y=int(a['x']),int(a['y']);
   if not (2<=x<782 and 2<=y<446): raise ValueError('Anchor out of bounds')
   d=unpack(depths,f'{i:05d}.exr',True)[y-2:y+3,x-2:x+3]; m=unpack(masks,f'{i:05d}.png',False)[y-2:y+3,x-2:x+3]
   ids,count=np.unique(m,return_counts=True); ranked=sorted([(int(n),int(v)) for v,n in zip(ids,count) if int(v) in person],reverse=True)
   if not ranked or ranked[0][0]<8: rows.append({'frame':i,'status':'unsupported'});continue
   n,instance=ranked[0]; z=float(np.median(d[m==instance])*scale); expected=float(centers[i,2]); rel=abs(z-expected)/expected
   rows.append({'frame':i,'status':'supported','instance':instance,'samples':n,'new_depth_in_old_units':z,'candidate_center_z':expected,'relative_difference':rel})
 supported=[v for v in rows if v['status']=='supported'];
 if len(supported)<40: raise ValueError(f'Only {len(supported)}/49 head patches have >=8 person pixels')
 errors=[v['relative_difference'] for v in supported]; med=float(np.median(errors));p90=float(np.percentile(errors,90))
 if med>.20 or p90>.40: raise ValueError(f'Head depth incompatible: median={med:.3f}, p90={p90:.3f}')
 # Native artifact depths are untouched; new coordinates are old/scale.
 # For W2C, R_new=R_old and t_new=t_old/scale. GGA uses new K and native depth.
 E[:,:,3]/=scale; row['ego_extrinsics']=E.tolist(); row['camera_intrinsics']=[[float(new[0]),0,float(new[2])],[0,float(new[1]),float(new[3])],[0,0,1]]
 output.mkdir(parents=True,exist_ok=False); (output/'meta.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
 report={'scale_old_over_new':scale,'old_intrinsics':oldk.tolist(),'new_intrinsics':new.tolist(),'head_supported_frames':len(supported),'head_anchor_max_reprojection_error_px':float(np.max(reprojection)),'head_relative_difference_median':med,'head_relative_difference_p90':p90,'rows':rows,'gate':'provisional >=40 frames; median <=20%, p90 <=40%; same-view depth agreement only; visual prior review remains necessary','transform':'native depth unchanged; camera intrinsic from new ViPE; ego W2C rotation unchanged, translation divided by scale'}
 (output/'geometry-check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--meta',type=Path,required=True);p.add_argument('--anchors',type=Path,required=True);p.add_argument('--scale',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 report=prepare(a.geometry,a.meta,a.anchors,a.scale,a.output);print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if __name__=='__main__':main()
