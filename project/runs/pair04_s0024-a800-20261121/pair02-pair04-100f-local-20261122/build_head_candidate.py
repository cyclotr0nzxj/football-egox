"""Continue legacy head pose from EXO red-shoe ground proxy; provisional only."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;R=P.parent
for pair,base,oldstart in [('Pair02',R/'multiclip-evidence/pair02_s0000/viewpoint',0),('Pair04',R/'geometry-008/viewpoint-008-upright',24)]:
 tr=json.loads((base/'trajectory-check.json').read_text()); template=json.loads((base/'meta.json').read_text()); original=template['test_datasets'][0]; E=np.array(original['ego_extrinsics']); rot=E[:,:,:3]; center=-np.einsum('nji,nj->ni',rot,E[:,:,3]); oldz=center[:,2]
 f=rot[:,2,:]; eye=np.eye(3); proj=eye-f[:,:,None]*f[:,None,:]; target=np.linalg.lstsq(proj.sum(axis=0),np.einsum('nij,nj->i',proj,center),rcond=None)[0]
 up=np.array(tr['ground_up']);up/=np.linalg.norm(up); head=np.array(json.loads((P/f'{pair}-head-track-corrected.json').read_text())['track']); foot=json.loads((P/f'{pair}-foot-depth.json').read_text()); z=np.array([v if v is not None else np.nan for v in foot['head_z']]); assert np.isfinite(z).all()
 scale=np.median(oldz[:49:2]/z[:49:2]); held=np.abs(scale*z[1:49:2]-oldz[1:49:2])/oldz[1:49:2]
 K=np.array(original['camera_intrinsics']); raw=np.stack(((head[:,0]-K[0,2])/K[0,0]*z*scale,(head[:,1]-K[1,2])/K[1,1]*z*scale,z*scale),axis=1)
 # Blend only the new tail toward the original endpoint; old 49 poses remain byte-for-byte inputs.
 bias=center[-1]-raw[48]; candidate=raw.copy()
 for i in range(49,101): candidate[i]+=bias*max(0,1-(i-48)/12)
 padded=np.pad(candidate,((2,2),(0,0)),mode='edge'); smoothed=sum(w*padded[i:i+101] for i,w in enumerate((1,2,3,2,1)))/9
 centers=np.vstack((center,smoothed[49:])); out=list(E)
 for c in centers[49:]:
  forward=target-c; forward/=np.linalg.norm(forward);right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
  counter=np.array([[0,1,0],[-1,0,0],[0,0,1]]); rr=counter@np.stack((right,down,forward));out.append(np.column_stack((rr,-rr@c)))
 out=np.array(out); heights=centers@up+tr['fitted_exo_camera_height_vipe_units']; steps=np.linalg.norm(np.diff(centers,axis=0),axis=1); errors=[]
 for i in range(49,101):
  p=K@centers[i]; p=p[:2]/p[2];errors.append(float(np.linalg.norm(p-head[i])))
 data={'frames':101,'source_start':oldstart,'legacy_first_49_exact':bool(np.array_equal(out[:49],E)),'world':'legacy ViPE coordinate approximation','target_fixed_from_legacy_rays':target.tolist(),'depth_scale_fit_even_legacy_frames':float(scale),'heldout_odd_legacy_z_relative_error_median':float(np.median(held)),'heldout_odd_legacy_z_relative_error_p90':float(np.percentile(held,90)),'candidate_head_centers':centers.tolist(),'ego_extrinsics':out.tolist(),'height_min_max': [float(heights.min()),float(heights.max())], 'max_center_step':float(max(steps)), 'boundary_48_to_49_step':float(steps[48]),'tail_reprojection_error_px_max':max(errors),'tail_reprojection_error_px_p90':float(np.percentile(errors,90)),'caution':'EXO 2D head and red-shoe ground proxy; fixed legacy target assumed, not real gaze. Not yet checked against new ViPE or render; no launch approval implied.'}
 (P/f'{pair}-trajectory-candidate.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 for off in (28,52):
  name=f'{pair.lower()}_s{oldstart+off:04d}'; meta=json.loads(json.dumps(template)); entry=meta['test_datasets'][0]; entry['exo_path']=f'./example/self_capture/videos/{name}/exo.mp4';entry['ego_prior_path']=f'./example/self_capture/ego_priors/{name}/ego_Prior.mp4';entry['ego_extrinsics']=out[off:off+49].tolist()
  (P/'windows'/name/'meta-candidate.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
 print(pair,'z holdout',np.median(held),'heights',data['height_min_max'],'step',data['max_center_step'],'seam',data['boundary_48_to_49_step'],'reprojection p90',data['tail_reprojection_error_px_p90'])
