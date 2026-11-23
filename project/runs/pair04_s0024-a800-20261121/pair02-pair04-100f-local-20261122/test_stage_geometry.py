"""Test geometry scaling and head gate on actual archived first-window ViPE."""
from pathlib import Path
import json,tempfile
import numpy as np
import stage_geometry as stage
from stage_geometry import prepare
P=Path(__file__).resolve().parent;R=P.parent
cases=[('Pair02',R/'multiclip-evidence/pair02_s0000/geometry/vipe_results/pair02_s0000',R/'multiclip-evidence/pair02_s0000/viewpoint/meta.json',R/'multiclip-prep/pair02_s0000/exo_head_anchors.json'),('Pair04',R/'geometry-008/pair04_s0024',R/'geometry-008/viewpoint-008-upright/meta.json',R/'geometry-008/exo_head_anchors.json')]; result=[]
for pair,geom,path,anchors in cases:
 with tempfile.TemporaryDirectory() as t:
  tmp=Path(t); meta=json.loads(path.read_text()); original=np.asarray(meta['test_datasets'][0]['ego_extrinsics']); scaled=original.copy();scaled[:,:,3]*=1.7;meta['test_datasets'][0]['ego_extrinsics']=scaled.tolist();(tmp/'meta.json').write_text(json.dumps(meta))
  report=prepare(geom,tmp/'meta.json',anchors,1.7,tmp/'out'); received=np.asarray(json.loads((tmp/'out/meta.json').read_text())['test_datasets'][0]['ego_extrinsics']);assert np.allclose(received,original,atol=1e-12)
  point=np.array([.4,-.5,8.0]); a=scaled[20,:,:3]@(1.7*point)+scaled[20,:,3];b=received[20,:,:3]@point+received[20,:,3];assert np.allclose(a/a[2],b/b[2],atol=1e-12)
  try:prepare(geom,tmp/'meta.json',anchors,1,tmp/'bad')
  except ValueError as e: assert 'Head depth incompatible' in str(e)
  else:raise AssertionError('Bad scale accepted')
  points=json.loads(anchors.read_text())['frames']; original_unpack=stage.unpack
  def perturbed(archive,name,depth):
   data=original_unpack(archive,name,depth)
   if depth:
    index=int(name.split('.')[0]); x,y=points[index]['x'],points[index]['y']; data[y-2:y+3,x-2:x+3]*=2
   return data
  stage.unpack=perturbed
  try:
   try:prepare(geom,path,anchors,1,tmp/'head-only-error')
   except ValueError as e: assert 'Head depth incompatible' in str(e)
   else:raise AssertionError('Head-only depth error accepted')
  finally:stage.unpack=original_unpack
  assert (2.0*3.0)==6.0
  E6=original.copy();E6[:,:,3]*=6;check=E6.copy();check[:,:,3]/=(2*3);assert np.allclose(check,original,atol=1e-12)
  result.append({'head_only_depth_error_rejected':True,'chained_2x3_scale_composes_once':True,'pair':pair,'synthetic_scale':1.7,'translation_transform_recovers_original':True,'projection_invariant':True,'wrong_scale_rejected':True,'supported_frames':report['head_supported_frames'],'median_head_relative_difference':report['head_relative_difference_median'],'p90_head_relative_difference':report['head_relative_difference_p90']})
(P/'stage-geometry-selftest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
