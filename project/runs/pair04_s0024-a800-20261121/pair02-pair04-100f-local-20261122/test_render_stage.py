"""Exercise staging against archived geometry with simulated renderer/converter only."""
from pathlib import Path
import json,shutil,tempfile,zipfile,subprocess,sys,hashlib
import numpy as np
import continue_008 as app
from check_overlap_geometry import unpack
P=Path(__file__).resolve().parent;R=P.parent
cases=[('Pair02',R/'multiclip-evidence/pair02_s0000/geometry/vipe_results/pair02_s0000',R/'multiclip-evidence/pair02_s0000/viewpoint/meta.json',R/'multiclip-prep/pair02_s0000/exo_head_anchors.json',R/'multiclip-prep/pair02_s0000/example/self_capture/videos/pair02_s0000/exo.mp4',R/'multiclip-evidence/pair02_s0000/prior/pair02_s0000/ego_Prior.mp4'),('Pair04',R/'geometry-008/pair04_s0024',R/'geometry-008/viewpoint-008-upright/meta.json',R/'geometry-008/exo_head_anchors.json',R.parents[1]/'data/self_capture/pair04_s0024/egox_input/example/self_capture/videos/pair04_s0024/exo.mp4',R/'geometry-008/prior-008-upright/pair04_s0024/ego_Prior.mp4')];results=[]
for pair,geom,meta,anchors,exo,prior in cases:
 with tempfile.TemporaryDirectory() as t:
  root=Path(t);run=root/'run'; stock=root/'stock';stock.mkdir();(stock/'infer.py').write_text('simulated');(stock/'core').mkdir(); app.STOCK=stock;app.RENDERER=root/'renderer.py';app.RENDERER.write_text('mock');app.RENDERER_SHA=app.digest(app.RENDERER);app.BASE=root
  leg=run/'legacy'/pair;leg.mkdir(parents=True);shutil.copyfile(prior,leg/'prior.mp4')
  for name in app.PAIRS[pair]:
   w=run/'windows'/name;w.mkdir(parents=True);(w/'geometry').mkdir();(w/'geometry'/name).symlink_to(geom.resolve(),target_is_directory=True);shutil.copyfile(exo,w/'exo.mp4');a=json.loads(anchors.read_text());a['name']=name;a['video_sha256']=app.digest(exo);(w/'exo_head_anchors.json').write_text(json.dumps(a));m=json.loads(meta.read_text());m['test_datasets'][0]['exo_path']=f'./example/self_capture/videos/{name}/exo.mp4';m['test_datasets'][0]['ego_prior_path']=f'./example/self_capture/ego_priors/{name}/ego_Prior.mp4';(w/'meta-candidate.json').write_text(json.dumps(m));(w/'overlap.json').write_text(json.dumps({'scale_to_old':1.0,'passed_provisional_gate':True,'old_start':int(name.rsplit('s',1)[1])-(28 if name==app.PAIRS[pair][0] else 24),'new_start':int(name.rsplit('s',1)[1])}))
  def fake_call(args,**kwargs):
   args=list(map(str,args))
   if '--out_dir' in args:
    out=Path(args[args.index('--out_dir')+1]);name=Path(args[args.index('--input_dir')+1]).name;(out/name).mkdir(parents=True);shutil.copyfile(prior,out/name/'ego_Prior.mp4')
   elif '--egox_depthmaps_path' in args:
    out=Path(args[args.index('--egox_depthmaps_path')+1])/'exo';out.mkdir(parents=True)
    with zipfile.ZipFile(geom/'depth/exo.zip') as z:
     for i in range(49):np.save(out/f'{i:05d}.npy',unpack(z,f'{i:05d}.exr',True))
   else:raise AssertionError(args)
  app.call=fake_call;app.render_pair(run,pair,app.env())
  for name in app.PAIRS[pair]:
   w=run/'windows'/name;rec=json.loads((w/'render-receipt.json').read_text()); assert rec['status']=='technical_pass_semantic_review_pending' and len(rec['depth_sha256'])==49 and (w/'inference-input/meta.json').exists()
  assert (run/'windows'/app.PAIRS[pair][0]/'overlap-legacy-prior.mp4').is_file() and (run/'windows'/app.PAIRS[pair][1]/'overlap-previous-prior.mp4').is_file()
  rawname='pair02_s0000' if pair=='Pair02' else 'pair04_s0024'
  def fake_infer(args,**kwargs):
   args=list(map(str,args));out=Path(args[args.index('--out')+1]); row=json.loads((Path(kwargs['cwd'])/'meta.json').read_text())['test_datasets'][0];name=Path(row['exo_path']).parent.name
   shutil.copyfile(R/f'four-egox-outputs-008/raw/{rawname}.mp4',out/f'{name}.mp4');shutil.copyfile(R/'four-egox-outputs-008/logs/runtime_audit.json',out/'runtime_audit.json')
  app.call=fake_infer;app.infer_pair(run,pair,app.env())
  for name in app.PAIRS[pair]:
   output=run/'windows'/name/'inference-output'; assert (output/'completed.json').is_file() and (output/'verified'/f'{name}_ego.mp4').is_file()
  archive=root/'evidence.zip'; subprocess.run([sys.executable,str(P/'collect_008.py'),'--run',str(run),'--output',str(archive)],check=True,stdout=subprocess.PIPE)
  with zipfile.ZipFile(archive) as z:
   manifest=json.loads(z.read('collection-manifest.json')); assert any(k.endswith('completed.json') for k in manifest['files']) and not any('depth-conversion' in k for k in manifest['files'])
  sha=hashlib.sha256(archive.read_bytes()).hexdigest(); destination=root/'received'; subprocess.run([sys.executable,str(P/'receive_008.py'),str(archive),str(destination),'--sha256',sha],check=True,stdout=subprocess.PIPE)
  assert json.loads((destination/'retrieval-receipt.json').read_text())['status']=='archive_and_each_member_verified'
  bad=subprocess.run([sys.executable,str(P/'receive_008.py'),str(archive),str(root/'wrong'),'--sha256','0'*64],stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert bad.returncode!=0 and not (root/'wrong').exists()
  results.append({'pair':pair,'collection_verified':True,'receiver_and_wrong_digest_rejection_passed':True,'archived_geometry':'real first window duplicated to simulate two windows','renderer_converter':'simulated, no GPU or model inference','both_windows_staged':True,'overlap_videos_written':True,'depth_arrays_verified_each':49,'inference_invocation':'simulated using archived raw output and audit, no inference','historical_result_verifier_passed':True})
(P/'render-stage-selftest.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
