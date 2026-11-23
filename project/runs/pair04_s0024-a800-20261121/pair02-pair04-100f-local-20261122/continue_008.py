"""Render verified new-window geometry and run the pinned EgoX for two EXO-only pairs."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,zipfile
from fractions import Fraction
from pathlib import Path
import numpy as np
from stage_geometry import prepare
from check_overlap_geometry import unpack
from assemble_extension import read as read_video, write as write_video
from verify_egox_outputs_008 import verify_take
BASE=Path('/root/autodl-tmp/football_egox/pair04-20261121');ROOT=BASE/'egox-shared-008';WEIGHTS=Path('/root/autodl-fs/football_egox/models');BIN=BASE/'renderer-env/bin'; PYTHON=ROOT/'egox-env/bin/python';STOCK=ROOT/'inference-four'; RENDERER=BASE/'Renderer-sparse/scripts/render_vipe_pointcloud.py'; RENDERER_SHA='177721241c123019e7f1932639c59efb510e314514f561ef9d54ce122f6a1ee8'; PAIRS={'Pair02':('pair02_s0028','pair02_s0052'),'Pair04':('pair04_s0052','pair04_s0076')}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda: f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def call(args,*,env=None,cwd=None,log=None,timeout=None):
 if log:
  with log.open('x') as f:return subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
 return subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=True)
def env():
 e={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')};e.update(CUDA_VISIBLE_DEVICES='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',IMAGEIO_FFMPEG_EXE=str(BIN/'ffmpeg'),PATH=str(BIN)+':'+e.get('PATH',''));return e
def check_prior(path):
 data=read_video(path);std=np.std(data.astype(np.float32),axis=(1,2,3));nonblack=np.mean(np.max(data,axis=3)>8,axis=(1,2))
 if np.any(std<2) or np.any(nonblack<.05):raise ValueError('A rendered prior frame is blank or nearly blank')
 return {'frames':49,'min_rgb_std':float(std.min()),'min_nonblack_fraction':float(nonblack.min()),'visual_semantics':'not assessed by this check'}
def render_pair(run,pair,e):
 names=PAIRS[pair]; previous=None; scales=[]
 for index,name in enumerate(names):
  win=run/'windows'/name; geom=win/'geometry'/name; overlap=json.loads((win/'overlap.json').read_text())
  if overlap.get('passed_provisional_gate') is not True:raise ValueError('Overlap gate has not passed')
  start=int(name.rsplit('s',1)[1]);expected_old=(start-28 if index==0 else start-24)
  if overlap.get('old_start')!=expected_old or overlap.get('new_start')!=start:raise ValueError('Overlap source frame mapping differs')
  annotation=json.loads((win/'exo_head_anchors.json').read_text())
  if annotation.get('name')!=name or annotation.get('video_sha256')!=digest(win/'exo.mp4'):raise ValueError('Head anchors do not match this EXO video')
  factor=float(overlap['scale_to_old']); scale=factor if index==0 else scales[0]*factor;scales.append(scale)
  prepared=win/'prepared';report=prepare(geom,win/'meta-candidate.json',win/'exo_head_anchors.json',scale,prepared)
  er=dict(e,PYTHONPATH=str(BASE/'Renderer-sparse'))
  if digest(RENDERER)!=RENDERER_SHA:raise ValueError('Official renderer differs from pinned version')
  call([BIN/'python','-B',RENDERER,'--input_dir',geom,'--out_dir',win/'prior','--meta_json_path',prepared/'meta.json','--point_size','5.0','--start_frame','0','--end_frame','48','--fish_eye_rendering','--use_mean_bg'],env=er,log=win/'render.log',timeout=360)
  prior=win/'prior'/name/'ego_Prior.mp4';checks=check_prior(prior)
  staged=win/'inference-input';staged.mkdir(); (staged/'infer.py').symlink_to((STOCK/'infer.py').resolve(strict=True));(staged/'core').symlink_to((STOCK/'core').resolve(strict=True));(staged/'meta.json').write_bytes((prepared/'meta.json').read_bytes())
  for relative,origin in [('videos',win/'exo.mp4'),('ego_priors',prior)]:
   folder=staged/'example/self_capture'/relative/name;folder.mkdir(parents=True);(folder/('exo.mp4' if relative=='videos' else 'ego_Prior.mp4')).symlink_to(origin.resolve(strict=True))
  call([BIN/'python','-B',BASE/'Renderer-sparse/scripts/convert_depth_zip_to_npy.py','--depth_path',geom/'depth','--egox_depthmaps_path',win/'depth-conversion'],env=er,log=win/'depth-conversion.log',timeout=180)
  depthdir=win/'depth-conversion/exo';files=sorted(depthdir.glob('*.npy'))
  if len(files)!=49:raise ValueError('Need all 49 indexed depth files')
  with zipfile.ZipFile(geom/'depth/exo.zip') as z:
   for i,f in enumerate(files):
    digits=''.join(c for c in f.stem if c.isdigit());raw=unpack(z,f'{i:05d}.exr',True);arr=np.load(f,allow_pickle=False)
    if not digits or int(digits)!=i or arr.dtype!=np.float32 or not np.array_equal(arr,raw):raise ValueError(f'Depth mismatch F{i}')
  target=staged/'example/self_capture/depth_maps'/name;target.parent.mkdir(parents=True);target.symlink_to(depthdir.resolve(strict=True),target_is_directory=True)
  receipt={'name':name,'frames':49,'scale_old_over_new':scale,'head_gate':{k:v for k,v in report.items() if k!='rows'},'prior':checks,'exo_sha256':digest(win/'exo.mp4'),'prior_sha256':digest(prior),'depth_sha256':[digest(f) for f in files],'meta_sha256':digest(staged/'meta.json'),'status':'technical_pass_semantic_review_pending'}
  (win/'render-receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
  if previous is None:
   old=read_video(run/'legacy'/pair/'prior.mp4');new=read_video(prior);write_video(win/'overlap-legacy-prior.mp4',np.concatenate((old[28:49],new[:21]),axis=2))
  else:
   old=read_video(previous);new=read_video(prior);write_video(win/'overlap-previous-prior.mp4',np.concatenate((old[24:49],new[:25]),axis=2))
  previous=prior
  print('PRIOR_STAGED',pair,name,'semantic review pending',flush=True)
def adapter_ok(path):
 events=json.loads(path.read_text())['events'];return [x.get('operation') for x in events]==['load_lora_weights','fuse_lora'] and all(x.get('active_adapters') and x.get('peft_config') and all(v.get('r')==256 for v in x['peft_config'].values()) for x in events)
def infer_pair(run,pair,e):
 for name in PAIRS[pair]:
  win=run/'windows'/name;receipt=json.loads((win/'render-receipt.json').read_text());stage=win/'inference-input';prior=win/'prior'/name/'ego_Prior.mp4'
  if receipt['status']!='technical_pass_semantic_review_pending' or digest(stage/'meta.json')!=receipt['meta_sha256'] or digest(prior)!=receipt['prior_sha256'] or digest(win/'exo.mp4')!=receipt['exo_sha256']:raise ValueError('Staged input changed')
  files=sorted((stage/'example/self_capture/depth_maps'/name).glob('*.npy'))
  if [digest(f) for f in files]!=receipt['depth_sha256']:raise ValueError('Staged depths changed')
  out=win/'inference-output';out.mkdir();(out/'tmp').mkdir();(out/'cache').mkdir()
  ei=dict(e,OMP_NUM_THREADS='8',PYTHONHASHSEED='846514',HF_HOME=str(out/'cache'),TMPDIR=str(out/'tmp'),EGOX_AUDIT_PATH=str(out/'runtime_audit.json'))
  print('EGOX_STARTED',pair,name,flush=True)
  cmd=['timeout','--signal=TERM','--kill-after=10s','3600s',PYTHON,'-B',ROOT/'scripts/instrumented_infer.py','--meta_data_file','./meta.json','--model_path',WEIGHTS/'Wan2.1-I2V-14B-480P-Diffusers','--lora_path',WEIGHTS/'EgoX/pytorch_lora_weights.safetensors','--lora_rank','256','--out',out,'--seed','846514','--use_GGA','--cos_sim_scaling_factor','3.0','--in_the_wild','--idx','-1']
  call(cmd,env=ei,cwd=stage,log=out/'inference.log',timeout=3650)
  if not adapter_ok(out/'runtime_audit.json'):raise ValueError('Official LoRA runtime audit missing or incorrect')
  verified=out/'verified';verified.mkdir();result=verify_take(name,out/f'{name}.mp4',verified,True);(out/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print('EGOX_VERIFIED',pair,name,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('render','infer','all'));p.add_argument('--run',type=Path,required=True);p.add_argument('--pair',choices=tuple(PAIRS),required=True);a=p.parse_args();run=a.run.resolve(strict=True); expected=(BASE/'pair02-pair04-100f-geometry-prep').resolve()
 if run!=expected:raise ValueError('Unexpected isolated run path')
 e=env(); os.environ['PATH']=e['PATH']
 with (ROOT/'.egox-inference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  call([PYTHON,'-I','-B',run/'check_runtime_008.py',ROOT,STOCK,WEIGHTS,BIN/'ffmpeg',BIN/'ffprobe'],env=e,timeout=180)
  if a.mode in ('render','all'):render_pair(run,a.pair,e)
  if a.mode in ('infer','all'):infer_pair(run,a.pair,e)
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
  print(f'CONTINUATION_STOPPED {type(exc).__name__}: {exc}',file=sys.stderr);raise SystemExit(1)
