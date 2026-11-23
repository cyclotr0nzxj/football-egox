"""EXO-only relative-depth experiment against held-out old ViPE frames."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import hashlib, json, subprocess, time, zipfile
import cv2
import numpy as np
import onnxruntime as ort

RUN=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
INPUT=RUN/'pair02-pair04-100f-prep-20261122/inputs'
MODEL=Path.home()/'.cache/football-course-cv/midas-v2_1-small/model-small.onnx'
EXPECTED='2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58'
assert hashlib.file_digest(MODEL.open('rb'),'sha256').hexdigest()==EXPECTED
cv2.setNumThreads(2)
opts=ort.SessionOptions(); opts.intra_op_num_threads=4; opts.inter_op_num_threads=1
session=ort.InferenceSession(str(MODEL),sess_options=opts,providers=['CPUExecutionProvider'])
for pair,old,tail in [
 ('Pair02',RUN/'multiclip-evidence/pair02_s0000/geometry/vipe_results/pair02_s0000',52),
 ('Pair04',RUN/'geometry-008/pair04_s0024',76),
]:
 def decode(path,n):
  b=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:v:0','-an','-pix_fmt','rgb24','-f','rawvideo','-'])
  assert len(b)==n*448*784*3
  return np.frombuffer(b,np.uint8).reshape(n,448,784,3)
 frames=decode(INPUT/pair/'exo-100.mp4',100)
 last=decode(INPUT/pair/f'exo-window-{tail:04d}.mp4',49)[-1:]
 frames=np.concatenate([frames,last])
 start=time.monotonic(); cached=OUT/f'{pair}-midas-relative.npz'
 if cached.exists():
  inverse=np.load(cached)['inverse_depth']
 else:
  inverse=[]
  for i,rgb in enumerate(frames):
   x=cv2.resize(rgb.astype(np.float32)/255,(256,256),interpolation=cv2.INTER_CUBIC)
   y,=session.run(None,{session.get_inputs()[0].name:np.ascontiguousarray(x.transpose(2,0,1)[None])})
   inverse.append(cv2.resize(y.reshape(256,256),(784,448),interpolation=cv2.INTER_CUBIC))
  inverse=np.asarray(inverse,dtype=np.float32)
  np.savez_compressed(cached,inverse_depth=inverse)
 with zipfile.ZipFile(old/'depth/exo.zip') as z:
  depths=[]
  for i in range(49):
   raw=z.read(f'{i:05d}.exr'); marker=b'Z\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00'
   assert raw.count(marker)==1
   a=cv2.imdecode(np.frombuffer(raw.replace(marker,b'Y'+marker[1:],1),np.uint8),cv2.IMREAD_UNCHANGED)
   if a.ndim==3: a=a[:,:,0]
   assert a.shape==(448,784) and np.isfinite(a).all()
   depths.append(a)
 depths=np.asarray(depths)
 yy,xx=np.mgrid[:448,:784]; mask=(yy>230)&(yy<430)&(xx%8==0)&(yy%8==0)
 # Training only even frames and valid green foreground; hold out all odd frames.
 xs=[]; ys=[]
 for i in range(0,49,2):
  im=frames[i].astype(np.int16); green=(im[:,:,1]>im[:,:,0]+7)&(im[:,:,1]>im[:,:,2]+4)
  valid=mask&green&(depths[i]>0.5)&(depths[i]<80)
  xs.append(inverse[i][valid]); ys.append(1/depths[i][valid])
 x=np.concatenate(xs); y=np.concatenate(ys); assert len(x)>1000; A=np.column_stack([x,np.ones_like(x)]); keep=np.ones(len(x),bool)
 for _ in range(5):
  a,b=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]; err=y-(a*x+b)
  mad=np.median(np.abs(err-np.median(err))); keep=np.abs(err)<max(3.5*mad,1e-4)
 rows=[]
 for i in range(49):
  im=frames[i].astype(np.int16); green=(im[:,:,1]>im[:,:,0]+7)&(im[:,:,1]>im[:,:,2]+4)
  valid=mask&green&(depths[i]>0.5)&(depths[i]<80)
  pred=a*inverse[i][valid]+b; ref=1/depths[i][valid]
  rows.append({'frame':i,'n':int(valid.sum()),'median_abs_inv_error':float(np.median(np.abs(pred-ref))),'median_relative_inv_error':float(np.median(np.abs(pred-ref)/ref)),'negative_prediction_fraction':float(np.mean(pred<=0))})
 track=np.asarray(json.loads((OUT/f'{pair}-head-track-corrected.json').read_text())['track'])
 trajectory=(RUN/'multiclip-evidence/pair02_s0000/viewpoint/trajectory-check.json' if pair=='Pair02' else RUN/'geometry-008/viewpoint-008-upright/trajectory-check.json')
 old_head=np.asarray(json.loads(trajectory.read_text())['head_centers_vipe_units'])[:,2]
 head_inverse=[]
 for i,(hx,hy) in enumerate(track[:49]):
  hx,hy=int(round(hx)),int(round(hy)); head_inverse.append(float(np.median(inverse[i,hy-2:hy+3,hx-2:hx+3])))
 fitted=1/(a*np.asarray(head_inverse)+b); rel=np.abs(fitted-old_head)/old_head
 result={'pair':pair, 'old_head_depth_comparison':{'method':'5x5 head patch fitted MiDaS Z versus legacy five-frame-smoothed head-center Z, same 49 frames','median_relative_error':float(np.median(rel)),'p90_relative_error':float(np.percentile(rel,90)),'median_predicted_to_legacy_ratio':float(np.median(fitted/old_head))},'method':'global affine MiDaS inverse depth to ViPE inverse depth, green field, even frames train, odd frames held out','a':float(a),'b':float(b),'train_count':len(x),'train_inlier_fraction':float(keep.mean()),'holdout_median_relative_inv_error':float(np.median([r['median_relative_inv_error'] for r in rows[1::2]])),'heldout_frames':rows[1::2], 'predicted_nonpositive_fraction_per_frame':[float(np.mean(a*z+b<=0)) for z in inverse], 'seconds':round(time.monotonic()-start,1),'caution':'relative-model fit to another estimate, not metric validation; independent unseen surfaces unverified'}
 (OUT/f'{pair}-depth-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(pair, result['holdout_median_relative_inv_error'],result['seconds'],flush=True)
