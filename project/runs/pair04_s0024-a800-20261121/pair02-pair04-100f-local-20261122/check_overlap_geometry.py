"""Align same-camera ViPE depths in overlap and reject unstable scale or intrinsics.

Inputs are ViPE artifact directories. Start indices are source-frame numbers.
Only a single scale is fitted to static background on alternating overlap frames.
This checks internal consistency, not metric correctness or head tracking.
"""
import argparse, json, os, zipfile
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
MARKER=b'Z\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00'

def unpack(archive, name, depth):
 raw=archive.read(name)
 if depth:
  if raw.count(MARKER)!=1: raise ValueError('Expected one EXR Z channel')
  raw=raw.replace(MARKER,b'Y'+MARKER[1:],1)
 array=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
 if array is None or array.shape!=(448,784): raise ValueError('Invalid artifact grid')
 if depth and (not np.isfinite(array).all() or np.any(array<=0)): raise ValueError('Invalid Z depth')
 return array

def load_k(root):
 with np.load(root/'intrinsics/exo.npz',allow_pickle=False) as z:
  a=z['data']; ids=z['inds']
  if a.shape!=(49,4) or not np.array_equal(ids,np.arange(49)) or not np.allclose(a,a[0],atol=1e-4): raise ValueError('K not constant/indexed')
  return a[0].astype(float)

def assess(old, new, old_start, new_start, synthetic_scale=None, synthetic_k=None):
 oldk=load_k(old); newk=load_k(new) if synthetic_k is None else synthetic_k
 if np.max(np.abs(newk[:2]/oldk[:2]-1))>.01 or np.max(np.abs(newk[2:]-oldk[2:]))>.5: raise ValueError('EXO K mismatch; a scale cannot repair different rays')
 overlap=range(max(old_start,new_start),min(old_start+49,new_start+49))
 if len(overlap)<8: raise ValueError('Insufficient overlap')
 summary=[]; training=[]
 with zipfile.ZipFile(old/'depth/exo.zip') as od, zipfile.ZipFile(old/'mask/exo.zip') as om, zipfile.ZipFile(new/'depth/exo.zip') as nd, zipfile.ZipFile(new/'mask/exo.zip') as nm:
  for frame in overlap:
   i=frame-old_start; j=frame-new_start
   a=unpack(od,f'{i:05d}.exr',True)[::4,::4]; ma=unpack(om,f'{i:05d}.png',False)[::4,::4]
   # Synthetic mode reads the same frame as the reference, then perturbs scale.
   if synthetic_scale is not None:
    factor=synthetic_scale(frame) if callable(synthetic_scale) else synthetic_scale
    b=a*factor; mb=ma
   else:
    b=unpack(nd,f'{j:05d}.exr',True)[::4,::4]; mb=unpack(nm,f'{j:05d}.png',False)[::4,::4]
   valid=(ma==0)&(mb==0)&(a>.5)&(a<80)&(b>.5)&(b<80)
   if valid.sum()<1000: raise ValueError(f'Insufficient background pixels at source frame {frame}')
   ratios=a[valid]/b[valid]; summary.append((frame,ratios));
   if (frame-overlap.start)%2==0: training.append(ratios)
 scale=float(np.median(np.concatenate(training)))
 held=[]
 for frame,ratios in summary:
  if (frame-overlap.start)%2: held.append({'frame':frame,'median_relative_error':float(np.median(np.abs(scale/ratios-1))),'p90_relative_error':float(np.percentile(np.abs(scale/ratios-1),90))})
 med=float(np.median([r['median_relative_error'] for r in held])); p90=float(np.percentile([r['p90_relative_error'] for r in held],90))
 return {'old_start':old_start,'new_start':new_start,'overlap':[overlap.start,overlap.stop-1], 'old_K':oldk.tolist(),'new_K':newk.tolist(),'scale_to_old':scale,'heldout_frames':held,'heldout_median_relative_error':med,'heldout_frame_p90_of_pixel_p90_error':p90,'passed_provisional_gate':bool(med<=.10 and p90<=.25),'gate':'Provisional consistency screen: median <= 10%, p90-of-frame-p90 <= 25%; not metric validation; inspect foreground and rendered seams separately.'}

def main():
 p=argparse.ArgumentParser(); p.add_argument('old',type=Path); p.add_argument('new',type=Path); p.add_argument('--old-start',type=int,required=True); p.add_argument('--new-start',type=int,required=True); p.add_argument('--output',type=Path,required=True)
 args=p.parse_args(); result=assess(args.old,args.new,args.old_start,args.new_start)
 args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='heldout_frames'})); return 0 if result['passed_provisional_gate'] else 2
if __name__=='__main__': raise SystemExit(main())
