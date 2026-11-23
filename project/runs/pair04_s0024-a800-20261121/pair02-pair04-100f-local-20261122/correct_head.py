"""Apply visually estimated sparse drift corrections; keep original 49 anchors."""
from pathlib import Path
import json, subprocess
import cv2
import numpy as np
P=Path(__file__).resolve().parent; RUN=P.parent; IN=RUN/'pair02-pair04-100f-prep-20261122/inputs'
corrections={'Pair02':[(48,0,0),(60,3,0),(72,6,0),(84,6,0),(100,7,0)],'Pair04':[(48,0,0),(63,-1,-1),(70,-5,-5),(80,-8,-10),(90,-11,-13),(100,-13,-18)]}
for pair in corrections:
 data=json.loads((P/f'{pair}-head-track-candidate.json').read_text()); track=np.array(data['track']); anchors=np.array(corrections[pair]);
 for axis in range(2): track[48:,axis]+=np.interp(np.arange(48,101),anchors[:,0],anchors[:,axis+1])
 data.update({'track':track.tolist(),'correction_keyframes_local_to_100_frame_clip':anchors.tolist(),'warning':'visually corrected EXO head-surface candidate, not individually manually marked; real head optical center and gaze unknown; 3D depth unverified'})
 (P/f'{pair}-head-track-corrected.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 def read(path,n):
  b=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(path),'-pix_fmt','rgb24','-f','rawvideo','-']); assert len(b)==n*448*784*3
  return np.frombuffer(b,np.uint8).reshape(n,448,784,3)
 tail=52 if pair=='Pair02' else 76
 imgs=np.concatenate((read(IN/pair/'exo-100.mp4',100),read(IN/pair/f'exo-window-{tail:04d}.mp4',49)[-1:])); crops=[]
 for i in range(48,101):
  im=imgs[i].copy(); x,y=np.rint(track[i]).astype(int); cv2.drawMarker(im,(x,y),(255,0,0),cv2.MARKER_CROSS,10,1)
  pad=cv2.copyMakeBorder(im,50,50,70,70,cv2.BORDER_CONSTANT); crop=pad[y:y+100,x:x+140]
  cv2.putText(crop,f'F{i}',(4,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
  crops.append(cv2.cvtColor(cv2.resize(crop,(280,200)),cv2.COLOR_RGB2BGR))
 for k in range(0,len(crops),15):
  c=crops[k:k+15]; c += [np.zeros_like(crops[0])]*(15-len(c))
  cv2.imwrite(str(P/f'{pair}-corrected-review-{k//15+1}.jpg'),np.vstack([np.hstack(c[j:j+5]) for j in (0,5,10)]),[cv2.IMWRITE_JPEG_QUALITY,92])
