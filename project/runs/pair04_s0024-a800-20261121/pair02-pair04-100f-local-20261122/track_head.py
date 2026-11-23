"""Track EXO head patches; retain old annotations; report uncertainty and overlays."""
from pathlib import Path
import json, subprocess
import cv2
import numpy as np
RUN=Path(__file__).resolve().parent.parent; OUT=Path(__file__).resolve().parent; IN=RUN/'pair02-pair04-100f-prep-20261122/inputs'
for pair,old,tail in [('Pair02',RUN/'multiclip-prep/pair02_s0000/exo_head_anchors.json',52),('Pair04',RUN/'geometry-008/exo_head_anchors.json',76)]:
 def read(path,n):
  b=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(path),'-pix_fmt','rgb24','-f','rawvideo','-'])
  assert len(b)==n*448*784*3
  return np.frombuffer(b,np.uint8).reshape(n,448,784,3)
 images=np.concatenate((read(IN/pair/'exo-100.mp4',100),read(IN/pair/f'exo-window-{tail:04d}.mp4',49)[-1:]))
 gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in images]
 old_rows=json.loads(old.read_text())['frames']; old_track=np.array([[r['x'],r['y']] for r in old_rows],float)
 def follow(i,j,point):
  x,y=point; m=np.zeros((448,784),np.uint8); cv2.circle(m,(round(x),round(y)),8,255,-1)
  pts=cv2.goodFeaturesToTrack(gray[i],maxCorners=30,qualityLevel=.01,minDistance=2,mask=m,blockSize=3)
  if pts is None or len(pts)<3: return point,0,None
  nxt,s,_=cv2.calcOpticalFlowPyrLK(gray[i],gray[j],pts,None,winSize=(21,21),maxLevel=2)
  back,t,_=cv2.calcOpticalFlowPyrLK(gray[j],gray[i],nxt,None,winSize=(21,21),maxLevel=2)
  good=(s.ravel()>0)&(t.ravel()>0)&(np.linalg.norm((back-pts).reshape(-1,2),axis=1)<1)
  if good.sum()<3: return point,int(good.sum()),None
  moves=(nxt-pts).reshape(-1,2)[good]; delta=np.median(moves,axis=0)
  return point+delta,int(good.sum()),float(np.median(np.linalg.norm(moves-delta,axis=1)))
 test=[]
 for i in range(48):
  est,n,spread=follow(i,i+1,old_track[i]); test.append(float(np.linalg.norm(est-old_track[i+1])))
 track=list(old_track); records=[]
 for i in range(48,100):
  est,n,spread=follow(i,i+1,np.array(track[-1])); track.append(est); records.append({'frame':i+1,'good_points':n,'spread':spread})
 track=np.asarray(track); (OUT/f'{pair}-head-track-candidate.json').write_text(json.dumps({'track':track.tolist(),'old_anchor_one_step_error_px':test,'new_flow':records,'warning':'uncorrected 2D flow candidate; accumulating drift and real gaze not validated'},indent=2)+'\n')
 crops=[]
 for i in range(48,101):
  im=images[i].copy(); x,y=track[i]; x=int(round(x)); y=int(round(y)); cv2.drawMarker(im,(x,y),(255,0,0),cv2.MARKER_CROSS,12,2)
  padded=cv2.copyMakeBorder(im,50,50,70,70,cv2.BORDER_CONSTANT); crop=padded[y:y+100,x:x+140]
  cv2.putText(crop,f'F{i}',(4,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
  crops.append(cv2.cvtColor(cv2.resize(crop,(280,200)),cv2.COLOR_RGB2BGR))
 for k in range(0,len(crops),15):
  c=crops[k:k+15]; c += [np.zeros_like(crops[0])]*(15-len(c)); sheet=np.vstack([np.hstack(c[j:j+5]) for j in (0,5,10)])
  cv2.imwrite(str(OUT/f'{pair}-head-review-{k//15+1}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,92])
 print(pair, 'one step median/max',np.median(test),max(test),'tail',track[-1].round(1),flush=True)
