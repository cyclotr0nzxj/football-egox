"""Freeze two corrected 2D tracks and stage four EXO-only window inputs."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
import cv2
import numpy as np
P=Path(__file__).resolve().parent; RUN=P.parent; INPUT=RUN/'pair02-pair04-100f-prep-20261122/inputs'; OUT=P/'windows'; OUT.mkdir(exist_ok=True)
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
manifest=[]
for pair,start,offsets,target in [('Pair02',0,[28,52],[290,183]),('Pair04',24,[28,52],[650,184])]:
 data=json.loads((P/f'{pair}-head-track-corrected.json').read_text()); tr=np.asarray(data['track']); assert tr.shape==(101,2)
 cap=cv2.VideoCapture(str(INPUT/pair/'exo-100.mp4')); frames=[]
 while True:
  ok,im=cap.read()
  if not ok:break
  frames.append(im)
 cap.release(); assert len(frames)==100
 gray=[cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY) for i in [0,99]]
 mask=np.ones(gray[0].shape,np.uint8)*255
 # Foreground exclusion deliberately generous; applies across the sequence.
 if pair=='Pair02': mask[95:330,240:590]=0
 else: mask[90:340,250:710]=0
 orb=cv2.ORB_create(nfeatures=3500); k1,d1=orb.detectAndCompute(gray[0],mask); k2,d2=orb.detectAndCompute(gray[1],mask)
 matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2); good=[a for a,b in matches if a.distance<.75*b.distance]
 if len(good)<20: raise ValueError('Insufficient static camera features')
 p1=np.float32([k1[m.queryIdx].pt for m in good]);p2=np.float32([k2[m.trainIdx].pt for m in good]); H,inliers=cv2.findHomography(p1,p2,cv2.RANSAC,2.0); inliers=inliers.ravel().astype(bool)
 shifts=np.linalg.norm(p1[inliers]-p2[inliers],axis=1)
 camera={'good_matches':len(good),'inliers':int(inliers.sum()),'inlier_median_identity_displacement_px':float(np.median(shifts)),'inlier_p90_identity_displacement_px':float(np.percentile(shifts,90)),'scope':'matched static points only; not lens calibration'}
 for off in offsets:
  source=start+off; name=f'{pair.lower()}_s{source:04d}'; exo=INPUT/pair/f'exo-window-{source:04d}.mp4'; assert exo.is_file()
  rows=[{'frame':i,'x':int(round(x)),'y':int(round(y))} for i,(x,y) in enumerate(tr[off:off+49])]
  assert len(rows)==49 and all(2<=r['x']<782 and 2<=r['y']<446 for r in rows)
  annotation={'name':name,'width':784,'height':448,'video_sha256':sha(exo),'target_pixel':target,'frames':rows,'provenance':'old anchors 0-48; optical flow with visually estimated sparse corrections after 48; not ground truth or gaze'}
  dst=OUT/name; dst.mkdir(exist_ok=True); (dst/'exo_head_anchors.json').write_text(json.dumps(annotation,indent=2)+'\n')
  manifest.append({'name':name,'source_start':source,'source_end':source+48,'exo':str(exo.relative_to(RUN)),'exo_sha256':sha(exo),'annotation_sha256':sha(dst/'exo_head_anchors.json'),'camera_check':camera})
(OUT/'manifest.json').write_text(json.dumps({'windows':manifest,'caution':'EXO and 2D anchor preparation only. No 3D trajectory, new ViPE geometry, rendered prior or EgoX launch readiness is asserted.'},indent=2,allow_nan=False)+'\n')
print(json.dumps(manifest,indent=2))
