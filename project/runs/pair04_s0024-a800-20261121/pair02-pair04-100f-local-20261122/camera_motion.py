"""Measure background image displacement, not physical camera calibration."""
from pathlib import Path
import cv2,json,numpy as np
P=Path(__file__).resolve().parent; I=P.parent/'pair02-pair04-100f-prep-20261122/inputs'; result={}
for pair in ('Pair02','Pair04'):
 cap=cv2.VideoCapture(str(I/pair/'exo-100.mp4')); frames=[]
 while True:
  ok,im=cap.read()
  if not ok: break
  frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
 cap.release(); assert len(frames)==100
 mask=np.ones((448,784),np.uint8)*255; mask[90:340,240:710]=0
 orb=cv2.ORB_create(nfeatures=3500); kp,des=orb.detectAndCompute(frames[0],mask); rows=[]
 for i in (0,14,28,40,48,52,60,76,90,99):
  kp2,des2=orb.detectAndCompute(frames[i],mask); pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des,des2,k=2)
  good=[a for a,b in pairs if a.distance<.75*b.distance]; a=np.float32([kp[m.queryIdx].pt for m in good]);b=np.float32([kp2[m.trainIdx].pt for m in good])
  mat,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.0); keep=inliers.ravel().astype(bool)
  centre=mat@np.array([392,224,1])-np.array([392,224]); residual=np.linalg.norm((np.column_stack([a,np.ones(len(a))])@mat.T)-b,axis=1)
  rows.append({'frame_local':i,'inliers':int(keep.sum()),'center_shift_px':centre.tolist(),'rotation_deg':float(np.degrees(np.arctan2(mat[1,0],mat[0,0]))),'scale':float(np.hypot(mat[0,0],mat[1,0])),'inlier_median_residual_px':float(np.median(residual[keep]))})
 result[pair]=rows
(P/'camera-motion.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); print(json.dumps(result,indent=2))
