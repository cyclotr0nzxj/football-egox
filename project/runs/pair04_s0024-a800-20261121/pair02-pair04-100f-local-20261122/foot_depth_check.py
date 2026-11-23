"""Test red-shoe ground-plane head depth against the legacy trajectory."""
from pathlib import Path
import json,subprocess
import cv2,numpy as np
P=Path(__file__).resolve().parent; R=P.parent; IN=R/'pair02-pair04-100f-prep-20261122/inputs'
for pair,tail,view,geo in [('Pair02',52,R/'multiclip-evidence/pair02_s0000/viewpoint/trajectory-check.json',R/'multiclip-evidence/pair02_s0000/geometry/vipe_results/pair02_s0000'),('Pair04',76,R/'geometry-008/viewpoint-008-upright/trajectory-check.json',R/'geometry-008/pair04_s0024')]:
 def read(path,n):
  b=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(path),'-pix_fmt','rgb24','-f','rawvideo','-']); assert len(b)==n*448*784*3
  return np.frombuffer(b,np.uint8).reshape(n,448,784,3)
 imgs=np.concatenate((read(IN/pair/'exo-100.mp4',100),read(IN/pair/f'exo-window-{tail:04d}.mp4',49)[-1:])); head=np.array(json.loads((P/f'{pair}-head-track-corrected.json').read_text())['track'])
 meta=json.loads(view.read_text());up=np.array(meta['ground_up']); up=up/np.linalg.norm(up);h=meta['fitted_exo_camera_height_vipe_units']; K=np.load(geo/'intrinsics/exo.npz')['data'][0]; fx,fy,cx,cy=K
 z=[]; foot=[]; crops=[]
 for i,(im,(hx,hy)) in enumerate(zip(imgs,head,strict=True)):
  hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV); yy,xx=np.mgrid[:448,:784]
  region=(abs(xx-hx)<90)&(yy>hy+55)&(yy<hy+210)
  red=(((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>105)&(hsv[:,:,2]>85)&region).astype(np.uint8)
  n,labels,stats,_=cv2.connectedComponentsWithStats(red,8); candidates=[]
  for j in range(1,n):
   x,y,w,ht,area=stats[j]
   if area>=3 and ht<=40 and w<=80:candidates.append((area,x,y,w,ht,j))
  candidates.sort(reverse=True); selected=candidates[:2]
  if not selected: foot.append(None); z.append(None); continue
  pix=np.isin(labels,[v[5] for v in selected]); ys,xs=np.where(pix); fy_px=float(np.percentile(ys,90)); fx_px=float(np.median(xs[ys>=np.percentile(ys,60)]));foot.append([fx_px,fy_px])
  ray=np.array([(fx_px-cx)/fx,(fy_px-cy)/fy,1.]); ground=ray*(-h/(up@ray)); direction=np.array([(hx-cx)/fx,(hy-cy)/fy,1.]); lateral=direction-(direction@up)*up
  value=float((ground@lateral)/(lateral@lateral)); z.append(value if value>0 else None)
  if i%5==0 or i in (48,99,100):
   img=im.copy();cv2.drawMarker(img,(round(hx),round(hy)),(255,0,0),cv2.MARKER_CROSS,10,2);cv2.drawMarker(img,(round(fx_px),round(fy_px)),(255,255,0),cv2.MARKER_CROSS,10,2)
   x0=max(0,int(hx)-100); x1=min(784,int(hx)+100);y0=max(0,int(hy)-20);y1=min(448,int(hy)+230); crop=img[y0:y1,x0:x1];crop=cv2.resize(crop,(200,250));cv2.putText(crop,f'F{i}',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);crops.append(cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
 old=np.array(meta['head_centers_vipe_units'])[:,2]; good=[i for i in range(49) if z[i] is not None]; rel=np.abs(np.array([z[i] for i in good])-old[good])/old[good]
 result={'pair':pair,'method':'automatic red-shoe contact proxy, same vertical line in fitted ViPE ground plane; not optical head center','feet':foot,'head_z':z,'old_window_support':len(good),'median_relative_z_disagreement':float(np.median(rel)),'p90_relative_z_disagreement':float(np.percentile(rel,90)),'caveat':'red threshold/foot contact, ground plane and person upright approximations require independent review'}
 (P/f'{pair}-foot-depth.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for k in range(0,len(crops),15):
  c=crops[k:k+15]; c += [np.zeros_like(crops[0])]*(15-len(c));cv2.imwrite(str(P/f'{pair}-foot-review-{k//15+1}.jpg'),np.vstack([np.hstack(c[j:j+5]) for j in (0,5,10)]))
 print(pair,'support',len(good),'median',np.median(rel),'p90',np.percentile(rel,90),flush=True)
