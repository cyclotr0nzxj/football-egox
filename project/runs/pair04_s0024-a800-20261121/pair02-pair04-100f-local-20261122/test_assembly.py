"""Use EXO-only frames as synthetic clips to exercise frame cuts; not model results."""
import json,subprocess,tempfile
from pathlib import Path
import numpy as np
from assemble_extension import read,run
P=Path(__file__).resolve().parent;I=P.parent/'pair02-pair04-100f-prep-20261122/inputs'; result=[]
for pair,start,mid,last in [('Pair02',0,28,52),('Pair04',24,52,76)]:
 with tempfile.TemporaryDirectory() as t:
  d=Path(t); sources=[I/pair/'exo-100.mp4',I/pair/f'exo-window-{mid:04d}.mp4',I/pair/f'exo-window-{last:04d}.mp4']; names=[]
  for i,src in enumerate(sources):
   out=d/f'window{i}.mp4';run(['ffmpeg','-v','error','-nostdin','-xerror','-n','-i',src,'-vf','crop=448:448:168:0,settb=1/30,setpts=N','-frames:v','49','-an','-fps_mode','passthrough','-enc_time_base','1:30','-c:v','libx264','-preset','ultrafast','-crf','12','-pix_fmt','yuv420p','-video_track_timescale','30000',out]);names.append(out)
  outdir=d/'out';run(['/usr/local/Caskroom/miniforge/base/bin/python3.12','-B',P/'assemble_extension.py','--first',names[0],'--middle',names[1],'--last',names[2],'--output',outdir,'--source-start',start]); rec=json.loads((outdir/'receipt.json').read_text()); assert rec['frame_count']==100 and rec['cuts_source_frames']==[start+49,start+76]
  fs=[read(n) for n in names]; expected=np.concatenate((fs[0],fs[1][21:48],fs[2][24:48])); raw=run(['ffmpeg','-v','error','-nostdin','-xerror','-i',outdir/'extension-draft-100f-25fps.mp4','-pix_fmt','rgb24','-f','rawvideo','-']); decoded=np.frombuffer(raw,np.uint8).reshape(100,448,448,3)
  mae=np.mean(np.abs(decoded.astype(np.int16)-expected.astype(np.int16)),axis=(1,2,3));assert mae.max()<5
  result.append({'pair':pair,'type':'EXO-only synthetic footage, not EgoX output','frames':100,'fps':25,'source_cuts':rec['cuts_source_frames'],'max_frame_rgb_mae_vs_selected_input':float(mae.max()),'decoded_and_pts_checked':True,'overlap_videos_frames':[21,25]})
(P/'assembly-selftest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
