"""Export a draft 100-frame extension and same-frame overlap evidence, without Nano."""
import argparse,hashlib,json,subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda: f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def run(args):return subprocess.run(list(map(str,args)),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout

def read(path):
 info=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries','stream=width,height,avg_frame_rate,time_base:frame=pts,width,height','-of','json',path])); s,=info['streams']; w,h=s['width'],s['height'];fps=Fraction(s['avg_frame_rate']); fs=info['frames']
 if (w not in (448,1232)) or h!=448 or fps!=30 or len(fs)!=49:raise ValueError(f'Expected 49 30-fps source frames: {path}')
 tb=Fraction(s['time_base']);
 if any(int(f['pts'])*tb!=Fraction(i,30) or f['width']!=w or f['height']!=h for i,f in enumerate(fs)):raise ValueError(f'Unexpected source PTS: {path}')
 raw=run(['ffmpeg','-v','error','-nostdin','-xerror','-i',path,'-map','0:v:0','-an','-pix_fmt','rgb24','-f','rawvideo','-']);
 if len(raw)!=49*448*w*3:raise ValueError('Unexpected raw frame count')
 frames=np.frombuffer(raw,np.uint8).reshape(49,448,w,3)
 return frames[:,:,w-448:,:].copy()

def write(path,frames):
 n,h,w,_=frames.shape
 proc=subprocess.run(['ffmpeg','-v','error','-nostdin','-xerror','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','25','-i','-','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','25000','-movflags','+faststart',str(path)],input=frames.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if proc.returncode:raise RuntimeError(proc.stderr.decode(errors='replace'))
 info=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries','stream=width,height,avg_frame_rate,time_base:frame=pts','-of','json',path]));s,=info['streams'];tb=Fraction(s['time_base'])
 if (s['width'],s['height'],Fraction(s['avg_frame_rate']),len(info['frames']))!=(w,h,25,n) or any(int(f['pts'])*tb!=Fraction(i,25) for i,f in enumerate(info['frames'])):raise ValueError('Output format/PTS mismatch')
 run(['ffmpeg','-v','error','-nostdin','-xerror','-i',path,'-f','null','-'])
 return sha256(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--middle',type=Path,required=True);p.add_argument('--last',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-start',type=int,choices=(0,24),required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a new output directory')
 frames=[read(x) for x in (a.first,a.middle,a.last)]; first,middle,last=frames
 # The first 49 remain from the existing result. Hard cuts are reviewable, not accepted automatically.
 final=np.concatenate((first,middle[21:48],last[24:48]));assert final.shape==(100,448,448,3)
 a.output.mkdir(parents=True)
 digest=write(a.output/'extension-draft-100f-25fps.mp4',final)
 overlap1=write(a.output/'overlap-first-middle-25fps.mp4',np.concatenate((first[28:49],middle[:21]),axis=2))
 overlap2=write(a.output/'overlap-middle-last-25fps.mp4',np.concatenate((middle[24:49],last[:25]),axis=2))
 receipt={'status':'draft_technical_pass_visual_review_pending','source_start':a.source_start,'source_end':a.source_start+99,'frame_count':100,'fps':25,'seconds':4.0,'cuts_source_frames':[a.source_start+49,a.source_start+76],'selection':'first[0:49] + middle[21:48] + last[24:48]','out_sha256':digest,'overlap_first_middle_sha256':overlap1,'overlap_middle_last_sha256':overlap2,'source_sha256':{n:sha256(path) for n,path in zip(('first','middle','last'),(a.first,a.middle,a.last))},'visual_acceptance':'pending; compare both 21/25-frame overlaps, same-frame identity, single ball, seams and original EXO; Nano only after freeze'}
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
