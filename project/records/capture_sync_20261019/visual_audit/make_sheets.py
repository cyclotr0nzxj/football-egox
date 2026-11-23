import io,json,math,pathlib,subprocess
import numpy as np
from PIL import Image,ImageDraw

ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(__file__).resolve().parent
PROBE=json.loads((ROOT/'records/capture_sync_20261019/source_probe.json').read_text())

def get_frames(path, start=0, duration=None, fps=1, width=384):
    height=width*9//16
    args=['ffmpeg','-v','error','-threads','2','-ss',str(start),'-i',str(ROOT/path),'-map','0:v:0']
    if duration is not None: args+=['-t',str(duration)]
    args+=['-vf',f'fps={fps},scale={width}:{height}','-f','rawvideo','-pix_fmt','rgb24','-']
    data=subprocess.run(args,check=True,stdout=subprocess.PIPE).stdout
    frames=np.frombuffer(data,dtype=np.uint8).reshape(-1,height,width,3)
    return frames

def sheet(frames,times,out,columns=4):
    h,w=frames.shape[1:3]
    canvas=Image.new('RGB',(w*columns,(h+24)*math.ceil(len(frames)/columns)),(20,20,20))
    draw=ImageDraw.Draw(canvas)
    for i,(frame,t) in enumerate(zip(frames,times)):
        x,y=(i%columns)*w,(i//columns)*(h+24)
        canvas.paste(Image.fromarray(frame),(x,y))
        draw.text((x+6,y+h+4),f'{t:.2f} s',fill='white')
    canvas.save(out,quality=83)

if __name__=='__main__':
    for item in PROBE:
        path=item['path']; name=pathlib.Path(path).stem
        short=name.split('_')[3] if 'ego/' in path else name
        frames=get_frames(path)
        duration=float(next(s for s in item['streams'] if s['codec_type']=='video')['duration'])
        indices=list(range(min(13,len(frames))))+list(range(max(13,len(frames)-9),len(frames)))
        indices=sorted(set(indices))
        sheet(frames[indices],[i+0.48 for i in indices],OUT/f'{short}_start_end.jpg')
        mids=list(range(0,len(frames),5))
        sheet(frames[mids],[i+0.48 for i in mids],OUT/f'{short}_overview.jpg')
        print(short,'duration',duration,'thumbnails',len(indices)+len(mids),flush=True)
