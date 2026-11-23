import pathlib,subprocess,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=pathlib.Path(__file__).resolve().parent;R=P.parents[2];probe=json.loads((R/'records/capture_sync_20261019/source_probe.json').read_text());font=ImageFont.truetype('/System/Library/Fonts/Monaco.ttf',16)
for ego,exo,first,D in [('170618','MVI_5688',3,19),('171228','MVI_5691',52,23),('171850','MVI_5693',44,31)]:
 out=Image.new('RGB',(3600,860),(15,15,15));draw=ImageDraw.Draw(out)
 for row,key,start in [(0,exo,first+D),(1,ego,first)]:
  path=next(p['path'] for p in probe if key in p['path'])
  cmd=['ffmpeg','-v','error','-threads','2','-ss',str(start/25),'-i',str(R/path),'-map','0:v:0','-vf','scale=720:405','-frames:v','5','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-']
  data=subprocess.run(cmd,check=True,stdout=subprocess.PIPE).stdout;frames=np.frombuffer(data,dtype=np.uint8).reshape(-1,405,720,3)
  for i,frame in enumerate(frames):
   x,y=i*720,row*430;out.paste(Image.fromarray(frame),(x,y));n=start+i;draw.text((x+4,y+408),f'{key} F{n} {n/25:.2f}s',font=font,fill='white')
 out.save(P/f'{exo}_start_clap.jpg',quality=86);print(exo,flush=True)
