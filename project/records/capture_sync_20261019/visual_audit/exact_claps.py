import importlib.util,pathlib,subprocess,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sheets',P/'make_sheets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROIS={'MVI_5688':(555,185,670,305),'MVI_5689':(610,170,710,280),'MVI_5691':(125,165,240,295),'MVI_5693':(15,65,180,265),'MVI_5694':(245,105,350,230)}
STARTS={'MVI_5688':47.08,'170618':46.20,'MVI_5689':25.12,'170820':24.08,'MVI_5691':29.4,'171228':28.36,'MVI_5693':22.28,'171850':20.96,'MVI_5694':42.76,'172108':41.8}
font=ImageFont.truetype('/System/Library/Fonts/Monaco.ttf',18)
for key,start in STARTS.items():
 path=next(p['path'] for p in m.PROBE if key in p['path'])
 x0,y0,x1,y1=ROIS.get(key,(290,275,745,540));x0*=4;y0*=4;x1*=4;y1*=4
 vf=f'crop={x1-x0}:{y1-y0}:{x0}:{y0},scale=360:280'
 cmd=['ffmpeg','-v','error','-threads','2','-ss',str(start),'-i',str(m.ROOT/path),'-map','0:v:0','-vf',vf,'-frames:v','11','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-']
 data=subprocess.run(cmd,check=True,stdout=subprocess.PIPE).stdout;frames=np.frombuffer(data,dtype=np.uint8).reshape(-1,280,360,3)
 out=Image.new('RGB',(1440,930),(15,15,15));draw=ImageDraw.Draw(out)
 for i,frame in enumerate(frames):
  x,y=(i%4)*360,(i//4)*310;out.paste(Image.fromarray(frame),(x,y));t=start+i/25;draw.text((x+4,y+284),f'{t:.2f}s / F{round(t*25)}',font=font,fill='white')
 out.save(P/f'{key}_end_exact.jpg',quality=88)
 print(key,flush=True)
