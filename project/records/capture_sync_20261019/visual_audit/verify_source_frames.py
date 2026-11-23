import importlib.util,pathlib,subprocess,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sheets',P/'make_sheets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROIS={'MVI_5688':(555,185,670,305),'MVI_5689':(610,170,710,280),'MVI_5691':(125,165,240,295),'MVI_5693':(15,65,180,265),'MVI_5694':(245,105,350,230)}
CENTERS={'MVI_5688':1180,'170618':1161,'MVI_5689':631,'170820':607,'MVI_5691':738,'171228':715,'MVI_5693':560,'171850':529,'MVI_5694':1072,'172108':1051}
font=ImageFont.truetype('/System/Library/Fonts/Monaco.ttf',18);results=[]
for key,center in CENTERS.items():
 path=next(p['path'] for p in m.PROBE if key in p['path']);first=center-2
 x0,y0,x1,y1=ROIS.get(key,(290,275,745,540));x0*=4;y0*=4;x1*=4;y1*=4
 vf=f'crop={x1-x0}:{y1-y0}:{x0}:{y0},scale=360:280'
 base=['ffmpeg','-v','error','-hwaccel','videotoolbox','-threads','4']
 suffix=['-map','0:v:0','-frames:v','5','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-']
 headcmd=base+['-i',str(m.ROOT/path),'-vf',f'select=between(n\\,{first}\\,{center+2}),'+vf]+suffix
 data=subprocess.run(headcmd,check=True,stdout=subprocess.PIPE).stdout;frames=np.frombuffer(data,dtype=np.uint8).reshape(-1,280,360,3)
 seekcmd=base+['-ss',str(first/25),'-i',str(m.ROOT/path),'-vf',vf]+suffix
 seekdata=subprocess.run(seekcmd,check=True,stdout=subprocess.PIPE).stdout;seekframes=np.frombuffer(seekdata,dtype=np.uint8).reshape(-1,280,360,3)
 equal=bool(np.array_equal(frames,seekframes));maxerr=int(np.max(np.abs(frames.astype('int16')-seekframes.astype('int16'))))
 out=Image.new('RGB',(1800,310),(15,15,15));draw=ImageDraw.Draw(out)
 for i,frame in enumerate(frames):
  x=i*360;out.paste(Image.fromarray(frame),(x,0));n=first+i;draw.text((x+4,284),f'F{n} / {n/25:.2f}s',font=font,fill='white')
 out.save(P/f'{key}_source_frame_verified.jpg',quality=88)
 row={'key':key,'source_path':path,'frame_indices':list(range(first,center+3)),'method':'decode from start and select between source frame n; videotoolbox; 25 fps source','seek_matches_absolute_frame_selection':equal,'max_pixel_error':maxerr}
 results.append(row);(P/'absolute_frame_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
 print(key,'exact_seek_match',equal,'maxerr',maxerr,flush=True)
