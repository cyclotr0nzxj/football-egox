import io,json,pathlib,subprocess,time
from PIL import Image,ImageDraw,ImageFont

OUT=pathlib.Path(__file__).resolve().parent
ROOT=OUT.parents[2]
MANIFEST=json.loads((ROOT/'records/capture_sync_20261019/sync_manifest.json').read_text())
FONT=ImageFont.truetype('/System/Library/Fonts/Monaco.ttf',18)
ROIS={'01':(555,185,670,305),'02':(610,170,710,280),'03':(125,165,240,295),'04':(15,65,180,265),'05':(245,105,350,230)}
STARTS={'01':6,'02':0,'03':55,'04':47,'05':0}
MIDS={'01':615,'02':430,'03':393,'04':304,'05':443}

def frame(path,index):
    cmd=['ffmpeg','-v','error','-threads','2','-ss',str(index/25),'-i',str(path),'-map','0:v:0','-frames:v','1','-f','image2pipe','-vcodec','mjpeg','-q:v','3','-']
    return Image.open(io.BytesIO(subprocess.run(cmd,check=True,stdout=subprocess.PIPE).stdout)).convert('RGB')

def thumb(im, width):
    return im.resize((width,round(im.height*width/im.width)),Image.Resampling.LANCZOS)

ready=[];overviews={};deadline=time.monotonic()+3600
while len(ready)<len(MANIFEST['pairs']):
    for pair in MANIFEST['pairs']:
        pid=pair['id']
        if any(row['id']==pid for row in ready):continue
        path=ROOT/MANIFEST['output_root']/pid/'comparison.mp4'
        if not path.is_file():continue
        probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],check=True,stdout=subprocess.PIPE).stdout)
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        center=pair['end_clap_source_frames_ego'][-1]-pair['ego_start_frame']
        ns=[STARTS[pid],MIDS[pid],center-1,center,center+1,pair['frame_count']-1]
        ims=[frame(path,n) for n in ns]
        h=round(ims[0].height*960/ims[0].width);sheet=Image.new('RGB',(1920,(h+32)*3),(12,12,12));draw=ImageDraw.Draw(sheet)
        for j,(im,n) in enumerate(zip(ims,ns)):
            x,y=(j%2)*960,(j//2)*(h+32);sheet.paste(thumb(im,960),(x,y));draw.text((x+8,y+h+4),f'Pair {pid} | output F{n} | {n/25:.2f}s',font=FONT,fill='white')
        sheet.save(OUT/f'pair_{pid}_samples.jpg',quality=86)
        contact=Image.new('RGB',(1080,620),(12,12,12));draw=ImageDraw.Draw(contact)
        for j,(im,n) in enumerate(zip(ims[2:5],ns[2:5])):
            for row,roi in [(0,ROIS[pid]),(1,(1250,275,1705,540))]:
                contact.paste(im.crop(roi).resize((360,280),Image.Resampling.LANCZOS),(j*360,row*310))
                draw.text((j*360+5,row*310+283),f'{"EXO" if row==0 else "EGO"} F{n} {n/25:.2f}s',font=FONT,fill='white')
        contact.save(OUT/f'pair_{pid}_end_clap.jpg',quality=87)
        overviews[pid]=[(ims[k],ns[k],label) for k,label in [(0,'opening'),(1,'middle'),(3,'ending clap')]]
        ready.append({'id':pid,'comparison_path':str(path.relative_to(ROOT)),'video':{'width':video['width'],'height':video['height'],'fps':video['avg_frame_rate'],'nb_frames':video.get('nb_frames'),'duration':video.get('duration')},'sample_output_frames':ns,'status':'extracted_pending_actual_visual_review','sample_evidence':f'pair_{pid}_samples.jpg','clap_evidence':f'pair_{pid}_end_clap.jpg'})
        (OUT/'extracted_samples.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2))
        oh=round(ims[0].height*640/ims[0].width);overview=Image.new('RGB',(1920,(oh+30)*len(overviews)),(12,12,12));od=ImageDraw.Draw(overview)
        for rr,pp in enumerate(sorted(overviews)):
            for cc,(im,n,label) in enumerate(overviews[pp]):
                x,y=cc*640,rr*(oh+30);overview.paste(thumb(im,640),(x,y));od.text((x+5,y+oh+3),f'{pp} {label} F{n} ({n/25:.2f}s)',font=FONT,fill='white')
        overview.save(OUT/'overview.jpg',quality=85)
        print('READY',pid,flush=True)
    if len(ready)==len(MANIFEST['pairs']):break
    if time.monotonic()>deadline:raise TimeoutError('Completed previews did not all appear within one hour')
    time.sleep(15)
print('ALL_EXTRACTED',flush=True)
