"""CPU-only native output probes and labeled contact sheets for visual review."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from build_artifacts import decode, sha, TAKE

root=Path('/root/autodl-tmp/football_egox')
runs=json.loads((root/'records/attempts.json').read_text())
run=next(r for r in runs if r['status']=='success')
raw=Path(run['output_directory'])/(TAKE+'.mp4')
assert sha(raw)==run['output_sha256']
generated,fps=decode(raw)
gt,_=decode(root/'EgoX/example/egoexo4D/videos'/TAKE/'ego_GT.mp4')
assert generated.shape==(49,448,1232,3) and fps==30
dest=root/'records/output_review';dest.mkdir(exist_ok=True)
for batch,indices in enumerate([list(range(0,16,4)),list(range(16,32,4)),list(range(32,49,4))]):
    sheet=Image.new('RGB',(896,len(indices)*480),(18,24,20))
    draw=ImageDraw.Draw(sheet)
    for row,i in enumerate(indices):
        draw.text((8,row*480+8),f'Frame {i:02d}: OFFICIAL GT',fill='white')
        draw.text((456,row*480+8),f'Frame {i:02d}: REAL EGOX / SEED 42',fill='white')
        sheet.paste(Image.fromarray(gt[i]),(0,row*480+32))
        sheet.paste(Image.fromarray(generated[i,:,784:]),(448,row*480+32))
    sheet.save(dest/f'gt_generated_{batch}.jpg',quality=95)
for i in (0,12,24,36,48):
    Image.fromarray(generated[i,:,784:]).save(dest/f'generated_{i:03d}.png')
probe={'raw_sha256':sha(raw),'raw_frames':49,'fps':fps,'shape':list(generated.shape),'generated_region_xywh':[784,0,448,448],'generated_rgb_min':int(generated[:,:,784:].min()),'generated_rgb_max':int(generated[:,:,784:].max()),'generated_rgb_mean':float(generated[:,:,784:].mean()),'generated_temporal_distinct_frames':len({f.tobytes() for f in generated[:,:,784:]})}
(dest/'probe.json').write_text(json.dumps(probe,indent=2))
print(json.dumps(probe,indent=2))
