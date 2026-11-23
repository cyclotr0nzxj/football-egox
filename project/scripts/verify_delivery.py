"""Verify exported pixels, frame counts and delivered file hashes on CPU."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from build_artifacts import decode, TAKE

root=Path('/root/autodl-tmp/football_egox')
a=root/'artifacts'
source=root/'EgoX/example/egoexo4D/videos'/TAKE
exo,_=decode(source/'exo.mp4');gt,_=decode(source/'ego_GT.mp4')
raw,fps=decode(a/'official_output.mp4')
assert raw.shape==(49,448,1232,3) and fps==30
for i in range(49):
    frame=np.asarray(Image.open(a/'frames'/f'{i:03d}.png'))
    assert np.array_equal(frame[:,:784],exo[i])
    assert np.array_equal(frame[:,784:1232],gt[i])
    assert np.array_equal(frame[:,1232:],raw[i,:,784:])
    assert np.array_equal(np.asarray(Image.open(a/'generated_frames'/f'{i:03d}.png')),raw[i,:,784:])
for name,shape in [('comparison.mp4',(49,528,1680,3)),('generated_ego.mp4',(49,448,448,3))]:
    video,fps=decode(a/name);assert video.shape==shape and fps==30
hashes=json.loads((a/'file_hashes.json').read_text())
for name,expected in hashes.items():
    assert hashlib.sha256((a/name).read_bytes()).hexdigest()==expected,name
result={'cpu_only':True,'triptych_frames_checked':49,'original_exo_pixels_exact':True,'official_gt_pixels_exact':True,'generated_branch_pixels_exact':True,'source_index_mapping':'0..48 without offset','comparison_video':'49 frames, 30fps, 1680x528','generated_video':'49 frames, 30fps, 448x448','artifact_files_hash_verified':len(hashes)}
(root/'records/delivery_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
